#pragma once

#ifndef __CUDA_ARCH__
#include "launch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace matrix_multiply_and_relu_launch {

using runtime_benchmark::check;
using runtime_benchmark::parse_size;

template <bool Fixed, class Launch>
int benchmark(int argc, char** argv, Launch launch_kernel) try {
    if (argc != (Fixed ? 5 : 4)) {
        std::fprintf(stderr, "Usage: %s M N K%s\n",
                     argv[0], Fixed ? " FIXED_BLOCK" : "");
        return 2;
    }

    const auto rows = parse_size(argv[1]);
    const auto columns = parse_size(argv[2]);
    const auto inner = parse_size(argv[3]);
    constexpr auto max_index = std::size_t(std::numeric_limits<int>::max());
    if (!rows || !columns || !inner || rows > max_index / columns ||
        rows > max_index / inner || inner > max_index / columns) {
        std::fprintf(stderr, "M, N and K must be positive and each matrix must fit 32-bit indexing\n");
        return 2;
    }

    constexpr unsigned min_tile_rows = warp_threads / tile_width;
    const auto grid_x = static_cast<unsigned>(1 + (columns - 1) / tile_width);
    int device;
    cudaDeviceProp properties{};
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&properties, device));
    if (grid_x > static_cast<unsigned>(properties.maxGridSize[0]) ||
        1 + (rows - 1) / min_tile_rows > static_cast<unsigned>(properties.maxGridSize[1])) {
        std::fprintf(stderr, "M or N exceeds the supported launch dimensions\n");
        return 2;
    }

    const auto work_items = rows * columns;
    const auto requested = Fixed ? parse_size(argv[4]) : 0;
    if (Fixed && (!requested || requested % warp_threads)) {
        std::fprintf(stderr, "FIXED_BLOCK must be a positive multiple of 32\n");
        return 2;
    }
    constexpr auto mode = Fixed ? "fixed" : "model";
    const auto select = [&] {
        if constexpr (Fixed) {
            const auto fixed = select_fixed(work_items, requested);
            return [fixed] { return fixed; };
        } else {
            return [&] { return select_model(rows, columns, inner, work_items); };
        }
    }();
    const auto initial = select();

    std::vector<float> weights(rows * inner), activations(inner * columns), bias(rows), output(work_items);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 8;
    }
    for (std::size_t i = 0; i < activations.size(); ++i) {
        activations[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 8;
    }
    for (std::size_t i = 0; i < bias.size(); ++i) {
        bias[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 8;
    }

    const auto upload = [](const std::vector<float>& host) {
        float* device_data;
        const auto bytes = host.size() * sizeof(float);
        check(cudaMalloc(&device_data, bytes));
        check(cudaMemcpy(device_data, host.data(), bytes, cudaMemcpyHostToDevice));
        return device_data;
    };
    float* device_weights = upload(weights);
    float* device_activations = upload(activations);
    float* device_bias = upload(bias);
    float* device_output;
    const auto output_bytes = output.size() * sizeof(float);
    check(cudaMalloc(&device_output, output_bytes));
    check(cudaMemset(device_output, 0xff, output_bytes));
    std::vector<float> preactivation(work_items);
    float* device_preactivation;
    check(cudaMalloc(&device_preactivation, output_bytes));
    check(cudaMemset(device_preactivation, 0xff, output_bytes));

    const auto launch = [&](unsigned threads) {
        launch_kernel(threads, grid_x, device_weights, device_activations, device_output, device_bias,
                      device_preactivation, static_cast<int>(rows),
                      static_cast<int>(columns), static_cast<int>(inner));
    };
    launch(initial);
    const auto timings = runtime_benchmark::measure(select, launch);
    check(cudaMemcpy(output.data(), device_output, output_bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(device_weights));
    check(cudaFree(device_activations));
    check(cudaFree(device_bias));
    check(cudaFree(device_output));

    check(cudaMemcpy(preactivation.data(), device_preactivation, output_bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(device_preactivation));
    for (std::size_t i = 0; i < work_items; ++i) {
        if (!std::isfinite(preactivation[i]) || output[i] != std::max(0.0f, preactivation[i])) {
            std::fprintf(stderr, "Output %zu has invalid preactivation or ReLU values\n", i);
            return 1;
        }
    }
    // shortcut: large runs compare 64 outputs to the CPU, increase coverage for accuracy studies.
    const std::size_t samples = output.size() <= (1 << 20) / inner
        ? output.size() : std::min(output.size(), std::size_t{64});
    for (std::size_t i = 0; i < samples; ++i) {
        const auto index = samples == 1 ? 0 : i * (output.size() - 1) / (samples - 1);
        const auto row = index / columns;
        const auto col = index % columns;
        double value = bias[row];
        for (std::size_t k = 0; k < inner; ++k) {
            value += static_cast<double>(weights[row * inner + k]) * activations[k * columns + col];
        }
        const double expected = value;
        const double tolerance = 1e-5 * std::max(1.0, std::abs(expected));
        const float actual = preactivation[index];
        if (std::abs(actual - expected) > tolerance) {
            std::fprintf(stderr, "Output [%zu,%zu] got %g, expected %g\n", row, col, actual, expected);
            return 1;
        }
    }

    std::printf("%s trials=%d M=%zu N=%zu K=%zu grid=%u grid_y=%u block=%u checked=%zu verified\n",
                mode, runtime_benchmark::trials, rows, columns, inner, grid_x, grid_rows(rows, initial), initial, samples);
    runtime_benchmark::print_timings(timings);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}

}  // namespace matrix_multiply_and_relu_launch
#endif
