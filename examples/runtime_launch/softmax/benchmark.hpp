#pragma once

#ifndef __CUDA_ARCH__
#include "launch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace softmax_launch {

using runtime_benchmark::check;
using runtime_benchmark::parse_size;

template <bool Fixed, bool Timed = true, class Launch>
int benchmark(int argc, char** argv, Launch launch_kernel) try {
    if (argc != (Fixed ? 4 : 3)) {
        std::fprintf(stderr, "Usage: %s M N%s\n",
                     argv[0], Fixed ? " FIXED_BLOCK" : "");
        return 2;
    }
    const auto vectors = parse_size(argv[1]);
    const auto width = parse_size(argv[2]);
    if (!vectors || !width || vectors > std::numeric_limits<std::size_t>::max() / sizeof(float) / width) {
        std::fprintf(stderr, "M and N must be positive with room for M*N floats\n");
        return 2;
    }

    const auto elements = vectors * width;
    const auto bytes = elements * sizeof(float);
    const auto select = [&] {
        if constexpr (Fixed) {
            const auto fixed = select_fixed(vectors, parse_size(argv[3]));
            return [fixed] { return fixed; };
        } else {
            return [&] { return select_model(vectors, width); };
        }
    }();
    const auto initial = select();

    std::vector<float> input(elements), output(elements);
    for (std::size_t row = 0; row < vectors; ++row) {
        for (std::size_t i = 0; i < width; ++i) {
            input[row * width + i] = 1000.0f + static_cast<float>((row + i) % 11);
        }
    }

    float *device_input, *device_output;
    check(cudaMalloc(&device_input, bytes));
    check(cudaMalloc(&device_output, bytes));
    check(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemset(device_output, 0, bytes));

    const auto launch = [&](Configuration configuration) {
        launch_kernel(configuration, device_input, vectors, width, device_output);
    };
    launch(initial);
    runtime_benchmark::Timings timings{};
    if constexpr (Timed) {
        timings = runtime_benchmark::measure(select, launch);
    }
    check(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(device_input));
    check(cudaFree(device_output));

    for (std::size_t row = 0; row < vectors; ++row) {
        const float* logits = input.data() + row * width;
        const double peak = *std::max_element(logits, logits + width);
        double total = 0.0;
        for (std::size_t i = 0; i < width; ++i) {
            total += std::exp(logits[i] - peak);
        }
        for (std::size_t i = 0; i < width; ++i) {
            const double expected = std::exp(logits[i] - peak) / total;
            const float actual = output[row * width + i];
            if (!(std::abs(actual - expected) <= 1e-6 * expected)) {
                std::fprintf(stderr, "Row %zu column %zu got %g, expected %g\n",
                             row, i, actual, expected);
                return 1;
            }
        }
    }

    const auto [grid, block] = initial;
    if constexpr (Timed) {
        std::printf("%s trials=%d ", Fixed ? "fixed" : "model", runtime_benchmark::trials);
    }
    std::printf("M=%zu N=%zu elements=%zu grid=%u block=%u verified\n",
                vectors, width, elements, grid, block);
    if constexpr (Timed) {
        runtime_benchmark::print_timings(timings);
    }
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}

}  // namespace softmax_launch
#endif
