#include <cuda_runtime.h>

namespace {
constexpr int tile_width = 32;
constexpr int warp_threads = 32;
static_assert(warp_threads % tile_width == 0);
}

// Adapted from LS-CAT 6580/8. Threads cover tile_width columns and blockDim.y rows.
__global__ __launch_bounds__(1024) void matrixMultiplyAndRelu(
    const float* weights, const float* activations, float* output, const float* bias,
    float* preactivation,
    int rows, int columns, int inner) {
    extern __shared__ float shared[];
    const int threads = tile_width * blockDim.y;
    float* weight_tile = shared;
    float* activation_tile = shared + threads;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tile_index = ty * tile_width + tx;
    const int row = blockIdx.y * blockDim.y + ty;
    const int column_start = blockIdx.x * tile_width;
    const int col = column_start + tx;
    const int tiles = 1 + (inner - 1) / tile_width;
    float value = 0.0f;

    for (int tile = 0; tile < tiles; ++tile) {
        const int k_start = tile * tile_width;
        const int weight_col = k_start + tx;
        weight_tile[tile_index] = row < rows && weight_col < inner
            ? weights[row * inner + weight_col] : 0.0f;

        for (int index = tile_index; index < tile_width * tile_width; index += threads) {
            const int activation_row = k_start + index / tile_width;
            const int activation_col = column_start + index % tile_width;
            activation_tile[index] = activation_row < inner && activation_col < columns
                ? activations[activation_row * columns + activation_col] : 0.0f;
        }
        __syncthreads();
        // Partially unroll the 32-step reduction to limit the generated code size.
        #pragma unroll 16
        for (int k = 0; k < tile_width; ++k) {
            value += weight_tile[ty * tile_width + k] * activation_tile[k * tile_width + tx];
        }
        __syncthreads();
    }

    if (row < rows && col < columns) {
        const int index = row * columns + col;
        const float biased = value + bias[row];
        preactivation[index] = biased;
        output[index] = fmaxf(biased, 0.0f);
    }
}

#ifndef __CUDA_ARCH__
#include "../runtime_benchmark.hpp"
#include "toolcall.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {
using runtime_benchmark::check;
using runtime_benchmark::parse_size;

unsigned grid_rows(std::size_t rows, unsigned threads) {
    return static_cast<unsigned>(1 + (rows - 1) / (threads / tile_width));
}
}

int main(int argc, char** argv) try {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s M N K\n", argv[0]);
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
    const auto select = [&] {
        return toolcall::model_output(matrixMultiplyAndRelu, {rows, columns, inner, work_items}).second;
    };
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
        const dim3 grid(grid_x, grid_rows(rows, threads));
        const dim3 block(tile_width, threads / tile_width);
        const auto shared_bytes = (threads + tile_width * tile_width) * sizeof(float);
        matrixMultiplyAndRelu<<<grid, block, shared_bytes>>>(
            device_weights, device_activations, device_output, device_bias, device_preactivation,
            static_cast<int>(rows), static_cast<int>(columns), static_cast<int>(inner));
        check(cudaGetLastError());
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
                "model", runtime_benchmark::trials, rows, columns, inner, grid_x, grid_rows(rows, initial), initial, samples);
    runtime_benchmark::print_timings(timings);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
#endif
