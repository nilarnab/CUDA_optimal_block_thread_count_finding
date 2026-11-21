#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

__global__ void softmax_kernel(const float* input, std::size_t vectors,
                               std::size_t width, float* output) {
    const auto vector = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vector >= vectors) return;

    const float* logits = input + vector * width;
    float* probabilities = output + vector * width;
    float peak = logits[0];
    for (std::size_t i = 1; i < width; ++i) {
        if (logits[i] > peak) peak = logits[i];
    }

    double total = 0.0;
    for (std::size_t i = 0; i < width; ++i) {
        const float value = expf(logits[i] - peak);
        probabilities[i] = value;
        total += value;
    }
    const auto inverse_total = static_cast<float>(1.0 / total);
    for (std::size_t i = 0; i < width; ++i) {
        probabilities[i] *= inverse_total;
    }
}

#ifndef __CUDA_ARCH__
#include "runtime_benchmark.hpp"
#ifndef RUNTIME_LAUNCH_FIXED
#include "toolcall.hpp"
#endif

using runtime_benchmark::check;
using runtime_benchmark::parse_size;

int main(int argc, char** argv) try {
#ifdef RUNTIME_LAUNCH_FIXED
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s M N FIXED_BLOCK\n", argv[0]);
#else
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s M N\n", argv[0]);
#endif
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
#ifdef RUNTIME_LAUNCH_FIXED
    const auto fixed = runtime_benchmark::fixed_configuration(softmax_kernel, vectors, parse_size(argv[3]));
    const auto select = [fixed] { return fixed; };
#else
    const auto select = [&] {
        return toolcall::model_output(softmax_kernel, {vectors, width, 1, vectors});
    };
#endif
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

    const auto launch = [&](runtime_benchmark::Configuration configuration) {
        const auto [grid, block] = configuration;
        softmax_kernel<<<grid, block>>>(device_input, vectors, width, device_output);
        check(cudaGetLastError());
    };
    launch(initial);
#ifdef RUNTIME_LAUNCH_BENCHMARK
    const auto timings = runtime_benchmark::measure(select, launch);
#endif
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
#ifdef RUNTIME_LAUNCH_BENCHMARK
#ifdef RUNTIME_LAUNCH_FIXED
    constexpr auto mode = "fixed";
#else
    constexpr auto mode = "model";
#endif
    std::printf("%s trials=%d ", mode, runtime_benchmark::trials);
#endif
    std::printf("M=%zu N=%zu elements=%zu grid=%u block=%u verified\n",
                vectors, width, elements, grid, block);
#ifdef RUNTIME_LAUNCH_BENCHMARK
    runtime_benchmark::print_timings(timings);
#endif
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
#endif
