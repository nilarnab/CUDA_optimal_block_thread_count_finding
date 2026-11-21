#include <cuda_runtime.h>

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

__global__ void accumulate_kernel(const float* input, std::size_t rows,
                                  std::size_t groups, float* output) {
    const auto group = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group >= groups) return;

    float sum = 0.0f;
    for (std::size_t row = 0; row < rows; ++row) {
        sum += input[row * groups + group];
    }
    output[group] = sum;
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
    const auto groups = parse_size(argv[1]);
    const auto rows = parse_size(argv[2]);
    if (!groups || !rows || groups > std::numeric_limits<std::size_t>::max() / sizeof(float) / rows) {
        std::fprintf(stderr, "M and N must be positive with room for M*N floats\n");
        return 2;
    }
    const auto elements = groups * rows;
    const auto input_bytes = elements * sizeof(float);
    const auto output_bytes = groups * sizeof(float);
#ifdef RUNTIME_LAUNCH_FIXED
    const auto fixed = runtime_benchmark::fixed_configuration(accumulate_kernel, groups, parse_size(argv[3]));
    const auto select = [fixed] { return fixed; };
#else
    const auto select = [&] {
        return toolcall::model_output(accumulate_kernel, {groups, rows, 1, groups});
    };
#endif
    const auto initial = select();

    std::vector<float> input(elements), expected(groups, 0.0f), output(groups);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t group = 0; group < groups; ++group) {
            const auto value = static_cast<float>((row + group) % 7 + 1);
            input[row * groups + group] = value;
            expected[group] += value;
        }
    }

    float *device_input, *device_output;
    check(cudaMalloc(&device_input, input_bytes));
    check(cudaMalloc(&device_output, output_bytes));
    check(cudaMemcpy(device_input, input.data(), input_bytes, cudaMemcpyHostToDevice));
    check(cudaMemset(device_output, 0, output_bytes));

    const auto launch = [&](runtime_benchmark::Configuration configuration) {
        const auto [grid, block] = configuration;
        accumulate_kernel<<<grid, block>>>(device_input, rows, groups, device_output);
        check(cudaGetLastError());
    };
    launch(initial);
#ifdef RUNTIME_LAUNCH_BENCHMARK
    const auto timings = runtime_benchmark::measure(select, launch);
#endif
    check(cudaMemcpy(output.data(), device_output, output_bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(device_input));
    check(cudaFree(device_output));

    for (std::size_t group = 0; group < groups; ++group) {
        if (output[group] != expected[group]) {
            std::fprintf(stderr, "Group %zu got %g, expected %g\n", group, output[group], expected[group]);
            return 1;
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
    std::printf("M=%zu N=%zu elements=%zu outputs=%zu grid=%u block=%u verified\n",
                groups, rows, elements, groups, grid, block);
#ifdef RUNTIME_LAUNCH_BENCHMARK
    runtime_benchmark::print_timings(timings);
#endif
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
#endif
