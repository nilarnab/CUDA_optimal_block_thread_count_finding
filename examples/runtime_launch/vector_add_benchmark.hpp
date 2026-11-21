#pragma once

#include "vector_add.cu"
#include "runtime_benchmark.hpp"

#ifndef __CUDA_ARCH__
#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace vector_add_benchmark {

struct LaunchLimits {
    unsigned block_x, grid_x;
};

inline std::size_t parse_size(std::string_view input) {
    std::size_t value = 0;
    const auto end = input.data() + input.size();
    const auto parsed = std::from_chars(input.data(), end, value);
    return parsed.ec == std::errc{} && parsed.ptr == end ? value : 0;
}

inline void check(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
    std::exit(1);
}

inline LaunchLimits launch_limits() {
    int device = 0;
    cudaDeviceProp properties{};
    cudaFuncAttributes attributes{};
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&properties, device));
    check(cudaFuncGetAttributes(&attributes, vector_add));
    return {
        static_cast<unsigned>(std::min({properties.maxThreadsPerBlock,
                                        properties.maxThreadsDim[0], attributes.maxThreadsPerBlock})),
        static_cast<unsigned>(properties.maxGridSize[0]),
    };
}

template <class Select>
int run(const char* mode, std::size_t side, Select select) try {
    if (!side || side > std::numeric_limits<std::size_t>::max() / sizeof(float) / side) {
        std::fprintf(stderr, "M must be a positive integer with room for M*M floats\n");
        return 2;
    }
    const auto work_items = side * side;
    const auto bytes = work_items * sizeof(float);
    const auto initial = select(side, work_items);

    std::vector<float> host(work_items, 1.0f);
    float *a, *b, *out;
    check(cudaMalloc(&a, bytes));
    check(cudaMalloc(&b, bytes));
    check(cudaMalloc(&out, bytes));
    check(cudaMemcpy(a, host.data(), bytes, cudaMemcpyHostToDevice));
    std::fill(host.begin(), host.end(), 2.0f);
    check(cudaMemcpy(b, host.data(), bytes, cudaMemcpyHostToDevice));

    const auto launch = [&](std::pair<unsigned, unsigned> configuration) {
        const auto [grid, block] = configuration;
        vector_add<<<grid, block>>>(a, b, out, work_items);
        check(cudaGetLastError());
    };
    const auto verify = [&] {
        check(cudaMemcpy(host.data(), out, bytes, cudaMemcpyDeviceToHost));
        if (!std::all_of(host.begin(), host.end(), [](float value) { return value == 3.0f; })) {
            throw std::runtime_error("vector add produced incorrect output");
        }
    };
    check(cudaMemset(out, 0, bytes));
    launch(initial);
    verify();

    const auto timings = runtime_benchmark::measure([&] { return select(side, work_items); }, launch);
    verify();
    check(cudaFree(a));
    check(cudaFree(b));
    check(cudaFree(out));

    const auto [grid, block] = initial;
    std::printf("%s M=%zu elements=%zu grid=%u block=%u trials=%d verified\n",
                mode, side, work_items, grid, block, runtime_benchmark::trials);
    runtime_benchmark::print_timings(timings);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}

}  // namespace vector_add_benchmark
#endif
