#pragma once

#ifndef __CUDA_ARCH__
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace runtime_benchmark {

using Configuration = std::pair<unsigned, unsigned>;

inline void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

inline std::size_t parse_size(std::string_view input) {
    std::size_t value = 0;
    const char* end = input.data() + input.size();
    const auto parsed = std::from_chars(input.data(), end, value);
    return parsed.ec == std::errc{} && parsed.ptr == end ? value : 0;
}

template <class Kernel>
Configuration fixed_configuration(Kernel kernel, std::size_t work_items, std::size_t block) {
    int device;
    cudaDeviceProp properties{};
    cudaFuncAttributes attributes{};
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&properties, device));
    check(cudaFuncGetAttributes(&attributes, kernel));
    const auto limit = std::min({properties.maxThreadsPerBlock,
                                 properties.maxThreadsDim[0], attributes.maxThreadsPerBlock});
    if (!block || block > static_cast<unsigned>(limit)) {
        throw std::invalid_argument("FIXED_BLOCK must be a positive number within the block limit");
    }
    const auto grid = 1 + (work_items - 1) / block;
    if (grid > static_cast<unsigned>(properties.maxGridSize[0])) {
        throw std::invalid_argument("FIXED_BLOCK produces a grid larger than the device limit");
    }
    return {static_cast<unsigned>(grid), static_cast<unsigned>(block)};
}

constexpr int warmups = 5, trials = 51;

struct Timings {
    double gpu_us, selection_us, total_us;
};

template <class Select, class Launch>
Timings measure(Select select, Launch launch) {
    using Clock = std::chrono::steady_clock;
    using Microseconds = std::chrono::duration<double, std::micro>;
    std::array<double, trials> selection_us{}, gpu_us{}, total_us{};
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&stop));
    for (int trial = 0; trial < warmups + trials; ++trial) {
        const auto begin = Clock::now();
        const auto configuration = select();
        const auto selected = Clock::now();
        launch(configuration);
        check(cudaStreamSynchronize(nullptr));
        const auto done = Clock::now();

        // A separate launch keeps event instrumentation out of the host timer.
        check(cudaEventRecord(start));
        launch(configuration);
        check(cudaEventRecord(stop));
        check(cudaEventSynchronize(stop));
        float elapsed_ms = 0;
        check(cudaEventElapsedTime(&elapsed_ms, start, stop));
        if (trial < warmups) continue;

        const auto index = trial - warmups;
        selection_us[index] = Microseconds(selected - begin).count();
        gpu_us[index] = 1000.0 * elapsed_ms;
        total_us[index] = Microseconds(done - begin).count();
    }
    check(cudaEventDestroy(start));
    check(cudaEventDestroy(stop));

    const auto median = [](auto& values) {
        auto middle = values.begin() + values.size() / 2;
        std::nth_element(values.begin(), middle, values.end());
        return *middle;
    };
    return {median(gpu_us), median(selection_us), median(total_us)};
}

inline void print_timings(Timings timings) {
    std::printf("gpu_us=%.3f selection_us=%.3f total_us=%.3f\n",
                timings.gpu_us, timings.selection_us, timings.total_us);
}

}  // namespace runtime_benchmark
#endif
