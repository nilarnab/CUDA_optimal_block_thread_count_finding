#include "kernel.cuh"

#ifndef __CUDA_ARCH__
#include "launch.hpp"

namespace softmax_launch {

Configuration select_fixed(std::size_t vectors, std::size_t block) {
    return runtime_benchmark::fixed_configuration(
        softmax_kernel, vectors, block);
}

void launch_fixed(Configuration configuration, const float* input, std::size_t vectors,
                  std::size_t width, float* output) {
    const auto [grid, block] = configuration;
    softmax_kernel<<<grid, block>>>(input, vectors, width, output);
    runtime_benchmark::check(cudaGetLastError());
}

}  // namespace softmax_launch
#endif
