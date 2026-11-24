#include "kernel.cuh"

#ifndef __CUDA_ARCH__
#include "launch.hpp"
#include "toolcall.hpp"

namespace softmax_launch {

Configuration select_model(std::size_t vectors, std::size_t width) {
    return toolcall::model_output(softmax_kernel,
                                  {vectors, width, 1, vectors});
}

void launch_model(Configuration configuration, const float* input, std::size_t vectors,
                  std::size_t width, float* output) {
    const auto [grid, block] = configuration;
    softmax_kernel<<<grid, block>>>(input, vectors, width, output);
    runtime_benchmark::check(cudaGetLastError());
}

}  // namespace softmax_launch
#endif
