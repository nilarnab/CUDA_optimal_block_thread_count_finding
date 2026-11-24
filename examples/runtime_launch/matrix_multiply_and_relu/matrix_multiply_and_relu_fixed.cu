#include "kernel.cuh"

#ifndef __CUDA_ARCH__
#include "launch.hpp"

namespace matrix_multiply_and_relu_launch {

unsigned select_fixed(std::size_t work_items, std::size_t block) {
    return runtime_benchmark::fixed_configuration(
        matrixMultiplyAndRelu, work_items, block).second;
}

void launch_fixed(unsigned threads, unsigned grid_x, const float* weights, const float* activations,
                  float* output, const float* bias, float* preactivation,
                  int rows, int columns, int inner) {
    const dim3 grid(grid_x, grid_rows(rows, threads));
    const dim3 block(tile_width, threads / tile_width);
    const auto shared_bytes = (threads + tile_width * tile_width) * sizeof(float);
    matrixMultiplyAndRelu<<<grid, block, shared_bytes>>>(
        weights, activations, output, bias, preactivation, rows, columns, inner);
    runtime_benchmark::check(cudaGetLastError());
}

}  // namespace matrix_multiply_and_relu_launch
#endif
