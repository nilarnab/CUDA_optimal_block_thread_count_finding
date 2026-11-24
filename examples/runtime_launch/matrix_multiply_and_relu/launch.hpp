#pragma once

namespace {
constexpr int tile_width = 32;
constexpr int warp_threads = 32;
static_assert(warp_threads % tile_width == 0);
}

#ifndef __CUDA_ARCH__
#include "../runtime_benchmark.hpp"

namespace matrix_multiply_and_relu_launch {

inline unsigned grid_rows(std::size_t rows, unsigned threads) {
    return static_cast<unsigned>(1 + (rows - 1) / (threads / tile_width));
}

unsigned select_model(std::size_t rows, std::size_t columns, std::size_t inner,
                      std::size_t work_items);
void launch_model(unsigned threads, unsigned grid_x, const float* weights, const float* activations,
                  float* output, const float* bias, float* preactivation,
                  int rows, int columns, int inner);

unsigned select_fixed(std::size_t work_items, std::size_t block);
void launch_fixed(unsigned threads, unsigned grid_x, const float* weights, const float* activations,
                  float* output, const float* bias, float* preactivation,
                  int rows, int columns, int inner);

}  // namespace matrix_multiply_and_relu_launch
#endif
