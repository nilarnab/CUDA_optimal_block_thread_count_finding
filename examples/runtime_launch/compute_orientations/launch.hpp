#pragma once

namespace {
constexpr unsigned lanes = 32;
constexpr unsigned max_block = 576;
}

#ifndef __CUDA_ARCH__
#include "../runtime_benchmark.hpp"

namespace compute_orientations_launch {

using runtime_benchmark::Configuration;

Configuration select_model(std::size_t count, std::size_t side, std::size_t work_items);
void launch_model(Configuration configuration, const float* image, const int* points,
                  float* output, int count, int side);

Configuration select_fixed(std::size_t work_items, std::size_t block);
void launch_fixed(Configuration configuration, const float* image, const int* points,
                  float* output, int count, int side);

}  // namespace compute_orientations_launch
#endif
