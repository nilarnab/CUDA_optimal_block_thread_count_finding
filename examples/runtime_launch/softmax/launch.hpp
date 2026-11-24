#pragma once

#ifndef __CUDA_ARCH__
#include "../runtime_benchmark.hpp"

namespace softmax_launch {

using runtime_benchmark::Configuration;

Configuration select_model(std::size_t vectors, std::size_t width);
void launch_model(Configuration configuration, const float* input, std::size_t vectors,
                  std::size_t width, float* output);

Configuration select_fixed(std::size_t vectors, std::size_t block);
void launch_fixed(Configuration configuration, const float* input, std::size_t vectors,
                  std::size_t width, float* output);

}  // namespace softmax_launch
#endif
