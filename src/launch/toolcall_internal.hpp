#pragma once

#include "toolcall.hpp"

#include <cstddef>

namespace toolcall::detail {

struct Kernel {
    const float* states;
    const float* gates;  // Raw for AOTInductor, logarithms for the native scorer.
    std::size_t rows;
    std::size_t width;
};

struct LaunchLimits {
    unsigned max_block_x;
    unsigned max_grid_x;
};

std::pair<unsigned, unsigned> choose_launch(
    ProblemSize problem, const Kernel& kernel, LaunchLimits limits);

}  // namespace toolcall::detail
