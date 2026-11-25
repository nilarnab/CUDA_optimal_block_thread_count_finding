#pragma once

#include <cstdint>
#include <utility>

namespace toolcall {

struct ProblemSize {
    std::uint64_t m, n, k;
    std::uint64_t work_items;
};

std::pair<unsigned, unsigned> model_output(const void* kernel, ProblemSize problem);

template <class... Args>
std::pair<unsigned, unsigned> model_output(void (*kernel)(Args...), ProblemSize problem) {
    return model_output(reinterpret_cast<const void*>(kernel), problem);
}

}
