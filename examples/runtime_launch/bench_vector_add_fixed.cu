#include "vector_add_benchmark.hpp"

#ifndef __CUDA_ARCH__
int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s M FIXED_BLOCK\n", argv[0]);
        return 2;
    }
    const auto block = vector_add_benchmark::parse_size(argv[2]);
    const auto limits = vector_add_benchmark::launch_limits();
    return vector_add_benchmark::run(
        "fixed", vector_add_benchmark::parse_size(argv[1]),
        [block, limits](std::size_t, std::size_t work_items) {
            if (!block || block > limits.block_x) {
                throw std::invalid_argument("FIXED_BLOCK must be a positive number within the block limit");
            }
            const auto grid = 1 + (work_items - 1) / block;
            if (grid > limits.grid_x) {
                throw std::invalid_argument("FIXED_BLOCK produces a grid larger than the device limit");
            }
            return std::pair{static_cast<unsigned>(grid), static_cast<unsigned>(block)};
        });
}
#endif
