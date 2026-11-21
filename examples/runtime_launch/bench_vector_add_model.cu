#include "vector_add_benchmark.hpp"

#ifndef __CUDA_ARCH__
#include "toolcall.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s M\n", argv[0]);
        return 2;
    }
    return vector_add_benchmark::run(
        "model", vector_add_benchmark::parse_size(argv[1]),
        [](std::size_t m, std::size_t work_items) {
            return toolcall::model_output(vector_add, {m, 1, 1, work_items});
        });
}
#endif
