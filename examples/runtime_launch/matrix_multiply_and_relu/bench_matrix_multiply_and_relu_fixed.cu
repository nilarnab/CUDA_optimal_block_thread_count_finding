#ifndef __CUDA_ARCH__
#include "benchmark.hpp"

int main(int argc, char** argv) {
    return matrix_multiply_and_relu_launch::benchmark<true>(argc, argv,
        matrix_multiply_and_relu_launch::launch_fixed);
}
#endif
