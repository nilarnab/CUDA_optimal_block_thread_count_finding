#ifndef __CUDA_ARCH__
#include "benchmark.hpp"

int main(int argc, char** argv) {
    return softmax_launch::benchmark<true>(argc, argv,
        softmax_launch::launch_fixed);
}
#endif
