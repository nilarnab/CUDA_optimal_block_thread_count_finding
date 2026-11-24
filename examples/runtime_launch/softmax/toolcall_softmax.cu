#ifndef __CUDA_ARCH__
#include "benchmark.hpp"

int main(int argc, char** argv) {
    return softmax_launch::benchmark<false, false>(argc, argv,
        softmax_launch::launch_model);
}
#endif
