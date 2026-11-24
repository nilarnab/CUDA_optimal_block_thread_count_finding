#ifndef __CUDA_ARCH__
#include "benchmark.hpp"

int main(int argc, char** argv) {
    return compute_orientations_launch::benchmark<true>(argc, argv,
        compute_orientations_launch::launch_fixed);
}
#endif
