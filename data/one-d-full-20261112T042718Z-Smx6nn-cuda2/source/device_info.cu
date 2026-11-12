#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    char* end = nullptr;
    const long selected = argc == 2 ? std::strtol(argv[1], &end, 10) : -1;
    int count = 0;
    cudaError_t result = cudaGetDeviceCount(&count);
    if (result != cudaSuccess) {
        std::cerr << "CUDA device discovery failed: " << cudaGetErrorString(result) << '\n';
        return 1;
    }
    if (argc != 2 || !end || end == argv[1] || *end || selected < 0 || selected >= count) {
        std::cerr << "Select a visible CUDA logical device from 0 through " << count - 1 << '\n';
        return 1;
    }
    cudaDeviceProp properties{};
    result = cudaGetDeviceProperties(&properties, static_cast<int>(selected));
    if (result != cudaSuccess) {
        std::cerr << "CUDA property query failed: " << cudaGetErrorString(result) << '\n';
        return 1;
    }
    std::cout << properties.major << properties.minor << '\n';
}
