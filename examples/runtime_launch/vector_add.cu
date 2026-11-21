#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* out, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
