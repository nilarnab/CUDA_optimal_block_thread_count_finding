#include <algorithm>

#include "vector_add.cu"

extern "C" const char* launch_error(int status) {
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}

extern "C" int launch_limits(int* max_threads, int* max_grid_x) {
    int device;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    cudaDeviceProp props;
    status = cudaGetDeviceProperties(&props, device);
    if (status != cudaSuccess) return status;
    cudaFuncAttributes attrs;
    status = cudaFuncGetAttributes(&attrs, vector_add);
    if (status != cudaSuccess) return status;
    *max_threads = std::min({attrs.maxThreadsPerBlock, props.maxThreadsPerBlock, props.maxThreadsDim[0]});
    *max_grid_x = props.maxGridSize[0];
    return cudaSuccess;
}

extern "C" int launch_vector_add(
    const float* a, const float* b, float* out, size_t n, int threads, cudaStream_t stream
) {
    // The host already checked the active device's block and grid limits.
    if (n == 0 || threads <= 0) return cudaErrorInvalidValue;
    const auto grid = 1 + (n - 1) / threads;
    vector_add<<<static_cast<unsigned>(grid), threads, 0, stream>>>(a, b, out, n);
    return cudaGetLastError();
}
