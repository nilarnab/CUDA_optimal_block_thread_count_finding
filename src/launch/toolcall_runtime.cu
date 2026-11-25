#ifndef __CUDA_ARCH__
#include "toolcall_kernel_data.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace toolcall {
namespace {

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

}

std::pair<unsigned, unsigned> model_output(const void* kernel, ProblemSize problem) {
    if (!kernel) throw std::invalid_argument("model_output expects a CUDA kernel");
    struct CachedLimits {
        const void* kernel{};
        int device = -1;
        detail::LaunchLimits limits{};
    };
    static thread_local CachedLimits cached;
    int device;
    check(cudaGetDevice(&device));
    if (cached.device != device || cached.kernel != kernel) {
        cudaDeviceProp properties{};
        cudaFuncAttributes attributes{};
        check(cudaGetDeviceProperties(&properties, device));
        check(cudaFuncGetAttributes(&attributes, kernel));
        cached = {kernel, device, {
            static_cast<unsigned>(std::min({properties.maxThreadsPerBlock,
                                            properties.maxThreadsDim[0],
                                            attributes.maxThreadsPerBlock})),
            static_cast<unsigned>(properties.maxGridSize[0]),
        }};
    }
    return detail::choose_launch(problem, detail::compiled_kernel, cached.limits);
}

}
#endif
