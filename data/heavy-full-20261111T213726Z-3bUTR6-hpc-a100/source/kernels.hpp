#ifndef HEAVY_KERNELS_HPP
#define HEAVY_KERNELS_HPP
namespace original_gemm {
#include "kernels/matrixMultiply.cu"
}
namespace original_sum {
#include "kernels/gpuSum.cu"
}
namespace original_mean {
#include "kernels/mean_kernel.cu"
}
namespace original_variance {
#include "kernels/variance_kernel.cu"
}
namespace polybench {
#define DATA_TYPE float
#define _PB_NI ni
#define _PB_NJ nj
#define NI ni
#define NJ nj
#include "kernels/syrk_kernel.cuh"
#undef _PB_NI
#undef _PB_NJ
#undef NI
#undef NJ
#undef DATA_TYPE
}
#ifndef HEAVY_CPU_TEST
#include "nbody.cuh"
#endif
#define HEAVY_FUNCTIONS(X) X(gemm,original_gemm::matrixMultiply) X(segmented_sum,original_sum::gpuSum) X(mean,original_mean::mean_kernel) X(variance,original_variance::variance_kernel) X(syrk,polybench::syrk_kernel) X(nbody,nbody_force)
#endif
