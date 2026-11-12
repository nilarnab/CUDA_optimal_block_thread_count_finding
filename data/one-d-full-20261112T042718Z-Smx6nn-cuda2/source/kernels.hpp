#ifndef PILOT_1D_KERNELS_HPP
#define PILOT_1D_KERNELS_HPP
using uint = unsigned int;
#ifdef HEAVY_CPU_TEST
using std::abs;
#endif
namespace original_add {
#include "kernels/add.cu"
}
namespace original_sum {
#include "kernels/gpuSum.cu"
}
namespace original_tile {
#include "kernels/kTile.cu"
}
namespace original_repeat {
#include "kernels/times.cu"
}
namespace original_transpose {
#include "kernels/flatten_kernel.cu"
}
namespace original_mean {
#include "kernels/mean_kernel.cu"
}
namespace original_variance {
#include "kernels/variance_kernel.cu"
}
namespace original_binary_weights {
#include "kernels/binarize_weights_kernel.cu"
}
namespace original_binary_input {
#include "kernels/binarize_input_kernel.cu"
}
namespace original_softmax {
#include "kernels/softmax_kernel.cu"
}
namespace original_accumulate {
#include "kernels/accumulate_kernel.cu"
}
#ifndef HEAVY_CPU_TEST
#include "nbody.cuh"
#endif
#define HEAVY_FUNCTIONS(X) X(vector_add,original_add::add) X(segmented_sum,original_sum::gpuSum) X(periodic_tile,original_tile::kTile) X(repeat_vector,original_repeat::times) X(matrix_transpose,original_transpose::flatten_kernel) X(mean,original_mean::mean_kernel) X(variance,original_variance::variance_kernel) X(nbody,nbody_force) X(binarize_weights,original_binary_weights::binarize_weights_kernel) X(binarize_input,original_binary_input::binarize_input_kernel) X(softmax,original_softmax::softmax_kernel) X(accumulate,original_accumulate::accumulate_kernel)
#endif
