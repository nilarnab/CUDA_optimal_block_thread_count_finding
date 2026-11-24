#include "kernel.cuh"

#ifndef __CUDA_ARCH__
#include "launch.hpp"
#include "toolcall.hpp"

namespace compute_orientations_launch {

Configuration select_model(std::size_t count, std::size_t side, std::size_t work_items) {
    return toolcall::model_output(ComputeOrientations,
                                  {count, side, side, work_items});
}

void launch_model(Configuration configuration, const float* image, const int* points,
                  float* output, int count, int side) {
    const auto [grid, block] = configuration;
    const auto shared_bytes = (block / lanes) * scratch_per_warp * sizeof(float);
    ComputeOrientations<<<grid, block, shared_bytes>>>(image, points, output, count, side, side);
    runtime_benchmark::check(cudaGetLastError());
}

}  // namespace compute_orientations_launch
#endif
