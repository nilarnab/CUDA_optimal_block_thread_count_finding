#ifndef PILOT_1D_DISPATCH_HPP
#define PILOT_1D_DISPATCH_HPP
#include "reference.hpp"
#include "kernels.hpp"
namespace heavy {
inline void legal(const Shape& s,dim3 grid,dim3 block) {
    const auto c=counts(s);
    if(grid.y!=1 || grid.z!=1 || block.y!=1 || block.z!=1 ||
       block.x<64 || block.x>1024 || block.x%64 || !grid.x)
        throw std::invalid_argument("requires one of the sixteen 1D blocks and a 1D grid");
    const std::uint64_t work=elementwise(s.id)?c.out:s.m;
    if(grid.x!=(work+block.x-1)/block.x)
        throw std::invalid_argument("requires the exact coverage grid");
}
#ifdef HEAVY_CPU_TEST
#define LAUNCH(fn,...) fn(__VA_ARGS__)
#else
#define LAUNCH(fn,...) fn<<<grid,block,0,stream>>>(__VA_ARGS__)
#endif
template<class T> void dispatch(const Shape& s,dim3 grid,dim3 block,cudaStream_t stream,T* a,T* b,T* out) {
    float* fa=reinterpret_cast<float*>(a);float* fb=reinterpret_cast<float*>(b);float* fo=reinterpret_cast<float*>(out);
    if(s.id=="vector_add")LAUNCH(original_add::add,s.m*s.m,fa,fb,fo);
    else if(s.id=="segmented_sum")LAUNCH(original_sum::gpuSum,reinterpret_cast<int*>(a),reinterpret_cast<int*>(out),s.m,s.k,s.m);
    else if(s.id=="periodic_tile")LAUNCH(original_tile::kTile,fa,fo,17,13,s.m,s.m);
    else if(s.id=="repeat_vector")LAUNCH(original_repeat::times,fa,s.m,fo,s.m);
    else if(s.id=="matrix_transpose")LAUNCH(original_transpose::flatten_kernel,s.m*s.m,fa,s.m,s.m,1,1,fo);
    else if(s.id=="mean")LAUNCH(original_mean::mean_kernel,fa,1,s.m,s.k,fo);
    else if(s.id=="variance")LAUNCH(original_variance::variance_kernel,fa,fb,1,s.m,s.k,fo);
    else if(s.id=="binarize_weights")LAUNCH(original_binary_weights::binarize_weights_kernel,fa,s.m,s.k,fo);
    else if(s.id=="binarize_input")LAUNCH(original_binary_input::binarize_input_kernel,fa,s.k,s.m,fo);
    else if(s.id=="softmax")LAUNCH(original_softmax::softmax_kernel,s.k,s.k,s.m,fa,1.0f,fo);
    else if(s.id=="accumulate")LAUNCH(original_accumulate::accumulate_kernel,fa,s.k,s.m,fo);
#ifndef HEAVY_CPU_TEST
    else if(s.id=="nbody")nbody_force<<<grid,block,block.x*sizeof(float4),stream>>>(reinterpret_cast<const float4*>(fa),fo,s.m);
#endif
    else throw std::invalid_argument("unknown dispatch");
}
#undef LAUNCH
}
#endif
