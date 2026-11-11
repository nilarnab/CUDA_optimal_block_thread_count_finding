#ifndef HEAVY_DISPATCH_HPP
#define HEAVY_DISPATCH_HPP
#include "reference.hpp"
#include "kernels.hpp"
namespace heavy {
inline void legal(const Shape& s,dim3 grid,dim3 block) {
    counts(s);
    if(!grid.x || !grid.y || grid.z!=1 || !block.x || !block.y || block.z!=1)throw std::invalid_argument("invalid launch dimensions");
    if(matrix(s.id)) {
        const std::uint64_t x=(s.n+block.x-1)/block.x,y=(s.m+block.y-1)/block.y;
        if(grid.x<x || grid.y<y || std::uint64_t(grid.x)*grid.y>2*x*y)throw std::invalid_argument("matrix grid skips work or exceeds search space");
    }else {
        if(block.y!=1)throw std::invalid_argument("linear block requires y=1");
        const std::uint64_t needed=(s.m+block.x-1)/block.x,count=std::uint64_t(grid.x)*grid.y;
        if((s.id=="segmented_sum" || s.id=="nbody") && grid.y!=1)throw std::invalid_argument("kernel requires grid_y=1");
        if(count>2*needed || (s.id!="nbody" && count<needed))throw std::invalid_argument("grid skips work or exceeds search space");
    }
}
#ifdef HEAVY_CPU_TEST
#define LAUNCH(fn,...) fn(__VA_ARGS__)
#else
#define LAUNCH(fn,...) fn<<<grid,block,0,stream>>>(__VA_ARGS__)
#endif
template<class T> void dispatch(const Shape& s,dim3 grid,dim3 block,cudaStream_t stream,T* a,T* b,T* out) {
    float* fa=reinterpret_cast<float*>(a);float* fb=reinterpret_cast<float*>(b);float* fo=reinterpret_cast<float*>(out);
    if(s.id=="gemm")LAUNCH(original_gemm::matrixMultiply,fa,fb,fo,s.m,s.k,s.k,s.n,s.m,s.n);
    else if(s.id=="segmented_sum")LAUNCH(original_sum::gpuSum,reinterpret_cast<int*>(a),reinterpret_cast<int*>(out),s.m,s.k,s.m);
    else if(s.id=="mean")LAUNCH(original_mean::mean_kernel,fa,1,s.m,s.k,fo);
    else if(s.id=="variance")LAUNCH(original_variance::variance_kernel,fa,fb,1,s.m,s.k,fo);
    else if(s.id=="syrk")LAUNCH(polybench::syrk_kernel,s.m,s.k,1.0f,.5f,fa,fo);
#ifndef HEAVY_CPU_TEST
    else if(s.id=="nbody")nbody_force<<<grid,block,block.x*sizeof(float4),stream>>>(reinterpret_cast<const float4*>(fa),fo,s.m);
#endif
    else throw std::invalid_argument("unknown dispatch");
}
#undef LAUNCH
}
#endif
