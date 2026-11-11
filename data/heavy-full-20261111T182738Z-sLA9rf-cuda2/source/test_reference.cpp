#define HEAVY_CPU_TEST
#define __global__
#define __host__
#define __device__
#include <cmath>
#include <iostream>
struct dim3 {unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
static dim3 blockIdx,threadIdx,blockDim,gridDim;
using cudaStream_t=void*;
struct float4{float x,y,z,w;};
inline float4 make_float4(float x,float y,float z,float w){return {x,y,z,w};}
inline float rsqrtf(float x){return 1.0f/std::sqrt(x);}
#include "dispatch.hpp"
#include "nbody.cuh"
static void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
template<class T> static void check(const heavy::Shape& s,dim3 grid,dim3 block) {
    const auto d=heavy::make_case<T>(s,12345);auto a=d.a,b=d.b,out=heavy::initial(d);
    heavy::legal(s,grid,block);gridDim=grid;blockDim=block;
    for(blockIdx.y=0;blockIdx.y<grid.y;blockIdx.y++)for(blockIdx.x=0;blockIdx.x<grid.x;blockIdx.x++)
        for(threadIdx.y=0;threadIdx.y<block.y;threadIdx.y++)for(threadIdx.x=0;threadIdx.x<block.x;threadIdx.x++)
            heavy::dispatch(s,grid,block,nullptr,a.data(),b.data(),out.data()+16);
    heavy::validate(d,out);require(a==d.a&&b==d.b,"input modified");
    auto broken=out;broken[16]=T(123456);bool rejected=false;
    try{heavy::validate(d,broken);}catch(const std::runtime_error&){rejected=true;}require(rejected,"incorrect value accepted");
    broken=out;broken[0]=0;rejected=false;
    try{heavy::validate(d,broken);}catch(const std::runtime_error&){rejected=true;}require(rejected,"guard overwrite accepted");
}
static void nbody_check(int n) {
    const auto d=heavy::make_case<float>({"nbody",n,1,1},42);auto out=heavy::initial(d);
    for(int i=0;i<n;i++) {
        const float4 p=make_float4(d.a[4*i],d.a[4*i+1],d.a[4*i+2],d.a[4*i+3]);float ax=0,ay=0,az=0;
        for(int j=0;j<n;j++)heavy_interaction(p,make_float4(d.a[4*j],d.a[4*j+1],d.a[4*j+2],d.a[4*j+3]),ax,ay,az);
        out[16+3*i]=ax;out[16+3*i+1]=ay;out[16+3*i+2]=az;
    }
    heavy::validate(d,out);
    if(n==2) {
        auto two=d;two.a={-.5f,0,0,.5f,.5f,0,0,.5f};
        require(std::abs(heavy::expected_at(two,0)-.5/std::pow(1.01,1.5))<1e-12,"analytic N-body result differs");
        require(std::abs(heavy::expected_at(two,3)+.5/std::pow(1.01,1.5))<1e-12,"pair force symmetry differs");
    }
}
int main(){try{
    for(int n:{5,17,35}) {
        check<float>({"gemm",n,n+2,n+4},dim3((n+2+7)/8,(n+7)/8),dim3(8,8));
        check<float>({"syrk",n,n,n+4},dim3(2*((n+7)/8),(n+7)/8),dim3(8,8));
        check<int>({"segmented_sum",n,1,67},dim3(2*((n+7)/8)),dim3(8));
        for(const auto& id:{"mean","variance"}) {
            check<float>({id,n,1,67},dim3((n+7)/8),dim3(8));
            check<float>({id,n,1,67},dim3((n+15)/16,2),dim3(8));
        }
    }
    // Exercise sampled matrix checks and long reduction dimensions without huge allocations.
    check<float>({"gemm",65,67,129},dim3(9,9),dim3(8,8));
    check<float>({"variance",17,1,4096},dim3(3),dim3(8));
    for(int n:{2,17,259})nbody_check(n);
    bool rejected=false;try{heavy::counts({"gemm",16384,16384,1048576});}catch(const std::invalid_argument&){rejected=true;}require(rejected,"unsafe indexing accepted");
    rejected=false;try{heavy::legal({"gemm",17,19,23},dim3(1),dim3(8,8));}catch(const std::invalid_argument&){rejected=true;}require(rejected,"uncovered matrix accepted");
    heavy::legal({"nbody",259,1,1},dim3(1),dim3(64));
    std::cout<<"All five original-kernel dispatches and N-body interaction/reference checks passed.\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
