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
template<class T> static void check(const heavy::Shape& s,int block) {
    const auto d=heavy::make_case<T>(s,12345);auto a=d.a,b=d.b,out=heavy::initial(d);
    const auto work=heavy::elementwise(s.id)?d.outputs:std::size_t(s.m);
    const dim3 grid((work+block-1)/block);heavy::legal(s,grid,dim3(block));
    gridDim=grid;blockDim=dim3(block);blockIdx.y=threadIdx.y=0;
    for(blockIdx.x=0;blockIdx.x<grid.x;blockIdx.x++)
        for(threadIdx.x=0;threadIdx.x<blockDim.x;threadIdx.x++)
            heavy::dispatch(s,grid,blockDim,nullptr,a.data(),b.data(),out.data()+16);
    heavy::validate(d,out);require(a==d.a&&b==d.b,"input modified");
    auto broken=out;broken[16]=T(123456);bool rejected=false;
    try{heavy::validate(d,broken);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"incorrect output accepted");
    broken=out;broken[0]=0;rejected=false;
    try{heavy::validate(d,broken);}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"guard overwrite accepted");
}
static void nbody_check(int n) {
    const auto d=heavy::make_case<float>({"nbody",n,1,1},42);auto out=heavy::initial(d);
    for(int i=0;i<n;i++) {
        const auto p=make_float4(d.a[4*i],d.a[4*i+1],d.a[4*i+2],d.a[4*i+3]);float ax=0,ay=0,az=0;
        for(int j=0;j<n;j++)heavy_interaction(p,make_float4(d.a[4*j],d.a[4*j+1],d.a[4*j+2],d.a[4*j+3]),ax,ay,az);
        out[16+3*i]=ax;out[16+3*i+1]=ay;out[16+3*i+2]=az;
    }
    heavy::validate(d,out);
    if(n==2) {
        auto two=d;two.a={-.5f,0,0,.5f,.5f,0,0,.5f};
        require(std::abs(heavy::expected_at(two,0)-.5/std::pow(1.01,1.5))<1e-12,"analytic force differs");
        require(std::abs(heavy::expected_at(two,3)+.5/std::pow(1.01,1.5))<1e-12,"pair symmetry differs");
    }
}
static void known_case(const std::string& id,const std::vector<float>& expected) {
    const heavy::Shape s{id,2,1,3};std::vector<float> input={1,-2,0,-3,4,5};
    std::vector<float> out(expected.size()+32,std::numeric_limits<float>::quiet_NaN());
    const dim3 block(64),grid(1);heavy::legal(s,grid,block);blockDim=block;gridDim=grid;blockIdx=dim3(0,0,0);
    for(threadIdx.x=0;threadIdx.x<block.x;threadIdx.x++)
        heavy::dispatch<float>(s,grid,block,nullptr,input.data(),nullptr,out.data()+16);
    for(std::size_t i=0;i<expected.size();i++)require(std::abs(out[i+16]-expected[i])<1e-6,"analytic new-kernel result differs");
    for(int i=0;i<16;i++)require(std::isnan(out[i])&&std::isnan(out[out.size()-1-i]),"analytic case guard overwritten");
}
int main(){try{
    for(int block=64;block<=1024;block+=64)for(int n:{5,17,65}) {
        for(const auto& id:{"vector_add","periodic_tile","repeat_vector","matrix_transpose"})check<float>({id,n,1,1},block);
        check<int>({"segmented_sum",n,1,67},block);
        for(const auto& id:{"mean","variance"})check<float>({id,n,1,67},block);
        for(const auto& id:{"binarize_weights","binarize_input","softmax","accumulate"})check<float>({id,n,1,67},block);
        heavy::legal({"nbody",259,1,1},dim3((259+block-1)/block),dim3(block));
    }
    check<float>({"variance",17,1,4096},64);
    for(int n:{2,17,259})nbody_check(n);
    known_case("binarize_weights",{1,-1,-1,-4,4,4});
    known_case("binarize_input",{5.f/3,-10.f/3,-5.f/3,-10.f/3,5.f/3,10.f/3});
    known_case("accumulate",{5,0});
    std::vector<float> probabilities;
    for(const auto& row:{std::vector<double>{1,-2,0},std::vector<double>{-3,4,5}}) {
        double sum=0;for(auto x:row)sum+=std::exp(x);
        for(auto x:row)probabilities.push_back(float(std::exp(x)/sum));
    }
    known_case("softmax",probabilities);
    require(heavy::tolerance("softmax",1e-8)<1e-8,"zero probability output could pass for large K");
    require(heavy::counts({"vector_add",32768,1,1}).out==1073741824,"large square count differs");
    require(heavy::counts({"segmented_sum",16,1,16777216}).a==268435456,"large reduction count differs");
    bool rejected=false;try{heavy::counts({"vector_add",32769,1,1});}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"unsafe square accepted");
    rejected=false;try{heavy::counts({"segmented_sum",16384,1,16777216});}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"unsafe reduction accepted");
    for(auto block:{dim3(32),dim3(64,2)}) {
        rejected=false;try{heavy::legal({"vector_add",17,1,1},dim3(1),block);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"unsupported block accepted");
    }
    rejected=false;try{heavy::legal({"vector_add",17,1,1},dim3(1),dim3(64));}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"uncovered grid accepted");
    std::cout<<"Eleven original 1D dispatches passed all sixteen blocks; analytic new-kernel cases, N-body arithmetic/reference and indexing/guard checks passed.\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
