#ifndef HEAVY_REFERENCE_HPP
#define HEAVY_REFERENCE_HPP
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
namespace heavy {
inline const std::vector<std::string>& names(){static const std::vector<std::string> a={"vector_add","segmented_sum","periodic_tile","repeat_vector","matrix_transpose","mean","variance","nbody","binarize_weights","binarize_input","softmax","accumulate"};return a;}
inline bool elementwise(const std::string& id){return id=="vector_add" || id=="periodic_tile" || id=="repeat_vector" || id=="matrix_transpose";}
inline bool tensor_output(const std::string& id){return id=="binarize_weights" || id=="binarize_input" || id=="softmax";}
struct Shape {std::string id;int m,n,k;};
struct Counts {std::size_t a,b,out;std::uint64_t bytes()const{return 4ULL*(a+b+out+32);}};
inline Counts counts(const Shape& s) {
    if(std::find(names().begin(),names().end(),s.id)==names().end() || s.m<1 || s.n<1 || s.k<1)throw std::invalid_argument("invalid workload");
    const std::uint64_t m=s.m,n=s.n,k=s.k;
    if(s.id=="nbody") {
        if(m<2 || m>4194304 || n!=1 || k!=1)throw std::invalid_argument("invalid body count");
        return {std::size_t(4*m),0,std::size_t(3*m)};
    }
    if(elementwise(s.id)) {
        if(m>32768 || n!=1 || k!=1)throw std::invalid_argument("invalid square output dimension");
        return {std::size_t(s.id=="periodic_tile"?221:s.id=="repeat_vector"?m:m*m),s.id=="vector_add"?std::size_t(m*m):0,std::size_t(m*m)};
    }
    if(m>16384 || n!=1 || k<2 || k>16777216 || m*k>2147483647)throw std::invalid_argument("invalid reduction/indexing dimensions");
    return {std::size_t(m*k),s.id=="variance"?std::size_t(m):0,std::size_t(tensor_output(s.id)?m*k:m)};
}
inline std::uint64_t random64(std::uint64_t& state){state+=0x9e3779b97f4a7c15ULL;auto z=state;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
template<class T> struct Case {
    Shape shape;std::vector<T>a,b;std::vector<std::size_t> indices;std::vector<double> expected;
    std::size_t outputs;std::string scope;std::vector<double> normalizers;
};
template<class T> double expected_at(const Case<T>& d,std::size_t index) {
    const auto& s=d.shape;double value=0;
    if(s.id=="vector_add")return double(d.a[index])+d.b[index];
    if(s.id=="periodic_tile")return d.a[((index/s.m)%13)*17+(index%s.m)%17];
    if(s.id=="repeat_vector")return d.a[index%s.m];
    if(s.id=="matrix_transpose")return d.a[(index%s.m)*s.m+index/s.m];
    if(s.id=="softmax")return std::exp(double(d.a[index]))/d.normalizers[index/s.k];
    if(s.id=="binarize_weights" || s.id=="binarize_input") {
        const auto group=s.id=="binarize_weights"?index/s.k:index%s.m;
        return (d.a[index]>0?1:-1)*d.normalizers[group];
    }
    if(s.id=="accumulate") {
        for(int j=0;j<s.k;j++)value+=d.a[std::size_t(j)*s.m+index];
        return value;
    }
    if(s.id=="nbody") {
        const std::size_t body=index/3,axis=index%3;
        for(int j=0;j<s.m;j++) {
            const double dx=double(d.a[4*j])-d.a[4*body],dy=double(d.a[4*j+1])-d.a[4*body+1],dz=double(d.a[4*j+2])-d.a[4*body+2];
            const double distance=dx*dx+dy*dy+dz*dz+0.01;
            const double r=axis==0?dx:axis==1?dy:dz;
            value+=double(d.a[4*j+3])*r/(distance*std::sqrt(distance));
        }
    }else {
        for(int j=0;j<s.k;j++) {
            const double x=d.a[index*s.k+j];
            value+=s.id=="variance"?(x-double(d.b[index]))*(x-double(d.b[index])):x;
        }
        if(s.id=="mean")value/=s.k;
        if(s.id=="variance")value/=(s.k-1);
    }
    return value;
}
template<class T> Case<T> make_case(const Shape& s,std::uint64_t seed) {
    const auto c=counts(s);
    if((s.id=="segmented_sum")!=std::is_same<T,int>::value)throw std::invalid_argument("incorrect scalar type");
    Case<T>d{s,std::vector<T>(c.a),std::vector<T>(c.b),{}, {},c.out,""};
    auto state=seed;
    for(auto& x:d.a)x=std::is_same<T,int>::value?T(int(random64(state)%17)-8):T(int(random64(state)%65)-32)/T(64);
    for(auto& x:d.b)x=T(int(random64(state)%65)-32)/T(64);
    if(s.id=="variance")for(int i=0;i<s.m;i++){double sum=0;for(int j=0;j<s.k;j++)sum+=d.a[std::size_t(i)*s.k+j];d.b[i]=T(sum/s.k);}
    if(s.id=="nbody")for(int i=0;i<s.m;i++)d.a[4*i+3]=T(1.0/s.m);
    if(tensor_output(s.id)) {
        d.normalizers.assign(s.m,0.0);
        for(int i=0;i<s.m;i++) {
            double sum=0;
            for(int j=0;j<s.k;j++) {
                const double x=d.a[s.id=="binarize_input"?std::size_t(j)*s.m+i:std::size_t(i)*s.k+j];
                sum+=s.id=="softmax"?std::exp(x):std::abs(x);
            }
            d.normalizers[i]=s.id=="softmax"?sum:sum/s.k;
        }
    }
    const bool full=c.out<=4096;
    std::set<std::size_t> selected;
    if(full)for(std::size_t i=0;i<c.out;i++)selected.insert(i);
    else {
        selected={0,1,c.out-1,c.out-2,c.out/2};
        if(elementwise(s.id))for(int row:{0,s.m/2,s.m-1})for(int col:{0,s.m/2,s.m-1})selected.insert(std::size_t(row)*s.m+col);
        while(selected.size()<64)selected.insert(random64(state)%c.out);
    }
    d.indices.assign(selected.begin(),selected.end());d.scope=full?"full_cpu_reference":"64_sampled_cpu_reference";
    for(auto index:d.indices)d.expected.push_back(expected_at(d,index));
    return d;
}
template<class T> std::vector<T> initial(const Case<T>& d) {
    std::vector<T> v(d.outputs+32,std::is_same<T,int>::value?T(std::numeric_limits<int>::lowest()):std::numeric_limits<T>::quiet_NaN());
    return v;
}
inline double tolerance(const std::string& id,double expected) {
    // Probabilities shrink with K; a fixed 3e-5 allowance would accept all-zero large softmax outputs.
    if(id=="softmax")return 1e-10+1e-3*std::abs(expected);
    return id=="segmented_sum"?0:id=="nbody"?2e-4+2e-3*std::abs(expected):3e-5+1e-3*std::abs(expected);
}
template<class T> std::pair<double,double> validate(const Case<T>& d,const std::vector<T>& output) {
    if(output.size()!=d.outputs+32)throw std::runtime_error("incorrect output length");
    auto guard=[](T x){return std::is_same<T,int>::value?x==T(std::numeric_limits<int>::lowest()):std::isnan(double(x));};
    for(std::size_t i=0;i<16;i++)if(!guard(output[i]) || !guard(output[output.size()-1-i]))throw std::runtime_error("output guard overwritten");
    for(std::size_t i=0;i<d.outputs;i++)if(!std::isfinite(double(output[i+16])) || (std::is_same<T,int>::value && output[i+16]==T(std::numeric_limits<int>::lowest())))throw std::runtime_error("unwritten/nonfinite output");
    double maximum=0,scaled=0;
    for(std::size_t j=0;j<d.indices.size();j++) {
        const double error=std::abs(double(output[d.indices[j]+16])-d.expected[j]),allowed=tolerance(d.shape.id,d.expected[j]);
        if(error>allowed)throw std::runtime_error("incorrect output at "+std::to_string(d.indices[j]));
        maximum=std::max(maximum,error);scaled=std::max(scaled,allowed?error/allowed:0);
    }
    return {maximum,scaled};
}
}
#endif
