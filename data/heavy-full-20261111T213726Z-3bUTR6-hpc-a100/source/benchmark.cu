#include <cuda_runtime.h>
#include "dispatch.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <tuple>

static void checked(cudaError_t e,const char* where){if(e!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));}
static std::string quote(const std::string& s){std::ostringstream o;o<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<c;else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else o<<c;}return o.str()+'"';}
static std::uint64_t integer(const std::string& s){if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("expected unsigned integer");return std::stoull(s);}
static cudaFuncAttributes attributes(const std::string& id){cudaFuncAttributes a{};
#define ATTR(name,fn) if(id==#name){checked(cudaFuncGetAttributes(&a,fn),"kernel attributes");return a;}
    HEAVY_FUNCTIONS(ATTR)
#undef ATTR
    throw std::invalid_argument("unknown kernel");
}
static cudaDeviceProp device(int i){int n=0;checked(cudaGetDeviceCount(&n),"device count");if(i<0||i>=n)throw std::invalid_argument("CUDA logical device unavailable");checked(cudaSetDevice(i),"select device");cudaDeviceProp p{};checked(cudaGetDeviceProperties(&p,i),"device properties");return p;}
static void describe(int i){const auto p=device(i);char pci[32]{};int driver=0,runtime=0;std::size_t free=0,total=0;
    checked(cudaDeviceGetPCIBusId(pci,sizeof(pci),i),"PCI identity");checked(cudaDriverGetVersion(&driver),"driver");checked(cudaRuntimeGetVersion(&runtime),"runtime");checked(cudaMemGetInfo(&free,&total),"memory info");
    std::cout<<"{\"gpu_name\":"<<quote(p.name)<<",\"pci_bus_id\":"<<quote(pci)<<",\"sm_count\":"<<p.multiProcessorCount<<",\"compute_capability\":"<<quote(std::to_string(p.major)+"."+std::to_string(p.minor))
        <<",\"driver_version\":"<<driver<<",\"runtime_version\":"<<runtime<<",\"free_bytes\":"<<free<<",\"total_bytes\":"<<total
        <<",\"max_threads_per_block\":"<<p.maxThreadsPerBlock<<",\"max_block_dim_x\":"<<p.maxThreadsDim[0]<<",\"max_block_dim_y\":"<<p.maxThreadsDim[1]<<",\"max_grid_dim_x\":"<<p.maxGridSize[0]<<",\"max_grid_dim_y\":"<<p.maxGridSize[1]<<",\"kernels\":{";
    bool first=true;for(const auto& id:heavy::names()){const auto a=attributes(id);if(!first)std::cout<<',';first=false;std::cout<<quote(id)<<":{\"max_threads_per_block\":"<<a.maxThreadsPerBlock<<",\"registers_per_thread\":"<<a.numRegs<<",\"static_shared_bytes\":"<<a.sharedSizeBytes<<'}';}std::cout<<"}}"<<std::endl;
}
struct Stream {cudaStream_t value=nullptr;Stream(){checked(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking),"create stream");}~Stream(){if(value)cudaStreamDestroy(value);}};
struct Event {cudaEvent_t value=nullptr;Event(){checked(cudaEventCreate(&value),"create event");}~Event(){if(value)cudaEventDestroy(value);}};
template<class T> struct Buffer {T* p=nullptr;explicit Buffer(std::size_t n){if(n)checked(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)),"allocate");}~Buffer(){if(p)cudaFree(p);}};
struct Job {int id,bx,by,gx,gy,round;};
static std::vector<Job> jobs(const std::string& path){std::ifstream f(path);if(!f)throw std::runtime_error("cannot open job file");std::vector<Job> v;std::set<int> ids;std::string line;
    while(std::getline(f,line)){Job j{};std::istringstream in(line);std::string extra;if(!(in>>j.id>>j.bx>>j.by>>j.gx>>j.gy>>j.round)||in>>extra||j.id<0||j.bx<1||j.by<1||j.gx<1||j.gy<1||j.round<0||!ids.insert(j.id).second)throw std::invalid_argument("invalid or duplicate job");v.push_back(j);if(v.size()>10000)throw std::invalid_argument("too many jobs");}if(v.empty())throw std::invalid_argument("empty plan");return v;
}
template<class T> static void worker(const heavy::Shape& shape,std::uint64_t seed,const cudaDeviceProp& p,const std::vector<Job>& plan,std::uint64_t cap){
    const auto c=heavy::counts(shape);std::size_t free=0,total=0;checked(cudaMemGetInfo(&free,&total),"memory info");
    if(c.bytes()>cap||c.bytes()>free*0.8)throw std::runtime_error("memory_limit: device buffers exceed requested cap or 80% of free GPU memory");
    const auto reference_start=std::chrono::steady_clock::now();const auto data=heavy::make_case<T>(shape,seed);
    const double reference_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-reference_start).count();
    auto host=heavy::initial(data);Buffer<T> a(c.a),b(c.b),out(c.out+32);Stream stream;Event start,stop;
    auto upload=[&](T* to,const std::vector<T>& from){if(!from.empty())checked(cudaMemcpyAsync(to,from.data(),from.size()*sizeof(T),cudaMemcpyHostToDevice,stream.value),"upload");};
    upload(a.p,data.a);upload(b.p,data.b);checked(cudaStreamSynchronize(stream.value),"input completion");
    const auto attr=attributes(shape.id);std::set<std::tuple<int,int,int,int>> warmed;
    for(const auto& j:plan){const auto wall=std::chrono::steady_clock::now();
        try{
            const dim3 block(j.bx,j.by),grid(j.gx,j.gy);heavy::legal(shape,grid,block);
            if(j.bx>p.maxThreadsDim[0]||j.by>p.maxThreadsDim[1]||std::int64_t(j.bx)*j.by>std::min(p.maxThreadsPerBlock,attr.maxThreadsPerBlock)||j.gx>p.maxGridSize[0]||j.gy>p.maxGridSize[1]||(shape.id=="nbody"&&j.bx*sizeof(float4)+attr.sharedSizeBytes>p.sharedMemPerBlock))throw std::invalid_argument("device/kernel launch limit");
            auto reset=[&](){host=heavy::initial(data);upload(out.p,host);};
            const auto key=std::make_tuple(j.bx,j.by,j.gx,j.gy);const bool warm=warmed.insert(key).second;
            reset();if(warm){heavy::dispatch(shape,grid,block,stream.value,a.p,b.p,out.p+16);checked(cudaGetLastError(),"warmup launch");checked(cudaStreamSynchronize(stream.value),"warmup completion");reset();}
            checked(cudaEventRecord(start.value,stream.value),"start event");
            heavy::dispatch(shape,grid,block,stream.value,a.p,b.p,out.p+16);checked(cudaGetLastError(),"kernel launch");
            checked(cudaEventRecord(stop.value,stream.value),"stop event");checked(cudaEventSynchronize(stop.value),"kernel completion");float ms=0;checked(cudaEventElapsedTime(&ms,start.value,stop.value),"elapsed time");
            if(!std::isfinite(ms)||ms<=0)throw std::runtime_error("invalid event timing");
            checked(cudaMemcpyAsync(host.data(),out.p,host.size()*sizeof(T),cudaMemcpyDeviceToHost,stream.value),"download");checked(cudaStreamSynchronize(stream.value),"output completion");const auto error=heavy::validate(data,host);
            const double query=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall).count();
            const auto timestamp=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            std::cout<<std::setprecision(12)<<"{\"trial_id\":"<<j.id<<",\"round\":"<<j.round<<",\"block_x\":"<<j.bx<<",\"block_y\":"<<j.by<<",\"grid_x\":"<<j.gx<<",\"grid_y\":"<<j.gy
                <<",\"status\":\"ok\",\"correctness\":\"passed\",\"guards\":\"passed\",\"iterations\":1,\"kernel_ms\":"<<ms<<",\"query_wall_ms\":"<<query<<",\"warmup_launches\":"<<(warm?1:0)
                <<",\"correctness_scope\":"<<quote(data.scope)<<",\"checked_outputs\":"<<data.indices.size()<<",\"output_elements\":"<<c.out<<",\"input_elements\":"<<c.a+c.b
                <<",\"device_bytes\":"<<c.bytes()<<",\"max_abs_error\":"<<error.first<<",\"max_scaled_error\":"<<error.second<<",\"cpu_setup_reference_ms\":"<<reference_ms<<",\"timestamp_unix_ns\":"<<timestamp<<"}"<<std::endl;
        }catch(const std::exception& e){std::cout<<"{\"trial_id\":"<<j.id<<",\"status\":\"failed\",\"error\":"<<quote(e.what())<<"}"<<std::endl;throw;}
    }
}
int main(int argc,char** argv){try{
    if(argc==3&&std::string(argv[1])=="--describe"){const auto i=integer(argv[2]);if(i>1024)throw std::invalid_argument("device index");describe(int(i));return 0;}
    if(argc==3&&std::string(argv[1])=="--selftest") {
        const auto i=integer(argv[2]);if(i>1024)throw std::invalid_argument("device index");const auto p=device(int(i));
        for(const auto& id:heavy::names()) {
            const heavy::Shape s=id=="nbody"?heavy::Shape{id,259,1,1}:heavy::matrix(id)?heavy::Shape{id,17,id=="gemm"?19:17,23}:heavy::Shape{id,35,1,67};
            std::vector<Job> plan=heavy::matrix(id)?std::vector<Job>{{0,8,8,(s.n+7)/8,(s.m+7)/8,0}}:std::vector<Job>{{0,64,1,(s.m+63)/64,1,0}};
            if(id=="nbody")plan.push_back({1,64,1,1,1,0});
            std::cout<<"Self-test: "<<id<<std::endl;
            if(id=="segmented_sum")worker<int>(s,12345,p,plan,64ULL*1024*1024);else worker<float>(s,12345,p,plan,64ULL*1024*1024);
        }
        std::cout<<"GPU self-tests passed, with full CPU references including capped/partial N-body tiles."<<std::endl;return 0;
    }
    if(argc!=10||std::string(argv[1])!="--worker")throw std::invalid_argument("Usage: benchmark --worker DEVICE KIND M N K SEED JOBS MAX_DEVICE_BYTES");
    const auto i=integer(argv[2]),m=integer(argv[4]),n=integer(argv[5]),k=integer(argv[6]),seed=integer(argv[7]),cap=integer(argv[9]);
    if(i>1024||m>4194304||n>16384||k>1048576||!cap)throw std::invalid_argument("invalid arguments");
    const heavy::Shape shape{argv[3],int(m),int(n),int(k)};heavy::counts(shape);const auto p=device(int(i));const auto plan=jobs(argv[8]);
    if(shape.id=="segmented_sum")worker<int>(shape,seed,p,plan,cap);else worker<float>(shape,seed,p,plan,cap);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
