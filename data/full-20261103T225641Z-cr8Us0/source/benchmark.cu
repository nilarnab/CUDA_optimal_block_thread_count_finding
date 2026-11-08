#include <cuda_runtime.h>
#include "reference.hpp"
#include <algorithm>
#include <climits>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace vector_add_source {
#include "kernels/10047/0/add.cu"
}
namespace segmented_sum_source {
#include "kernels/10040/1/gpuSum.cu"
}
namespace periodic_tile_source {
using uint = unsigned int;
#include "kernels/10073/223/kTile.cu"
}
namespace repeat_vector_source {
#include "kernels/1018/5/times.cu"
}
namespace transpose_source {
#include "kernels/10194/34/flatten_kernel.cu"
}
namespace matrix_add_source {
#include "kernels/10273/0/sumMat.cu"
}

namespace {
using pilot::Kernel;

void checked(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

std::string json_string(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
        else out << c;
    }
    return out.str() + '"';
}

struct Arguments {
    bool describe = false;
    int device = 0, size = 0, block_x = 0, block_y = 0;
    std::uint64_t seed = 0;
    std::string kernel;
};

std::uint64_t integer(const std::string& value) {
    if (value.empty() || !std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; }))
        throw std::invalid_argument("expected an unsigned decimal integer: " + value);
    return std::stoull(value);
}

Arguments arguments(int argc, char** argv) {
    Arguments out;
    std::vector<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (std::find(seen.begin(), seen.end(), key) != seen.end())
            throw std::invalid_argument("duplicate option: " + key);
        seen.push_back(key);
        if (key == "--describe") { out.describe = true; continue; }
        if (key != "--device" && key != "--kernel" && key != "--size" &&
            key != "--block-x" && key != "--block-y" && key != "--seed")
            throw std::invalid_argument("unknown option: " + key);
        if (++i >= argc) throw std::invalid_argument("missing value for " + key);
        if (key == "--kernel") { out.kernel = argv[i]; continue; }
        const std::uint64_t value = integer(argv[i]);
        if (key == "--seed") out.seed = value;
        else {
            if (value > INT_MAX) throw std::invalid_argument("integer too large for " + key);
            if (key == "--device") out.device = static_cast<int>(value);
            if (key == "--size") out.size = static_cast<int>(value);
            if (key == "--block-x") out.block_x = static_cast<int>(value);
            if (key == "--block-y") out.block_y = static_cast<int>(value);
        }
    }
    if (out.describe) {
        for (const auto& key : seen)
            if (key != "--describe" && key != "--device")
                throw std::invalid_argument("--describe accepts only --device");
    } else if (out.kernel.empty() || !pilot::allowed_size(out.size) || out.block_x < 1 || out.block_y < 1) {
        throw std::invalid_argument("require --kernel ID --size in [240,496,784,1016,1232,1680,2024] --block-x BX --block-y BY");
    }
    return out;
}

cudaFuncAttributes attributes(Kernel kernel) {
    cudaFuncAttributes result{};
    cudaError_t status = cudaErrorInvalidValue;
    switch (kernel) {
    case Kernel::VectorAdd: status = cudaFuncGetAttributes(&result, vector_add_source::add); break;
    case Kernel::SegmentedSum: status = cudaFuncGetAttributes(&result, segmented_sum_source::gpuSum); break;
    case Kernel::PeriodicTile: status = cudaFuncGetAttributes(&result, periodic_tile_source::kTile); break;
    case Kernel::RepeatVector: status = cudaFuncGetAttributes(&result, repeat_vector_source::times); break;
    case Kernel::Transpose: status = cudaFuncGetAttributes(&result, transpose_source::flatten_kernel); break;
    case Kernel::MatrixAdd: status = cudaFuncGetAttributes(&result, matrix_add_source::sumMat); break;
    }
    checked(status, "cudaFuncGetAttributes");
    return result;
}

void describe(const cudaDeviceProp& properties, int device) {
    int driver = 0, runtime = 0;
    char bus[32]{};
    checked(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    checked(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    checked(cudaDeviceGetPCIBusId(bus, sizeof(bus), device), "cudaDeviceGetPCIBusId");
    std::vector<cudaFuncAttributes> resources;
    for (const auto& name : pilot::kernel_names()) resources.push_back(attributes(pilot::parse_kernel(name)));
    std::cout << "{\"gpu_name\":" << json_string(properties.name)
              << ",\"compute_capability\":" << json_string(std::to_string(properties.major) + "." + std::to_string(properties.minor))
              << ",\"sm_count\":" << properties.multiProcessorCount
              << ",\"max_threads_per_block\":" << properties.maxThreadsPerBlock
              << ",\"max_block_dim_x\":" << properties.maxThreadsDim[0]
              << ",\"max_block_dim_y\":" << properties.maxThreadsDim[1]
              << ",\"driver_version\":" << driver << ",\"runtime_version\":" << runtime
              << ",\"pci_bus_id\":" << json_string(bus) << ",\"kernels\":{";
    bool first = true;
    for (std::size_t i = 0; i < resources.size(); ++i) {
        const auto& name = pilot::kernel_names()[i];
        const auto& info = resources[i];
        if (!first) std::cout << ',';
        first = false;
        std::cout << json_string(name) << ":{\"max_threads_per_block\":" << info.maxThreadsPerBlock
                  << ",\"registers_per_thread\":" << info.numRegs
                  << ",\"static_shared_bytes\":" << info.sharedSizeBytes << '}';
    }
    std::cout << "}}\n";
}

struct Event {
    cudaEvent_t value = nullptr;
    Event() { checked(cudaEventCreate(&value), "cudaEventCreate"); }
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

template <typename T>
struct DeviceArray {
    T* value = nullptr;
    explicit DeviceArray(std::size_t size) {
        if (size) checked(cudaMalloc(reinterpret_cast<void**>(&value), size * sizeof(T)), "cudaMalloc");
    }
    ~DeviceArray() { if (value) cudaFree(value); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

template <typename T>
void launch(Kernel kernel, int n, dim3 grid, dim3 block, T* a, T* b, T* output) {
    switch (kernel) {
    case Kernel::VectorAdd:
        vector_add_source::add<<<grid, block>>>(n * n, reinterpret_cast<float*>(a), reinterpret_cast<float*>(b), reinterpret_cast<float*>(output));
        break;
    case Kernel::SegmentedSum:
        segmented_sum_source::gpuSum<<<grid, block>>>(reinterpret_cast<int*>(a), reinterpret_cast<int*>(output), n, n, n);
        break;
    case Kernel::PeriodicTile:
        periodic_tile_source::kTile<<<grid, block>>>(reinterpret_cast<float*>(a), reinterpret_cast<float*>(output), 17, 13, n, n);
        break;
    case Kernel::RepeatVector:
        repeat_vector_source::times<<<grid, block>>>(reinterpret_cast<float*>(a), n, reinterpret_cast<float*>(output), n);
        break;
    case Kernel::Transpose:
        transpose_source::flatten_kernel<<<grid, block>>>(n * n, reinterpret_cast<float*>(a), n, n, 1, 1, reinterpret_cast<float*>(output));
        break;
    case Kernel::MatrixAdd:
        matrix_add_source::sumMat<<<grid, block>>>(reinterpret_cast<double*>(a), reinterpret_cast<double*>(b), reinterpret_cast<double*>(output), n);
        break;
    }
    checked(cudaGetLastError(), "kernel launch");
}

dim3 grid_for(Kernel kernel, int n, int bx, int by) {
    if (kernel == Kernel::MatrixAdd) return dim3(n / bx + (n % bx != 0), n / by + (n % by != 0));
    const int count = kernel == Kernel::SegmentedSum ? n : n * n;
    return dim3(count / bx + (count % bx != 0));
}

void validate_action(const Arguments& args, Kernel kernel, const cudaDeviceProp& properties, dim3 grid) {
    if (args.block_x > properties.maxThreadsDim[0] || args.block_y > properties.maxThreadsDim[1] ||
        static_cast<long long>(args.block_x) * args.block_y > properties.maxThreadsPerBlock ||
        static_cast<long long>(args.block_x) * args.block_y > attributes(kernel).maxThreadsPerBlock)
        throw std::invalid_argument("block exceeds GPU or kernel limits");
    if (grid.x > static_cast<unsigned>(properties.maxGridSize[0]) ||
        grid.y > static_cast<unsigned>(properties.maxGridSize[1]))
        throw std::invalid_argument("grid exceeds GPU limits");
    if (kernel == Kernel::MatrixAdd) {
        const bool one_dimensional = args.block_x == 1 && args.block_y >= 64 && args.block_y <= 1024 && args.block_y % 64 == 0;
        const bool square = args.block_x == args.block_y &&
            (args.block_x == 8 || args.block_x == 16 || args.block_x == 24 || args.block_x == 32);
        if (!one_dimensional && !square) throw std::invalid_argument("matrix_add requires one of the 20 LS-CAT block shapes");
    } else if (args.block_y != 1 || args.block_x < 64 || args.block_x > 1024 || args.block_x % 64 != 0) {
        throw std::invalid_argument("linear kernels require block-x in 64..1024 by 64 and block-y=1");
    }
}

template <typename T>
void benchmark(const Arguments& args, Kernel kernel, dim3 grid) {
    const auto data = pilot::make_case<T>(kernel, args.size, args.seed);
    DeviceArray<T> a(data.a.size()), b(data.b.size()), output(data.expected.size());
    const std::size_t bytes = data.expected.size() * sizeof(T);
    checked(cudaMemcpy(a.value, data.a.data(), data.a.size() * sizeof(T), cudaMemcpyHostToDevice), "copy input a");
    if (!data.b.empty()) checked(cudaMemcpy(b.value, data.b.data(), data.b.size() * sizeof(T), cudaMemcpyHostToDevice), "copy input b");
    const T sentinel = std::numeric_limits<T>::has_quiet_NaN ? std::numeric_limits<T>::quiet_NaN() : std::numeric_limits<T>::lowest();
    std::vector<T> actual(data.expected.size(), sentinel);
    checked(cudaMemcpy(output.value, actual.data(), bytes, cudaMemcpyHostToDevice), "initialize output");
    const dim3 block(args.block_x, args.block_y);
    const auto issue = [&]() { launch(kernel, args.size, grid, block, a.value, b.value, output.value); };
    for (int i = 0; i < 3; ++i) issue();
    checked(cudaDeviceSynchronize(), "warmup completion");
    checked(cudaMemcpy(actual.data(), output.value, bytes, cudaMemcpyDeviceToHost), "copy warmup output");
    double error = pilot::validate(actual, data.expected);

    Event start, stop;
    const auto timed = [&](int count) {
        checked(cudaEventRecord(start.value), "record start event");
        for (int i = 0; i < count; ++i) issue();
        checked(cudaEventRecord(stop.value), "record stop event");
        checked(cudaEventSynchronize(stop.value), "timed kernel completion");
        float milliseconds = 0;
        checked(cudaEventElapsedTime(&milliseconds, start.value, stop.value), "read event duration");
        return milliseconds;
    };
    const float calibration_ms = timed(1);
    if (!std::isfinite(calibration_ms) || calibration_ms < 0) throw std::runtime_error("invalid calibration duration");
    const int iterations = static_cast<int>(std::max(1.0, std::min(1000.0, std::ceil(10.0 / std::max(0.000001, static_cast<double>(calibration_ms))))));
    std::fill(actual.begin(), actual.end(), sentinel);
    checked(cudaMemcpy(output.value, actual.data(), bytes, cudaMemcpyHostToDevice), "reset output before timing");
    const float batch_ms = timed(iterations);
    if (!std::isfinite(batch_ms) || batch_ms <= 0) throw std::runtime_error("nonpositive or non-finite batch duration");
    checked(cudaMemcpy(actual.data(), output.value, bytes, cudaMemcpyDeviceToHost), "copy timed output");
    error = std::max(error, pilot::validate(actual, data.expected));
    std::cout << std::setprecision(10)
              << "{\"status\":\"ok\",\"correctness\":\"passed\",\"kernel_ms\":" << batch_ms / iterations
              << ",\"batch_ms\":" << batch_ms << ",\"iterations\":" << iterations
              << ",\"grid_x\":" << grid.x << ",\"grid_y\":" << grid.y << ",\"grid_z\":" << grid.z
              << ",\"input_elements\":" << data.input_elements() << ",\"output_elements\":" << data.expected.size()
              << ",\"max_abs_error\":" << error << ",\"warmup_launches\":3,\"calibration_launches\":1}\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto args = arguments(argc, argv);
        int count = 0;
        checked(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (args.device >= count) throw std::invalid_argument("device index unavailable");
        checked(cudaSetDevice(args.device), "cudaSetDevice");
        cudaDeviceProp properties{};
        checked(cudaGetDeviceProperties(&properties, args.device), "cudaGetDeviceProperties");
        if (args.describe) { describe(properties, args.device); return 0; }
        const auto kernel = pilot::parse_kernel(args.kernel);
        const auto grid = grid_for(kernel, args.size, args.block_x, args.block_y);
        validate_action(args, kernel, properties, grid);
        if (kernel == Kernel::SegmentedSum) benchmark<int>(args, kernel, grid);
        else if (kernel == Kernel::MatrixAdd) benchmark<double>(args, kernel, grid);
        else benchmark<float>(args, kernel, grid);
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"status\":\"error\",\"message\":" << json_string(error.what()) << "}\n";
        return 1;
    }
}
