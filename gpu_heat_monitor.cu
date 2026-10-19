#include <cuda_runtime.h>
#include <nvml.h>

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

volatile std::sig_atomic_t stop_requested = 0;

void handle_signal(int) { stop_requested = 1; }

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
    }
}

void check_nvml(nvmlReturn_t result, const char* operation) {
    if (result != NVML_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 nvmlErrorString(result));
    }
}

struct Options {
    int device = 0;
    unsigned int target_temp_c = 75;
    unsigned int max_temp_c = 80;
    int duration_s = 300;
    int sample_ms = 1000;
    int threads = 256;
    int blocks_per_sm = 8;
    int iterations_per_chunk = 20000;
    std::string csv_path;
};

int parse_int(const char* value, const char* option) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (*value == '\0' || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("Invalid value for ") + option);
    }
    return static_cast<int>(parsed);
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "  --device N          CUDA device index (default: 0)\n"
        << "  --target-temp C     Stop after reaching this temperature (default: 75)\n"
        << "  --max-temp C        Emergency stop temperature (default: 80)\n"
        << "  --duration S        Maximum run time in seconds (default: 300)\n"
        << "  --sample-ms MS      Monitoring output interval (default: 1000)\n"
        << "  --threads N         Threads per heating block (default: 256)\n"
        << "  --blocks-per-sm N   Blocks launched per SM (default: 8)\n"
        << "  --iterations N      Work in each responsive chunk (default: 20000)\n"
        << "  --csv PATH          Also write samples to a CSV file\n"
        << "  --help              Show this help\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value after " + arg);
        }
        const char* value = argv[++i];
        if (arg == "--device") {
            options.device = parse_int(value, "--device");
        } else if (arg == "--target-temp") {
            options.target_temp_c = parse_int(value, "--target-temp");
        } else if (arg == "--max-temp") {
            options.max_temp_c = parse_int(value, "--max-temp");
        } else if (arg == "--duration") {
            options.duration_s = parse_int(value, "--duration");
        } else if (arg == "--sample-ms") {
            options.sample_ms = parse_int(value, "--sample-ms");
        } else if (arg == "--threads") {
            options.threads = parse_int(value, "--threads");
        } else if (arg == "--blocks-per-sm") {
            options.blocks_per_sm = parse_int(value, "--blocks-per-sm");
        } else if (arg == "--iterations") {
            options.iterations_per_chunk = parse_int(value, "--iterations");
        } else if (arg == "--csv") {
            options.csv_path = value;
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (options.target_temp_c >= options.max_temp_c) {
        throw std::runtime_error("--target-temp must be lower than --max-temp");
    }
    if (options.max_temp_c > 90) {
        throw std::runtime_error("Refusing --max-temp above 90 C");
    }
    if (options.duration_s <= 0 || options.sample_ms <= 0 ||
        options.threads <= 0 || options.threads > 1024 ||
        options.blocks_per_sm <= 0 || options.iterations_per_chunk <= 0) {
        throw std::runtime_error("Duration and workload settings must be positive; "
                                 "threads must be in the range 1..1024");
    }
    return options;
}

// Multiple independent accumulators keep the arithmetic pipelines busy. The
// output write prevents the compiler from deleting the work.
__global__ void heat_kernel(float* output, int iterations) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 1.0001f + static_cast<float>(index & 7u) * 0.00001f;
    float b = 0.9999f;
    float c = 1.0002f;
    float d = 0.9998f;

    for (int i = 0; i < iterations; ++i) {
        a = fmaf(a, 1.0000001f, b * 0.0000001f);
        b = fmaf(b, 0.9999999f, c * 0.0000001f);
        c = fmaf(c, 1.0000001f, d * 0.0000001f);
        d = fmaf(d, 0.9999999f, a * 0.0000001f);
    }
    output[index] = a + b + c + d;
}

}  // namespace

int main(int argc, char** argv) {
    nvmlDevice_t nvml_device{};
    float* device_output = nullptr;
    bool nvml_initialized = false;

    try {
        const Options options = parse_options(argc, argv);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        check_cuda(cudaSetDevice(options.device), "cudaSetDevice");

        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, options.device),
                   "cudaGetDeviceProperties");
        if (options.threads > properties.maxThreadsPerBlock) {
            throw std::runtime_error("--threads exceeds this GPU's per-block limit of " +
                                     std::to_string(properties.maxThreadsPerBlock));
        }

        char pci_bus_id[32]{};
        check_cuda(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id),
                                         options.device),
                   "cudaDeviceGetPCIBusId");

        check_nvml(nvmlInit_v2(), "nvmlInit_v2");
        nvml_initialized = true;
        check_nvml(nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, &nvml_device),
                   "nvmlDeviceGetHandleByPciBusId_v2");

        const int blocks = properties.multiProcessorCount * options.blocks_per_sm;
        const std::size_t value_count =
            static_cast<std::size_t>(blocks) * options.threads;
        check_cuda(cudaMalloc(&device_output, value_count * sizeof(float)),
                   "cudaMalloc");

        std::ofstream csv;
        if (!options.csv_path.empty()) {
            csv.open(options.csv_path);
            if (!csv) {
                throw std::runtime_error("Cannot open CSV file: " + options.csv_path);
            }
            csv << "elapsed_s,temp_c,power_w,gpu_util_pct,sm_clock_mhz,"
                   "memory_clock_mhz,throttle_reasons_hex\n";
        }

        std::cout << "GPU: " << properties.name << " (PCI " << pci_bus_id << ")\n"
                  << "Load: " << blocks << " blocks x " << options.threads
                  << " threads; target=" << options.target_temp_c
                  << " C; hard stop=" << options.max_temp_c << " C\n"
                  << "Press Ctrl-C to stop.\n\n"
                  << "elapsed_s  temp_C  power_W  util_%  sm_MHz  mem_MHz  throttle\n";

        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        auto next_sample = start;
        std::string stop_reason = "maximum duration reached";

        while (!stop_requested) {
            heat_kernel<<<blocks, options.threads>>>(device_output,
                                                     options.iterations_per_chunk);
            check_cuda(cudaGetLastError(), "heat_kernel launch");
            check_cuda(cudaDeviceSynchronize(), "heat_kernel execution");

            const auto now = Clock::now();
            const double elapsed =
                std::chrono::duration<double>(now - start).count();

            unsigned int temperature = 0;
            check_nvml(nvmlDeviceGetTemperature(
                           nvml_device, NVML_TEMPERATURE_GPU, &temperature),
                       "nvmlDeviceGetTemperature");

            if (temperature >= options.max_temp_c) {
                stop_reason = "hard temperature limit reached";
                break;
            }
            if (temperature >= options.target_temp_c) {
                stop_reason = "target temperature reached";
                break;
            }
            if (elapsed >= options.duration_s) {
                break;
            }

            if (now >= next_sample) {
                unsigned int power_mw = 0;
                const nvmlReturn_t power_result =
                    nvmlDeviceGetPowerUsage(nvml_device, &power_mw);

                nvmlUtilization_t utilization{};
                const nvmlReturn_t utilization_result =
                    nvmlDeviceGetUtilizationRates(nvml_device, &utilization);

                unsigned int sm_clock = 0;
                const nvmlReturn_t sm_clock_result = nvmlDeviceGetClockInfo(
                    nvml_device, NVML_CLOCK_SM, &sm_clock);

                unsigned int memory_clock = 0;
                const nvmlReturn_t memory_clock_result = nvmlDeviceGetClockInfo(
                    nvml_device, NVML_CLOCK_MEM, &memory_clock);

                unsigned long long throttle_reasons = 0;
                const nvmlReturn_t throttle_result =
                    nvmlDeviceGetCurrentClocksThrottleReasons(
                        nvml_device, &throttle_reasons);

                const std::string power = power_result == NVML_SUCCESS
                    ? std::to_string(power_mw / 1000.0)
                    : "NA";
                const std::string util = utilization_result == NVML_SUCCESS
                    ? std::to_string(utilization.gpu)
                    : "NA";
                const std::string sm = sm_clock_result == NVML_SUCCESS
                    ? std::to_string(sm_clock)
                    : "NA";
                const std::string mem = memory_clock_result == NVML_SUCCESS
                    ? std::to_string(memory_clock)
                    : "NA";

                std::ostringstream throttle_stream;
                if (throttle_result == NVML_SUCCESS) {
                    throttle_stream << "0x" << std::hex << throttle_reasons;
                } else {
                    throttle_stream << "NA";
                }

                std::cout << std::fixed << std::setprecision(1)
                          << std::setw(9) << elapsed << "  "
                          << std::setw(6) << temperature << "  "
                          << std::setw(7) << power << "  "
                          << std::setw(6) << util << "  "
                          << std::setw(6) << sm << "  "
                          << std::setw(7) << mem << "  "
                          << throttle_stream.str() << '\n';

                if (csv) {
                    csv << std::fixed << std::setprecision(3) << elapsed << ','
                        << temperature << ',' << power << ',' << util << ','
                        << sm << ',' << mem << ',' << throttle_stream.str() << '\n';
                    csv.flush();
                }

                do {
                    next_sample += std::chrono::milliseconds(options.sample_ms);
                } while (next_sample <= now);
            }
        }

        if (stop_requested) {
            stop_reason = "interrupt received";
        }
        std::cout << "\nStopped: " << stop_reason << ".\n";

        check_cuda(cudaFree(device_output), "cudaFree");
        device_output = nullptr;
        check_nvml(nvmlShutdown(), "nvmlShutdown");
        nvml_initialized = false;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        if (device_output != nullptr) {
            cudaFree(device_output);
        }
        if (nvml_initialized) {
            nvmlShutdown();
        }
        return 1;
    }
}
