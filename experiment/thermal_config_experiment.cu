#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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

int parse_nonnegative_int(const std::string& value, const char* option) {
    std::size_t parsed_chars = 0;
    const long parsed = std::stol(value, &parsed_chars);
    if (parsed_chars != value.size() || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("Invalid value for ") + option +
                                 ": " + value);
    }
    return static_cast<int>(parsed);
}

std::size_t parse_size(const std::string& value, const char* option) {
    std::size_t parsed_chars = 0;
    const unsigned long long parsed = std::stoull(value, &parsed_chars);
    if (parsed_chars != value.size() || parsed == 0) {
        throw std::runtime_error(std::string("Invalid value for ") + option +
                                 ": " + value);
    }
    return static_cast<std::size_t>(parsed);
}

std::vector<int> parse_int_list(const std::string& text, const char* option) {
    std::vector<int> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const int value = parse_nonnegative_int(item, option);
        if (value == 0) {
            throw std::runtime_error(std::string(option) +
                                     " values must be positive");
        }
        values.push_back(value);
    }
    if (values.empty()) {
        throw std::runtime_error(std::string(option) + " cannot be empty");
    }
    return values;
}

struct Options {
    int device = 0;
    std::vector<int> temperatures{45, 55, 65};
    std::vector<int> thread_counts{64, 128, 256, 512};
    std::vector<int> blocks_per_sm{2, 4};
    int repetitions = 5;
    int launches_per_trial = 10;
    std::size_t elements = 1u << 24;
    int inner_iterations = 512;
    int temperature_band_c = 1;
    int stabilize_seconds = 3;
    int prepare_timeout_seconds = 300;
    int max_temp_c = 75;
    int heat_iterations = 5000;
    unsigned int seed = 42;
    std::string csv_path = "results.csv";
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "  --device N                 CUDA device index (default: 0)\n"
        << "  --temperatures A,B,C       Target temperatures (default: 45,55,65)\n"
        << "  --threads A,B,C            Thread counts (default: 64,128,256,512)\n"
        << "  --blocks-per-sm A,B        Grid sizes in blocks/SM (default: 2,4)\n"
        << "  --repetitions N            Trials per configuration/temperature (default: 5)\n"
        << "  --launches N               Timed launches in each trial (default: 10)\n"
        << "  --elements N               Output elements per launch (default: 16777216)\n"
        << "  --inner-iters N            Compute work per element (default: 512)\n"
        << "  --temp-band C              Allowed target deviation (default: 1)\n"
        << "  --stabilize-seconds S      Required time inside band (default: 3)\n"
        << "  --prepare-timeout S        Limit for reaching each target (default: 300)\n"
        << "  --max-temp C               Emergency stop temperature (default: 75)\n"
        << "  --heat-iters N             Work per heating chunk (default: 5000)\n"
        << "  --seed N                   Randomization seed (default: 42)\n"
        << "  --csv PATH                 Output file (default: results.csv)\n"
        << "  --help                     Show this help\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value after " + argument);
        }
        const std::string value = argv[++i];
        if (argument == "--device") {
            options.device = parse_nonnegative_int(value, "--device");
        } else if (argument == "--temperatures") {
            options.temperatures = parse_int_list(value, "--temperatures");
        } else if (argument == "--threads") {
            options.thread_counts = parse_int_list(value, "--threads");
        } else if (argument == "--blocks-per-sm") {
            options.blocks_per_sm = parse_int_list(value, "--blocks-per-sm");
        } else if (argument == "--repetitions") {
            options.repetitions = parse_nonnegative_int(value, "--repetitions");
        } else if (argument == "--launches") {
            options.launches_per_trial = parse_nonnegative_int(value, "--launches");
        } else if (argument == "--elements") {
            options.elements = parse_size(value, "--elements");
        } else if (argument == "--inner-iters") {
            options.inner_iterations = parse_nonnegative_int(value, "--inner-iters");
        } else if (argument == "--temp-band") {
            options.temperature_band_c = parse_nonnegative_int(value, "--temp-band");
        } else if (argument == "--stabilize-seconds") {
            options.stabilize_seconds =
                parse_nonnegative_int(value, "--stabilize-seconds");
        } else if (argument == "--prepare-timeout") {
            options.prepare_timeout_seconds =
                parse_nonnegative_int(value, "--prepare-timeout");
        } else if (argument == "--max-temp") {
            options.max_temp_c = parse_nonnegative_int(value, "--max-temp");
        } else if (argument == "--heat-iters") {
            options.heat_iterations = parse_nonnegative_int(value, "--heat-iters");
        } else if (argument == "--seed") {
            options.seed = static_cast<unsigned int>(
                parse_nonnegative_int(value, "--seed"));
        } else if (argument == "--csv") {
            options.csv_path = value;
        } else {
            throw std::runtime_error("Unknown option: " + argument);
        }
    }

    if (options.repetitions <= 0 || options.launches_per_trial <= 0 ||
        options.inner_iterations <= 0 || options.prepare_timeout_seconds <= 0 ||
        options.heat_iterations <= 0 || options.csv_path.empty()) {
        throw std::runtime_error("Repetitions, launches, work settings, timeout, "
                                 "and CSV path must be positive/nonempty");
    }
    if (options.max_temp_c > 90) {
        throw std::runtime_error("Refusing --max-temp above 90 C");
    }
    for (const int target : options.temperatures) {
        if (target + options.temperature_band_c >= options.max_temp_c) {
            throw std::runtime_error(
                "Every target temperature plus its band must be below --max-temp");
        }
    }
    return options;
}

struct Configuration {
    int threads;
    int blocks_per_sm;
};

struct Trial {
    int repetition;
    Configuration configuration;
};

struct Telemetry {
    unsigned int temperature_c = 0;
    std::optional<double> power_w;
    std::optional<unsigned int> utilization_pct;
    std::optional<unsigned int> sm_clock_mhz;
    std::optional<unsigned int> memory_clock_mhz;
    std::optional<unsigned long long> throttle_reasons;
};

Telemetry read_telemetry(nvmlDevice_t device) {
    Telemetry sample;
    check_nvml(nvmlDeviceGetTemperature(
                   device, NVML_TEMPERATURE_GPU, &sample.temperature_c),
               "nvmlDeviceGetTemperature");

    unsigned int value = 0;
    if (nvmlDeviceGetPowerUsage(device, &value) == NVML_SUCCESS) {
        sample.power_w = static_cast<double>(value) / 1000.0;
    }

    nvmlUtilization_t utilization{};
    if (nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS) {
        sample.utilization_pct = utilization.gpu;
    }

    if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &value) == NVML_SUCCESS) {
        sample.sm_clock_mhz = value;
    }
    if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &value) == NVML_SUCCESS) {
        sample.memory_clock_mhz = value;
    }

    unsigned long long throttle_reasons = 0;
    if (nvmlDeviceGetCurrentClocksThrottleReasons(device, &throttle_reasons) ==
        NVML_SUCCESS) {
        sample.throttle_reasons = throttle_reasons;
    }
    return sample;
}

std::string optional_value(const std::optional<unsigned int>& value) {
    return value ? std::to_string(*value) : "NA";
}

std::string optional_value(const std::optional<double>& value) {
    if (!value) {
        return "NA";
    }
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << *value;
    return stream.str();
}

std::string throttle_value(
    const std::optional<unsigned long long>& throttle_reasons) {
    if (!throttle_reasons) {
        return "NA";
    }
    std::ostringstream stream;
    stream << "0x" << std::hex << *throttle_reasons;
    return stream.str();
}

// This kernel is used only to move the GPU toward a requested temperature.
__global__ void heating_kernel(float* output, int iterations) {
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

// A grid-stride loop keeps the problem identical while allowing block count
// and threads per block to vary independently.
__global__ void compute_benchmark_kernel(float* output, std::size_t elements,
                                         int inner_iterations) {
    const std::size_t start =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t index = start; index < elements; index += stride) {
        float x = 1.0001f + static_cast<float>(index & 15u) * 0.000001f;
        float y = 0.9999f;
        for (int i = 0; i < inner_iterations; ++i) {
            x = fmaf(x, 1.0000001f, y * 0.0000001f);
            y = fmaf(y, 0.9999999f, x * 0.0000001f);
        }
        output[index] = x + y;
    }
}

bool prepare_temperature(const Options& options, nvmlDevice_t nvml_device,
                         float* heat_output, int heat_blocks,
                         int target_temperature) {
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    std::optional<Clock::time_point> stable_since;
    auto next_status = started;

    while (!stop_requested) {
        const auto now = Clock::now();
        const int elapsed_seconds = static_cast<int>(
            std::chrono::duration_cast<std::chrono::seconds>(now - started).count());
        if (elapsed_seconds >= options.prepare_timeout_seconds) {
            throw std::runtime_error(
                "Timed out while trying to stabilize at " +
                std::to_string(target_temperature) + " C. The target may be "
                "below the idle temperature or above the reachable temperature.");
        }

        const Telemetry sample = read_telemetry(nvml_device);
        const int temperature = static_cast<int>(sample.temperature_c);
        if (temperature >= options.max_temp_c) {
            throw std::runtime_error("Emergency temperature limit reached");
        }

        const bool inside_band =
            temperature >= target_temperature - options.temperature_band_c &&
            temperature <= target_temperature + options.temperature_band_c;

        if (now >= next_status) {
            std::cout << "  preparing " << target_temperature << " C: now "
                      << temperature << " C"
                      << (inside_band ? " (stabilizing)" : "") << '\n';
            next_status = now + std::chrono::seconds(1);
        }

        if (inside_band) {
            if (!stable_since) {
                stable_since = now;
            }
            const int stable_seconds = static_cast<int>(
                std::chrono::duration_cast<std::chrono::seconds>(
                    now - *stable_since)
                    .count());
            if (stable_seconds >= options.stabilize_seconds) {
                return true;
            }

            if (temperature <= target_temperature) {
                heating_kernel<<<heat_blocks, 256>>>(
                    heat_output, options.heat_iterations);
                check_cuda(cudaGetLastError(), "heating kernel launch");
                check_cuda(cudaDeviceSynchronize(), "heating kernel execution");
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        } else {
            stable_since.reset();
            if (temperature < target_temperature - options.temperature_band_c) {
                heating_kernel<<<heat_blocks, 256>>>(
                    heat_output, options.heat_iterations);
                check_cuda(cudaGetLastError(), "heating kernel launch");
                check_cuda(cudaDeviceSynchronize(), "heating kernel execution");
            } else {
                // Cooling cannot be forced safely; wait for the installed cooler.
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
    }
    return false;
}

void write_telemetry(std::ofstream& csv, const Telemetry& sample) {
    csv << sample.temperature_c << ',' << optional_value(sample.power_w) << ','
        << optional_value(sample.utilization_pct) << ','
        << optional_value(sample.sm_clock_mhz) << ','
        << optional_value(sample.memory_clock_mhz) << ','
        << throttle_value(sample.throttle_reasons);
}

}  // namespace

int main(int argc, char** argv) {
    float* benchmark_output = nullptr;
    float* heat_output = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    bool nvml_initialized = false;

    try {
        const Options options = parse_options(argc, argv);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        check_cuda(cudaSetDevice(options.device), "cudaSetDevice");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, options.device),
                   "cudaGetDeviceProperties");

        for (const int threads : options.thread_counts) {
            if (threads > properties.maxThreadsPerBlock) {
                throw std::runtime_error(
                    "Requested thread count " + std::to_string(threads) +
                    " exceeds this GPU's per-block limit of " +
                    std::to_string(properties.maxThreadsPerBlock));
            }
        }

        char pci_bus_id[32]{};
        check_cuda(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id),
                                         options.device),
                   "cudaDeviceGetPCIBusId");
        check_nvml(nvmlInit_v2(), "nvmlInit_v2");
        nvml_initialized = true;
        nvmlDevice_t nvml_device{};
        check_nvml(nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, &nvml_device),
                   "nvmlDeviceGetHandleByPciBusId_v2");

        const int heat_blocks = properties.multiProcessorCount * 8;
        const std::size_t heat_values =
            static_cast<std::size_t>(heat_blocks) * 256;
        check_cuda(cudaMalloc(&heat_output, heat_values * sizeof(float)),
                   "cudaMalloc heat buffer");
        check_cuda(cudaMalloc(&benchmark_output,
                              options.elements * sizeof(float)),
                   "cudaMalloc benchmark buffer");
        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");

        std::vector<Configuration> configurations;
        for (const int threads : options.thread_counts) {
            for (const int blocks_per_sm : options.blocks_per_sm) {
                configurations.push_back({threads, blocks_per_sm});
            }
        }

        std::ofstream csv(options.csv_path);
        if (!csv) {
            throw std::runtime_error("Cannot open CSV file: " + options.csv_path);
        }
        csv << "sequence,repetition,target_temp_c,threads,blocks_per_sm,blocks,"
               "launches,elements,inner_iterations,total_ms,runtime_ms,"
               "pre_temp_c,pre_power_w,pre_util_pct,pre_sm_clock_mhz,"
               "pre_memory_clock_mhz,pre_throttle_reasons,post_temp_c,"
               "post_power_w,post_util_pct,post_sm_clock_mhz,"
               "post_memory_clock_mhz,post_throttle_reasons\n";

        std::cout << "GPU: " << properties.name << " (PCI " << pci_bus_id
                  << ", " << properties.multiProcessorCount << " SMs)\n"
                  << "Configurations: " << configurations.size()
                  << "; repetitions: " << options.repetitions
                  << "; output: " << options.csv_path << "\n"
                  << "Press Ctrl-C to stop safely between kernel launches.\n";

        // Initialize the context and executable before recording any trial.
        compute_benchmark_kernel<<<properties.multiProcessorCount * 2, 256>>>(
            benchmark_output, std::min<std::size_t>(options.elements, 1u << 20),
            std::min(options.inner_iterations, 32));
        check_cuda(cudaGetLastError(), "warm-up launch");
        check_cuda(cudaDeviceSynchronize(), "warm-up execution");

        std::mt19937 random(options.seed);
        std::size_t sequence = 0;

        for (const int target_temperature : options.temperatures) {
            std::vector<Trial> trials;
            for (int repetition = 1; repetition <= options.repetitions;
                 ++repetition) {
                for (const Configuration& configuration : configurations) {
                    trials.push_back({repetition, configuration});
                }
            }
            std::shuffle(trials.begin(), trials.end(), random);

            std::cout << "\nTarget temperature: " << target_temperature
                      << " C (" << trials.size() << " randomized trials)\n";

            for (const Trial& trial : trials) {
                if (!prepare_temperature(options, nvml_device, heat_output,
                                         heat_blocks, target_temperature)) {
                    break;
                }

                const int blocks = properties.multiProcessorCount *
                                   trial.configuration.blocks_per_sm;
                const Telemetry before = read_telemetry(nvml_device);
                if (before.temperature_c >=
                    static_cast<unsigned int>(options.max_temp_c)) {
                    throw std::runtime_error(
                        "Emergency temperature limit reached before benchmark");
                }

                check_cuda(cudaEventRecord(start_event), "record start event");
                for (int launch = 0; launch < options.launches_per_trial;
                     ++launch) {
                    compute_benchmark_kernel<<<blocks,
                                               trial.configuration.threads>>>(
                        benchmark_output, options.elements,
                        options.inner_iterations);
                }
                check_cuda(cudaGetLastError(), "benchmark kernel launch");
                check_cuda(cudaEventRecord(stop_event), "record stop event");
                check_cuda(cudaEventSynchronize(stop_event),
                           "synchronize stop event");

                float total_ms = 0.0f;
                check_cuda(cudaEventElapsedTime(&total_ms, start_event, stop_event),
                           "measure benchmark time");
                const float runtime_ms =
                    total_ms / static_cast<float>(options.launches_per_trial);
                const Telemetry after = read_telemetry(nvml_device);
                if (after.temperature_c >=
                    static_cast<unsigned int>(options.max_temp_c)) {
                    throw std::runtime_error(
                        "Emergency temperature limit reached after benchmark");
                }

                ++sequence;
                csv << sequence << ',' << trial.repetition << ','
                    << target_temperature << ',' << trial.configuration.threads
                    << ',' << trial.configuration.blocks_per_sm << ',' << blocks
                    << ',' << options.launches_per_trial << ',' << options.elements
                    << ',' << options.inner_iterations << ',' << std::fixed
                    << std::setprecision(6) << total_ms << ',' << runtime_ms << ',';
                write_telemetry(csv, before);
                csv << ',';
                write_telemetry(csv, after);
                csv << '\n';
                csv.flush();

                std::cout << "  trial " << sequence << ": "
                          << trial.configuration.threads << " threads, "
                          << trial.configuration.blocks_per_sm
                          << " blocks/SM -> " << std::fixed
                          << std::setprecision(3) << runtime_ms << " ms/launch"
                          << " (" << before.temperature_c << " -> "
                          << after.temperature_c << " C)\n";
            }
            if (stop_requested) {
                break;
            }
        }

        std::cout << "\n"
                  << (stop_requested ? "Stopped by user. Partial results saved to "
                                     : "Experiment complete. Results saved to ")
                  << options.csv_path << "\n";

        check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy start");
        start_event = nullptr;
        check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop");
        stop_event = nullptr;
        check_cuda(cudaFree(benchmark_output), "cudaFree benchmark buffer");
        benchmark_output = nullptr;
        check_cuda(cudaFree(heat_output), "cudaFree heat buffer");
        heat_output = nullptr;
        check_nvml(nvmlShutdown(), "nvmlShutdown");
        nvml_initialized = false;
        return stop_requested ? 130 : 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        if (start_event != nullptr) {
            cudaEventDestroy(start_event);
        }
        if (stop_event != nullptr) {
            cudaEventDestroy(stop_event);
        }
        if (benchmark_output != nullptr) {
            cudaFree(benchmark_output);
        }
        if (heat_output != nullptr) {
            cudaFree(heat_output);
        }
        if (nvml_initialized) {
            nvmlShutdown();
        }
        return 1;
    }
}
