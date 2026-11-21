#include <cuda_runtime.h>

namespace {
constexpr unsigned lanes = 32;
constexpr unsigned max_block = 576;
constexpr unsigned patch_stride = 16;
constexpr unsigned patch_height = 15;
constexpr unsigned histogram_rows = 13;
constexpr unsigned scratch_per_warp = patch_stride * patch_height + lanes * histogram_rows + patch_stride;
static_assert(max_block / lanes * scratch_per_warp * sizeof(float) <= 48 * 1024);
}

// Adapted from LS-CAT 10319/29. Each warp computes one keypoint orientation.
__global__ __launch_bounds__(max_block) void ComputeOrientations(
    const float* image, const int* points, float* orientations, int count, int width, int height) {
    const int lane = threadIdx.x % lanes;
    const int warp = threadIdx.x / lanes;
    const int point = blockIdx.x * (blockDim.x / lanes) + warp;
    if (point >= count) return;

    extern __shared__ float scratch[];
    float* patch = scratch + warp * scratch_per_warp;
    float* hist = patch + patch_stride * patch_height;
    float* gauss = hist + lanes * histogram_rows;
    for (int column = 0; column < 13; ++column) hist[column * 32 + lane] = 0.0f;
    if (lane < 15) gauss[lane] = expf(-(lane - 7) * (lane - 7) / 18.0f);

    const int position = points[point];
    const int y_start = position / width - 7;
    const int x_start = position % width - 7;
    const int x = lane - (x_start & 15);
    if (x >= 0 && x < 15) {
        int ix = x_start + x;
        if (ix < 0) ix = 0;
        if (ix >= width) ix = width - 1;
        for (int y = 0; y < 15; ++y) {
            int iy = y_start + y;
            if (iy < 0) iy = 0;
            if (iy >= height) iy = height - 1;
            patch[16 * y + x] = image[iy * width + ix];
        }
    }
    __syncwarp();

    if (x >= 1 && x < 14) {
        for (int y = 1; y < 14; ++y) {
            const int p = 16 * y + x;
            const float dy = patch[p + 16] - patch[p - 16];
            const float dx = patch[p + 1] - patch[p - 1];
            const int bin = static_cast<int>(16.0f * atan2f(dy, dx) / 3.1416f + 16.5f) & 31;
            const float weight = sqrtf(dx * dx + dy * dy) * gauss[x] * gauss[y];
            hist[32 * (x - 1) + bin] += weight;
        }
    }
    __syncwarp();
    for (int i = 0; i < 5; ++i) hist[i * 32 + lane] += hist[(i + 8) * 32 + lane];
    for (int i = 0; i < 4; ++i) hist[i * 32 + lane] += hist[(i + 4) * 32 + lane];
    for (int i = 0; i < 2; ++i) hist[i * 32 + lane] += hist[(i + 2) * 32 + lane];
    hist[lane] += hist[32 + lane];
    __syncwarp();

    const float smoothed = 6 * hist[lane] + 4 * (hist[(lane + 1) & 31] + hist[(lane + 31) & 31]) +
                           hist[(lane + 2) & 31] + hist[(lane + 30) & 31];
    hist[32 + lane] = smoothed;
    __syncwarp();
    hist[lane] = smoothed > hist[32 + ((lane + 1) & 31)] &&
                 smoothed >= hist[32 + ((lane + 31) & 31)] ? smoothed : 0.0f;
    __syncwarp();

    if (lane == 0) {
        int first = -1, second = -1;
        float first_value = 0.0f, second_value = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const float value = hist[i];
            if (value > first_value) {
                second = first;
                second_value = first_value;
                first = i;
                first_value = value;
            } else if (value > second_value) {
                second = i;
                second_value = value;
            }
        }
        orientations[point] = -1.0f;
        orientations[point + count] = -1.0f;
        if (first >= 0) {
            const float right = hist[32 + ((first + 1) & 31)];
            const float left = hist[32 + ((first + 31) & 31)];
            const float peak = first + 0.5f * (right - left) / (2.0f * first_value - right - left);
            orientations[point] = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);
        }
        if (second >= 0 && second_value >= 0.8f * first_value) {
            const float right = hist[32 + ((second + 1) & 31)];
            const float left = hist[32 + ((second + 31) & 31)];
            const float peak = second + 0.5f * (right - left) / (2.0f * second_value - right - left);
            orientations[point + count] = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);
        }
    }
}

#ifndef __CUDA_ARCH__
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "runtime_benchmark.hpp"
#ifndef RUNTIME_LAUNCH_FIXED
#include "toolcall.hpp"
#endif

using runtime_benchmark::check;
using runtime_benchmark::parse_size;

namespace {
std::pair<float, float> reference(const std::vector<float>& image, int side, int position) {
    std::array<float, 15> gauss;
    for (int i = 0; i < 15; ++i) gauss[i] = std::exp(-(i - 7) * (i - 7) / 18.0f);
    const int origin_x = position % side - 7;
    const int origin_y = position / side - 7;
    const auto pixel = [&](int x, int y) {
        return image[std::clamp(origin_y + y, 0, side - 1) * side +
                     std::clamp(origin_x + x, 0, side - 1)];
    };

    std::array<float, lanes> histogram{};
    for (int x = 1; x < 14; ++x) {
        for (int y = 1; y < 14; ++y) {
            const float dx = pixel(x + 1, y) - pixel(x - 1, y);
            const float dy = pixel(x, y + 1) - pixel(x, y - 1);
            const int bin = static_cast<int>(16.0f * std::atan2(dy, dx) / 3.1416f + 16.5f) & 31;
            histogram[bin] += std::sqrt(dx * dx + dy * dy) * gauss[x] * gauss[y];
        }
    }
    std::array<float, lanes> smooth;
    for (int i = 0; i < 32; ++i) {
        smooth[i] = 6 * histogram[i] + 4 * (histogram[(i + 1) & 31] + histogram[(i + 31) & 31]) +
                    histogram[(i + 2) & 31] + histogram[(i + 30) & 31];
    }
    int first = -1, second = -1;
    for (int i = 0; i < 32; ++i) {
        if (smooth[i] <= smooth[(i + 1) & 31] || smooth[i] < smooth[(i + 31) & 31]) continue;
        if (first < 0 || smooth[i] > smooth[first]) {
            second = first;
            first = i;
        } else if (second < 0 || smooth[i] > smooth[second]) {
            second = i;
        }
    }
    const auto angle = [&](int i) {
        if (i < 0) return -1.0f;
        const float right = smooth[(i + 1) & 31], left = smooth[(i + 31) & 31];
        const float peak = i + 0.5f * (right - left) / (2.0f * smooth[i] - right - left);
        return 11.25f * (peak < 0.0f ? peak + 32.0f : peak);
    };
    if (second >= 0 && smooth[second] < 0.8f * smooth[first]) second = -1;
    return {angle(first), angle(second)};
}

bool matches_angle(float actual, float expected) {
    if (actual < 0.0f || expected < 0.0f) return actual == expected;
    return std::abs(std::remainder(actual - expected, 360.0f)) <= 0.02f;
}
}

int main(int argc, char** argv) try {
#ifdef RUNTIME_LAUNCH_FIXED
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s KEYPOINTS IMAGE_SIDE FIXED_BLOCK\n", argv[0]);
#else
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s KEYPOINTS IMAGE_SIDE\n", argv[0]);
#endif
        return 2;
    }

    const auto count = parse_size(argv[1]);
    const auto side = parse_size(argv[2]);
    constexpr auto max_index = std::size_t(std::numeric_limits<int>::max());
    if (!count || !side || count > max_index / 2 || side > max_index / side) {
        std::fprintf(stderr, "KEYPOINTS and IMAGE_SIDE must be positive with room for 32-bit indexing\n");
        return 2;
    }

    const auto work_items = count * lanes;
#ifdef RUNTIME_LAUNCH_FIXED
    constexpr auto mode = "fixed";
    const auto requested = parse_size(argv[3]);
    if (!requested || requested % lanes || requested > max_block) {
        std::fprintf(stderr, "FIXED_BLOCK must be a multiple of 32 between 32 and 576\n");
        return 2;
    }
    const auto fixed = runtime_benchmark::fixed_configuration(ComputeOrientations, work_items, requested);
    const auto select = [fixed] { return fixed; };
#else
    constexpr auto mode = "model";
    const auto select = [&] { return toolcall::model_output(ComputeOrientations, {count, side, side, work_items}); };
#endif
    const auto initial = select();

    std::vector<float> image(side * side), orientations(count * 2);
    std::vector<int> points(count);
    for (std::size_t i = 0; i < image.size(); ++i) {
        image[i] = static_cast<float>((i ^ (i >> 5) ^ (i >> 13)) & 255);
    }
    for (std::size_t i = 0; i < points.size(); ++i) {
        points[i] = static_cast<int>((std::uint64_t(i) * 104729) % image.size());
    }

    float *device_image, *device_output;
    int* device_points;
    const auto image_bytes = image.size() * sizeof(float);
    const auto point_bytes = points.size() * sizeof(int);
    const auto output_bytes = orientations.size() * sizeof(float);
    check(cudaMalloc(&device_image, image_bytes));
    check(cudaMalloc(&device_points, point_bytes));
    check(cudaMalloc(&device_output, output_bytes));
    check(cudaMemcpy(device_image, image.data(), image_bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_points, points.data(), point_bytes, cudaMemcpyHostToDevice));
    check(cudaMemset(device_output, 0xff, output_bytes));

    const auto launch = [&](runtime_benchmark::Configuration configuration) {
        const auto [grid, block] = configuration;
        const auto shared_bytes = (block / lanes) * scratch_per_warp * sizeof(float);
        ComputeOrientations<<<grid, block, shared_bytes>>>(
            device_image, device_points, device_output, static_cast<int>(count),
            static_cast<int>(side), static_cast<int>(side));
        check(cudaGetLastError());
    };
    launch(initial);
    const auto timings = runtime_benchmark::measure(select, launch);
    check(cudaMemcpy(orientations.data(), device_output, output_bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(device_image));
    check(cudaFree(device_points));
    check(cudaFree(device_output));

    if (!std::all_of(orientations.begin(), orientations.end(),
                     [](float angle) { return angle == -1.0f || (angle >= 0.0f && angle <= 360.0f); })) {
        std::fprintf(stderr, "ComputeOrientations produced an unwritten or invalid angle\n");
        return 1;
    }
    // shortcut: large runs compare 64 keypoints to the CPU, increase coverage for accuracy studies.
    const auto samples = std::min(count, std::size_t{64});
    for (std::size_t i = 0; i < samples; ++i) {
        const auto point = samples == 1 ? 0 : i * (count - 1) / (samples - 1);
        const auto [primary, secondary] = reference(image, static_cast<int>(side), points[point]);
        if (!matches_angle(orientations[point], primary) ||
            !matches_angle(orientations[point + count], secondary)) {
            std::fprintf(stderr, "Keypoint %zu got (%g, %g), expected (%g, %g)\n", point,
                         orientations[point], orientations[point + count], primary, secondary);
            return 1;
        }
    }

    const auto [grid, block] = initial;
    std::printf("%s trials=%d M=%zu N=%zu elements=%zu grid=%u block=%u checked=%zu verified\n",
                mode, runtime_benchmark::trials, count, side, image.size(), grid, block, samples);
    runtime_benchmark::print_timings(timings);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
#endif
