#pragma once

#ifndef __CUDA_ARCH__
#include "launch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace compute_orientations_launch {

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

template <bool Fixed, class Launch>
int benchmark(int argc, char** argv, Launch launch_kernel) try {
    if (argc != (Fixed ? 4 : 3)) {
        std::fprintf(stderr, "Usage: %s KEYPOINTS IMAGE_SIDE%s\n",
                     argv[0], Fixed ? " FIXED_BLOCK" : "");
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
    const auto requested = Fixed ? parse_size(argv[3]) : 0;
    if (Fixed && (!requested || requested % lanes || requested > max_block)) {
        std::fprintf(stderr, "FIXED_BLOCK must be a multiple of 32 between 32 and 576\n");
        return 2;
    }
    constexpr auto mode = Fixed ? "fixed" : "model";
    const auto select = [&] {
        if constexpr (Fixed) {
            const auto fixed = select_fixed(work_items, requested);
            return [fixed] { return fixed; };
        } else {
            return [&] { return select_model(count, side, work_items); };
        }
    }();
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

    const auto launch = [&](Configuration configuration) {
        launch_kernel(configuration, device_image, device_points, device_output,
                      static_cast<int>(count), static_cast<int>(side));
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

}  // namespace compute_orientations_launch
#endif
