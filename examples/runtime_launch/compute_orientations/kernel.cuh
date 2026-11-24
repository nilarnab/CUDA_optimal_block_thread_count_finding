#pragma once

#include <cuda_runtime.h>

#include "launch.hpp"

namespace {
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
