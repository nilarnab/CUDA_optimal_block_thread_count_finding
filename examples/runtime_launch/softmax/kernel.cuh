#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

__global__ void softmax_kernel(const float* input, std::size_t vectors,
                               std::size_t width, float* output) {
    const auto vector = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vector >= vectors) return;

    const float* logits = input + vector * width;
    float* probabilities = output + vector * width;
    float peak = logits[0];
    for (std::size_t i = 1; i < width; ++i) {
        if (logits[i] > peak) peak = logits[i];
    }

    double total = 0.0;
    for (std::size_t i = 0; i < width; ++i) {
        const float value = expf(logits[i] - peak);
        probabilities[i] = value;
        total += value;
    }
    const auto inverse_total = static_cast<float>(1.0 / total);
    for (std::size_t i = 0; i < width; ++i) {
        probabilities[i] *= inverse_total;
    }
}
