#pragma once

#include <cuda_runtime.h>

#include "launch.hpp"

// Adapted from LS-CAT 6580/8. Threads cover tile_width columns and blockDim.y rows.
__global__ __launch_bounds__(1024) void matrixMultiplyAndRelu(
    const float* weights, const float* activations, float* output, const float* bias,
    float* preactivation,
    int rows, int columns, int inner) {
    extern __shared__ float shared[];
    const int threads = tile_width * blockDim.y;
    float* weight_tile = shared;
    float* activation_tile = shared + threads;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tile_index = ty * tile_width + tx;
    const int row = blockIdx.y * blockDim.y + ty;
    const int column_start = blockIdx.x * tile_width;
    const int col = column_start + tx;
    const int tiles = 1 + (inner - 1) / tile_width;
    float value = 0.0f;

    for (int tile = 0; tile < tiles; ++tile) {
        const int k_start = tile * tile_width;
        const int weight_col = k_start + tx;
        weight_tile[tile_index] = row < rows && weight_col < inner
            ? weights[row * inner + weight_col] : 0.0f;

        for (int index = tile_index; index < tile_width * tile_width; index += threads) {
            const int activation_row = k_start + index / tile_width;
            const int activation_col = column_start + index % tile_width;
            activation_tile[index] = activation_row < inner && activation_col < columns
                ? activations[activation_row * columns + activation_col] : 0.0f;
        }
        __syncthreads();
        // Partially unroll the 32-step reduction to limit the generated code size.
        #pragma unroll 16
        for (int k = 0; k < tile_width; ++k) {
            value += weight_tile[ty * tile_width + k] * activation_tile[k * tile_width + tx];
        }
        __syncthreads();
    }

    if (row < rows && col < columns) {
        const int index = row * columns + col;
        const float biased = value + bias[row];
        preactivation[index] = biased;
        output[index] = fmaxf(biased, 0.0f);
    }
}
