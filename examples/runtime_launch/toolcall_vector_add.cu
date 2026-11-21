#include "vector_add.cu"
#include "toolcall.hpp"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string_view>
#include <vector>

void check(cudaError_t status) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
    std::exit(1);
}

int main(int argc, char** argv) try {
    std::size_t side = 0;
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s M\n", argv[0]);
        return 2;
    }
    const std::string_view input(argv[1]);
    const auto parsed = std::from_chars(input.data(), input.data() + input.size(), side);
    if (parsed.ec != std::errc{} || parsed.ptr != input.data() + input.size() ||
        !side || side > std::numeric_limits<std::size_t>::max() / sizeof(float) / side) {
        std::fprintf(stderr, "M must be a positive integer with room for M*M floats\n");
        return 2;
    }
    const std::size_t work_items = side * side;
    const std::size_t bytes = work_items * sizeof(float);
    const auto [grid, block] = toolcall::model_output(vector_add, {side, 1, 1, work_items});

    std::vector<float> input_values(work_items, 2.0f), result(work_items);
    float *input_device, *result_device;
    check(cudaMalloc(&input_device, bytes));
    check(cudaMalloc(&result_device, bytes));
    check(cudaMemcpy(input_device, input_values.data(), bytes, cudaMemcpyHostToDevice));
    vector_add<<<grid, block>>>(input_device, input_device, result_device, work_items);
    check(cudaGetLastError());
    check(cudaMemcpy(result.data(), result_device, bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(input_device));
    check(cudaFree(result_device));

    if (!std::all_of(result.begin(), result.end(), [](float value) { return value == 4.0f; })) {
        std::fprintf(stderr, "Vector add produced incorrect output\n");
        return 1;
    }
    std::printf("M=%zu elements=%zu grid=%u block=%u verified\n", side, work_items, grid, block);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
