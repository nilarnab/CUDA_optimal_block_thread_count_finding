#ifndef LS_CAT_PILOT_REFERENCE_HPP
#define LS_CAT_PILOT_REFERENCE_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace pilot {

enum class Kernel { VectorAdd, SegmentedSum, PeriodicTile, RepeatVector, Transpose, MatrixAdd };

inline const std::vector<std::string>& kernel_names() {
    static const std::vector<std::string> names = {
        "vector_add", "segmented_sum", "periodic_tile", "repeat_vector",
        "matrix_transpose", "matrix_add"};
    return names;
}

inline Kernel parse_kernel(const std::string& name) {
    const auto& names = kernel_names();
    for (std::size_t i = 0; i < names.size(); ++i)
        if (name == names[i]) return static_cast<Kernel>(i);
    throw std::invalid_argument("unknown kernel: " + name);
}

inline bool allowed_size(int n) {
    for (int value : {240, 496, 784, 1016, 1232, 1680, 2024})
        if (n == value) return true;
    return false;
}

template <typename T>
struct Case {
    std::vector<T> a, b, expected;
    std::size_t input_elements() const { return a.size() + b.size(); }
};

template <typename T>
inline void fill_values(std::vector<T>& values, std::mt19937_64& random) {
    for (auto& value : values)
        value = static_cast<T>(static_cast<int>(random() % 2049) - 1024) / static_cast<T>(128);
}

template <typename T>
inline Case<T> make_case(Kernel kernel, int n, std::uint64_t seed) {
    if (n < 1 || n > 2024) throw std::invalid_argument("reference dimension out of range");
    const std::size_t count = static_cast<std::size_t>(n) * n;
    Case<T> data;
    data.expected.resize(kernel == Kernel::SegmentedSum ? static_cast<std::size_t>(n) : count);
    data.a.resize(kernel == Kernel::PeriodicTile ? 17 * 13 :
                  kernel == Kernel::RepeatVector ? static_cast<std::size_t>(n) : count);
    if (kernel == Kernel::VectorAdd || kernel == Kernel::MatrixAdd) data.b.resize(count);
    std::mt19937_64 random(seed);
    fill_values(data.a, random);
    fill_values(data.b, random);
    if (kernel == Kernel::SegmentedSum) {
        for (auto& value : data.a) value = static_cast<T>(static_cast<int>(random() % 17) - 8);
    }
    for (std::size_t i = 0; i < data.expected.size(); ++i) {
        switch (kernel) {
        case Kernel::VectorAdd:
        case Kernel::MatrixAdd:
            data.expected[i] = data.a[i] + data.b[i];
            break;
        case Kernel::SegmentedSum: {
            T sum = 0;
            for (int j = 0; j < n; ++j) sum += data.a[i * n + j];
            data.expected[i] = sum;
            break;
        }
        case Kernel::PeriodicTile:
            data.expected[i] = data.a[((i / n) % 13) * 17 + (i % n) % 17];
            break;
        case Kernel::RepeatVector:
            data.expected[i] = data.a[i % n];
            break;
        case Kernel::Transpose:
            data.expected[i] = data.a[(i % n) * n + i / n];
            break;
        }
    }
    return data;
}

template <typename T>
inline double validate(const std::vector<T>& actual, const std::vector<T>& expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("output length mismatch");
    double maximum_error = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double a = static_cast<double>(actual[i]);
        const double e = static_cast<double>(expected[i]);
        const double error = std::abs(a - e);
        if (!std::isfinite(a) || !std::isfinite(e) || error > 1e-6 + 1e-5 * std::abs(e))
            throw std::runtime_error("incorrect or non-finite output at element " + std::to_string(i));
        if (error > maximum_error) maximum_error = error;
    }
    return maximum_error;
}

} // namespace pilot
#endif
