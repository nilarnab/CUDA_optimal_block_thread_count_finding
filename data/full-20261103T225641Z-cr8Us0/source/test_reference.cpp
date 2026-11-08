#include "reference.hpp"
#include <cassert>
#include <iostream>
#include <limits>

int main() {
    using namespace pilot;
    const int n = 3;
    for (const auto& name : kernel_names()) {
        const auto kernel = parse_kernel(name);
        const auto data = make_case<double>(kernel, n, 101);
        const auto again = make_case<double>(kernel, n, 101);
        assert(data.a == again.a && data.b == again.b && data.expected == again.expected);
        assert(validate(data.expected, data.expected) == 0);
        if (kernel == Kernel::VectorAdd || kernel == Kernel::MatrixAdd) {
            for (int i = 0; i < n * n; ++i) assert(data.expected[i] == data.a[i] + data.b[i]);
        } else if (kernel == Kernel::SegmentedSum) {
            for (int day = 0; day < n; ++day)
                assert(data.expected[day] == data.a[day * n] + data.a[day * n + 1] + data.a[day * n + 2]);
        } else if (kernel == Kernel::Transpose) {
            for (int row = 0; row < n; ++row)
                for (int col = 0; col < n; ++col)
                    assert(data.expected[col * n + row] == data.a[row * n + col]);
        } else if (kernel == Kernel::RepeatVector) {
            for (int repeat = 0; repeat < n; ++repeat)
                for (int i = 0; i < n; ++i) assert(data.expected[repeat * n + i] == data.a[i]);
        } else {
            const auto tile = make_case<float>(kernel, 19, 101);
            assert(tile.expected[17] == tile.a[0]);
            assert(tile.expected[13 * 19 + 18] == tile.a[1]);
        }
    }
    bool rejected = false;
    try { validate(std::vector<float>{std::numeric_limits<float>::quiet_NaN()}, std::vector<float>{0}); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { validate(std::vector<int>{5}, std::vector<int>{4}); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    assert(allowed_size(240) && allowed_size(2024) && !allowed_size(3));
    std::cout << "CPU references: all six kernels and invalid-output checks passed\n";
}
