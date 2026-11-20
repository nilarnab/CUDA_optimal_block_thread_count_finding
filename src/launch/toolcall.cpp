#include "toolcall_internal.hpp"
#include "toolcall_weights.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace toolcall::detail {
namespace {

using namespace weights;

template <std::size_t Out, std::size_t In>
std::array<float, Out> linear(const float (&weight)[Out][In], const float (&bias)[Out],
                             const std::array<float, In>& input) {
    std::array<float, Out> result;
    for (std::size_t i = 0; i < Out; ++i) {
        float sum = 0;
        for (std::size_t j = 0; j < In; ++j) sum += weight[i][j] * input[j];
        result[i] = sum + bias[i];
    }
    return result;
}

template <std::size_t Size>
std::array<float, Size> relu(std::array<float, Size> values) {
    for (auto& value : values) value = std::max(0.0f, value);
    return values;
}

auto encode_context(ProblemSize problem) {
    const std::array<float, 3> log_dimensions{
        std::log1p(static_cast<float>(problem.m)),
        std::log1p(static_cast<float>(problem.n)),
        std::log1p(static_cast<float>(problem.k)),
    };
    auto context = relu(linear(context_w, context_b, log_dimensions));
    const float mean = std::accumulate(context.begin(), context.end(), 0.0f) / context.size();
    float variance = 0;
    for (auto value : context) variance += (value - mean) * (value - mean);
    const float scale = 1.0f / std::sqrt(variance / context.size() + norm_eps);
    for (std::size_t i = 0; i < context.size(); ++i) {
        context[i] = (context[i] - mean) * scale * norm_w[i] + norm_b[i];
    }
    return context;
}

auto score(ProblemSize problem, const Kernel& kernel) {
    const auto context = encode_context(problem);
    std::array<float, state_width + std::tuple_size_v<decltype(context)>> input;
    std::copy(context.begin(), context.end(), input.begin() + state_width);
    std::array<float, std::size(block_sizes)> scores{};
    float total = 0;
    float max_logit = -std::numeric_limits<float>::infinity();

    for (std::size_t row = 0; row < kernel.rows; ++row) {
        std::copy_n(kernel.states + row * state_width, state_width, input.begin());
        const auto first = relu(linear(dense0_w, dense0_b, input));
        const auto second = relu(linear(dense1_w, dense1_b, first));
        const auto third = relu(linear(dense2_w, dense2_b, second));
        const auto row_scores = linear(dense3_w, dense3_b, third);
        if (kernel.rows == 1) return row_scores;

        auto hidden = linear(attention0_w, attention0_b, input);
        for (auto& value : hidden) value = std::tanh(value);
        const float logit = linear(attention1_w, attention1_b, hidden)[0] + kernel.log_gates[row];
        // Accumulate softmax-weighted scores without allocating per-row buffers.
        const float next_max = std::max(max_logit, logit);
        const float old_scale = std::exp(max_logit - next_max);
        const float row_scale = std::exp(logit - next_max);
        for (std::size_t i = 0; i < scores.size(); ++i) {
            scores[i] = scores[i] * old_scale + row_scores[i] * row_scale;
        }
        total = total * old_scale + row_scale;
        max_logit = next_max;
    }
    for (auto& value : scores) value /= total;
    return scores;
}

}  // namespace

std::pair<unsigned, unsigned> choose_launch(
    ProblemSize problem, const Kernel& kernel, LaunchLimits limits) {
    if (!problem.work_items || !kernel.states || !kernel.log_gates ||
        !kernel.rows || kernel.width != state_width) {
        throw std::invalid_argument("invalid problem size or kernel representation");
    }
    const auto scores = score(problem, kernel);
    float best_score = -std::numeric_limits<float>::infinity();
    std::pair<unsigned, unsigned> best{};
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (!std::isfinite(scores[i])) throw std::runtime_error("non-finite launch score");
        const unsigned block = block_sizes[i];
        const auto grid = 1 + (problem.work_items - 1) / block;
        if (block <= limits.max_block_x && grid <= limits.max_grid_x && scores[i] > best_score) {
            best = {static_cast<unsigned>(grid), block};
            best_score = scores[i];
        }
    }
    if (!best.second) throw std::runtime_error("no legal block size in the checkpoint catalog");
    return best;
}

}  // namespace toolcall::detail
