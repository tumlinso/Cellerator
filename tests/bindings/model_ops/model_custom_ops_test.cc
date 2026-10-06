#include <Cellerator/bindings/torch/model_ops.hh>

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace ops = ::cellerator::bindings::torch::model_ops;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool close(double lhs, double rhs, double atol = 2.0e-4, double rtol = 2.0e-4) {
    return std::abs(lhs - rhs) <= atol + rtol * std::abs(rhs);
}

void require_close(const torch::Tensor& actual, const torch::Tensor& expected, const char* message,
    double atol = 2.0e-4, double rtol = 2.0e-4) {
    const auto a = actual.detach().to(torch::kCPU).contiguous();
    const auto e = expected.detach().to(torch::kCPU).contiguous();
    require(a.sizes() == e.sizes(), message);
    const auto* ap = a.data_ptr<float>();
    const auto* ep = e.data_ptr<float>();
    for (std::int64_t i = 0; i < a.numel(); ++i) require(close(ap[i], ep[i], atol, rtol), message);
}

struct PairReference {
    float local = 0.0f;
    float far = 0.0f;
    int local_count = 0;
    int far_count = 0;
};

PairReference pair_reference(
    const std::vector<std::int64_t>& rows,
    const std::vector<std::int64_t>& cols,
    const std::vector<float>& latent,
    const std::vector<float>& time,
    std::int64_t latent_dim,
    float local_window,
    float far_window,
    float margin) {
    PairReference result;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto r = rows[i], c = cols[i];
        float dot = 0.0f;
        for (std::int64_t d = 0; d < latent_dim; ++d) dot += latent[r * latent_dim + d] * latent[c * latent_dim + d];
        const float sqdist = std::max(2.0f - 2.0f * dot, 0.0f);
        const float delta = std::abs(time[r] - time[c]);
        if (delta <= local_window) {
            result.local += sqdist;
            ++result.local_count;
        }
        if (delta >= far_window) {
            const float dist = std::sqrt(sqdist + 1.0e-12f);
            result.far += margin > dist ? margin - dist : 0.0f;
            ++result.far_count;
        }
    }
    if (result.local_count) result.local /= static_cast<float>(result.local_count);
    if (result.far_count) result.far /= static_cast<float>(result.far_count);
    return result;
}

std::pair<float, float> pair_losses(const std::vector<float>& latent) {
    const std::vector<std::int64_t> rows{0, 0, 1, 2, 1};
    const std::vector<std::int64_t> cols{1, 2, 2, 3, 3};
    const std::vector<float> time{0.0f, 0.05f, 0.4f, 0.8f};
    const auto reference = pair_reference(rows, cols, latent, time, 2, 0.1f, 0.3f, 0.7f);
    return {reference.local, reference.far};
}

void check_pair_forward_and_gradient() {
    const std::vector<std::int64_t> rows{0, 0, 1, 2, 1};
    const std::vector<std::int64_t> cols{1, 2, 2, 3, 3};
    const std::vector<float> latent_values{1.0f, 0.0f, 0.9f, 0.4358899f, 0.8f, 0.6f, 0.7f, 0.7141428f};
    const std::vector<float> time_values{0.0f, 0.05f, 0.4f, 0.8f};
    const auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32);
    auto latent = torch::from_blob(const_cast<float*>(latent_values.data()), {4, 2}, cpu_options).clone()
        .to(torch::kCUDA).set_requires_grad(true);
    auto times = torch::from_blob(const_cast<float*>(time_values.data()), {4}, cpu_options).clone().to(torch::kCUDA);
    auto pair_rows = torch::from_blob(const_cast<std::int64_t*>(rows.data()), {5}, torch::kInt64).clone().to(torch::kCUDA);
    auto pair_cols = torch::from_blob(const_cast<std::int64_t*>(cols.data()), {5}, torch::kInt64).clone().to(torch::kCUDA);

    auto losses = ops::dense_reduce_pair_losses(pair_rows, pair_cols, latent, times, 0.1, 0.3, 0.7);
    const auto expected = pair_reference(rows, cols, latent_values, time_values, 2, 0.1f, 0.3f, 0.7f);
    require(close(std::get<0>(losses).item<float>(), expected.local), "pair local forward mismatch");
    require(close(std::get<1>(losses).item<float>(), expected.far), "pair far forward mismatch");
    (std::get<0>(losses) * 0.7f + std::get<1>(losses) * 1.3f).backward();

    constexpr float epsilon = 1.0e-3f;
    const auto grad = latent.grad().detach().to(torch::kCPU).contiguous();
    const auto* actual_grad = grad.data_ptr<float>();
    for (std::size_t i = 0; i < latent_values.size(); ++i) {
        auto plus = latent_values;
        auto minus = latent_values;
        plus[i] += epsilon;
        minus[i] -= epsilon;
        const auto lp = pair_losses(plus);
        const auto lm = pair_losses(minus);
        const float expected_grad = (0.7f * (lp.first - lm.first) + 1.3f * (lp.second - lm.second)) / (2.0f * epsilon);
        require(close(actual_grad[i], expected_grad, 2.0e-3, 2.0e-3), "pair gradient finite-difference mismatch");
    }
}

struct BucketReference {
    float ranking = 0.0f;
    float anchor = 0.0f;
    float spread = 0.0f;
};

BucketReference bucket_reference(
    const std::vector<float>& stage,
    const std::vector<std::int64_t>& buckets,
    float ranking_margin,
    float min_std,
    bool neighbors_only,
    std::int64_t declared_bucket_count) {
    const auto bucket_count = std::max<std::int64_t>(declared_bucket_count,
        buckets.empty() ? 0 : *std::max_element(buckets.begin(), buckets.end()) + 1);
    std::vector<float> sums(bucket_count, 0.0f), sumsq(bucket_count, 0.0f), means(bucket_count, 0.0f);
    std::vector<int> counts(bucket_count, 0);
    for (std::size_t i = 0; i < stage.size(); ++i) {
        const auto b = buckets[i];
        sums[b] += stage[i];
        sumsq[b] += stage[i] * stage[i];
        ++counts[b];
    }
    int active = 0;
    for (auto n : counts) active += n > 0;
    if (!active) return {};
    const auto total_buckets = declared_bucket_count > 0 ? declared_bucket_count : bucket_count;
    float anchor_sum = 0.0f;
    float spread_sum = 0.0f;
    for (std::int64_t b = 0; b < bucket_count; ++b) {
        if (counts[b] == 0) continue;
        const float inv = 1.0f / counts[b];
        means[b] = sums[b] * inv;
        const float variance = std::max(sumsq[b] * inv - means[b] * means[b], 0.0f);
        const float stddev = std::sqrt(variance + 1.0e-12f);
        const float anchor_value = total_buckets > 1 ? static_cast<float>(b) / (total_buckets - 1) : 0.5f;
        const float error = means[b] - anchor_value;
        anchor_sum += error * error;
        if (counts[b] > 1 && min_std > stddev) spread_sum += min_std - stddev;
    }
    // Accumulate ranking independently after choosing adjacent or all active pairs.
    float ranking_sum = 0.0f;
    int pairs = 0;
    std::vector<std::int64_t> active_buckets;
    for (std::int64_t b = 0; b < bucket_count; ++b) if (counts[b] > 0) active_buckets.push_back(b);
    for (std::size_t i = 0; i < active_buckets.size(); ++i) {
        const auto end = neighbors_only ? std::min(i + 2, active_buckets.size()) : active_buckets.size();
        for (std::size_t j = i + 1; j < end; ++j) {
            const auto low = active_buckets[i], high = active_buckets[j];
            const float penalty = ranking_margin - (means[high] - means[low]);
            if (penalty > 0.0f) { ranking_sum += penalty; ++pairs; }
        }
    }
    return {pairs ? ranking_sum / pairs : 0.0f, anchor_sum / active, spread_sum / active};
}

std::array<float, 3> bucket_losses(const std::vector<float>& stage, bool neighbors_only) {
    const std::vector<std::int64_t> buckets{0, 0, 1, 1, 3, 3};
    const auto result = bucket_reference(stage, buckets, 0.5f, 0.7f, neighbors_only, 5);
    return {result.ranking, result.anchor, result.spread};
}

void check_bucket_forward_and_gradient(bool neighbors_only) {
    const std::vector<float> stage_values{0.18f, 0.52f, 0.35f, 0.50f, 0.60f, 0.75f};
    const std::vector<std::int64_t> bucket_values{0, 0, 1, 1, 3, 3}; // buckets 2 and 4 are empty
    const auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32);
    auto stage = torch::from_blob(const_cast<float*>(stage_values.data()), {6}, cpu_options).clone()
        .to(torch::kCUDA).set_requires_grad(true);
    auto buckets = torch::from_blob(const_cast<std::int64_t*>(bucket_values.data()), {6}, torch::kInt64).clone().to(torch::kCUDA);
    auto losses = ops::developmental_stage_bucket_losses(stage, buckets, 0.5, 0.7, neighbors_only, 5);
    const auto expected = bucket_reference(stage_values, bucket_values, 0.5f, 0.7f, neighbors_only, 5);
    require(close(std::get<0>(losses).item<float>(), expected.ranking), "bucket ranking forward mismatch");
    require(close(std::get<1>(losses).item<float>(), expected.anchor), "bucket anchor forward mismatch");
    require(close(std::get<2>(losses).item<float>(), expected.spread), "bucket spread forward mismatch");
    (std::get<0>(losses) * 0.8f + std::get<1>(losses) * 0.6f + std::get<2>(losses) * 1.2f).backward();

    constexpr float epsilon = 1.0e-3f;
    const auto grad = stage.grad().detach().to(torch::kCPU).contiguous();
    const auto* actual_grad = grad.data_ptr<float>();
    for (std::size_t i = 0; i < stage_values.size(); ++i) {
        auto plus = stage_values;
        auto minus = stage_values;
        plus[i] += epsilon;
        minus[i] -= epsilon;
        const auto lp = bucket_losses(plus, neighbors_only);
        const auto lm = bucket_losses(minus, neighbors_only);
        const float expected_grad = (0.8f * (lp[0] - lm[0]) + 0.6f * (lp[1] - lm[1])
            + 1.2f * (lp[2] - lm[2])) / (2.0f * epsilon);
        if (!close(actual_grad[i], expected_grad, 3.0e-3, 3.0e-3)) {
            std::ostringstream message;
            message << "bucket gradient finite-difference mismatch (neighbors_only=" << neighbors_only
                    << ", stage_index=" << i << ", actual=" << actual_grad[i]
                    << ", expected=" << expected_grad << ")";
            throw std::runtime_error(message.str());
        }
    }
}

void check_weighted_target_and_bounds() {
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    const auto reference = torch::tensor({{1.0f, 10.0f}, {2.0f, 20.0f}, {4.0f, 40.0f}}, options);
    const auto rows = torch::tensor({{0, 1, -1}, {2, -7, -1}}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    const auto weights = torch::tensor({{0.25f, 0.75f, 3.0f}, {1.0f, 1.0f, 0.0f}}, options);
    const auto actual = ops::weighted_future_target(reference, rows, weights);
    const auto expected = torch::tensor({{1.75f, 17.5f}, {4.0f, 40.0f}}, options);
    require_close(actual, expected, "weighted future target mismatch");

    const auto scalar_reference = torch::tensor({{2.5f}}, options);
    const auto scalar_neighbor = torch::tensor({{0}}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    const auto scalar_weight = torch::ones({1, 1}, options);
    const auto scalar_target = ops::weighted_future_target(scalar_reference, scalar_neighbor, scalar_weight);
    require_close(scalar_target, scalar_reference, "single-row single-feature weighted target mismatch");

    auto bad_rows = rows.clone();
    bad_rows[0][0] = 3;
    bool rejected = false;
    try { (void)ops::weighted_future_target(reference, bad_rows, weights); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "out-of-range weighted neighbor was accepted");

    auto pair_rows = torch::tensor({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto pair_cols = torch::tensor({4}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto latent = torch::ones({3, 2}, options);
    auto time = torch::zeros({3}, options);
    rejected = false;
    try { (void)ops::dense_reduce_pair_losses(pair_rows, pair_cols, latent, time, 0.1, 0.3, 0.7); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "out-of-range pair row was accepted");

    auto extra_col = torch::tensor({0, 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    rejected = false;
    try { (void)ops::dense_reduce_pair_losses(pair_rows, extra_col, latent, time, 0.1, 0.3, 0.7); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "misaligned pair arrays were accepted");

    auto bad_buckets = torch::tensor({-1, 0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto short_stage = torch::zeros({2}, options);
    rejected = false;
    try { (void)ops::developmental_stage_bucket_losses(short_stage, bad_buckets, 0.4, 0.2, false, 2); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "negative bucket id was accepted");

    auto oversized_buckets = torch::tensor(
        {static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto one_stage = torch::zeros({1}, options);
    rejected = false;
    try { (void)ops::developmental_stage_bucket_losses(one_stage, oversized_buckets, 0.4, 0.2, false, 2); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "bucket label beyond the native index range was accepted");

    auto wrong_weight_type = weights.to(torch::kFloat64);
    rejected = false;
    try { (void)ops::weighted_future_target(reference, rows, wrong_weight_type); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "unexpected weighted-target dtype was accepted");

    auto wrong_weight_shape = torch::zeros({2, 2}, options);
    rejected = false;
    try { (void)ops::weighted_future_target(reference, rows, wrong_weight_shape); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "misaligned neighbor arrays were accepted");

    if (torch::cuda::device_count() > 1) {
        auto other_device_weights = weights.to(torch::Device(torch::kCUDA, 1));
        rejected = false;
        try { (void)ops::weighted_future_target(reference, rows, other_device_weights); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "mixed-device weighted-target inputs were accepted");
    }
}

void check_empty_inputs() {
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto latent = torch::ones({2, 2}, options).set_requires_grad(true);
    auto times = torch::zeros({2}, options);
    auto empty_indices = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto pair_losses_out = ops::dense_reduce_pair_losses(empty_indices, empty_indices, latent, times, 0.1, 0.3, 0.7);
    require(std::get<0>(pair_losses_out).item<float>() == 0.0f, "empty pair local loss must be zero");
    require(std::get<1>(pair_losses_out).item<float>() == 0.0f, "empty pair far loss must be zero");
    (std::get<0>(pair_losses_out) + std::get<1>(pair_losses_out)).backward();
    require(latent.grad().abs().sum().item<float>() == 0.0f, "empty pair gradient must be zero");

    auto empty_stage = torch::empty({0}, options).set_requires_grad(true);
    auto empty_buckets = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto bucket_losses_out = ops::developmental_stage_bucket_losses(empty_stage, empty_buckets, 0.5, 0.7, false, 3);
    require(std::get<0>(bucket_losses_out).item<float>() == 0.0f, "empty bucket ranking must be zero");
    require(std::get<1>(bucket_losses_out).item<float>() == 0.0f, "empty bucket anchor must be zero");
    require(std::get<2>(bucket_losses_out).item<float>() == 0.0f, "empty bucket spread must be zero");
    (std::get<0>(bucket_losses_out) + std::get<1>(bucket_losses_out) + std::get<2>(bucket_losses_out)).backward();
    require(empty_stage.grad().numel() == 0, "empty bucket gradient must remain empty");
}

} // namespace

int main() {
    require(torch::cuda::is_available(), "modelCustomOpsTest requires CUDA");
    check_pair_forward_and_gradient();
    check_bucket_forward_and_gradient(false);
    check_bucket_forward_and_gradient(true);
    check_weighted_target_and_bounds();
    check_empty_inputs();
    return 0;
}
