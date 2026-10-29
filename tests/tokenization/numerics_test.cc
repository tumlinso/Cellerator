#include <Cellerator/compute/operators/aten_ops.hh>

#include <ATen/ATen.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

bool close(const at::Tensor &actual, const at::Tensor &expected) {
    return at::allclose(actual.to(at::kDouble), expected.to(at::kDouble), 1e-6, 1e-6, true);
}

at::Tensor matrix(std::initializer_list<double> values, int64_t rows,
    int64_t columns, at::ScalarType dtype) {
    return at::tensor(std::vector<double>(values), at::TensorOptions().dtype(at::kDouble))
        .reshape({rows, columns}).to(dtype);
}
}

int main() {
    using namespace cellerator::aten_ops;

    auto dense = matrix({0.0, 2.0, 4.0, 1.0, 0.0, 9.0}, 2, 3, at::kDouble);
    auto med = nonzero_feature_median({dense}, 2);
    check(close(med.medians, at::tensor({1.0, 2.0, 6.5}, at::kDouble)), "dense median values");
    check(close(med.nonzero_counts, at::tensor({1, 1, 2}, at::kLong)), "dense median counts");
    check(at::all(med.valid).item<bool>(), "dense median validity");

    auto empty = at::zeros({2, 3}, at::kFloat);
    auto empty_stats = nonzero_feature_median({empty});
    check(at::all(empty_stats.nonzero_counts.eq(0)).item<bool>(), "empty median counts");
    check(!at::any(empty_stats.valid).item<bool>(), "empty median validity");
    check(at::isnan(empty_stats.medians).all().item<bool>(), "empty median values");

    const auto largest = std::numeric_limits<double>::max();
    const auto least_subnormal = std::numeric_limits<double>::denorm_min();
    auto extreme = matrix({largest * 0.5, -largest, -largest, least_subnormal,
        largest, -largest * 0.5, largest, least_subnormal}, 2, 4, at::kDouble);
    auto extreme_stats = nonzero_feature_median({extreme}, 2);
    auto first_three = extreme_stats.medians.slice(0, 0, 3);
    check(at::isfinite(first_three).all().item<bool>(),
        "finite extreme values must not overflow their medians");
    check(close(first_three, at::tensor({largest * 0.75, -largest * 0.75, 0.0},
        at::TensorOptions().dtype(at::kDouble))), "extreme median midpoint values");
    check(extreme_stats.medians[3].item<double>() == least_subnormal,
        "equal subnormal values survive median calculation");

    auto ids = at::tensor({30, 10, 20}, at::kLong);
    auto scores = matrix({3.0, 3.0, 2.0, 1.0, 9.0, 9.0}, 2, 3, at::kFloat);
    auto selected = select_ranked_features(scores, ids, std::nullopt, 2);
    check(close(selected.flat_indices, at::tensor({1, 0, 4, 5}, at::kLong)),
        "stable feature ranking and cutoff");
    check(close(selected.offsets, at::tensor({0, 2, 4}, at::kLong)), "rank offsets");
    auto eligible = matrix({0, 1, 1, 0, 1, 0}, 2, 3, at::kByte).ne(0);
    auto masked = select_ranked_features(scores, ids, eligible, -1);
    check(close(masked.flat_indices, at::tensor({1, 2, 4}, at::kLong)), "eligibility mask");

    auto totals = at::tensor({2.0, 10.0}, at::kDouble);
    auto factors = at::tensor({1.0, 2.0, 3.0}, at::kDouble);
    auto normalized = normalize_features(dense, totals, factors, 10000.0);
    auto normalized_expected = matrix({0.0, 5000.0, 20000.0 / 3.0,
        1000.0, 0.0, 3000.0}, 2, 3, at::kDouble);
    check(close(normalized, normalized_expected), "dense normalization");
    auto double_storage = matrix({1.234567890123, 2.345678901234}, 1, 2, at::kDouble);
    auto float_compute = normalize_features(double_storage, std::nullopt,
        std::nullopt, 1.0, at::kFloat);
    check(float_compute.scalar_type() == at::kDouble,
        "explicit compute type preserves floating storage dtype");
    check(at::equal(float_compute, float_compute.to(at::kFloat).to(at::kDouble)),
        "explicit FP32 arithmetic controls result precision");
    auto integer_counts = at::tensor({2, 4, 0, 8}, at::kLong).reshape({2, 2});
    auto integer_scores = normalize_features(
        integer_counts, at::tensor({2, 8}, at::kLong), std::nullopt, 100.0);
    check(integer_scores.scalar_type() == at::kFloat, "integer normalization promotes to FP32");
    check(close(integer_scores, matrix({100.0, 200.0, 0.0, 100.0}, 2, 2, at::kFloat)),
        "integer normalization values");

    auto differentiable = matrix({2.0, 3.0, 4.0}, 1, 3, at::kFloat).set_requires_grad(true);
    auto differentiable_selection = select_ranked_features(
        differentiable, at::tensor({10, 20, 30}, at::kLong), std::nullopt, 2);
    differentiable.reshape({-1}).index_select(0, differentiable_selection.flat_indices)
        .sum().backward();
    check(close(differentiable.grad(), matrix({0.0, 1.0, 1.0}, 1, 3, at::kFloat)),
        "selected values preserve ATen gradients");

    auto crow = at::tensor({0, 2, 4}, at::kLong);
    auto columns = at::tensor({1, 2, 0, 2}, at::kLong);
    auto vals = at::tensor({2.0, 4.0, 1.0, 9.0}, at::kFloat);
    auto csr = at::sparse_csr_tensor(crow, columns, vals, {2, 3},
        at::TensorOptions().layout(at::kSparseCsr).dtype(at::kFloat));
    validate_feature_matrix(csr);
    auto csr_selected = select_ranked_features(csr, ids, std::nullopt, 1);
    check(close(csr_selected.flat_indices, at::tensor({1, 3}, at::kLong)), "CSR selection slots");
    check(close(csr_selected.offsets, at::tensor({0, 1, 2}, at::kLong)), "CSR selection offsets");
    auto csr_medians = nonzero_feature_median(std::vector<at::Tensor>{csr});
    check(close(csr_medians.medians, at::tensor({1.0, 2.0, 6.5}, at::kFloat)), "CSR median values");
    auto csr_scaled = normalize_features(csr, at::tensor({2.0, 10.0}), factors.to(at::kFloat), 100.0);
    check(csr_scaled.is_sparse_csr(), "CSR normalization preserves layout");
    check(close(csr_scaled.to_dense(), matrix({0.0, 50.0, 200.0 / 3.0,
        10.0, 0.0, 30.0}, 2, 3, at::kFloat)), "CSR normalization values");

    bool rejected = false;
    try {
        (void)select_ranked_features(scores, at::tensor({1, 1, 2}, at::kLong));
    } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "duplicate feature IDs rejected");

    std::cout << "ATen numerical tokenization tests passed\n";
}
