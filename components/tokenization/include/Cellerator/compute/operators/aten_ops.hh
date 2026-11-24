#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace cellerator::aten_ops {

// Indices identify source values: dense flattened row-major positions or CSR
// values slots. Offsets delimit the selected entries for each matrix row.
struct RankedSelection {
    at::Tensor flat_indices;
    at::Tensor offsets;
};

struct FeatureStatistics {
    at::Tensor medians;
    at::Tensor nonzero_counts;
    at::Tensor valid;
};

// Scale each row by its supplied total (if present), multiply by target_sum,
// then divide by supplied feature factors (if present). Dense and CSR inputs
// retain their layout; floating inputs retain storage dtype while integer
// inputs return their computation dtype (FP32 by default). compute_type selects
// arithmetic precision; it does not change floating input storage dtype.
at::Tensor normalize_features(
    const at::Tensor &values,
    std::optional<at::Tensor> row_totals = std::nullopt,
    std::optional<at::Tensor> feature_factors = std::nullopt,
    double target_sum = 1.0,
    std::optional<at::ScalarType> compute_type = std::nullopt);

// Rank eligible matrix entries within each observation by descending score,
// breaking ties by ascending feature ID. Dense scores are rank-2 BxF tensors;
// sparse scores are canonical rank-2 CSR tensors. Feature IDs must be a unique
// int64 vector of length F. A dense eligibility mask is BxF; a CSR mask is a
// bool vector aligned with scores.values(). A negative maximum keeps all
// eligible values; zero emits an empty result per row.
RankedSelection select_ranked_features(
    const at::Tensor &scores,
    const at::Tensor &feature_ids,
    std::optional<at::Tensor> eligibility = std::nullopt,
    int64_t max_per_row = -1);

// Compute exact per-feature medians over finite, nonzero entries in the supplied
// dense or canonical CSR batches. Batches must share feature width and device.
// Even samples use the arithmetic mean of the middle pair; empty features have
// NaN medians, zero counts, and valid=false. Feature blocking bounds temporary
// storage. Values are computed in compute_type or FP64 for FP64 inputs and
// FP32 otherwise.
FeatureStatistics nonzero_feature_median(
    const std::vector<at::Tensor> &batches,
    int64_t feature_block_size = 256,
    std::optional<at::ScalarType> compute_type = std::nullopt);

// Validate dense rank-2 or canonical CSR rank-2 input. This check may
// synchronize when CSR indices live on CUDA.
void validate_feature_matrix(const at::Tensor &matrix);

} // namespace cellerator::aten_ops
