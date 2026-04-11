#pragma once

#include <torch/torch.h>

#include <tuple>

namespace cellerator::compute::model_ops {

std::tuple<torch::Tensor, torch::Tensor> dense_reduce_pair_losses(
    const torch::Tensor &pair_rows,
    const torch::Tensor &pair_cols,
    const torch::Tensor &latent_unit,
    const torch::Tensor &developmental_time,
    double local_time_window,
    double far_time_window,
    double margin);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> developmental_stage_bucket_losses(
    const torch::Tensor &stage,
    const torch::Tensor &day_buckets,
    double ranking_margin,
    double min_within_day_std,
    bool use_neighbor_day_pairs_only,
    std::int64_t num_day_buckets);

torch::Tensor weighted_future_target(
    const torch::Tensor &reference_dense,
    const torch::Tensor &neighbor_row_indices,
    const torch::Tensor &neighbor_weights);

} // namespace cellerator::compute::model_ops
