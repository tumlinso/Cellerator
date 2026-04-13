#include "model_ops.hh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/autograd/custom_function.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace cellerator::compute::model_ops {

namespace {

inline void require_cuda_(const torch::Tensor &tensor, const char *label) {
    if (!tensor.defined() || !tensor.is_cuda()) {
        throw std::invalid_argument(std::string(label) + " must be a defined CUDA tensor");
    }
}

inline void require_contiguous_(const torch::Tensor &tensor, const char *label) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
}

inline void require_dtype_(const torch::Tensor &tensor, torch::ScalarType dtype, const char *label) {
    if (tensor.scalar_type() != dtype) {
        throw std::invalid_argument(std::string(label) + " has an unexpected dtype");
    }
}

inline void require_dim_(const torch::Tensor &tensor, std::int64_t dim, const char *label) {
    if (tensor.dim() != dim) {
        throw std::invalid_argument(std::string(label) + " has an unexpected rank");
    }
}

inline void cuda_check_(cudaError_t status, const char *label) {
    if (status == cudaSuccess) return;
    std::ostringstream message;
    message << label << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

constexpr int kThreads = 256;
constexpr float kPairEps = 1.0e-12f;
constexpr float kStdEps = 1.0e-12f;

#include "kernels/dense_reduce_pair_forward_kernel_.cuh"
#include "kernels/dense_reduce_pair_backward_kernel_.cuh"
#include "kernels/stage_bucket_accumulate_kernel_.cuh"
#include "kernels/stage_bucket_finalize_kernel_.cuh"
#include "kernels/stage_bucket_backward_kernel_.cuh"
#include "kernels/weighted_future_target_kernel_.cuh"

std::tuple<torch::Tensor, torch::Tensor> dense_reduce_pair_forward_cuda_(
    const torch::Tensor &pair_rows,
    const torch::Tensor &pair_cols,
    const torch::Tensor &latent_unit,
    const torch::Tensor &developmental_time,
    double local_time_window,
    double far_time_window,
    double margin) {
    require_cuda_(pair_rows, "pair_rows");
    require_cuda_(pair_cols, "pair_cols");
    require_cuda_(latent_unit, "latent_unit");
    require_cuda_(developmental_time, "developmental_time");
    require_contiguous_(pair_rows, "pair_rows");
    require_contiguous_(pair_cols, "pair_cols");
    require_contiguous_(latent_unit, "latent_unit");
    require_contiguous_(developmental_time, "developmental_time");
    require_dtype_(pair_rows, torch::kInt64, "pair_rows");
    require_dtype_(pair_cols, torch::kInt64, "pair_cols");
    require_dtype_(latent_unit, torch::kFloat32, "latent_unit");
    require_dtype_(developmental_time, torch::kFloat32, "developmental_time");
    require_dim_(latent_unit, 2, "latent_unit");
    require_dim_(developmental_time, 1, "developmental_time");
    require_dim_(pair_rows, 1, "pair_rows");
    require_dim_(pair_cols, 1, "pair_cols");

    if (pair_rows.numel() != pair_cols.numel()) {
        throw std::invalid_argument("pair_rows and pair_cols must align");
    }
    if (latent_unit.size(0) != developmental_time.size(0)) {
        throw std::invalid_argument("latent_unit and developmental_time must align");
    }

    c10::cuda::CUDAGuard guard(latent_unit.device());
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(latent_unit.device());
    auto count_options = torch::TensorOptions().dtype(torch::kInt32).device(latent_unit.device());
    torch::Tensor local_sum = torch::zeros({}, options);
    torch::Tensor far_sum = torch::zeros({}, options);
    torch::Tensor local_count = torch::zeros({}, count_options);
    torch::Tensor far_count = torch::zeros({}, count_options);

    const std::int64_t pair_count = pair_rows.numel();
    if (pair_count > 0) {
        const int blocks = static_cast<int>((pair_count + kThreads - 1) / kThreads);
        // Pairwise loss forward is launch-light but atomic-heavy; cost grows with sampled pairs and becomes memory/atomic bound before math bound.
        dense_reduce_pair_forward_kernel_<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            pair_rows.data_ptr<std::int64_t>(),
            pair_cols.data_ptr<std::int64_t>(),
            latent_unit.data_ptr<float>(),
            developmental_time.data_ptr<float>(),
            pair_count,
            latent_unit.size(1),
            static_cast<float>(local_time_window),
            static_cast<float>(far_time_window),
            static_cast<float>(margin),
            local_sum.data_ptr<float>(),
            local_count.data_ptr<int>(),
            far_sum.data_ptr<float>(),
            far_count.data_ptr<int>());
        cuda_check_(cudaGetLastError(), "dense_reduce_pair_forward_kernel_");
    }

    // These scalar reads force a device-to-host round trip and synchronize the current stream; cheap in bytes, not in latency.
    const int local_count_host = local_count.to(torch::kCPU).item<int>();
    const int far_count_host = far_count.to(torch::kCPU).item<int>();
    torch::Tensor local_loss = local_count_host > 0
        ? local_sum / static_cast<double>(local_count_host)
        : torch::zeros({}, options);
    torch::Tensor far_loss = far_count_host > 0
        ? far_sum / static_cast<double>(far_count_host)
        : torch::zeros({}, options);
    return { std::move(local_loss), std::move(far_loss) };
}

torch::Tensor dense_reduce_pair_backward_cuda_(
    const torch::Tensor &pair_rows,
    const torch::Tensor &pair_cols,
    const torch::Tensor &latent_unit,
    const torch::Tensor &developmental_time,
    double local_time_window,
    double far_time_window,
    double margin,
    float grad_local_scale,
    float grad_far_scale) {
    c10::cuda::CUDAGuard guard(latent_unit.device());
    torch::Tensor grad_latent = torch::zeros_like(latent_unit);
    const std::int64_t pair_count = pair_rows.numel();
    if (pair_count == 0 || (grad_local_scale == 0.0f && grad_far_scale == 0.0f)) return grad_latent;

    const int blocks = static_cast<int>((pair_count + kThreads - 1) / kThreads);
    // Backward mirrors the forward pair sweep and adds atomics into latent gradients, so contention rises with pair reuse.
    dense_reduce_pair_backward_kernel_<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        pair_rows.data_ptr<std::int64_t>(),
        pair_cols.data_ptr<std::int64_t>(),
        latent_unit.data_ptr<float>(),
        developmental_time.data_ptr<float>(),
        pair_count,
        latent_unit.size(1),
        static_cast<float>(local_time_window),
        static_cast<float>(far_time_window),
        static_cast<float>(margin),
        grad_local_scale,
        grad_far_scale,
        grad_latent.data_ptr<float>());
    cuda_check_(cudaGetLastError(), "dense_reduce_pair_backward_kernel_");
    return grad_latent;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> developmental_stage_forward_cuda_(
    const torch::Tensor &stage,
    const torch::Tensor &day_buckets,
    double ranking_margin,
    double min_within_day_std,
    bool use_neighbor_day_pairs_only,
    std::int64_t num_day_buckets,
    torch::Tensor *bucket_mean,
    torch::Tensor *row_anchor_scale,
    torch::Tensor *row_rank_scale,
    torch::Tensor *spread_row_scale) {
    require_cuda_(stage, "stage");
    require_cuda_(day_buckets, "day_buckets");
    require_contiguous_(stage, "stage");
    require_contiguous_(day_buckets, "day_buckets");
    require_dtype_(stage, torch::kFloat32, "stage");
    require_dtype_(day_buckets, torch::kInt64, "day_buckets");
    require_dim_(stage, 1, "stage");
    require_dim_(day_buckets, 1, "day_buckets");
    if (stage.numel() != day_buckets.numel()) {
        throw std::invalid_argument("stage and day_buckets must align");
    }

    c10::cuda::CUDAGuard guard(stage.device());
    const std::int64_t row_count = stage.numel();
    // This max() read pulls one scalar to host to size the scratch tensors, so the forward path has an unavoidable sync today.
    const std::int64_t inferred_bucket_count = row_count == 0
        ? 0
        : (day_buckets.max().to(torch::kCPU).item<std::int64_t>() + 1);
    const std::int64_t bucket_count = std::max<std::int64_t>(
        inferred_bucket_count,
        num_day_buckets > 0 ? num_day_buckets : inferred_bucket_count);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(stage.device());
    auto count_options = torch::TensorOptions().dtype(torch::kInt32).device(stage.device());
    *bucket_mean = torch::zeros({ bucket_count }, options);
    *row_anchor_scale = torch::zeros({ bucket_count }, options);
    *row_rank_scale = torch::zeros({ bucket_count }, options);
    *spread_row_scale = torch::zeros({ bucket_count }, options);
    torch::Tensor bucket_sum = torch::zeros({ bucket_count }, options);
    torch::Tensor bucket_sumsq = torch::zeros({ bucket_count }, options);
    torch::Tensor bucket_rows = torch::zeros({ bucket_count }, count_options);
    torch::Tensor ranking = torch::zeros({}, options);
    torch::Tensor anchor = torch::zeros({}, options);
    torch::Tensor spread = torch::zeros({}, options);

    if (row_count > 0 && bucket_count > 0) {
        const int blocks = static_cast<int>((row_count + kThreads - 1) / kThreads);
        // First pass is a flat reduction into per-bucket stats; bandwidth/atomic cost dominates for large row counts.
        stage_bucket_accumulate_kernel_<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            stage.data_ptr<float>(),
            day_buckets.data_ptr<std::int64_t>(),
            row_count,
            bucket_count,
            bucket_sum.data_ptr<float>(),
            bucket_sumsq.data_ptr<float>(),
            bucket_rows.data_ptr<int>());
        cuda_check_(cudaGetLastError(), "stage_bucket_accumulate_kernel_");

        // Single-thread finalize is intentionally serial because bucket_count is small; launch latency matters more than throughput here.
        stage_bucket_finalize_kernel_<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
            bucket_sum.data_ptr<float>(),
            bucket_sumsq.data_ptr<float>(),
            bucket_rows.data_ptr<int>(),
            bucket_count,
            static_cast<float>(ranking_margin),
            static_cast<float>(min_within_day_std),
            use_neighbor_day_pairs_only ? 1 : 0,
            num_day_buckets,
            bucket_mean->data_ptr<float>(),
            row_anchor_scale->data_ptr<float>(),
            row_rank_scale->data_ptr<float>(),
            spread_row_scale->data_ptr<float>(),
            ranking.data_ptr<float>(),
            anchor.data_ptr<float>(),
            spread.data_ptr<float>());
        cuda_check_(cudaGetLastError(), "stage_bucket_finalize_kernel_");
    }

    return { std::move(ranking), std::move(anchor), std::move(spread) };
}

torch::Tensor developmental_stage_backward_cuda_(
    const torch::Tensor &stage,
    const torch::Tensor &day_buckets,
    const torch::Tensor &bucket_mean,
    const torch::Tensor &row_anchor_scale,
    const torch::Tensor &row_rank_scale,
    const torch::Tensor &spread_row_scale,
    float grad_ranking,
    float grad_anchor,
    float grad_spread) {
    c10::cuda::CUDAGuard guard(stage.device());
    torch::Tensor grad_stage = torch::zeros_like(stage);
    const std::int64_t row_count = stage.numel();
    if (row_count == 0 || (grad_ranking == 0.0f && grad_anchor == 0.0f && grad_spread == 0.0f)) return grad_stage;

    const int blocks = static_cast<int>((row_count + kThreads - 1) / kThreads);
    // Backward is a single row-parallel sweep over cached bucket scales, so it is usually cheaper than the forward accumulation path.
    stage_bucket_backward_kernel_<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        stage.data_ptr<float>(),
        day_buckets.data_ptr<std::int64_t>(),
        bucket_mean.data_ptr<float>(),
        row_anchor_scale.data_ptr<float>(),
        row_rank_scale.data_ptr<float>(),
        spread_row_scale.data_ptr<float>(),
        row_count,
        grad_ranking,
        grad_anchor,
        grad_spread,
        grad_stage.data_ptr<float>());
    cuda_check_(cudaGetLastError(), "stage_bucket_backward_kernel_");
    return grad_stage;
}

torch::Tensor weighted_future_target_cuda_(
    const torch::Tensor &reference_dense,
    const torch::Tensor &neighbor_row_indices,
    const torch::Tensor &neighbor_weights) {
    require_cuda_(reference_dense, "reference_dense");
    require_cuda_(neighbor_row_indices, "neighbor_row_indices");
    require_cuda_(neighbor_weights, "neighbor_weights");
    require_contiguous_(reference_dense, "reference_dense");
    require_contiguous_(neighbor_row_indices, "neighbor_row_indices");
    require_contiguous_(neighbor_weights, "neighbor_weights");
    require_dtype_(reference_dense, torch::kFloat32, "reference_dense");
    require_dtype_(neighbor_row_indices, torch::kInt64, "neighbor_row_indices");
    require_dtype_(neighbor_weights, torch::kFloat32, "neighbor_weights");
    require_dim_(reference_dense, 2, "reference_dense");
    require_dim_(neighbor_row_indices, 2, "neighbor_row_indices");
    require_dim_(neighbor_weights, 2, "neighbor_weights");
    if (neighbor_row_indices.sizes() != neighbor_weights.sizes()) {
        throw std::invalid_argument("neighbor_row_indices and neighbor_weights must align");
    }

    c10::cuda::CUDAGuard guard(reference_dense.device());
    const std::int64_t query_rows = neighbor_row_indices.size(0);
    const std::int64_t top_k = neighbor_row_indices.size(1);
    const std::int64_t genes = reference_dense.size(1);
    torch::Tensor target = torch::zeros(
        { query_rows, genes },
        torch::TensorOptions().dtype(torch::kFloat32).device(reference_dense.device()));

    const std::int64_t total = query_rows * genes;
    if (total > 0) {
        const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
        // One output thread per query/gene cell; cost is dominated by gathering top-k rows from global memory rather than arithmetic.
        weighted_future_target_kernel_<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reference_dense.data_ptr<float>(),
            neighbor_row_indices.data_ptr<std::int64_t>(),
            neighbor_weights.data_ptr<float>(),
            query_rows,
            top_k,
            genes,
            target.data_ptr<float>());
        cuda_check_(cudaGetLastError(), "weighted_future_target_kernel_");
    }
    return target;
}

class DenseReducePairLossFunction : public torch::autograd::Function<DenseReducePairLossFunction> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor pair_rows,
        torch::Tensor pair_cols,
        torch::Tensor latent_unit,
        torch::Tensor developmental_time,
        double local_time_window,
        double far_time_window,
        double margin) {
        auto losses = dense_reduce_pair_forward_cuda_(
            pair_rows, pair_cols, latent_unit, developmental_time, local_time_window, far_time_window, margin);
        ctx->save_for_backward({ pair_rows, pair_cols, latent_unit, developmental_time });
        ctx->saved_data["local_time_window"] = local_time_window;
        ctx->saved_data["far_time_window"] = far_time_window;
        ctx->saved_data["margin"] = margin;
        return {
            std::move(std::get<0>(losses)),
            std::move(std::get<1>(losses))
        };
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const torch::Tensor &pair_rows = saved[0];
        const torch::Tensor &pair_cols = saved[1];
        const torch::Tensor &latent_unit = saved[2];
        const torch::Tensor &developmental_time = saved[3];

        const float grad_local = grad_outputs[0].defined()
            ? grad_outputs[0].to(torch::kFloat32).item<float>()
            : 0.0f;
        const float grad_far = grad_outputs[1].defined()
            ? grad_outputs[1].to(torch::kFloat32).item<float>()
            : 0.0f;

        torch::Tensor grad_latent = dense_reduce_pair_backward_cuda_(
            pair_rows,
            pair_cols,
            latent_unit,
            developmental_time,
            ctx->saved_data["local_time_window"].toDouble(),
            ctx->saved_data["far_time_window"].toDouble(),
            ctx->saved_data["margin"].toDouble(),
            grad_local,
            grad_far);

        return {
            torch::Tensor(),
            torch::Tensor(),
            std::move(grad_latent),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor()
        };
    }
};

class DevelopmentalStageBucketLossFunction
    : public torch::autograd::Function<DevelopmentalStageBucketLossFunction> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor stage,
        torch::Tensor day_buckets,
        double ranking_margin,
        double min_within_day_std,
        bool use_neighbor_day_pairs_only,
        std::int64_t num_day_buckets) {
        torch::Tensor bucket_mean;
        torch::Tensor row_anchor_scale;
        torch::Tensor row_rank_scale;
        torch::Tensor spread_row_scale;
        auto losses = developmental_stage_forward_cuda_(
            stage,
            day_buckets,
            ranking_margin,
            min_within_day_std,
            use_neighbor_day_pairs_only,
            num_day_buckets,
            &bucket_mean,
            &row_anchor_scale,
            &row_rank_scale,
            &spread_row_scale);
        ctx->save_for_backward({ stage, day_buckets, bucket_mean, row_anchor_scale, row_rank_scale, spread_row_scale });
        return {
            std::move(std::get<0>(losses)),
            std::move(std::get<1>(losses)),
            std::move(std::get<2>(losses))
        };
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const float grad_ranking = grad_outputs[0].defined()
            ? grad_outputs[0].to(torch::kFloat32).item<float>()
            : 0.0f;
        const float grad_anchor = grad_outputs[1].defined()
            ? grad_outputs[1].to(torch::kFloat32).item<float>()
            : 0.0f;
        const float grad_spread = grad_outputs[2].defined()
            ? grad_outputs[2].to(torch::kFloat32).item<float>()
            : 0.0f;

        torch::Tensor grad_stage = developmental_stage_backward_cuda_(
            saved[0],
            saved[1],
            saved[2],
            saved[3],
            saved[4],
            saved[5],
            grad_ranking,
            grad_anchor,
            grad_spread);
        return {
            std::move(grad_stage),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor()
        };
    }
};

} // namespace

std::tuple<torch::Tensor, torch::Tensor> dense_reduce_pair_losses(
    const torch::Tensor &pair_rows,
    const torch::Tensor &pair_cols,
    const torch::Tensor &latent_unit,
    const torch::Tensor &developmental_time,
    double local_time_window,
    double far_time_window,
    double margin) {
    auto outputs = DenseReducePairLossFunction::apply(
        pair_rows,
        pair_cols,
        latent_unit,
        developmental_time,
        local_time_window,
        far_time_window,
        margin);
    return { outputs[0], outputs[1] };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> developmental_stage_bucket_losses(
    const torch::Tensor &stage,
    const torch::Tensor &day_buckets,
    double ranking_margin,
    double min_within_day_std,
    bool use_neighbor_day_pairs_only,
    std::int64_t num_day_buckets) {
    auto outputs = DevelopmentalStageBucketLossFunction::apply(
        stage,
        day_buckets,
        ranking_margin,
        min_within_day_std,
        use_neighbor_day_pairs_only,
        num_day_buckets);
    return { outputs[0], outputs[1], outputs[2] };
}

torch::Tensor weighted_future_target(
    const torch::Tensor &reference_dense,
    const torch::Tensor &neighbor_row_indices,
    const torch::Tensor &neighbor_weights) {
    return weighted_future_target_cuda_(reference_dense, neighbor_row_indices, neighbor_weights);
}

} // namespace cellerator::compute::model_ops
