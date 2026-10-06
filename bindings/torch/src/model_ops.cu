#include <Cellerator/bindings/torch/model_ops.hh>

#include <Cellerator/compute/operation/model_ops/model_ops.hh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/autograd/custom_function.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

namespace cellerator::bindings::torch::model_ops {
namespace {
namespace native = ::cellerator::compute::operation::model_ops;

void require_cuda(const ::torch::Tensor& tensor, const char* label) {
    if (!tensor.defined() || !tensor.is_cuda()) {
        throw std::invalid_argument(std::string(label) + " must be a defined CUDA tensor");
    }
}
void require_contiguous(const ::torch::Tensor& tensor, const char* label) {
    if (!tensor.is_contiguous()) throw std::invalid_argument(std::string(label) + " must be contiguous");
}
void require_dtype(const ::torch::Tensor& tensor, ::torch::ScalarType dtype, const char* label) {
    if (tensor.scalar_type() != dtype) throw std::invalid_argument(std::string(label) + " has an unexpected dtype");
}
void require_dim(const ::torch::Tensor& tensor, std::int64_t dim, const char* label) {
    if (tensor.dim() != dim) throw std::invalid_argument(std::string(label) + " has an unexpected rank");
}
void require_same_device(const ::torch::Tensor& lhs, const ::torch::Tensor& rhs, const char* label) {
    if (lhs.device() != rhs.device()) throw std::invalid_argument(std::string(label) + " must be on the same device");
}
void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

void require_pair_indices_in_range(
    const ::torch::Tensor& pair_rows,
    const ::torch::Tensor& pair_cols,
    std::int64_t row_count) {
    if (pair_rows.numel() == 0) return;
    if (row_count == 0) throw std::invalid_argument("pair indices cannot address an empty latent matrix");
    const auto rows_min = pair_rows.min().item<std::int64_t>();
    const auto rows_max = pair_rows.max().item<std::int64_t>();
    const auto cols_min = pair_cols.min().item<std::int64_t>();
    const auto cols_max = pair_cols.max().item<std::int64_t>();
    if (rows_min < 0 || rows_max >= row_count || cols_min < 0 || cols_max >= row_count) {
        throw std::invalid_argument("pair indices must address rows in latent_unit");
    }
}

void require_bucket_indices_nonnegative(const ::torch::Tensor& day_buckets) {
    if (day_buckets.numel() == 0) return;
    const auto minimum = day_buckets.min().item<std::int64_t>();
    const auto maximum = day_buckets.max().item<std::int64_t>();
    if (minimum < 0) throw std::invalid_argument("day_buckets must be nonnegative");
    if (maximum > native::maximum_indexed_count) {
        throw std::invalid_argument("day bucket labels exceed the native 32-bit count range");
    }
}

void require_neighbor_indices_in_range(const ::torch::Tensor& neighbors, std::int64_t reference_rows) {
    if (neighbors.numel() == 0) return;
    const auto maximum = neighbors.max().item<std::int64_t>();
    if (maximum >= reference_rows) {
        throw std::invalid_argument("nonnegative neighbor indices must address rows in reference_dense");
    }
}

std::tuple<::torch::Tensor, ::torch::Tensor, ::torch::Tensor, ::torch::Tensor> dense_reduce_pair_forward(
    const ::torch::Tensor& pair_rows,
    const ::torch::Tensor& pair_cols,
    const ::torch::Tensor& latent_unit,
    const ::torch::Tensor& developmental_time,
    double local_time_window,
    double far_time_window,
    double margin) {
    require_cuda(pair_rows, "pair_rows");
    require_cuda(pair_cols, "pair_cols");
    require_cuda(latent_unit, "latent_unit");
    require_cuda(developmental_time, "developmental_time");
    for (const auto* item : {&pair_rows, &pair_cols, &latent_unit, &developmental_time}) {
        require_contiguous(*item, "input");
        require_same_device(latent_unit, *item, "model-op inputs");
    }
    require_dtype(pair_rows, ::torch::kInt64, "pair_rows");
    require_dtype(pair_cols, ::torch::kInt64, "pair_cols");
    require_dtype(latent_unit, ::torch::kFloat32, "latent_unit");
    require_dtype(developmental_time, ::torch::kFloat32, "developmental_time");
    require_dim(pair_rows, 1, "pair_rows");
    require_dim(pair_cols, 1, "pair_cols");
    require_dim(latent_unit, 2, "latent_unit");
    require_dim(developmental_time, 1, "developmental_time");
    if (pair_rows.numel() != pair_cols.numel()) throw std::invalid_argument("pair_rows and pair_cols must align");
    if (latent_unit.size(0) != developmental_time.size(0)) throw std::invalid_argument("latent_unit and developmental_time must align");
    require_pair_indices_in_range(pair_rows, pair_cols, latent_unit.size(0));

    c10::cuda::CUDAGuard guard(latent_unit.device());
    const auto options = ::torch::TensorOptions().dtype(::torch::kFloat32).device(latent_unit.device());
    const auto count_options = ::torch::TensorOptions().dtype(::torch::kInt32).device(latent_unit.device());
    auto local_sum = ::torch::empty({}, options);
    auto far_sum = ::torch::empty({}, options);
    auto local_count = ::torch::empty({}, count_options);
    auto far_count = ::torch::empty({}, count_options);
    auto local_loss = ::torch::empty({}, options);
    auto far_loss = ::torch::empty({}, options);
    check_cuda(native::dense_reduce_pair_forward(
        pair_rows.data_ptr<std::int64_t>(), pair_cols.data_ptr<std::int64_t>(),
        latent_unit.data_ptr<float>(), developmental_time.data_ptr<float>(),
        pair_rows.numel(), latent_unit.size(0), latent_unit.size(1),
        static_cast<float>(local_time_window), static_cast<float>(far_time_window), static_cast<float>(margin),
        local_sum.data_ptr<float>(), local_count.data_ptr<std::int32_t>(),
        far_sum.data_ptr<float>(), far_count.data_ptr<std::int32_t>(),
        local_loss.data_ptr<float>(), far_loss.data_ptr<float>(),
        static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream())), "dense_reduce_pair_forward");
    return {std::move(local_loss), std::move(far_loss), std::move(local_count), std::move(far_count)};
}

::torch::Tensor dense_reduce_pair_backward(
    const ::torch::Tensor& pair_rows,
    const ::torch::Tensor& pair_cols,
    const ::torch::Tensor& latent_unit,
    const ::torch::Tensor& developmental_time,
    const ::torch::Tensor& local_count,
    const ::torch::Tensor& far_count,
    double local_time_window,
    double far_time_window,
    double margin,
    float grad_local,
    float grad_far) {
    c10::cuda::CUDAGuard guard(latent_unit.device());
    auto grad_latent = ::torch::zeros_like(latent_unit);
    check_cuda(native::dense_reduce_pair_backward(
        pair_rows.data_ptr<std::int64_t>(), pair_cols.data_ptr<std::int64_t>(),
        latent_unit.data_ptr<float>(), developmental_time.data_ptr<float>(),
        local_count.data_ptr<std::int32_t>(), far_count.data_ptr<std::int32_t>(),
        pair_rows.numel(), latent_unit.size(0), latent_unit.size(1),
        static_cast<float>(local_time_window), static_cast<float>(far_time_window), static_cast<float>(margin),
        grad_local, grad_far, grad_latent.data_ptr<float>(),
        static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream())), "dense_reduce_pair_backward");
    return grad_latent;
}

struct bucket_forward_result {
    ::torch::Tensor ranking;
    ::torch::Tensor anchor;
    ::torch::Tensor spread;
    ::torch::Tensor bucket_mean;
    ::torch::Tensor row_anchor_scale;
    ::torch::Tensor row_rank_scale;
    ::torch::Tensor spread_row_scale;
};

bucket_forward_result developmental_stage_forward(
    const ::torch::Tensor& stage,
    const ::torch::Tensor& day_buckets,
    double ranking_margin,
    double min_within_day_std,
    bool use_neighbor_day_pairs_only,
    std::int64_t num_day_buckets) {
    require_cuda(stage, "stage");
    require_cuda(day_buckets, "day_buckets");
    require_contiguous(stage, "stage");
    require_contiguous(day_buckets, "day_buckets");
    require_dtype(stage, ::torch::kFloat32, "stage");
    require_dtype(day_buckets, ::torch::kInt64, "day_buckets");
    require_dim(stage, 1, "stage");
    require_dim(day_buckets, 1, "day_buckets");
    require_same_device(stage, day_buckets, "stage and day_buckets");
    if (stage.numel() != day_buckets.numel()) throw std::invalid_argument("stage and day_buckets must align");
    if (num_day_buckets < 0) throw std::invalid_argument("num_day_buckets must be nonnegative");
    require_bucket_indices_nonnegative(day_buckets);

    c10::cuda::CUDAGuard guard(stage.device());
    const auto row_count = stage.numel();
    const auto inferred_bucket_count = row_count == 0
        ? std::int64_t{0}
        : day_buckets.max().item<std::int64_t>() + 1;
    const auto bucket_count = std::max(inferred_bucket_count, num_day_buckets);
    const auto options = ::torch::TensorOptions().dtype(::torch::kFloat32).device(stage.device());
    const auto count_options = ::torch::TensorOptions().dtype(::torch::kInt32).device(stage.device());
    auto bucket_mean = ::torch::empty({bucket_count}, options);
    auto row_anchor_scale = ::torch::empty({bucket_count}, options);
    auto row_rank_scale = ::torch::empty({bucket_count}, options);
    auto spread_row_scale = ::torch::empty({bucket_count}, options);
    auto bucket_sum = ::torch::empty({bucket_count}, options);
    auto bucket_sumsq = ::torch::empty({bucket_count}, options);
    auto bucket_rows = ::torch::empty({bucket_count}, count_options);
    auto ranking = ::torch::empty({}, options);
    auto anchor = ::torch::empty({}, options);
    auto spread = ::torch::empty({}, options);
    check_cuda(native::developmental_stage_bucket_forward(
        stage.data_ptr<float>(), day_buckets.data_ptr<std::int64_t>(), row_count, bucket_count,
        static_cast<float>(ranking_margin), static_cast<float>(min_within_day_std),
        use_neighbor_day_pairs_only, num_day_buckets,
        bucket_sum.data_ptr<float>(), bucket_sumsq.data_ptr<float>(), bucket_rows.data_ptr<std::int32_t>(),
        bucket_mean.data_ptr<float>(), row_anchor_scale.data_ptr<float>(), row_rank_scale.data_ptr<float>(),
        spread_row_scale.data_ptr<float>(), ranking.data_ptr<float>(), anchor.data_ptr<float>(),
        spread.data_ptr<float>(), static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream())),
        "developmental_stage_bucket_forward");
    return {std::move(ranking), std::move(anchor), std::move(spread), std::move(bucket_mean),
        std::move(row_anchor_scale), std::move(row_rank_scale), std::move(spread_row_scale)};
}

::torch::Tensor developmental_stage_backward(
    const ::torch::Tensor& stage,
    const ::torch::Tensor& day_buckets,
    const ::torch::Tensor& bucket_mean,
    const ::torch::Tensor& row_anchor_scale,
    const ::torch::Tensor& row_rank_scale,
    const ::torch::Tensor& spread_row_scale,
    float grad_ranking,
    float grad_anchor,
    float grad_spread) {
    c10::cuda::CUDAGuard guard(stage.device());
    auto grad_stage = ::torch::zeros_like(stage);
    check_cuda(native::developmental_stage_bucket_backward(
        stage.data_ptr<float>(), day_buckets.data_ptr<std::int64_t>(), bucket_mean.data_ptr<float>(),
        row_anchor_scale.data_ptr<float>(), row_rank_scale.data_ptr<float>(), spread_row_scale.data_ptr<float>(),
        stage.numel(), bucket_mean.numel(), grad_ranking, grad_anchor, grad_spread,
        grad_stage.data_ptr<float>(), static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream())),
        "developmental_stage_bucket_backward");
    return grad_stage;
}

::torch::Tensor weighted_future_target_native(
    const ::torch::Tensor& reference_dense,
    const ::torch::Tensor& neighbor_row_indices,
    const ::torch::Tensor& neighbor_weights) {
    require_cuda(reference_dense, "reference_dense");
    require_cuda(neighbor_row_indices, "neighbor_row_indices");
    require_cuda(neighbor_weights, "neighbor_weights");
    for (const auto* item : {&reference_dense, &neighbor_row_indices, &neighbor_weights}) {
        require_contiguous(*item, "input");
        require_same_device(reference_dense, *item, "weighted-target inputs");
    }
    require_dtype(reference_dense, ::torch::kFloat32, "reference_dense");
    require_dtype(neighbor_row_indices, ::torch::kInt64, "neighbor_row_indices");
    require_dtype(neighbor_weights, ::torch::kFloat32, "neighbor_weights");
    require_dim(reference_dense, 2, "reference_dense");
    require_dim(neighbor_row_indices, 2, "neighbor_row_indices");
    require_dim(neighbor_weights, 2, "neighbor_weights");
    if (neighbor_row_indices.sizes() != neighbor_weights.sizes()) {
        throw std::invalid_argument("neighbor_row_indices and neighbor_weights must align");
    }
    require_neighbor_indices_in_range(neighbor_row_indices, reference_dense.size(0));

    c10::cuda::CUDAGuard guard(reference_dense.device());
    auto target = ::torch::empty({neighbor_row_indices.size(0), reference_dense.size(1)},
        ::torch::TensorOptions().dtype(::torch::kFloat32).device(reference_dense.device()));
    check_cuda(native::weighted_future_target(
        reference_dense.data_ptr<float>(), neighbor_row_indices.data_ptr<std::int64_t>(),
        neighbor_weights.data_ptr<float>(), reference_dense.size(0), neighbor_row_indices.size(0),
        neighbor_row_indices.size(1), reference_dense.size(1), target.data_ptr<float>(),
        static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream())), "weighted_future_target");
    return target;
}

class DenseReducePairLossFunction : public ::torch::autograd::Function<DenseReducePairLossFunction> {
public:
    static ::torch::autograd::variable_list forward(
        ::torch::autograd::AutogradContext* ctx,
        ::torch::Tensor pair_rows,
        ::torch::Tensor pair_cols,
        ::torch::Tensor latent_unit,
        ::torch::Tensor developmental_time,
        double local_time_window,
        double far_time_window,
        double margin) {
        auto outputs = dense_reduce_pair_forward(pair_rows, pair_cols, latent_unit, developmental_time,
            local_time_window, far_time_window, margin);
        ctx->save_for_backward({pair_rows, pair_cols, latent_unit, developmental_time,
            std::get<2>(outputs), std::get<3>(outputs)});
        ctx->saved_data["local_time_window"] = local_time_window;
        ctx->saved_data["far_time_window"] = far_time_window;
        ctx->saved_data["margin"] = margin;
        return {std::get<0>(outputs), std::get<1>(outputs)};
    }

    static ::torch::autograd::variable_list backward(
        ::torch::autograd::AutogradContext* ctx,
        ::torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const float grad_local = grad_outputs[0].defined() ? grad_outputs[0].item<float>() : 0.0f;
        const float grad_far = grad_outputs[1].defined() ? grad_outputs[1].item<float>() : 0.0f;
        auto grad_latent = dense_reduce_pair_backward(saved[0], saved[1], saved[2], saved[3],
            saved[4], saved[5], ctx->saved_data["local_time_window"].toDouble(),
            ctx->saved_data["far_time_window"].toDouble(), ctx->saved_data["margin"].toDouble(),
            grad_local, grad_far);
        return {::torch::Tensor(), ::torch::Tensor(), std::move(grad_latent), ::torch::Tensor(),
            ::torch::Tensor(), ::torch::Tensor(), ::torch::Tensor()};
    }
};

class DevelopmentalStageBucketLossFunction
    : public ::torch::autograd::Function<DevelopmentalStageBucketLossFunction> {
public:
    static ::torch::autograd::variable_list forward(
        ::torch::autograd::AutogradContext* ctx,
        ::torch::Tensor stage,
        ::torch::Tensor day_buckets,
        double ranking_margin,
        double min_within_day_std,
        bool use_neighbor_day_pairs_only,
        std::int64_t num_day_buckets) {
        auto outputs = developmental_stage_forward(stage, day_buckets, ranking_margin,
            min_within_day_std, use_neighbor_day_pairs_only, num_day_buckets);
        ctx->save_for_backward({stage, day_buckets, outputs.bucket_mean, outputs.row_anchor_scale,
            outputs.row_rank_scale, outputs.spread_row_scale});
        return {std::move(outputs.ranking), std::move(outputs.anchor), std::move(outputs.spread)};
    }

    static ::torch::autograd::variable_list backward(
        ::torch::autograd::AutogradContext* ctx,
        ::torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const float grad_ranking = grad_outputs[0].defined() ? grad_outputs[0].item<float>() : 0.0f;
        const float grad_anchor = grad_outputs[1].defined() ? grad_outputs[1].item<float>() : 0.0f;
        const float grad_spread = grad_outputs[2].defined() ? grad_outputs[2].item<float>() : 0.0f;
        auto grad_stage = developmental_stage_backward(saved[0], saved[1], saved[2], saved[3],
            saved[4], saved[5], grad_ranking, grad_anchor, grad_spread);
        return {std::move(grad_stage), ::torch::Tensor(), ::torch::Tensor(), ::torch::Tensor(),
            ::torch::Tensor(), ::torch::Tensor()};
    }
};

} // namespace

std::tuple<::torch::Tensor, ::torch::Tensor> dense_reduce_pair_losses(
    const ::torch::Tensor& pair_rows,
    const ::torch::Tensor& pair_cols,
    const ::torch::Tensor& latent_unit,
    const ::torch::Tensor& developmental_time,
    double local_time_window,
    double far_time_window,
    double margin) {
    auto outputs = DenseReducePairLossFunction::apply(pair_rows, pair_cols, latent_unit,
        developmental_time, local_time_window, far_time_window, margin);
    return {outputs[0], outputs[1]};
}

std::tuple<::torch::Tensor, ::torch::Tensor, ::torch::Tensor> developmental_stage_bucket_losses(
    const ::torch::Tensor& stage,
    const ::torch::Tensor& day_buckets,
    double ranking_margin,
    double min_within_day_std,
    bool use_neighbor_day_pairs_only,
    std::int64_t num_day_buckets) {
    auto outputs = DevelopmentalStageBucketLossFunction::apply(stage, day_buckets,
        ranking_margin, min_within_day_std, use_neighbor_day_pairs_only, num_day_buckets);
    return {outputs[0], outputs[1], outputs[2]};
}

::torch::Tensor weighted_future_target(
    const ::torch::Tensor& reference_dense,
    const ::torch::Tensor& neighbor_row_indices,
    const ::torch::Tensor& neighbor_weights) {
    return weighted_future_target_native(reference_dense, neighbor_row_indices, neighbor_weights);
}

} // namespace cellerator::bindings::torch::model_ops
