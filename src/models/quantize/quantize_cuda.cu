#include "quantize.hh"

#include "../../compute/autograd/autograd.hh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace cellerator::models::quantize::detail {

namespace autograd = ::cellerator::compute::autograd;

namespace {

inline void assign_or_accumulate_grad_(torch::Tensor &param, const torch::Tensor &grad) {
    torch::Tensor grad_value = grad.detach();
    if (param.grad().defined()) {
        param.mutable_grad().add_(grad_value);
    } else {
        param.mutable_grad() = grad_value;
    }
}

} // namespace

SparseCudaLossBundle sparse_cuda_reconstruction_range_backward_(
    GeneQuantizerModel &model,
    const QuantizerBatch &batch,
    const GeneQuantizerLossConfig &config) {
    if (!can_use_sparse_cuda_quantizer_(batch.features)) {
        throw std::invalid_argument("sparse_cuda_reconstruction_range_backward requires a CUDA sparse CSR batch");
    }

    torch::Tensor log_scale_param = model->log_scale_parameter();
    torch::Tensor offset_param = model->offset_parameter();
    if (!log_scale_param.is_cuda() || !offset_param.is_cuda()) {
        throw std::invalid_argument("sparse CUDA quantizer training requires CUDA model parameters");
    }
    if (log_scale_param.device() != batch.features.device() || offset_param.device() != batch.features.device()) {
        throw std::invalid_argument("sparse CUDA quantizer training requires features and parameters on the same device");
    }

    c10::cuda::CUDAGuard guard(batch.features.device());
    torch::Tensor crow = batch.features.crow_indices().to(
        torch::TensorOptions().dtype(torch::kInt32).device(batch.features.device())).contiguous();
    torch::Tensor col = batch.features.col_indices().to(
        torch::TensorOptions().dtype(torch::kInt32).device(batch.features.device())).contiguous();
    torch::Tensor values = batch.features.values().to(
        torch::TensorOptions().dtype(torch::kFloat16).device(batch.features.device())).contiguous();
    torch::Tensor log_scale = log_scale_param.contiguous();
    torch::Tensor offset = offset_param.contiguous();

    torch::Tensor reconstruction_loss = torch::zeros(
        {},
        torch::TensorOptions().dtype(torch::kFloat32).device(batch.features.device()));
    torch::Tensor range_loss = torch::zeros(
        {},
        torch::TensorOptions().dtype(torch::kFloat32).device(batch.features.device()));
    torch::Tensor grad_log_scale = torch::zeros_like(log_scale, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor grad_offset = torch::zeros_like(offset, torch::TensorOptions().dtype(torch::kFloat32));

    autograd::feature_affine_quantize_config kernel_config;
    kernel_config.bits = static_cast<std::uint32_t>(model->bits());
    kernel_config.scale_floor = static_cast<float>(model->scale_floor());
    kernel_config.reconstruction_weight = static_cast<float>(config.reconstruction_weight);
    kernel_config.range_weight = static_cast<float>(config.range_weight);
    kernel_config.min_dynamic_range = static_cast<float>(config.min_dynamic_range);

    autograd::execution_context ctx;
    autograd::init(&ctx, batch.features.get_device(), at::cuda::getCurrentCUDAStream());
    autograd::base::csr_feature_affine_quantize_fwd_bwd_f16_f32(
        ctx,
        reinterpret_cast<const std::uint32_t *>(crow.data_ptr<std::int32_t>()),
        reinterpret_cast<const std::uint32_t *>(col.data_ptr<std::int32_t>()),
        reinterpret_cast<const __half *>(values.data_ptr<at::Half>()),
        static_cast<std::uint32_t>(batch.features.size(0)),
        static_cast<std::uint32_t>(batch.features.size(1)),
        log_scale.data_ptr<float>(),
        offset.data_ptr<float>(),
        kernel_config,
        reconstruction_loss.data_ptr<float>(),
        range_loss.data_ptr<float>(),
        grad_log_scale.data_ptr<float>(),
        grad_offset.data_ptr<float>());
    autograd::clear(&ctx);

    assign_or_accumulate_grad_(log_scale_param, grad_log_scale);
    assign_or_accumulate_grad_(offset_param, grad_offset);
    return SparseCudaLossBundle{ std::move(reconstruction_loss), std::move(range_loss) };
}

} // namespace cellerator::models::quantize::detail
