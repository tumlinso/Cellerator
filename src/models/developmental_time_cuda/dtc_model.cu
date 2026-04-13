#include "dtc_model.hh"

#include "../../compute/autograd/autograd.hh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace cellerator::models::developmental_time_cuda {

namespace autograd = ::cellerator::compute::autograd;
namespace dt = ::cellerator::models::developmental_time;

namespace {

struct forward_cache {
    torch::Tensor stem_pre;
    torch::Tensor stem_act;
    torch::Tensor hidden_pre;
    torch::Tensor hidden_act;
    torch::Tensor predicted_time;
    torch::Tensor time_bin_logits;
};

torch::Tensor silu_grad_(const torch::Tensor &value) {
    torch::Tensor sigma = torch::sigmoid(value);
    return sigma * (1.0f + value * (1.0f - sigma));
}

torch::Tensor to_sparse_csr_cuda_(const torch::Tensor &features, const torch::Device &device) {
    torch::Tensor out = features;
    if (out.layout() == torch::kSparse) out = out.to_sparse_csr();
    if (out.layout() != torch::kSparseCsr) {
        throw std::invalid_argument("DevelopmentalTimeCudaModel requires sparse COO or CSR input");
    }
    if (!out.is_cuda() || out.device() != device) out = out.to(device);
    return out;
}

forward_cache forward_with_cache_(
    DevelopmentalTimeCudaModel &model,
    const torch::Tensor &sparse_csr_batch) {
    torch::Tensor features = to_sparse_csr_cuda_(sparse_csr_batch, model.device());
    if (features.size(1) != model.config().input_genes) {
        throw std::invalid_argument("DevelopmentalTimeCudaModel input gene dimension does not match config");
    }

    c10::cuda::CUDAGuard guard(model.device());
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(model.device());

    torch::Tensor crow = features.crow_indices().to(
        torch::TensorOptions().dtype(torch::kInt32).device(model.device())).contiguous();
    torch::Tensor col = features.col_indices().to(
        torch::TensorOptions().dtype(torch::kInt32).device(model.device())).contiguous();
    torch::Tensor values = features.values().to(
        torch::TensorOptions().dtype(torch::kFloat16).device(model.device())).contiguous();
    torch::Tensor stem_weight = model.stem_weight().contiguous();

    forward_cache cache;
    cache.stem_pre = torch::zeros({ features.size(0), model.config().stem_dim }, options);

    autograd::execution_context ctx;
    autograd::init(&ctx, model.device().index(), at::cuda::getCurrentCUDAStream());
    autograd::base::csr_spmm_fwd_f16_f32(
        ctx,
        reinterpret_cast<const std::uint32_t *>(crow.data_ptr<std::int32_t>()),
        reinterpret_cast<const std::uint32_t *>(col.data_ptr<std::int32_t>()),
        reinterpret_cast<const __half *>(values.data_ptr<at::Half>()),
        static_cast<std::uint32_t>(features.size(0)),
        static_cast<std::uint32_t>(features.size(1)),
        stem_weight.data_ptr<float>(),
        stem_weight.stride(0),
        stem_weight.size(1),
        cache.stem_pre.data_ptr<float>(),
        cache.stem_pre.stride(0));
    autograd::clear(&ctx);

    if (model.config().use_bias) cache.stem_pre = cache.stem_pre + model.stem_bias();
    cache.stem_act = torch::nn::functional::silu(cache.stem_pre);
    cache.hidden_pre = torch::matmul(cache.stem_act, model.hidden_weight());
    if (model.config().use_bias) cache.hidden_pre = cache.hidden_pre + model.hidden_bias();
    cache.hidden_act = torch::nn::functional::silu(cache.hidden_pre);
    cache.predicted_time = torch::matmul(cache.hidden_act, model.time_weight().view({ -1, 1 })).squeeze(-1);
    if (model.config().use_bias) cache.predicted_time = cache.predicted_time + model.time_bias();
    cache.time_bin_logits = torch::matmul(cache.hidden_act, model.bin_weight());
    if (model.config().use_bias) cache.time_bin_logits = cache.time_bin_logits + model.bin_bias();
    return cache;
}

torch::Tensor smooth_l1_grad_(
    const torch::Tensor &prediction,
    const torch::Tensor &target,
    double beta) {
    const std::int64_t count = prediction.numel();
    if (count == 0) return torch::zeros_like(prediction);
    torch::Tensor diff = prediction - target;
    if (beta <= 0.0) return diff.sign() / static_cast<double>(count);
    torch::Tensor abs_diff = diff.abs();
    torch::Tensor small = diff / beta;
    torch::Tensor large = diff.sign();
    return torch::where(abs_diff < beta, small, large) / static_cast<double>(count);
}

void clip_gradients_if_needed_(
    std::vector<torch::Tensor> *grads,
    const DevelopmentalTimeCudaTrainConfig &config) {
    if (!config.clip_gradients || grads == nullptr || grads->empty()) return;
    torch::Tensor total_sq = torch::zeros({}, grads->front().options());
    for (const torch::Tensor &grad : *grads) total_sq = total_sq + grad.square().sum();
    const double total_norm = std::sqrt(total_sq.to(torch::kCPU).item<double>());
    if (total_norm <= config.max_grad_norm || total_norm == 0.0) return;
    const double scale = config.max_grad_norm / total_norm;
    for (torch::Tensor &grad : *grads) grad.mul_(scale);
}

void apply_sgd_update_(
    torch::Tensor *param,
    const torch::Tensor &grad,
    const DevelopmentalTimeCudaTrainConfig &config) {
    if (param == nullptr) throw std::invalid_argument("apply_sgd_update_ requires a parameter");
    torch::NoGradGuard no_grad;
    if (config.weight_decay != 0.0) param->mul_(1.0 - config.learning_rate * config.weight_decay);
    param->add_(grad, -config.learning_rate);
}

} // namespace

DevelopmentalTimeCudaModel::DevelopmentalTimeCudaModel(DevelopmentalTimeCudaConfig config)
    : config_(std::move(config)) {
    if (config_.input_genes <= 0) throw std::invalid_argument("DevelopmentalTimeCudaConfig.input_genes must be > 0");
    if (config_.stem_dim <= 0) throw std::invalid_argument("DevelopmentalTimeCudaConfig.stem_dim must be > 0");
    if (config_.hidden_dim <= 0) throw std::invalid_argument("DevelopmentalTimeCudaConfig.hidden_dim must be > 0");
    if (config_.num_time_bins <= 1) throw std::invalid_argument("DevelopmentalTimeCudaConfig.num_time_bins must be > 1");
    if (!config_.device.is_cuda()) throw std::invalid_argument("DevelopmentalTimeCudaModel requires a CUDA device");

    to(config_.device);
    reset_parameters_();
}

void DevelopmentalTimeCudaModel::to(const torch::Device &device) {
    if (!device.is_cuda()) throw std::invalid_argument("DevelopmentalTimeCudaModel::to requires a CUDA device");
    config_.device = device;
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    stem_weight_ = stem_weight_.defined() ? stem_weight_.to(options).contiguous() : torch::empty({ config_.input_genes, config_.stem_dim }, options);
    stem_bias_ = stem_bias_.defined() ? stem_bias_.to(options).contiguous() : torch::zeros({ config_.stem_dim }, options);
    hidden_weight_ = hidden_weight_.defined() ? hidden_weight_.to(options).contiguous() : torch::empty({ config_.stem_dim, config_.hidden_dim }, options);
    hidden_bias_ = hidden_bias_.defined() ? hidden_bias_.to(options).contiguous() : torch::zeros({ config_.hidden_dim }, options);
    time_weight_ = time_weight_.defined() ? time_weight_.to(options).contiguous() : torch::empty({ config_.hidden_dim }, options);
    time_bias_ = time_bias_.defined() ? time_bias_.to(options).contiguous() : torch::zeros({}, options);
    bin_weight_ = bin_weight_.defined() ? bin_weight_.to(options).contiguous() : torch::empty({ config_.hidden_dim, config_.num_time_bins }, options);
    bin_bias_ = bin_bias_.defined() ? bin_bias_.to(options).contiguous() : torch::zeros({ config_.num_time_bins }, options);
}

const DevelopmentalTimeCudaConfig &DevelopmentalTimeCudaModel::config() const { return config_; }
torch::Device DevelopmentalTimeCudaModel::device() const { return config_.device; }

torch::Tensor &DevelopmentalTimeCudaModel::stem_weight() { return stem_weight_; }
torch::Tensor &DevelopmentalTimeCudaModel::stem_bias() { return stem_bias_; }
torch::Tensor &DevelopmentalTimeCudaModel::hidden_weight() { return hidden_weight_; }
torch::Tensor &DevelopmentalTimeCudaModel::hidden_bias() { return hidden_bias_; }
torch::Tensor &DevelopmentalTimeCudaModel::time_weight() { return time_weight_; }
torch::Tensor &DevelopmentalTimeCudaModel::time_bias() { return time_bias_; }
torch::Tensor &DevelopmentalTimeCudaModel::bin_weight() { return bin_weight_; }
torch::Tensor &DevelopmentalTimeCudaModel::bin_bias() { return bin_bias_; }

const torch::Tensor &DevelopmentalTimeCudaModel::stem_weight() const { return stem_weight_; }
const torch::Tensor &DevelopmentalTimeCudaModel::stem_bias() const { return stem_bias_; }
const torch::Tensor &DevelopmentalTimeCudaModel::hidden_weight() const { return hidden_weight_; }
const torch::Tensor &DevelopmentalTimeCudaModel::hidden_bias() const { return hidden_bias_; }
const torch::Tensor &DevelopmentalTimeCudaModel::time_weight() const { return time_weight_; }
const torch::Tensor &DevelopmentalTimeCudaModel::time_bias() const { return time_bias_; }
const torch::Tensor &DevelopmentalTimeCudaModel::bin_weight() const { return bin_weight_; }
const torch::Tensor &DevelopmentalTimeCudaModel::bin_bias() const { return bin_bias_; }

std::vector<torch::Tensor *> DevelopmentalTimeCudaModel::parameters() {
    return {
        &stem_weight_,
        &stem_bias_,
        &hidden_weight_,
        &hidden_bias_,
        &time_weight_,
        &time_bias_,
        &bin_weight_,
        &bin_bias_,
    };
}

std::vector<const torch::Tensor *> DevelopmentalTimeCudaModel::parameters() const {
    return {
        &stem_weight_,
        &stem_bias_,
        &hidden_weight_,
        &hidden_bias_,
        &time_weight_,
        &time_bias_,
        &bin_weight_,
        &bin_bias_,
    };
}

void DevelopmentalTimeCudaModel::reset_parameters_() {
    torch::nn::init::kaiming_uniform_(stem_weight_, std::sqrt(5.0));
    torch::nn::init::kaiming_uniform_(hidden_weight_, std::sqrt(5.0));
    torch::nn::init::uniform_(time_weight_, -0.1, 0.1);
    torch::nn::init::kaiming_uniform_(bin_weight_, std::sqrt(5.0));
    stem_bias_.zero_();
    hidden_bias_.zero_();
    time_bias_.zero_();
    bin_bias_.zero_();
}

DevelopmentalTimeCudaOutput forward(
    DevelopmentalTimeCudaModel &model,
    const torch::Tensor &sparse_csr_batch) {
    const forward_cache cache = forward_with_cache_(model, sparse_csr_batch);
    return DevelopmentalTimeCudaOutput{
        cache.predicted_time,
        cache.time_bin_logits,
        cache.hidden_act
    };
}

torch::Tensor predict_time(
    DevelopmentalTimeCudaModel &model,
    const torch::Tensor &sparse_csr_batch) {
    return forward(model, sparse_csr_batch).predicted_time;
}

DevelopmentalTimeCudaLoss compute_loss(
    const DevelopmentalTimeCudaOutput &output,
    const dt::TimeBatch &batch,
    const DevelopmentalTimeCudaLossConfig &config) {
    torch::Tensor predicted_time = output.predicted_time.to(torch::kFloat32).view({ -1 });
    torch::Tensor target_time = batch.day_labels.to(predicted_time.options()).view({ -1 });
    torch::Tensor time_bins = batch.day_buckets.to(
        torch::TensorOptions().dtype(torch::kInt64).device(predicted_time.device())).view({ -1 });

    torch::Tensor regression = torch::nn::functional::smooth_l1_loss(
        predicted_time,
        target_time,
        torch::nn::functional::SmoothL1LossFuncOptions()
            .reduction(torch::kMean)
            .beta(config.huber_delta));
    torch::Tensor bin_classification = torch::nn::functional::cross_entropy(
        output.time_bin_logits.to(torch::kFloat32),
        time_bins,
        torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kMean));
    torch::Tensor total = config.regression_weight * regression
        + config.bin_weight * bin_classification;
    return DevelopmentalTimeCudaLoss{
        std::move(total),
        std::move(regression),
        std::move(bin_classification)
    };
}

DevelopmentalTimeCudaTrainStep train_step(
    DevelopmentalTimeCudaModel &model,
    const dt::TimeBatch &batch,
    const DevelopmentalTimeCudaLossConfig &loss_config,
    const DevelopmentalTimeCudaTrainConfig &train_config) {
    const forward_cache cache = forward_with_cache_(model, batch.features);
    DevelopmentalTimeCudaOutput output{
        cache.predicted_time,
        cache.time_bin_logits,
        cache.hidden_act
    };
    DevelopmentalTimeCudaLoss loss = compute_loss(output, batch, loss_config);

    torch::Tensor target_time = batch.day_labels.to(cache.predicted_time.options()).view({ -1 });
    torch::Tensor time_bins = batch.day_buckets.to(
        torch::TensorOptions().dtype(torch::kInt64).device(model.device())).view({ -1 });
    torch::Tensor grad_predicted_time = smooth_l1_grad_(cache.predicted_time.view({ -1 }), target_time, loss_config.huber_delta);
    grad_predicted_time.mul_(loss_config.regression_weight);

    torch::Tensor probs = torch::softmax(cache.time_bin_logits, 1);
    torch::Tensor one_hot = torch::zeros_like(cache.time_bin_logits);
    one_hot.scatter_(1, time_bins.unsqueeze(1), 1.0f);
    torch::Tensor grad_time_bin_logits = (probs - one_hot) * (loss_config.bin_weight / static_cast<double>(cache.time_bin_logits.size(0)));

    torch::Tensor grad_time_weight = torch::matmul(cache.hidden_act.t(), grad_predicted_time.unsqueeze(1)).view_as(model.time_weight());
    torch::Tensor grad_time_bias = grad_predicted_time.sum();
    torch::Tensor grad_bin_weight = torch::matmul(cache.hidden_act.t(), grad_time_bin_logits);
    torch::Tensor grad_bin_bias = grad_time_bin_logits.sum(0);

    torch::Tensor grad_hidden = grad_predicted_time.unsqueeze(1) * model.time_weight().view({ 1, -1 });
    grad_hidden = grad_hidden + torch::matmul(grad_time_bin_logits, model.bin_weight().t());
    torch::Tensor grad_hidden_pre = grad_hidden * silu_grad_(cache.hidden_pre);
    torch::Tensor grad_hidden_weight = torch::matmul(cache.stem_act.t(), grad_hidden_pre);
    torch::Tensor grad_hidden_bias = grad_hidden_pre.sum(0);

    torch::Tensor grad_stem_act = torch::matmul(grad_hidden_pre, model.hidden_weight().t());
    torch::Tensor grad_stem_pre = grad_stem_act * silu_grad_(cache.stem_pre);
    torch::Tensor grad_stem_bias = grad_stem_pre.sum(0);

    torch::Tensor grad_stem_pre_contig = grad_stem_pre.contiguous();
    torch::Tensor features = to_sparse_csr_cuda_(batch.features, model.device()).to(torch::kFloat32);
    torch::Tensor sparse_features = features.layout() == torch::kSparseCsr
        ? features.to_sparse()
        : features;
    torch::Tensor grad_stem_weight = torch::matmul(sparse_features.transpose(0, 1), grad_stem_pre_contig);

    std::vector<torch::Tensor> grads{
        grad_stem_weight,
        grad_stem_bias,
        grad_hidden_weight,
        grad_hidden_bias,
        grad_time_weight,
        grad_time_bias.view_as(model.time_bias()),
        grad_bin_weight,
        grad_bin_bias
    };
    clip_gradients_if_needed_(&grads, train_config);

    apply_sgd_update_(&model.stem_weight(), grads[0], train_config);
    apply_sgd_update_(&model.stem_bias(), grads[1], train_config);
    apply_sgd_update_(&model.hidden_weight(), grads[2], train_config);
    apply_sgd_update_(&model.hidden_bias(), grads[3], train_config);
    apply_sgd_update_(&model.time_weight(), grads[4], train_config);
    apply_sgd_update_(&model.time_bias(), grads[5], train_config);
    apply_sgd_update_(&model.bin_weight(), grads[6], train_config);
    apply_sgd_update_(&model.bin_bias(), grads[7], train_config);

    return DevelopmentalTimeCudaTrainStep{ std::move(output), std::move(loss) };
}

} // namespace cellerator::models::developmental_time_cuda
