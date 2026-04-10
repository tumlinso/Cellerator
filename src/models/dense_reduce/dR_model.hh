#pragma once

#include "dR_dataloader.hh"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace cellerator::models::dense_reduce {

struct SparseDenseReduceConfig {
    std::int64_t input_genes = 0;
    std::int64_t hidden_dim = 256;
    std::int64_t bottleneck_dim = 128;
    std::int64_t latent_dim = 32;
    bool use_bias = true;
    double dropout = 0.1;
    double corruption_rate = 0.1;
    bool fp16_input = true;
};

struct DenseReduceLossConfig {
    double recon_weight = 1.0;
    double nonzero_recon_weight = 4.0;
    double local_time_window = 0.05;
    double far_time_window = 0.25;
    double local_weight = 0.15;
    double far_weight = 0.05;
    double margin = 0.35;
    std::int64_t max_sampled_pairs = 4096;
};

struct DenseReduceOutput {
    torch::Tensor reconstruction;
    torch::Tensor latent_raw;
    torch::Tensor latent_unit;
};

struct DenseReduceLoss {
    torch::Tensor total;
    torch::Tensor reconstruction;
    torch::Tensor local_smoothness;
    torch::Tensor far_separation;
};

class DenseReduceModelImpl : public torch::nn::Module {
public:
    explicit DenseReduceModelImpl(SparseDenseReduceConfig config = SparseDenseReduceConfig())
        : config_(std::move(config)),
          enc_norm_0_(torch::nn::LayerNormOptions({ config_.hidden_dim })),
          enc_hidden_(config_.hidden_dim, config_.bottleneck_dim),
          enc_norm_1_(torch::nn::LayerNormOptions({ config_.bottleneck_dim })),
          enc_latent_(config_.bottleneck_dim, config_.latent_dim),
          dec_hidden_0_(config_.latent_dim, config_.bottleneck_dim),
          dec_norm_0_(torch::nn::LayerNormOptions({ config_.bottleneck_dim })),
          dec_hidden_1_(config_.bottleneck_dim, config_.hidden_dim),
          dec_norm_1_(torch::nn::LayerNormOptions({ config_.hidden_dim })),
          dec_output_(config_.hidden_dim, config_.input_genes),
          dropout_(config_.dropout) {
        if (config_.input_genes <= 0) throw std::invalid_argument("SparseDenseReduceConfig.input_genes must be > 0");
        if (config_.hidden_dim <= 0) throw std::invalid_argument("SparseDenseReduceConfig.hidden_dim must be > 0");
        if (config_.bottleneck_dim <= 0) {
            throw std::invalid_argument("SparseDenseReduceConfig.bottleneck_dim must be > 0");
        }
        if (config_.latent_dim <= 0) throw std::invalid_argument("SparseDenseReduceConfig.latent_dim must be > 0");
        if (config_.corruption_rate < 0.0 || config_.corruption_rate >= 1.0) {
            throw std::invalid_argument("SparseDenseReduceConfig.corruption_rate must be in [0, 1)");
        }

        projection_weight_ = register_parameter(
            "projection_weight",
            torch::empty({ config_.input_genes, config_.hidden_dim }, torch::TensorOptions().dtype(torch::kFloat32)));
        if (config_.use_bias) {
            projection_bias_ = register_parameter(
                "projection_bias",
                torch::empty({ config_.hidden_dim }, torch::TensorOptions().dtype(torch::kFloat32)));
        }

        register_module("enc_norm_0", enc_norm_0_);
        register_module("enc_hidden", enc_hidden_);
        register_module("enc_norm_1", enc_norm_1_);
        register_module("enc_latent", enc_latent_);
        register_module("dec_hidden_0", dec_hidden_0_);
        register_module("dec_norm_0", dec_norm_0_);
        register_module("dec_hidden_1", dec_hidden_1_);
        register_module("dec_norm_1", dec_norm_1_);
        register_module("dec_output", dec_output_);
        register_module("dropout", dropout_);
        reset_parameters();
    }

    void reset_parameters() {
        torch::nn::init::kaiming_uniform_(projection_weight_, std::sqrt(5.0));
        if (projection_bias_.defined()) {
            const double bound = 1.0 / std::sqrt(static_cast<double>(config_.input_genes));
            torch::nn::init::uniform_(projection_bias_, -bound, bound);
        }
        enc_hidden_->reset_parameters();
        enc_latent_->reset_parameters();
        dec_hidden_0_->reset_parameters();
        dec_hidden_1_->reset_parameters();
        dec_output_->reset_parameters();
    }

    DenseReduceOutput forward(const torch::Tensor &sparse_csr_batch) {
        torch::Tensor latent_raw = encode_raw_(sparse_csr_batch);
        torch::Tensor latent_unit = torch::nn::functional::normalize(
            latent_raw,
            torch::nn::functional::NormalizeFuncOptions().p(2.0).dim(1).eps(1.0e-6));

        torch::Tensor hidden = dec_hidden_0_(latent_raw);
        hidden = dec_norm_0_(hidden);
        hidden = torch::nn::functional::silu(hidden);

        hidden = dec_hidden_1_(hidden);
        hidden = dec_norm_1_(hidden);
        hidden = torch::nn::functional::silu(hidden);

        torch::Tensor reconstruction = dec_output_(hidden);
        return DenseReduceOutput{
            std::move(reconstruction),
            std::move(latent_raw),
            std::move(latent_unit)
        };
    }

    torch::Tensor encode(const torch::Tensor &sparse_csr_batch) {
        return forward(sparse_csr_batch).latent_unit;
    }

private:
    torch::Tensor maybe_corrupt_sparse_(const torch::Tensor &sparse_batch) const {
        if (!is_training() || config_.corruption_rate <= 0.0) return sparse_batch;

        torch::Tensor sparse_input = sparse_batch;
        if (sparse_input.layout() == torch::kSparseCsr) sparse_input = sparse_input.to_sparse();
        sparse_input = sparse_input.coalesce();

        torch::Tensor values = sparse_input.values();
        if (values.numel() == 0) return sparse_input;

        const double keep_prob = 1.0 - config_.corruption_rate;
        torch::Tensor mask = torch::rand(values.sizes(), values.options()) < keep_prob;
        torch::Tensor scaled_values = values * mask.to(values.dtype()) / keep_prob;
        return torch::sparse_coo_tensor(
            sparse_input.indices(),
            scaled_values,
            sparse_input.sizes(),
            sparse_input.options()).coalesce();
    }

    torch::Tensor encode_raw_(const torch::Tensor &sparse_csr_batch) {
        if (sparse_csr_batch.dim() != 2) {
            throw std::invalid_argument("DenseReduceModel expects a rank-2 sparse batch");
        }
        if (sparse_csr_batch.size(1) != config_.input_genes) {
            throw std::invalid_argument("DenseReduceModel input gene dimension does not match config");
        }
        if (sparse_csr_batch.layout() != torch::kSparse && sparse_csr_batch.layout() != torch::kSparseCsr) {
            throw std::invalid_argument("DenseReduceModel expects sparse COO or CSR input");
        }

        torch::Tensor sparse_input = sparse_csr_batch.to(torch::kFloat32);
        sparse_input = maybe_corrupt_sparse_(sparse_input);
        if (sparse_input.layout() == torch::kSparseCsr) sparse_input = sparse_input.to_sparse();

        torch::Tensor projected = torch::matmul(sparse_input, projection_weight_);
        if (projection_bias_.defined()) projected = projected + projection_bias_;
        projected = enc_norm_0_(projected);
        projected = torch::nn::functional::silu(projected);
        projected = dropout_(projected);

        torch::Tensor hidden = enc_hidden_(projected);
        hidden = enc_norm_1_(hidden);
        hidden = torch::nn::functional::silu(hidden);
        hidden = dropout_(hidden);
        return enc_latent_(hidden).to(torch::kFloat32);
    }

    SparseDenseReduceConfig config_;
    torch::Tensor projection_weight_;
    torch::Tensor projection_bias_;
    torch::nn::LayerNorm enc_norm_0_{ nullptr };
    torch::nn::Linear enc_hidden_{ nullptr };
    torch::nn::LayerNorm enc_norm_1_{ nullptr };
    torch::nn::Linear enc_latent_{ nullptr };
    torch::nn::Linear dec_hidden_0_{ nullptr };
    torch::nn::LayerNorm dec_norm_0_{ nullptr };
    torch::nn::Linear dec_hidden_1_{ nullptr };
    torch::nn::LayerNorm dec_norm_1_{ nullptr };
    torch::nn::Linear dec_output_{ nullptr };
    torch::nn::Dropout dropout_{ nullptr };
};

TORCH_MODULE(DenseReduceModel);

inline DenseReduceLoss compute_dense_reduce_loss(
    const DenseReduceOutput &output,
    const DenseReduceBatch &batch,
    const DenseReduceLossConfig &config = DenseReduceLossConfig()) {
    torch::Tensor target = batch.features.to_dense().to(torch::kFloat32);
    torch::Tensor reconstruction = output.reconstruction.to(torch::kFloat32);
    torch::Tensor weights = torch::ones_like(target);
    weights = torch::where(target != 0, torch::full_like(target, config.nonzero_recon_weight), weights);
    torch::Tensor reconstruction_loss = (weights * torch::pow(reconstruction - target, 2)).mean();

    torch::Tensor time = batch.developmental_time.to(output.latent_unit.device(), torch::kFloat32).view({ -1 });
    torch::Tensor latent = output.latent_unit.to(torch::kFloat32);
    torch::Tensor scalar_zero = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(latent.device()));
    torch::Tensor local_loss = scalar_zero.clone();
    torch::Tensor far_loss = scalar_zero.clone();

    if (latent.dim() != 2) throw std::invalid_argument("DenseReduceOutput.latent_unit must be rank-2");
    if (latent.size(0) != time.size(0)) {
        throw std::invalid_argument("batch developmental_time must align with latent rows");
    }
    if (config.max_sampled_pairs < 0) {
        throw std::invalid_argument("DenseReduceLossConfig.max_sampled_pairs must be >= 0");
    }

    const std::int64_t batch_size = latent.size(0);
    if (batch_size > 1 && config.max_sampled_pairs != 0) {
        torch::Tensor pair_index = torch::triu_indices(
            batch_size,
            batch_size,
            1,
            torch::TensorOptions().dtype(torch::kInt64).device(latent.device()));
        torch::Tensor pair_rows = pair_index.select(0, 0);
        torch::Tensor pair_cols = pair_index.select(0, 1);
        const std::int64_t total_pairs = pair_rows.numel();
        if (config.max_sampled_pairs < total_pairs) {
            torch::Tensor perm = torch::randperm(
                total_pairs,
                torch::TensorOptions().dtype(torch::kInt64).device(latent.device()));
            torch::Tensor keep = perm.slice(0, 0, config.max_sampled_pairs);
            pair_rows = pair_rows.index_select(0, keep);
            pair_cols = pair_cols.index_select(0, keep);
        }

        torch::Tensor pair_time = torch::abs(time.index_select(0, pair_rows) - time.index_select(0, pair_cols));
        torch::Tensor pair_sim = (latent.index_select(0, pair_rows) * latent.index_select(0, pair_cols)).sum(1);
        torch::Tensor pair_sqdist = torch::clamp(2.0f - 2.0f * pair_sim, 0.0f);
        torch::Tensor pair_dist = torch::sqrt(pair_sqdist + 1.0e-12f);

        torch::Tensor local_mask = pair_time <= config.local_time_window;
        if (local_mask.any().item<bool>()) {
            local_loss = pair_sqdist.masked_select(local_mask).mean();
        }

        torch::Tensor far_mask = pair_time >= config.far_time_window;
        if (far_mask.any().item<bool>()) {
            torch::Tensor far_dist = pair_dist.masked_select(far_mask);
            far_loss = torch::relu(torch::full_like(far_dist, config.margin) - far_dist).mean();
        }
    }

    torch::Tensor total = config.recon_weight * reconstruction_loss
        + config.local_weight * local_loss
        + config.far_weight * far_loss;
    return DenseReduceLoss{
        std::move(total),
        std::move(reconstruction_loss),
        std::move(local_loss),
        std::move(far_loss)
    };
}

} // namespace cellerator::models::dense_reduce
