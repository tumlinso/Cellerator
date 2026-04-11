#pragma once

#include "dT_dataloader.hh"
#include "../../compute/model_ops/model_ops.hh"

#include <torch/torch.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cellerator::models::developmental_time {

namespace model_ops = ::cellerator::compute::model_ops;

struct SparseTimeEncoderConfig {
    std::int64_t input_genes = 0;
    std::int64_t hidden_dim = 256;
    std::int64_t proj_dim = 64;
    bool use_bias = true;
    double dropout = 0.1;
    bool fp16_input = true;
};

struct DevelopmentalStageHeadConfig {
    std::int64_t input_dim = 0;
    std::int64_t hidden_dim = 64;
    bool bounded_output = true;
};

struct DevelopmentalStageLossConfig {
    double ranking_weight = 1.0;
    double anchor_weight = 0.25;
    double spread_weight = 0.05;
    double ranking_margin = 0.02;
    double min_within_day_std = 0.01;
    bool use_neighbor_day_pairs_only = true;
    std::int64_t num_day_buckets = 0;
};

struct SparseTimeEncoderOutput {
    torch::Tensor embedding;
};

struct DevelopmentalStageOutput {
    torch::Tensor stage_logit;
    torch::Tensor stage;
    torch::Tensor embedding;
};

struct DevelopmentalStageLoss {
    torch::Tensor total;
    torch::Tensor ranking;
    torch::Tensor anchor;
    torch::Tensor spread;
};

struct SpeciesCalibration {
    std::string species_id;
    float slope = 1.0f;
    float intercept = 0.0f;
};

class SpeciesTimeCalibrator {
public:
    void set_calibration(SpeciesCalibration calibration) {
        calibrations_[calibration.species_id] = std::move(calibration);
    }

    bool has_species(const std::string &species_id) const {
        return calibrations_.find(species_id) != calibrations_.end();
    }

    torch::Tensor stage_to_time(const torch::Tensor &stage, const std::string &species_id) const {
        const auto it = calibrations_.find(species_id);
        if (it == calibrations_.end()) {
            throw std::invalid_argument("no developmental-time calibration registered for species '" + species_id + "'");
        }
        return stage.to(torch::kFloat32) * std::max(it->second.slope, 0.0f) + it->second.intercept;
    }

private:
    std::unordered_map<std::string, SpeciesCalibration> calibrations_;
};

class SparseTimeEncoderImpl : public torch::nn::Module {
public:
    explicit SparseTimeEncoderImpl(SparseTimeEncoderConfig config = SparseTimeEncoderConfig())
        : config_(std::move(config)),
          proj_norm_(torch::nn::LayerNormOptions({ config_.hidden_dim })),
          refine_(config_.hidden_dim, config_.proj_dim),
          out_norm_(torch::nn::LayerNormOptions({ config_.proj_dim })),
          dropout_(config_.dropout) {
        if (config_.input_genes <= 0) throw std::invalid_argument("SparseTimeEncoderConfig.input_genes must be > 0");
        if (config_.hidden_dim <= 0) throw std::invalid_argument("SparseTimeEncoderConfig.hidden_dim must be > 0");
        if (config_.proj_dim <= 0) throw std::invalid_argument("SparseTimeEncoderConfig.proj_dim must be > 0");

        projection_weight_ = register_parameter(
            "projection_weight",
            torch::empty({ config_.input_genes, config_.hidden_dim }, torch::TensorOptions().dtype(torch::kFloat32)));
        if (config_.use_bias) {
            projection_bias_ = register_parameter(
                "projection_bias",
                torch::empty({ config_.hidden_dim }, torch::TensorOptions().dtype(torch::kFloat32)));
        }

        register_module("proj_norm", proj_norm_);
        register_module("refine", refine_);
        register_module("out_norm", out_norm_);
        register_module("dropout", dropout_);
        reset_parameters();
    }

    void reset_parameters() {
        torch::nn::init::kaiming_uniform_(projection_weight_, std::sqrt(5.0));
        if (projection_bias_.defined()) {
            const double bound = 1.0 / std::sqrt(static_cast<double>(config_.input_genes));
            torch::nn::init::uniform_(projection_bias_, -bound, bound);
        }
        refine_->reset_parameters();
    }

    SparseTimeEncoderOutput forward(const torch::Tensor &sparse_csr_batch) {
        if (sparse_csr_batch.dim() != 2) {
            throw std::invalid_argument("SparseTimeEncoder expects a rank-2 sparse batch");
        }
        if (sparse_csr_batch.size(1) != config_.input_genes) {
            throw std::invalid_argument("SparseTimeEncoder input gene dimension does not match config");
        }
        if (sparse_csr_batch.layout() != torch::kSparse && sparse_csr_batch.layout() != torch::kSparseCsr) {
            throw std::invalid_argument("SparseTimeEncoder expects sparse COO or CSR input");
        }

        // Main overhead is sparse dtype/layout churn before the first
        // sparse-dense projection: values widen to f32 and CSR batches become
        // COO before matmul.
        torch::Tensor sparse_input = sparse_csr_batch.to(torch::kFloat32);
        if (sparse_input.layout() == torch::kSparseCsr) sparse_input = sparse_input.to_sparse();

        torch::Tensor projected = torch::matmul(sparse_input, projection_weight_);
        if (projection_bias_.defined()) projected = projected + projection_bias_;
        projected = proj_norm_(projected);
        projected = torch::nn::functional::silu(projected);
        projected = dropout_(projected);

        torch::Tensor embedding = refine_(projected);
        embedding = out_norm_(embedding);
        embedding = torch::nn::functional::silu(embedding);
        return SparseTimeEncoderOutput{ std::move(embedding) };
    }

private:
    SparseTimeEncoderConfig config_;
    torch::Tensor projection_weight_;
    torch::Tensor projection_bias_;
    torch::nn::LayerNorm proj_norm_{ nullptr };
    torch::nn::Linear refine_{ nullptr };
    torch::nn::LayerNorm out_norm_{ nullptr };
    torch::nn::Dropout dropout_{ nullptr };
};

TORCH_MODULE(SparseTimeEncoder);

class DevelopmentalStageHeadImpl : public torch::nn::Module {
public:
    explicit DevelopmentalStageHeadImpl(DevelopmentalStageHeadConfig config = DevelopmentalStageHeadConfig())
        : config_(std::move(config)),
          hidden_(config_.input_dim, config_.hidden_dim),
          hidden_norm_(torch::nn::LayerNormOptions({ config_.hidden_dim })),
          output_(config_.hidden_dim, 1) {
        if (config_.input_dim <= 0) throw std::invalid_argument("DevelopmentalStageHeadConfig.input_dim must be > 0");
        if (config_.hidden_dim <= 0) throw std::invalid_argument("DevelopmentalStageHeadConfig.hidden_dim must be > 0");

        register_module("hidden", hidden_);
        register_module("hidden_norm", hidden_norm_);
        register_module("output", output_);
    }

    DevelopmentalStageOutput forward(const torch::Tensor &embedding) {
        if (embedding.dim() != 2) throw std::invalid_argument("DevelopmentalStageHead expects a rank-2 embedding tensor");

        // Dense head; cheaper than the sparse encoder path above.
        torch::Tensor hidden = hidden_(embedding.to(torch::kFloat32));
        hidden = hidden_norm_(hidden);
        hidden = torch::nn::functional::silu(hidden);

        torch::Tensor stage_logit = output_(hidden).squeeze(-1);
        torch::Tensor stage = config_.bounded_output ? torch::sigmoid(stage_logit) : stage_logit;
        return DevelopmentalStageOutput{
            std::move(stage_logit),
            std::move(stage),
            embedding.to(torch::kFloat32)
        };
    }

private:
    DevelopmentalStageHeadConfig config_;
    torch::nn::Linear hidden_{ nullptr };
    torch::nn::LayerNorm hidden_norm_{ nullptr };
    torch::nn::Linear output_{ nullptr };
};

TORCH_MODULE(DevelopmentalStageHead);

class DevelopmentalStageModelImpl : public torch::nn::Module {
public:
    DevelopmentalStageModelImpl(
        SparseTimeEncoderConfig encoder_config = SparseTimeEncoderConfig(),
        DevelopmentalStageHeadConfig head_config = DevelopmentalStageHeadConfig())
        : encoder_(encoder_config) {
        if (head_config.input_dim == 0) head_config.input_dim = encoder_config.proj_dim;
        head_ = DevelopmentalStageHead(std::move(head_config));
        register_module("encoder", encoder_);
        register_module("head", head_);
    }

    DevelopmentalStageOutput forward(const torch::Tensor &sparse_csr_batch) {
        SparseTimeEncoderOutput encoder_out = encoder_->forward(sparse_csr_batch);
        DevelopmentalStageOutput output = head_->forward(encoder_out.embedding);
        output.embedding = std::move(encoder_out.embedding);
        return output;
    }

    torch::Tensor predict_stage(const torch::Tensor &sparse_csr_batch) {
        return forward(sparse_csr_batch).stage;
    }

private:
    SparseTimeEncoder encoder_{ nullptr };
    DevelopmentalStageHead head_{ nullptr };
};

TORCH_MODULE(DevelopmentalStageModel);

inline DevelopmentalStageLoss compute_developmental_stage_loss(
    const DevelopmentalStageOutput &output,
    const TimeBatch &batch,
    const DevelopmentalStageLossConfig &config = DevelopmentalStageLossConfig()) {
    torch::Tensor stage = output.stage.to(torch::kFloat32).view({ -1 });
    torch::Tensor day_buckets = batch.day_buckets.to(torch::kInt64).view({ -1 });
    torch::Tensor scalar_zero = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(stage.device()));
    torch::Tensor ranking = scalar_zero.clone();
    torch::Tensor anchor = scalar_zero.clone();
    torch::Tensor spread = scalar_zero.clone();
    std::vector<std::int64_t> bucket_ids;
    std::vector<torch::Tensor> bucket_means;
    std::size_t ranking_pairs = 0;

    if (stage.numel() != day_buckets.numel()) {
        throw std::invalid_argument("stage predictions and day buckets must have the same length");
    }
    if (stage.numel() == 0) {
        return DevelopmentalStageLoss{ scalar_zero.clone(), scalar_zero.clone(), scalar_zero.clone(), scalar_zero.clone() };
    }

    // Prefer the CUDA custom-op path when tensors stay on device.
    if (stage.is_cuda() && day_buckets.is_cuda()) {
        std::tie(ranking, anchor, spread) = model_ops::developmental_stage_bucket_losses(
            stage.contiguous(),
            day_buckets.contiguous(),
            config.ranking_margin,
            config.min_within_day_std,
            config.use_neighbor_day_pairs_only,
            config.num_day_buckets);
        torch::Tensor total = config.ranking_weight * ranking
            + config.anchor_weight * anchor
            + config.spread_weight * spread;
        return DevelopmentalStageLoss{ std::move(total), std::move(ranking), std::move(anchor), std::move(spread) };
    }

    torch::Tensor min_std = torch::full({}, config.min_within_day_std, scalar_zero.options());
    torch::Tensor margin = torch::full({}, config.ranking_margin, scalar_zero.options());

    {
        // CPU fallback pays for bucket copy, sorting, and eager masked selects.
        torch::Tensor bucket_cpu = day_buckets.to(torch::kCPU).contiguous();
        bucket_ids.resize(static_cast<std::size_t>(bucket_cpu.numel()));
        std::memcpy(bucket_ids.data(), bucket_cpu.data_ptr<std::int64_t>(), bucket_ids.size() * sizeof(std::int64_t));
        std::sort(bucket_ids.begin(), bucket_ids.end());
        bucket_ids.erase(std::unique(bucket_ids.begin(), bucket_ids.end()), bucket_ids.end());
    }

    bucket_means.reserve(bucket_ids.size());
    for (const std::int64_t bucket_id : bucket_ids) {
        torch::Tensor mask = day_buckets == bucket_id;
        torch::Tensor bucket_stage = stage.masked_select(mask);
        torch::Tensor mean = bucket_stage.mean();

        bucket_means.push_back(mean);
        if (bucket_stage.numel() > 1) {
            torch::Tensor stddev = bucket_stage.std(false);
            spread = spread + torch::relu(min_std - stddev);
        }

        const std::int64_t total_buckets = config.num_day_buckets > 0 ? config.num_day_buckets : (bucket_ids.back() + 1);
        const float anchor_value = total_buckets > 1
            ? static_cast<float>(bucket_id) / static_cast<float>(total_buckets - 1)
            : 0.5f;
        anchor = anchor + torch::pow(mean - anchor_value, 2);
    }

    if (bucket_means.size() > 1u) {
        for (std::size_t i = 0; i + 1u < bucket_means.size(); ++i) {
            const std::size_t j_begin = config.use_neighbor_day_pairs_only ? i + 1u : i + 1u;
            const std::size_t j_end = config.use_neighbor_day_pairs_only ? i + 2u : bucket_means.size();
            for (std::size_t j = j_begin; j < j_end; ++j) {
                ranking = ranking + torch::relu(margin - (bucket_means[j] - bucket_means[i]));
                ++ranking_pairs;
            }
        }
    }

    if (!bucket_means.empty()) {
        anchor = anchor / static_cast<double>(bucket_means.size());
        spread = spread / static_cast<double>(bucket_means.size());
    }
    if (ranking_pairs != 0u) ranking = ranking / static_cast<double>(ranking_pairs);

    torch::Tensor total = config.ranking_weight * ranking
        + config.anchor_weight * anchor
        + config.spread_weight * spread;
    return DevelopmentalStageLoss{ std::move(total), std::move(ranking), std::move(anchor), std::move(spread) };
}

} // namespace cellerator::models::developmental_time
