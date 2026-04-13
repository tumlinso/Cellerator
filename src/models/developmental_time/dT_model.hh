#pragma once

#include "dT_dataloader.hh"

#include <torch/torch.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace cellerator::models::developmental_time {

struct SparseTimeEncoderConfig {
    std::int64_t input_genes = 0;
    std::int64_t hidden_dim = 256;
    std::int64_t proj_dim = 128;
    bool use_bias = true;
    double dropout = 0.1;
};

struct DevelopmentalTimeHeadConfig {
    std::int64_t input_dim = 0;
    std::int64_t hidden_dim = 128;
    std::int64_t num_time_bins = 8;
};

struct DevelopmentalTimeLossConfig {
    double regression_weight = 1.0;
    double bin_weight = 0.25;
    double huber_delta = 0.1;
};

using DevelopmentalStageHeadConfig = DevelopmentalTimeHeadConfig;
using DevelopmentalStageLossConfig = DevelopmentalTimeLossConfig;

struct SparseTimeEncoderOutput {
    torch::Tensor embedding;
};

struct DevelopmentalTimeOutput {
    torch::Tensor predicted_time;
    torch::Tensor time_bin_logits;
    torch::Tensor embedding;
};

struct DevelopmentalTimeLoss {
    torch::Tensor total;
    torch::Tensor regression;
    torch::Tensor bin_classification;
};

using DevelopmentalStageOutput = DevelopmentalTimeOutput;
using DevelopmentalStageLoss = DevelopmentalTimeLoss;

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

    torch::Tensor stage_to_time(const torch::Tensor &predicted_time, const std::string &species_id) const {
        const auto it = calibrations_.find(species_id);
        if (it == calibrations_.end()) {
            throw std::invalid_argument("no developmental-time calibration registered for species '" + species_id + "'");
        }
        return predicted_time.to(torch::kFloat32) * std::max(it->second.slope, 0.0f) + it->second.intercept;
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

class DevelopmentalTimeHeadImpl : public torch::nn::Module {
public:
    explicit DevelopmentalTimeHeadImpl(DevelopmentalTimeHeadConfig config = DevelopmentalTimeHeadConfig())
        : config_(std::move(config)),
          hidden_(config_.input_dim, config_.hidden_dim),
          hidden_norm_(torch::nn::LayerNormOptions({ config_.hidden_dim })),
          time_output_(config_.hidden_dim, 1),
          bin_output_(config_.hidden_dim, config_.num_time_bins) {
        if (config_.input_dim <= 0) throw std::invalid_argument("DevelopmentalTimeHeadConfig.input_dim must be > 0");
        if (config_.hidden_dim <= 0) throw std::invalid_argument("DevelopmentalTimeHeadConfig.hidden_dim must be > 0");
        if (config_.num_time_bins <= 1) throw std::invalid_argument("DevelopmentalTimeHeadConfig.num_time_bins must be > 1");

        register_module("hidden", hidden_);
        register_module("hidden_norm", hidden_norm_);
        register_module("time_output", time_output_);
        register_module("bin_output", bin_output_);
    }

    DevelopmentalTimeOutput forward(const torch::Tensor &embedding) {
        if (embedding.dim() != 2) throw std::invalid_argument("DevelopmentalTimeHead expects a rank-2 embedding tensor");

        torch::Tensor hidden = hidden_(embedding.to(torch::kFloat32));
        hidden = hidden_norm_(hidden);
        hidden = torch::nn::functional::silu(hidden);

        torch::Tensor predicted_time = time_output_(hidden).squeeze(-1);
        torch::Tensor time_bin_logits = bin_output_(hidden);
        return DevelopmentalTimeOutput{
            std::move(predicted_time),
            std::move(time_bin_logits),
            embedding.to(torch::kFloat32)
        };
    }

private:
    DevelopmentalTimeHeadConfig config_;
    torch::nn::Linear hidden_{ nullptr };
    torch::nn::LayerNorm hidden_norm_{ nullptr };
    torch::nn::Linear time_output_{ nullptr };
    torch::nn::Linear bin_output_{ nullptr };
};

TORCH_MODULE(DevelopmentalTimeHead);
using DevelopmentalStageHead = DevelopmentalTimeHead;

class DevelopmentalTimeModelImpl : public torch::nn::Module {
public:
    DevelopmentalTimeModelImpl(
        SparseTimeEncoderConfig encoder_config = SparseTimeEncoderConfig(),
        DevelopmentalTimeHeadConfig head_config = DevelopmentalTimeHeadConfig())
        : encoder_(encoder_config) {
        if (head_config.input_dim == 0) head_config.input_dim = encoder_config.proj_dim;
        head_ = DevelopmentalTimeHead(std::move(head_config));
        register_module("encoder", encoder_);
        register_module("head", head_);
    }

    DevelopmentalTimeOutput forward(const torch::Tensor &sparse_csr_batch) {
        SparseTimeEncoderOutput encoder_out = encoder_->forward(sparse_csr_batch);
        DevelopmentalTimeOutput output = head_->forward(encoder_out.embedding);
        output.embedding = std::move(encoder_out.embedding);
        return output;
    }

    torch::Tensor predict_time(const torch::Tensor &sparse_csr_batch) {
        return forward(sparse_csr_batch).predicted_time;
    }

private:
    SparseTimeEncoder encoder_{ nullptr };
    DevelopmentalTimeHead head_{ nullptr };
};

TORCH_MODULE(DevelopmentalTimeModel);
using DevelopmentalStageModel = DevelopmentalTimeModel;

inline DevelopmentalTimeLoss compute_developmental_time_loss(
    const DevelopmentalTimeOutput &output,
    const TimeBatch &batch,
    const DevelopmentalTimeLossConfig &config = DevelopmentalTimeLossConfig()) {
    torch::Tensor predicted_time = output.predicted_time.to(torch::kFloat32).view({ -1 });
    torch::Tensor target_time = batch.day_labels.to(predicted_time.options()).view({ -1 });
    torch::Tensor time_bins = batch.day_buckets.to(
        torch::TensorOptions().dtype(torch::kInt64).device(predicted_time.device())).view({ -1 });

    if (predicted_time.numel() != target_time.numel()) {
        throw std::invalid_argument("predicted_time and day_labels must have the same length");
    }
    if (predicted_time.numel() != time_bins.numel()) {
        throw std::invalid_argument("predicted_time and day_buckets must have the same length");
    }

    torch::Tensor zero = torch::zeros({}, predicted_time.options());
    if (predicted_time.numel() == 0) {
        return DevelopmentalTimeLoss{ zero.clone(), zero.clone(), zero.clone() };
    }

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
    return DevelopmentalTimeLoss{
        std::move(total),
        std::move(regression),
        std::move(bin_classification)
    };
}

inline DevelopmentalStageLoss compute_developmental_stage_loss(
    const DevelopmentalStageOutput &output,
    const TimeBatch &batch,
    const DevelopmentalStageLossConfig &config = DevelopmentalStageLossConfig()) {
    return compute_developmental_time_loss(output, batch, config);
}

} // namespace cellerator::models::developmental_time
