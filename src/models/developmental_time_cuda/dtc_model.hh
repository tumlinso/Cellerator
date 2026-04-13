#pragma once

#include "../developmental_time/dT_dataloader.hh"

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace cellerator::models::developmental_time_cuda {

struct DevelopmentalTimeCudaConfig {
    std::int64_t input_genes = 0;
    std::int64_t stem_dim = 256;
    std::int64_t hidden_dim = 128;
    std::int64_t num_time_bins = 8;
    bool use_bias = true;
    torch::Device device = torch::Device(torch::kCUDA, 0);
};

struct DevelopmentalTimeCudaLossConfig {
    double regression_weight = 1.0;
    double bin_weight = 0.25;
    double huber_delta = 0.1;
};

struct DevelopmentalTimeCudaTrainConfig {
    double learning_rate = 1.0e-2;
    double weight_decay = 1.0e-4;
    double max_grad_norm = 1.0;
    bool clip_gradients = true;
};

struct DevelopmentalTimeCudaOutput {
    torch::Tensor predicted_time;
    torch::Tensor time_bin_logits;
    torch::Tensor embedding;
};

struct DevelopmentalTimeCudaLoss {
    torch::Tensor total;
    torch::Tensor regression;
    torch::Tensor bin_classification;
};

struct DevelopmentalTimeCudaTrainStep {
    DevelopmentalTimeCudaOutput output;
    DevelopmentalTimeCudaLoss loss;
};

class DevelopmentalTimeCudaModel {
public:
    explicit DevelopmentalTimeCudaModel(DevelopmentalTimeCudaConfig config = DevelopmentalTimeCudaConfig());

    void to(const torch::Device &device);
    const DevelopmentalTimeCudaConfig &config() const;
    torch::Device device() const;

    torch::Tensor &stem_weight();
    torch::Tensor &stem_bias();
    torch::Tensor &hidden_weight();
    torch::Tensor &hidden_bias();
    torch::Tensor &time_weight();
    torch::Tensor &time_bias();
    torch::Tensor &bin_weight();
    torch::Tensor &bin_bias();

    const torch::Tensor &stem_weight() const;
    const torch::Tensor &stem_bias() const;
    const torch::Tensor &hidden_weight() const;
    const torch::Tensor &hidden_bias() const;
    const torch::Tensor &time_weight() const;
    const torch::Tensor &time_bias() const;
    const torch::Tensor &bin_weight() const;
    const torch::Tensor &bin_bias() const;

    std::vector<torch::Tensor *> parameters();
    std::vector<const torch::Tensor *> parameters() const;

private:
    void reset_parameters_();

    DevelopmentalTimeCudaConfig config_;
    torch::Tensor stem_weight_;
    torch::Tensor stem_bias_;
    torch::Tensor hidden_weight_;
    torch::Tensor hidden_bias_;
    torch::Tensor time_weight_;
    torch::Tensor time_bias_;
    torch::Tensor bin_weight_;
    torch::Tensor bin_bias_;
};

DevelopmentalTimeCudaOutput forward(
    DevelopmentalTimeCudaModel &model,
    const torch::Tensor &sparse_csr_batch);

torch::Tensor predict_time(
    DevelopmentalTimeCudaModel &model,
    const torch::Tensor &sparse_csr_batch);

DevelopmentalTimeCudaLoss compute_loss(
    const DevelopmentalTimeCudaOutput &output,
    const ::cellerator::models::developmental_time::TimeBatch &batch,
    const DevelopmentalTimeCudaLossConfig &config = DevelopmentalTimeCudaLossConfig());

DevelopmentalTimeCudaTrainStep train_step(
    DevelopmentalTimeCudaModel &model,
    const ::cellerator::models::developmental_time::TimeBatch &batch,
    const DevelopmentalTimeCudaLossConfig &loss_config = DevelopmentalTimeCudaLossConfig(),
    const DevelopmentalTimeCudaTrainConfig &train_config = DevelopmentalTimeCudaTrainConfig());

} // namespace cellerator::models::developmental_time_cuda
