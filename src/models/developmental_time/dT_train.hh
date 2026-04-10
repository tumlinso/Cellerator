#pragma once

#include "dT_model.hh"

#include <torch/torch.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace cellerator::models::developmental_time {

struct DevelopmentalStageTrainConfig {
    double learning_rate = 1.0e-3;
    double weight_decay = 1.0e-4;
    double loss_scale = 128.0;
    double max_grad_norm = 1.0;
    bool clip_gradients = true;
    bool skip_non_finite_updates = true;
};

struct DevelopmentalStageTrainStep {
    DevelopmentalStageOutput output;
    DevelopmentalStageLoss loss;
};

inline torch::optim::AdamW make_developmental_stage_optimizer(
    DevelopmentalStageModel &model,
    const DevelopmentalStageTrainConfig &config = DevelopmentalStageTrainConfig()) {
    return torch::optim::AdamW(
        model->parameters(),
        torch::optim::AdamWOptions(config.learning_rate).weight_decay(config.weight_decay));
}

inline DevelopmentalStageTrainStep train_developmental_stage_step(
    DevelopmentalStageModel &model,
    torch::optim::Optimizer &optimizer,
    const TimeBatch &batch,
    const DevelopmentalStageLossConfig &loss_config = DevelopmentalStageLossConfig(),
    const DevelopmentalStageTrainConfig &train_config = DevelopmentalStageTrainConfig()) {
    if (train_config.loss_scale <= 0.0) throw std::invalid_argument("loss_scale must be > 0");

    model->train();
    optimizer.zero_grad();

    DevelopmentalStageOutput output = model->forward(batch.features);
    DevelopmentalStageLoss loss = compute_developmental_stage_loss(output, batch, loss_config);
    (loss.total * train_config.loss_scale).backward();

    for (torch::Tensor &param : model->parameters()) {
        if (param.grad().defined()) param.grad().div_(train_config.loss_scale);
    }
    if (train_config.skip_non_finite_updates) {
        for (const torch::Tensor &param : model->parameters()) {
            if (param.grad().defined() && !torch::isfinite(param.grad()).all().item<bool>()) {
                optimizer.zero_grad();
                return DevelopmentalStageTrainStep{ std::move(output), std::move(loss) };
            }
        }
    }
    if (train_config.clip_gradients) {
        torch::nn::utils::clip_grad_norm_(model->parameters(), train_config.max_grad_norm);
    }
    optimizer.step();
    return DevelopmentalStageTrainStep{ std::move(output), std::move(loss) };
}

inline DevelopmentalStageTrainStep evaluate_developmental_stage_step(
    DevelopmentalStageModel &model,
    const TimeBatch &batch,
    const DevelopmentalStageLossConfig &loss_config = DevelopmentalStageLossConfig()) {
    torch::NoGradGuard no_grad;
    model->eval();

    DevelopmentalStageOutput output = model->forward(batch.features);
    DevelopmentalStageLoss loss = compute_developmental_stage_loss(output, batch, loss_config);
    return DevelopmentalStageTrainStep{ std::move(output), std::move(loss) };
}

} // namespace cellerator::models::developmental_time
