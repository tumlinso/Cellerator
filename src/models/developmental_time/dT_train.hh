#pragma once

#include "dT_model.hh"

#include <torch/torch.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace cellerator::models::developmental_time {

struct DevelopmentalTimeTrainConfig {
    double learning_rate = 1.0e-2;
    double weight_decay = 1.0e-4;
    double momentum = 0.0;
    double loss_scale = 128.0;
    double max_grad_norm = 1.0;
    bool clip_gradients = true;
    bool skip_non_finite_updates = true;
};

using DevelopmentalStageTrainConfig = DevelopmentalTimeTrainConfig;

struct DevelopmentalTimeTrainStep {
    DevelopmentalTimeOutput output;
    DevelopmentalTimeLoss loss;
};

using DevelopmentalStageTrainStep = DevelopmentalTimeTrainStep;

inline torch::optim::SGD make_developmental_time_optimizer(
    DevelopmentalTimeModel &model,
    const DevelopmentalTimeTrainConfig &config = DevelopmentalTimeTrainConfig()) {
    return torch::optim::SGD(
        model->parameters(),
        torch::optim::SGDOptions(config.learning_rate)
            .weight_decay(config.weight_decay)
            .momentum(config.momentum));
}

inline torch::optim::SGD make_developmental_stage_optimizer(
    DevelopmentalStageModel &model,
    const DevelopmentalStageTrainConfig &config = DevelopmentalStageTrainConfig()) {
    return make_developmental_time_optimizer(model, config);
}

inline DevelopmentalTimeTrainStep train_developmental_time_step(
    DevelopmentalTimeModel &model,
    torch::optim::Optimizer &optimizer,
    const TimeBatch &batch,
    const DevelopmentalTimeLossConfig &loss_config = DevelopmentalTimeLossConfig(),
    const DevelopmentalTimeTrainConfig &train_config = DevelopmentalTimeTrainConfig()) {
    if (train_config.loss_scale <= 0.0) throw std::invalid_argument("loss_scale must be > 0");

    model->train();
    optimizer.zero_grad();

    DevelopmentalTimeOutput output = model->forward(batch.features);
    DevelopmentalTimeLoss loss = compute_developmental_time_loss(output, batch, loss_config);
    (loss.total * train_config.loss_scale).backward();

    for (torch::Tensor &param : model->parameters()) {
        if (param.grad().defined()) param.grad().div_(train_config.loss_scale);
    }
    if (train_config.skip_non_finite_updates) {
        for (const torch::Tensor &param : model->parameters()) {
            if (param.grad().defined() && !torch::isfinite(param.grad()).all().item<bool>()) {
                optimizer.zero_grad();
                return DevelopmentalTimeTrainStep{ std::move(output), std::move(loss) };
            }
        }
    }
    if (train_config.clip_gradients) {
        torch::nn::utils::clip_grad_norm_(model->parameters(), train_config.max_grad_norm);
    }
    optimizer.step();
    return DevelopmentalTimeTrainStep{ std::move(output), std::move(loss) };
}

inline DevelopmentalStageTrainStep train_developmental_stage_step(
    DevelopmentalStageModel &model,
    torch::optim::Optimizer &optimizer,
    const TimeBatch &batch,
    const DevelopmentalStageLossConfig &loss_config = DevelopmentalStageLossConfig(),
    const DevelopmentalStageTrainConfig &train_config = DevelopmentalStageTrainConfig()) {
    return train_developmental_time_step(model, optimizer, batch, loss_config, train_config);
}

inline DevelopmentalTimeTrainStep evaluate_developmental_time_step(
    DevelopmentalTimeModel &model,
    const TimeBatch &batch,
    const DevelopmentalTimeLossConfig &loss_config = DevelopmentalTimeLossConfig()) {
    torch::NoGradGuard no_grad;
    model->eval();

    DevelopmentalTimeOutput output = model->forward(batch.features);
    DevelopmentalTimeLoss loss = compute_developmental_time_loss(output, batch, loss_config);
    return DevelopmentalTimeTrainStep{ std::move(output), std::move(loss) };
}

inline DevelopmentalStageTrainStep evaluate_developmental_stage_step(
    DevelopmentalStageModel &model,
    const TimeBatch &batch,
    const DevelopmentalStageLossConfig &loss_config = DevelopmentalStageLossConfig()) {
    return evaluate_developmental_time_step(model, batch, loss_config);
}

} // namespace cellerator::models::developmental_time
