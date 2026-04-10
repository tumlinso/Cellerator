#pragma once

#include "dR_model.hh"

#include <torch/torch.h>

#include <stdexcept>
#include <utility>

namespace cellerator::models::dense_reduce {

struct DenseReduceTrainConfig {
    double learning_rate = 1.0e-3;
    double weight_decay = 1.0e-4;
    double loss_scale = 128.0;
    double max_grad_norm = 1.0;
    bool clip_gradients = true;
    bool skip_non_finite_updates = true;
};

struct DenseReduceTrainStep {
    DenseReduceOutput output;
    DenseReduceLoss loss;
};

inline torch::optim::AdamW make_dense_reduce_optimizer(
    DenseReduceModel &model,
    const DenseReduceTrainConfig &config = DenseReduceTrainConfig()) {
    return torch::optim::AdamW(
        model->parameters(),
        torch::optim::AdamWOptions(config.learning_rate).weight_decay(config.weight_decay));
}

inline DenseReduceTrainStep train_dense_reduce_step(
    DenseReduceModel &model,
    torch::optim::Optimizer &optimizer,
    const DenseReduceBatch &batch,
    const DenseReduceLossConfig &loss_config = DenseReduceLossConfig(),
    const DenseReduceTrainConfig &train_config = DenseReduceTrainConfig()) {
    if (train_config.loss_scale <= 0.0) throw std::invalid_argument("loss_scale must be > 0");

    model->train();
    optimizer.zero_grad();

    DenseReduceOutput output = model->forward(batch.features);
    DenseReduceLoss loss = compute_dense_reduce_loss(output, batch, loss_config);
    (loss.total * train_config.loss_scale).backward();

    for (torch::Tensor &param : model->parameters()) {
        if (param.grad().defined()) param.grad().div_(train_config.loss_scale);
    }
    if (train_config.skip_non_finite_updates) {
        for (const torch::Tensor &param : model->parameters()) {
            if (param.grad().defined() && !torch::isfinite(param.grad()).all().item<bool>()) {
                optimizer.zero_grad();
                return DenseReduceTrainStep{ std::move(output), std::move(loss) };
            }
        }
    }
    if (train_config.clip_gradients) {
        torch::nn::utils::clip_grad_norm_(model->parameters(), train_config.max_grad_norm);
    }
    optimizer.step();
    return DenseReduceTrainStep{ std::move(output), std::move(loss) };
}

inline DenseReduceTrainStep evaluate_dense_reduce_step(
    DenseReduceModel &model,
    const DenseReduceBatch &batch,
    const DenseReduceLossConfig &loss_config = DenseReduceLossConfig()) {
    torch::NoGradGuard no_grad;
    model->eval();

    DenseReduceOutput output = model->forward(batch.features);
    DenseReduceLoss loss = compute_dense_reduce_loss(output, batch, loss_config);
    return DenseReduceTrainStep{ std::move(output), std::move(loss) };
}

} // namespace cellerator::models::dense_reduce
