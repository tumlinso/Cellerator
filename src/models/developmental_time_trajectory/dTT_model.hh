#pragma once

#include "dTT_dataloader.hh"
#include "../developmental_time/dT_model.hh"

namespace cellerator::models::developmental_time_trajectory {

struct DevelopmentalTimeTrajectoryModelConfig {
    std::uint32_t input_genes = 0u;
    std::uint32_t stem_dim = 32u;
    std::uint32_t hidden_dim = 32u;
    bool use_encoder_bias = true;
    bool use_hidden_bias = true;
    bool use_output_bias = true;
    int device = -1;
    dt::DevelopmentalTimeBackend backend = dt::DevelopmentalTimeBackend::tensor_cusparse;
};

struct DevelopmentalTimeTrajectoryLossConfig {
    float huber_delta = 0.15f;
    float smoothness_weight = 0.10f;
    float order_weight = 0.20f;
    float order_margin = 0.05f;
};

using DevelopmentalTimeTrajectoryOptimizerConfig = dt::DevelopmentalTimeOptimizerConfig;

struct DevelopmentalTimeTrajectoryMetrics {
    float total = 0.0f;
    float regression = 0.0f;
    float smoothness = 0.0f;
    float order = 0.0f;
};

using DevelopmentalTimeTrajectoryModel = dt::DevelopmentalTimeModel;

void init(
    DevelopmentalTimeTrajectoryModel *model,
    DevelopmentalTimeTrajectoryModelConfig config = DevelopmentalTimeTrajectoryModelConfig());

void clear(DevelopmentalTimeTrajectoryModel *model);

dt::autograd::device_buffer<float> infer_time(
    DevelopmentalTimeTrajectoryModel *model,
    const DevelopmentalTimeTrajectoryBatchView &batch,
    float graph_mix = 0.5f);

DevelopmentalTimeTrajectoryMetrics evaluate(
    DevelopmentalTimeTrajectoryModel *model,
    const DevelopmentalTimeTrajectoryBatchView &batch,
    const DevelopmentalTimeTrajectoryLossConfig &loss_config = DevelopmentalTimeTrajectoryLossConfig());

DevelopmentalTimeTrajectoryMetrics train_step(
    DevelopmentalTimeTrajectoryModel *model,
    const DevelopmentalTimeTrajectoryBatchView &batch,
    const DevelopmentalTimeTrajectoryLossConfig &loss_config = DevelopmentalTimeTrajectoryLossConfig(),
    const DevelopmentalTimeTrajectoryOptimizerConfig &optimizer_config = DevelopmentalTimeTrajectoryOptimizerConfig());

} // namespace cellerator::models::developmental_time_trajectory
