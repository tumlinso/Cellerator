#pragma once

#include "dT_dataloader.hh"

#include <cstdint>

namespace cellerator::models::developmental_time {

enum class DevelopmentalTimeBackend {
    tensor_cusparse,
    custom_wmma
};

struct DevelopmentalTimeModelConfig {
    std::uint32_t input_genes = 0u;
    std::uint32_t stem_dim = 32u;
    std::uint32_t hidden_dim = 32u;
    bool use_encoder_bias = true;
    bool use_hidden_bias = true;
    bool use_output_bias = true;
    int device = -1;
    DevelopmentalTimeBackend backend = DevelopmentalTimeBackend::tensor_cusparse;
};

struct DevelopmentalTimeLossConfig {
    float huber_delta = 0.15f;
};

struct DevelopmentalTimeOptimizerConfig {
    float learning_rate = 1.0e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1.0e-8f;
    float weight_decay = 1.0e-4f;
};

struct DevelopmentalTimeMetrics {
    float total = 0.0f;
    float regression = 0.0f;
};

struct DevelopmentalTimeModel {
    DevelopmentalTimeModelConfig config{};
    autograd::execution_context ctx{};
    autograd::scratch_arena scratch{};
    autograd::cusparse_cache sparse_cache{};
    std::uint64_t step = 0u;

    autograd::device_buffer<float> encoder_weight;
    autograd::device_buffer<float> encoder_bias;
    autograd::device_buffer<float> hidden_weight;
    autograd::device_buffer<float> hidden_bias;
    autograd::device_buffer<float> output_weight;
    autograd::device_buffer<float> output_bias;

    autograd::device_buffer<float> encoder_weight_m1;
    autograd::device_buffer<float> encoder_weight_m2;
    autograd::device_buffer<float> encoder_bias_m1;
    autograd::device_buffer<float> encoder_bias_m2;
    autograd::device_buffer<float> hidden_weight_m1;
    autograd::device_buffer<float> hidden_weight_m2;
    autograd::device_buffer<float> hidden_bias_m1;
    autograd::device_buffer<float> hidden_bias_m2;
    autograd::device_buffer<float> output_weight_m1;
    autograd::device_buffer<float> output_weight_m2;
    autograd::device_buffer<float> output_bias_m1;
    autograd::device_buffer<float> output_bias_m2;
};

void init(
    DevelopmentalTimeModel *model,
    DevelopmentalTimeModelConfig config = DevelopmentalTimeModelConfig());

void clear(DevelopmentalTimeModel *model);

autograd::device_buffer<float> infer_time(
    DevelopmentalTimeModel *model,
    const DevelopmentalTimeBatchView &batch);

DevelopmentalTimeMetrics evaluate(
    DevelopmentalTimeModel *model,
    const DevelopmentalTimeBatchView &batch,
    const DevelopmentalTimeLossConfig &loss_config = DevelopmentalTimeLossConfig());

DevelopmentalTimeMetrics train_step(
    DevelopmentalTimeModel *model,
    const DevelopmentalTimeBatchView &batch,
    const DevelopmentalTimeLossConfig &loss_config = DevelopmentalTimeLossConfig(),
    const DevelopmentalTimeOptimizerConfig &optimizer_config = DevelopmentalTimeOptimizerConfig());

using DevelopmentalStageMetrics = DevelopmentalTimeMetrics;

} // namespace cellerator::models::developmental_time
