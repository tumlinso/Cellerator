#pragma once

#include "../../compute/autograd/autograd.hh"
#include "../../../extern/CellShard/include/CellShard/runtime/device/sharded_device.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace cellerator::models::state_reduce {

namespace autograd = ::cellerator::compute::autograd;
namespace csv = ::cellshard::device;

enum class StateReduceBackend {
    wmma_fused,
    cusparse_heavy
};

enum class StateReduceLayout {
    blocked_ell,
    sliced_ell
};

struct StateReduceModelConfig {
    std::uint32_t input_genes = 0u;
    std::uint32_t latent_dim = 16u;
    std::uint32_t factor_dim = 16u;
    bool use_encoder_bias = true;
    int device = -1;
    StateReduceBackend backend = StateReduceBackend::wmma_fused;
};

struct StateReduceLossConfig {
    float reconstruction_weight = 1.0f;
    float nonzero_weight = 4.0f;
    float zero_weight = 0.25f;
    float graph_weight = 0.20f;
};

struct StateReduceOptimizerConfig {
    float learning_rate = 1.0e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1.0e-8f;
    float weight_decay = 1.0e-4f;
};

struct StateReduceDistributedConfig {
    bool enable_nccl = true;
    unsigned int requested_slots = 1u;
    bool pair_local_first = true;
};

struct StateReduceGraphView {
    std::uint32_t edge_count = 0u;
    const std::uint32_t *src = nullptr;
    const std::uint32_t *dst = nullptr;
    const float *weight = nullptr;
};

struct StateReduceBatchView {
    StateReduceLayout layout = StateReduceLayout::blocked_ell;
    std::uint32_t rows = 0u;
    csv::blocked_ell_view blocked_ell{};
    csv::sliced_ell_view sliced_ell{};
    StateReduceGraphView graph{};
};

inline StateReduceBatchView make_state_reduce_blocked_ell_batch(
    const csv::blocked_ell_view &view,
    StateReduceGraphView graph = StateReduceGraphView()) {
    return StateReduceBatchView{
        StateReduceLayout::blocked_ell,
        view.rows,
        view,
        csv::sliced_ell_view{},
        graph
    };
}

inline StateReduceBatchView make_state_reduce_sliced_ell_batch(
    const csv::sliced_ell_view &view,
    StateReduceGraphView graph = StateReduceGraphView()) {
    return StateReduceBatchView{
        StateReduceLayout::sliced_ell,
        view.rows,
        csv::blocked_ell_view{},
        view,
        graph
    };
}

struct StateReduceTrainMetrics {
    float total = 0.0f;
    float reconstruction = 0.0f;
    float graph = 0.0f;
};

struct StateReduceModel {
    StateReduceModelConfig config{};
    StateReduceDistributedConfig distributed{};
    autograd::execution_context ctx{};
    autograd::scratch_arena scratch{};
    autograd::cusparse_cache sparse_cache{};
    autograd::fleet_context fleet{};
    bool fleet_ready = false;
    std::uint64_t step = 0u;

    autograd::device_buffer<float> encoder_weight;
    autograd::device_buffer<float> encoder_bias;
    autograd::device_buffer<float> decoder_factor;
    autograd::device_buffer<float> gene_dictionary;

    autograd::device_buffer<float> encoder_weight_m1;
    autograd::device_buffer<float> encoder_weight_m2;
    autograd::device_buffer<float> encoder_bias_m1;
    autograd::device_buffer<float> encoder_bias_m2;
    autograd::device_buffer<float> decoder_factor_m1;
    autograd::device_buffer<float> decoder_factor_m2;
    autograd::device_buffer<float> gene_dictionary_m1;
    autograd::device_buffer<float> gene_dictionary_m2;

    autograd::device_buffer<__half> decoder_factor_half;
    autograd::device_buffer<__half> gene_dictionary_half;
};

void init(
    StateReduceModel *model,
    StateReduceModelConfig config = StateReduceModelConfig(),
    StateReduceDistributedConfig distributed = StateReduceDistributedConfig());

void clear(StateReduceModel *model);

autograd::device_buffer<float> infer_embeddings(
    StateReduceModel *model,
    const StateReduceBatchView &batch);

StateReduceTrainMetrics evaluate(
    StateReduceModel *model,
    const StateReduceBatchView &batch,
    const StateReduceLossConfig &loss_config = StateReduceLossConfig());

StateReduceTrainMetrics train_step(
    StateReduceModel *model,
    const StateReduceBatchView &batch,
    const StateReduceLossConfig &loss_config = StateReduceLossConfig(),
    const StateReduceOptimizerConfig &optimizer_config = StateReduceOptimizerConfig());

} // namespace cellerator::models::state_reduce
