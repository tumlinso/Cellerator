#pragma once

#include <Cellerator/runtime/multi_gpu/fleet.cuh>
#include <Cellerator/memory/cuda/shared_buffer.hh>
#include <Cellerator/tensor/base/device_views.cuh>

#include <cuda_fp16.h>

#include <cstdint>

namespace cellerator::models::state_reduce {

namespace runtime = ::cellerator::runtime;
namespace csv = ::cellerator::tensor::base::device;

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
    csv::BlockedEllView blocked_ell{};
    csv::SlicedEllView sliced_ell{};
    StateReduceGraphView graph{};
};

inline StateReduceBatchView make_state_reduce_blocked_ell_batch(
    const csv::BlockedEllView &view,
    StateReduceGraphView graph = StateReduceGraphView()) {
    return StateReduceBatchView{
        StateReduceLayout::blocked_ell,
        view.rows,
        view,
        csv::SlicedEllView{},
        graph
    };
}

inline StateReduceBatchView make_state_reduce_sliced_ell_batch(
    const csv::SlicedEllView &view,
    StateReduceGraphView graph = StateReduceGraphView()) {
    return StateReduceBatchView{
        StateReduceLayout::sliced_ell,
        view.rows,
        csv::BlockedEllView{},
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
    runtime::execution_context ctx{};
    runtime::scratch_arena scratch{};
    runtime::cusparse_cache sparse_cache{};
    runtime::cublas_cache dense_cache{};
    runtime::fleet_context fleet{};
    bool fleet_ready = false;
    std::uint64_t step = 0u;

    ::cellerator::memory::cuda::SharedDeviceBuffer<float> encoder_weight;
    ::cellerator::memory::cuda::SharedDeviceBuffer<float> encoder_bias;
    ::cellerator::memory::cuda::SharedDeviceBuffer<float> decoder_factor;
    ::cellerator::memory::cuda::SharedDeviceBuffer<float> gene_dictionary;

    ::cellerator::memory::cuda::SharedDeviceBuffer<float> encoder_weight_m1;
    ::cellerator::memory::cuda::SharedDeviceBuffer<float> encoder_weight_m2;
    ::cellerator::memory::cuda::SharedDeviceBuffer<float> encoder_bias_m1;
    ::cellerator::memory::cuda::SharedDeviceBuffer<float> encoder_bias_m2;
    ::cellerator::memory::cuda::SharedDeviceBuffer<float> decoder_factor_m1;
    ::cellerator::memory::cuda::SharedDeviceBuffer<float> decoder_factor_m2;
    ::cellerator::memory::cuda::SharedDeviceBuffer<float> gene_dictionary_m1;
    ::cellerator::memory::cuda::SharedDeviceBuffer<float> gene_dictionary_m2;

    ::cellerator::memory::cuda::SharedDeviceBuffer<__half> decoder_factor_half;
    ::cellerator::memory::cuda::SharedDeviceBuffer<__half> gene_dictionary_half;
    ::cellerator::memory::cuda::SharedDeviceBuffer<float> row_ones;
};

void init(
    StateReduceModel *model,
    StateReduceModelConfig config = StateReduceModelConfig(),
    StateReduceDistributedConfig distributed = StateReduceDistributedConfig());

void clear(StateReduceModel *model);

::cellerator::memory::cuda::SharedDeviceBuffer<float> infer_embeddings(
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
