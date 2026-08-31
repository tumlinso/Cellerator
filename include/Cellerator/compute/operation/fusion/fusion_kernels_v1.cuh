#pragma once

#include <Cellerator/compute/operation/fusion/prepared_stage_graph_v1.hh>

#include <cuda_runtime.h>

#include <cstdint>

namespace cellerator::compute::operation::fusion {

struct relation_edge_v1 {
    std::uint32_t source_local = 0u;
    std::uint32_t destination_local = 0u;
    std::uint32_t projection_slot_local = 0u;
};

struct pack_apply_request_v1 {
    const relation_edge_v1 *logical_edges = nullptr;
    const std::uint32_t *logical_to_projection = nullptr;
    const float *logical_edge_values = nullptr;
    float *projection_edge_values = nullptr;
    const float *source = nullptr;
    float *destination = nullptr;
    std::uint64_t global_edge_begin = 0u;
    std::uint32_t local_edge_count = 0u;
    std::uint32_t source_count = 0u;
    std::uint32_t destination_count = 0u;
    std::uint32_t component_count = 0u;
    std::uint64_t structure_id = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t value_generation = 0u;
    std::uint64_t profiler_correlation_id = 0u;
    cudaStream_t stream = nullptr;
};

status_v1 validate_pack_apply_request_v1(
    const pack_apply_request_v1 &request) noexcept;
status_v1 enqueue_value_pack_unfused_v1(
    const pack_apply_request_v1 &request) noexcept;
status_v1 enqueue_apply_from_packed_unfused_v1(
    const pack_apply_request_v1 &request) noexcept;
status_v1 enqueue_value_pack_apply_fused_v1(
    const pack_apply_request_v1 &request) noexcept;

struct row_edge_v1 {
    std::uint32_t source_local = 0u;
    std::uint32_t projection_slot_local = 0u;
};

struct apply_epilogue_request_v1 {
    const std::uint32_t *destination_row_offsets = nullptr;
    const row_edge_v1 *edges = nullptr;
    const float *projection_edge_values = nullptr;
    const float *source = nullptr;
    const float *prior_destination = nullptr;
    const float *bias = nullptr;
    float *accumulation_workspace = nullptr;
    float *destination = nullptr;
    std::uint32_t edge_count = 0u;
    std::uint32_t source_count = 0u;
    std::uint32_t destination_count = 0u;
    std::uint32_t component_count = 0u;
    float alpha = 1.0f;
    float beta = 0.0f;
    bool relu = false;
    std::uint8_t reserved[3]{};
    cudaStream_t stream = nullptr;
};

status_v1 validate_apply_epilogue_request_v1(
    const apply_epilogue_request_v1 &request) noexcept;
status_v1 enqueue_relation_apply_unfused_v1(
    const apply_epilogue_request_v1 &request) noexcept;
status_v1 enqueue_epilogue_unfused_v1(
    const apply_epilogue_request_v1 &request) noexcept;
status_v1 enqueue_apply_epilogue_fused_v1(
    const apply_epilogue_request_v1 &request) noexcept;

struct mma_residual_request_v1 {
    const float *mma_contribution = nullptr;
    const float *same_owner_residual = nullptr;
    float *output = nullptr;
    std::uint64_t global_output_begin = 0u;
    std::uint32_t local_output_count = 0u;
    std::uint64_t owner_order_id = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t value_generation = 0u;
    std::uint64_t profiler_correlation_id = 0u;
    cudaStream_t stream = nullptr;
};

status_v1 validate_mma_residual_request_v1(
    const mma_residual_request_v1 &request) noexcept;
status_v1 enqueue_mma_contribution_unfused_v1(
    const mma_residual_request_v1 &request) noexcept;
status_v1 enqueue_same_owner_residual_unfused_v1(
    const mma_residual_request_v1 &request) noexcept;
status_v1 enqueue_mma_same_owner_residual_fused_v1(
    const mma_residual_request_v1 &request) noexcept;

struct contract_edge_v1 {
    std::uint32_t source_local = 0u;
    std::uint32_t destination_local = 0u;
};

struct contract_composition_request_v1 {
    const contract_edge_v1 *edges = nullptr;
    const float *source = nullptr;
    const float *destination = nullptr;
    const float *per_edge_gate = nullptr;
    const std::uint32_t *segment_offsets = nullptr;
    float *contraction_workspace = nullptr;
    float *mapped_output = nullptr;
    float *segment_sum_output = nullptr;
    float *segment_maximum_output = nullptr;
    std::uint64_t global_edge_begin = 0u;
    std::uint32_t edge_count = 0u;
    std::uint32_t source_count = 0u;
    std::uint32_t destination_count = 0u;
    std::uint32_t dense_width = 0u;
    std::uint32_t segment_count = 0u;
    float map_scale = 1.0f;
    float map_bias = 0.0f;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t value_generation = 0u;
    cudaStream_t stream = nullptr;
};

status_v1 validate_contract_composition_request_v1(
    const contract_composition_request_v1 &request) noexcept;
status_v1 enqueue_contraction_unfused_v1(
    const contract_composition_request_v1 &request) noexcept;
status_v1 enqueue_edge_map_unfused_v1(
    const contract_composition_request_v1 &request) noexcept;
status_v1 enqueue_contract_edge_map_fused_v1(
    const contract_composition_request_v1 &request) noexcept;
status_v1 enqueue_segment_statistic_unfused_v1(
    const contract_composition_request_v1 &request) noexcept;
status_v1 enqueue_contract_segment_statistic_fused_v1(
    const contract_composition_request_v1 &request) noexcept;

struct sparse_exchange_request_v1 {
    const contract_edge_v1 *edges = nullptr;
    const std::uint32_t *destination_segment_offsets = nullptr;
    const float *source_key = nullptr;
    const float *destination_query = nullptr;
    const float *source_value = nullptr;
    const float *per_edge_gate = nullptr;
    float *contraction_workspace = nullptr;
    float *mapped_workspace = nullptr;
    float *normalized_workspace = nullptr;
    float *destination_output = nullptr;
    std::uint32_t edge_count = 0u;
    std::uint32_t source_count = 0u;
    std::uint32_t destination_count = 0u;
    std::uint32_t contraction_width = 0u;
    std::uint32_t value_width = 0u;
    float map_scale = 1.0f;
    float map_bias = 0.0f;
    std::uint64_t structure_id = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t value_generation = 0u;
    cudaStream_t stream = nullptr;
};

status_v1 validate_sparse_exchange_request_v1(
    const sparse_exchange_request_v1 &request) noexcept;
status_v1 enqueue_exchange_contraction_unfused_v1(
    const sparse_exchange_request_v1 &request) noexcept;
status_v1 enqueue_exchange_map_gate_unfused_v1(
    const sparse_exchange_request_v1 &request) noexcept;
status_v1 enqueue_exchange_normalization_unfused_v1(
    const sparse_exchange_request_v1 &request) noexcept;
status_v1 enqueue_exchange_apply_unfused_v1(
    const sparse_exchange_request_v1 &request) noexcept;
status_v1 enqueue_normalize_apply_fused_v1(
    const sparse_exchange_request_v1 &request) noexcept;
status_v1 enqueue_sparse_exchange_fused_v1(
    const sparse_exchange_request_v1 &request) noexcept;

} // namespace cellerator::compute::operation::fusion
