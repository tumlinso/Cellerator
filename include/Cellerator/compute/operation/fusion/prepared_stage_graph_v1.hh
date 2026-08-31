#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::operation::fusion {

enum class status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    invalid_identity = 2u,
    invalid_dependency = 3u,
    incompatible_order = 4u,
    incompatible_lifetime = 5u,
    unsupported = 6u,
    cuda_failure = 7u
};

enum class stage_kind_v1 : std::uint8_t {
    value_generation = 0u,
    value_pack,
    relation_apply,
    epilogue,
    mma_contribution,
    residual_contribution,
    contraction,
    edge_map_or_gate,
    segment_statistic,
    normalization,
    relation_bundle,
    shared_destination_accumulation,
    relation_moments
};

enum class composition_kind_v1 : std::uint8_t {
    value_generation_to_pack = 0u,
    value_pack_to_relation_apply,
    mma_to_same_owner_residual,
    relation_apply_to_epilogue,
    contraction_to_edge_map,
    contraction_to_segment_statistic,
    normalization_to_relation_apply,
    sparse_exchange,
    bundle_to_shared_destination,
    relation_moments_pair
};

enum class order_kind_v1 : std::uint8_t {
    logical = 0u,
    projection_native = 1u,
    persistent_physical = 2u
};

enum resource_flag_v1 : std::uint32_t {
    resource_none_v1 = 0u,
    resource_cuda_execution_v1 = 1u << 0u,
    resource_transient_workspace_v1 = 1u << 1u,
    resource_persistent_projection_v1 = 1u << 2u
};

struct stage_descriptor_v1 {
    std::uint64_t stable_stage_id = 0u;
    stage_kind_v1 kind = stage_kind_v1::value_generation;
    order_kind_v1 input_order = order_kind_v1::logical;
    order_kind_v1 output_order = order_kind_v1::logical;
    std::uint8_t reserved = 0u;
    std::uint32_t required_resources = resource_none_v1;
    std::uint64_t structure_id = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t input_value_generation = 0u;
    std::uint64_t output_value_generation = 0u;
    std::uint64_t global_item_begin = 0u;
    std::uint32_t local_item_count = 0u;
    std::uint32_t profiler_stage_index = 0u;
};

struct dependency_v1 {
    std::uint32_t producer_stage = 0u;
    std::uint32_t consumer_stage = 0u;
};

struct prepared_stage_graph_v1 {
    std::uint64_t stable_graph_id = 0u;
    composition_kind_v1 composition =
        composition_kind_v1::value_generation_to_pack;
    bool fused = false;
    bool experimental = true;
    bool requires_measurement = true;
    bool explicitly_selectable = true;
    bool auto_promoted = false;
    bool unfused_stages_available = true;
    std::uint8_t reserved[2]{};
    const stage_descriptor_v1 *stages = nullptr;
    std::uint32_t stage_count = 0u;
    const dependency_v1 *dependencies = nullptr;
    std::uint32_t dependency_count = 0u;
};

struct resource_availability_v1 {
    std::uint32_t available_flags = resource_none_v1;
    std::uint64_t transient_workspace_bytes = 0u;
};

status_v1 validate_prepared_stage_graph_v1(
    const prepared_stage_graph_v1 &graph) noexcept;
status_v1 validate_graph_resources_v1(const prepared_stage_graph_v1 &graph,
    resource_availability_v1 resources) noexcept;

static_assert(std::is_trivially_copyable<stage_descriptor_v1>::value,
    "stage descriptors are data-only contracts");
static_assert(std::is_trivially_copyable<prepared_stage_graph_v1>::value,
    "prepared graphs are non-owning data-only views");

} // namespace cellerator::compute::operation::fusion
