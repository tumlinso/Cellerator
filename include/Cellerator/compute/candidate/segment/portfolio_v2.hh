#pragma once

#include <Cellerator/compute/candidate/segment/mechanism_v2.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::segment {

inline constexpr std::uint32_t segment_candidate_schema_version_v2 = 2u;

struct segment_candidate_descriptor_v2 {
    std::uint32_t schema_version = segment_candidate_schema_version_v2;
    segment_operation_v2 operation = segment_operation_v2::reduce;
    segment_direction_v2 direction = segment_direction_v2::forward;
    segment_reduce_kind_v2 reduction = segment_reduce_kind_v2::sum;
    segment_normalize_kind_v2 normalization =
        segment_normalize_kind_v2::softmax;
    segment_mechanism_v2 mechanism = segment_mechanism_v2::cta_per_output;
    segment_storage_order_v2 storage_order =
        segment_storage_order_v2::logical_edge;
    std::uint8_t reserved0[2]{};
    std::uint64_t candidate_identity = 0u;
    std::uint64_t stage_identity = 0u;
    std::uint32_t stage_count = 1u;
    std::uint32_t launch_count_per_component = 1u;
    std::uint32_t threads_per_cta = 0u;
    std::uint32_t warps_per_cta = 0u;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
    bool graph_capture_compatible = true;
    bool requires_measurement = true;
    bool production_promoted = false;
    std::uint8_t reserved1[5]{};
};

struct segment_candidate_buffer_v2 {
    segment_candidate_descriptor_v2 *data = nullptr;
    std::uint32_t capacity = 0u;
    std::uint32_t count = 0u;
};

struct segment_prepared_manifest_v2 {
    std::uint32_t schema_version = segment_candidate_schema_version_v2;
    std::uint32_t reserved0 = 0u;
    std::uint64_t candidate_identity = 0u;
    std::uint64_t operation_identity = 0u;
    std::uint64_t stage_identity = 0u;
    std::uint64_t partition_identity = 0u;
    segment_operation_v2 operation = segment_operation_v2::reduce;
    segment_direction_v2 direction = segment_direction_v2::forward;
    segment_mechanism_v2 mechanism = segment_mechanism_v2::cta_per_output;
    segment_storage_order_v2 storage_order =
        segment_storage_order_v2::logical_edge;
    std::uint8_t reserved1[4]{};
    std::uint64_t logical_values = 0u;
    std::uint64_t physical_slots = 0u;
    std::uint64_t physical_holes = 0u;
    std::uint64_t useful_interactions = 0u;
    std::uint64_t input_bytes = 0u;
    std::uint64_t output_bytes = 0u;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
    std::uint32_t threads_per_cta = 0u;
    std::uint32_t warps_per_cta = 0u;
    std::uint32_t launch_count = 1u;
    bool graph_capture_compatible = true;
    bool requires_measurement = true;
    bool production_promoted = false;
    std::uint8_t reserved2[1]{};
};

std::uint32_t segment_candidate_count_v2() noexcept;

segment_result_v2 enumerate_segment_candidates_v2(
    segment_candidate_buffer_v2 &buffer) noexcept;

segment_result_v2 validate_segment_candidate_catalog_v2(
    const segment_candidate_descriptor_v2 *candidates,
    std::uint32_t count) noexcept;

segment_result_v2 build_segment_prepared_manifest_v2(
    const segment_plan_v2 &plan,
    std::uint64_t physical_slot_count,
    segment_prepared_manifest_v2 &manifest) noexcept;

std::uint64_t segment_candidate_identity_v2(
    segment_operation_v2 operation,
    segment_direction_v2 direction,
    segment_reduce_kind_v2 reduction,
    segment_normalize_kind_v2 normalization,
    segment_mechanism_v2 mechanism,
    segment_storage_order_v2 storage_order) noexcept;

static_assert(std::is_trivially_copyable<segment_candidate_descriptor_v2>::value,
    "segment candidate descriptor must remain pointer-free");
static_assert(std::is_trivially_copyable<segment_prepared_manifest_v2>::value,
    "segment prepared manifest must remain pointer-free");

} // namespace cellerator::compute::segment
