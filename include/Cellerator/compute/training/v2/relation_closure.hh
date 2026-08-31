#pragma once

#include <Cellerator/execution/training_program_v2/program.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::training_v2 {

using execution::axis_identity;
using execution::structure_epoch;
using execution::structure_handle;
using execution::training_v2::training_order_mode_v2;
using execution::training_v2::training_result_v2;
using execution::training_v2::training_status_v2;
using execution::value_generation;

struct projection_edge_v2 {
    std::uint64_t source_index = 0u;
    std::uint64_t destination_index = 0u;
    std::uint64_t logical_edge_index = 0u;
    std::uint64_t physical_slot = 0u;
};

// Both schedules contain exactly the occupied physical slots. Forward is
// destination-owned; transpose is source-owned. Holes occur only in the value
// plane and therefore have neither a schedule entry nor biological identity.
struct projection_relation_v2 {
    structure_handle structure{};
    structure_epoch epoch{};
    value_generation generation{};
    axis_identity source_axis{};
    axis_identity destination_axis{};
    std::uint64_t source_count = 0u;
    std::uint64_t destination_count = 0u;
    std::uint64_t logical_edge_count = 0u;
    std::uint64_t physical_slot_count = 0u;
    const float *physical_values = nullptr;
    const projection_edge_v2 *forward_edges = nullptr;
    const projection_edge_v2 *transpose_edges = nullptr;
};

struct relation_validation_workspace_v2 {
    std::uint64_t *logical_to_forward = nullptr;
    std::uint64_t logical_to_forward_capacity = 0u;
    std::uint8_t *physical_seen = nullptr;
    std::uint64_t physical_seen_capacity = 0u;
};

struct dense_vector_view_v2 {
    axis_identity axis{};
    training_order_mode_v2 order = training_order_mode_v2::canonical;
    std::uint64_t extent = 0u;
    const float *data = nullptr;
};

struct mutable_dense_vector_view_v2 {
    axis_identity axis{};
    training_order_mode_v2 order = training_order_mode_v2::canonical;
    std::uint64_t extent = 0u;
    float *data = nullptr;
};

struct projection_gradient_view_v2 {
    structure_handle structure{};
    structure_epoch epoch{};
    value_generation source_generation{};
    std::uint64_t physical_slot_count = 0u;
    float *physical_gradients = nullptr;
};

struct relation_closure_receipt_v2 {
    std::uint64_t logical_edges_visited = 0u;
    std::uint64_t physical_slots_written = 0u;
    bool permanent_holes_untouched = true;
    bool deterministic_fp32 = true;
    std::uint8_t reserved[6]{};
};

training_result_v2 validate_projection_relation_v2(
    const projection_relation_v2 &relation,
    relation_validation_workspace_v2 workspace) noexcept;

training_result_v2 relation_forward_v2(const projection_relation_v2 &relation,
    dense_vector_view_v2 source, mutable_dense_vector_view_v2 destination,
    relation_closure_receipt_v2 &receipt) noexcept;

training_result_v2 relation_transpose_v2(
    const projection_relation_v2 &relation, dense_vector_view_v2 destination,
    mutable_dense_vector_view_v2 source,
    relation_closure_receipt_v2 &receipt) noexcept;

training_result_v2 logical_edge_gradient_v2(
    const projection_relation_v2 &relation, dense_vector_view_v2 source,
    dense_vector_view_v2 destination_gradient,
    projection_gradient_view_v2 physical_gradient,
    relation_closure_receipt_v2 &receipt) noexcept;

static_assert(std::is_trivially_copyable<projection_relation_v2>::value,
    "projection relation view must remain trivially copyable");

} // namespace cellerator::compute::training_v2
