#pragma once

#include <Cellerator/execution/training_program_v2/program.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::training_v2 {

using execution::structure_epoch;
using execution::structure_handle;
using execution::training_v2::training_result_v2;
using execution::training_v2::training_value_mode_v2;
using execution::value_generation;

struct training_value_binding_v2 {
    structure_handle structure{};
    structure_epoch epoch{};
    value_generation generation{};
    training_value_mode_v2 mode = training_value_mode_v2::logical_primary;
    std::uint64_t logical_edge_count = 0u;
    std::uint64_t physical_slot_count = 0u;
    const std::uint64_t *logical_to_physical = nullptr;
    const float *logical_values = nullptr;
    float *physical_values = nullptr;
};

struct value_mode_workspace_v2 {
    std::uint8_t *physical_seen = nullptr;
    std::uint64_t physical_seen_capacity = 0u;
};

struct value_mode_receipt_v2 {
    std::uint64_t occupied_slots_visited = 0u;
    std::uint64_t values_written = 0u;
    bool permanent_holes_untouched = true;
    bool direct_projection_binding = false;
    std::uint8_t reserved[6]{};
};

training_result_v2 validate_training_value_binding_v2(
    const training_value_binding_v2 &binding,
    value_mode_workspace_v2 workspace) noexcept;

// Logical-primary mode packs only occupied slots. Projection-primary mode
// validates a direct binding and performs no copy.
training_result_v2 prepare_training_values_v2(
    const training_value_binding_v2 &binding,
    value_mode_receipt_v2 &receipt) noexcept;

training_result_v2 export_logical_values_v2(
    const training_value_binding_v2 &binding, float *logical_output,
    std::uint64_t logical_capacity,
    value_mode_receipt_v2 &receipt) noexcept;

training_result_v2 export_logical_gradients_v2(
    const training_value_binding_v2 &binding, const float *physical_gradients,
    std::uint64_t physical_gradient_count, float *logical_gradients,
    std::uint64_t logical_gradient_capacity,
    value_mode_receipt_v2 &receipt) noexcept;

static_assert(std::is_trivially_copyable<training_value_binding_v2>::value,
    "training value binding must remain trivially copyable");

} // namespace cellerator::compute::training_v2
