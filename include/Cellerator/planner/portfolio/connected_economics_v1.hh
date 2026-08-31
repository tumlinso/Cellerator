#pragma once

#include <Cellerator/planner/end_to_end_planner.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::planner::portfolio {

enum class planner_value_mode_v1 : std::uint8_t {
    logical_primary = 1u,
    projection_primary = 2u,
};

enum class economics_status_code_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_order,
    invalid_cost,
    arithmetic_overflow,
};

struct economics_status_v1 {
    economics_status_code_v1 code = economics_status_code_v1::success;
    std::uint64_t subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == economics_status_code_v1::success;
    }
};

struct operation_economics_v1 {
    operation_core::stable_id candidate{};
    execution::order_id input_order{};
    execution::order_id output_order{};
    phase_costs phases{};
    planner_value_mode_v1 value_mode = planner_value_mode_v1::logical_primary;
    std::uint8_t reserved0[7]{};
    std::uint64_t frequency = 1u;
    std::uint64_t repetitions = 1u;
    std::uint64_t structure_reuse = 1u;
    std::uint64_t projection_reuse = 1u;
    std::uint64_t value_reuse = 1u;
    bool canonical_output_required = false;
    bool graph_capture_required = false;
    std::uint8_t reserved1[6]{};
};

struct layout_transition_economics_v1 {
    execution::order_id source_order{};
    execution::order_id destination_order{};
    double transform_ns = 0.0;
    double fusion_savings_ns = 0.0;
    std::uint64_t transient_bytes = 0u;
    bool fused = false;
    std::uint8_t reserved[7]{};
};

struct connected_program_economics_v1 {
    const operation_economics_v1 *operations = nullptr;
    std::uint64_t operation_count = 0u;
    const layout_transition_economics_v1 *transitions = nullptr;
    std::uint64_t transition_count = 0u;
    execution::order_id canonical_output_order{};
    double final_canonicalization_ns = 0.0;
    std::uint64_t final_canonicalization_bytes = 0u;
};

struct connected_economics_result_v1 {
    double complete_cost_ns = 0.0;
    double operation_cost_ns = 0.0;
    double layout_cost_ns = 0.0;
    double value_pack_cost_ns = 0.0;
    double fusion_savings_ns = 0.0;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t peak_transient_bytes = 0u;
    std::uint64_t launch_count = 0u;
};

economics_status_v1 compute_connected_economics_v1(
    const connected_program_economics_v1 &program,
    connected_economics_result_v1 *result) noexcept;

static_assert(std::is_trivially_copyable<operation_economics_v1>::value,
    "operation economics must remain caller-owned records");
static_assert(std::is_trivially_copyable<connected_program_economics_v1>::value,
    "connected economics must remain a pointer-plus-count view");

}  // namespace cellerator::planner::portfolio
