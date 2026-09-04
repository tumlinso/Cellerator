#pragma once

#include <Cellerator/planner/portfolio/connected_economics_v1.hh>

#include <cstdint>

namespace Cellerator::compiler::planning {

enum connected_operation_effect_v1 : std::uint32_t {
    connected_order_transform_v1 = 1u << 0u,
    connected_materialization_v1 = 1u << 1u,
    connected_shared_traversal_v1 = 1u << 2u,
    connected_fusion_v1 = 1u << 3u,
    connected_common_output_ownership_v1 = 1u << 4u,
    connected_canonicalization_v1 = 1u << 5u,
    connected_field_boundary_v1 = 1u << 6u,
};

struct connected_operation_transition_cost_v1 {
    cellerator::execution::order_id source_order{};
    cellerator::execution::order_id destination_order{};
    std::uint32_t effects = 0u;
    double order_transform_nanoseconds = 0.0;
    double materialization_nanoseconds = 0.0;
    double shared_traversal_savings_nanoseconds = 0.0;
    double fusion_savings_nanoseconds = 0.0;
    double common_output_ownership_savings_nanoseconds = 0.0;
    double canonicalization_nanoseconds = 0.0;
    double field_boundary_nanoseconds = 0.0;
    std::uint64_t transient_bytes = 0u;
};

enum class connected_operation_cost_code_v1 : std::uint8_t {
    ok = 0u,
    invalid_effect,
    invalid_order,
    invalid_cost,
    excessive_savings,
};

struct connected_operation_cost_result_v1 {
    connected_operation_cost_code_v1 code =
        connected_operation_cost_code_v1::invalid_effect;
    double gross_nanoseconds = 0.0;
    double savings_nanoseconds = 0.0;
    double complete_nanoseconds = 0.0;
    cellerator::planner::portfolio::layout_transition_economics_v1 transition{};

    constexpr explicit operator bool() const noexcept {
        return code == connected_operation_cost_code_v1::ok;
    }
};

[[nodiscard]] connected_operation_cost_result_v1
adapt_transition_and_connected_operation_costs_v1(
    const connected_operation_transition_cost_v1& cost) noexcept;

}  // namespace Cellerator::compiler::planning
