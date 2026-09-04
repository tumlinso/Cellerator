#pragma once

#include <Cellerator/execution/joint_compiler/external_binding_v1.hh>

#include <cstdint>

namespace Cellerator::compiler::planning {

enum class multi_extent_binding_route_v1 : std::uint8_t {
    direct = 1u,
    assembled,
};

struct multi_extent_candidate_capability_v1 {
    std::uint64_t candidate_identity = 0u;
    bool supports_multi_extent_binding = false;
    std::uint64_t maximum_extent_count = 0u;
};

struct multi_extent_assembly_cost_v1 {
    std::uint64_t bytes_copied = 0u;
    std::uint64_t copy_operations = 0u;
    std::uint64_t predicted_nanoseconds = 0u;
};

struct multi_extent_binding_plan_v1 {
    multi_extent_binding_route_v1 route = multi_extent_binding_route_v1::assembled;
    std::uint64_t candidate_identity = 0u;
    std::uint64_t extent_count = 0u;
    std::uint64_t total_bytes = 0u;
    std::uint64_t candidate_execution_nanoseconds = 0u;
    std::uint64_t total_predicted_nanoseconds = 0u;
    multi_extent_assembly_cost_v1 assembly{};
    bool assembly_profiler_stage_visible = false;
};

enum class multi_extent_binding_plan_code_v1 : std::uint8_t {
    ok = 0u,
    invalid_candidate,
    invalid_binding,
    invalid_cost_model,
    cost_overflow,
};

struct multi_extent_binding_plan_result_v1 {
    multi_extent_binding_plan_code_v1 code =
        multi_extent_binding_plan_code_v1::invalid_binding;
    multi_extent_binding_plan_v1 plan{};

    constexpr explicit operator bool() const noexcept {
        return code == multi_extent_binding_plan_code_v1::ok;
    }
};

[[nodiscard]] multi_extent_binding_plan_result_v1
plan_multi_extent_binding_v1(
    const cellerator::execution::joint_compiler::external_binding_v1& binding,
    const multi_extent_candidate_capability_v1& candidate,
    std::uint64_t candidate_execution_nanoseconds,
    std::uint64_t assembly_fixed_nanoseconds,
    std::uint64_t assembly_bytes_per_nanosecond) noexcept;

}  // namespace Cellerator::compiler::planning
