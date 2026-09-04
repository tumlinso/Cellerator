#include <Cellerator/compiler/planning/adapt_multi_extent_direct_binding_and_assembly_fallback_v1.hh>

#include <limits>

namespace Cellerator::compiler::planning {
namespace {

constexpr std::uint64_t ceil_div(
    std::uint64_t numerator,
    std::uint64_t denominator) noexcept {
    return numerator / denominator + (numerator % denominator != 0u ? 1u : 0u);
}

}  // namespace

multi_extent_binding_plan_result_v1 plan_multi_extent_binding_v1(
    const cellerator::execution::joint_compiler::external_binding_v1& binding,
    const multi_extent_candidate_capability_v1& candidate,
    std::uint64_t candidate_execution_nanoseconds,
    std::uint64_t assembly_fixed_nanoseconds,
    std::uint64_t assembly_bytes_per_nanosecond) noexcept {
    multi_extent_binding_plan_result_v1 result{};
    if (candidate.candidate_identity == 0u) {
        result.code = multi_extent_binding_plan_code_v1::invalid_candidate;
        return result;
    }
    if (!cellerator::execution::joint_compiler::validate_external_binding_v1(binding)) {
        result.code = multi_extent_binding_plan_code_v1::invalid_binding;
        return result;
    }

    auto& plan = result.plan;
    plan.candidate_identity = candidate.candidate_identity;
    plan.extent_count = binding.extent_count;
    plan.total_bytes = binding.total_bytes;
    plan.candidate_execution_nanoseconds = candidate_execution_nanoseconds;

    const bool direct = candidate.supports_multi_extent_binding &&
        candidate.maximum_extent_count >= binding.extent_count;
    if (direct) {
        plan.route = multi_extent_binding_route_v1::direct;
        plan.total_predicted_nanoseconds = candidate_execution_nanoseconds;
        result.code = multi_extent_binding_plan_code_v1::ok;
        return result;
    }
    if (assembly_bytes_per_nanosecond == 0u) {
        result.code = multi_extent_binding_plan_code_v1::invalid_cost_model;
        return result;
    }

    plan.route = multi_extent_binding_route_v1::assembled;
    plan.assembly.bytes_copied = binding.total_bytes;
    plan.assembly.copy_operations = binding.extent_count;
    const auto variable_cost = ceil_div(binding.total_bytes, assembly_bytes_per_nanosecond);
    if (assembly_fixed_nanoseconds > std::numeric_limits<std::uint64_t>::max() - variable_cost) {
        result.code = multi_extent_binding_plan_code_v1::cost_overflow;
        return result;
    }
    plan.assembly.predicted_nanoseconds = assembly_fixed_nanoseconds + variable_cost;
    if (candidate_execution_nanoseconds >
        std::numeric_limits<std::uint64_t>::max() - plan.assembly.predicted_nanoseconds) {
        result.code = multi_extent_binding_plan_code_v1::cost_overflow;
        return result;
    }
    plan.total_predicted_nanoseconds =
        candidate_execution_nanoseconds + plan.assembly.predicted_nanoseconds;
    plan.assembly_profiler_stage_visible = true;
    result.code = multi_extent_binding_plan_code_v1::ok;
    return result;
}

}  // namespace Cellerator::compiler::planning
