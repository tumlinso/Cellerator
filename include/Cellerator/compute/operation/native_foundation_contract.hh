#pragma once

#include <Cellerator/compute/operation/operation_core_v2/schema.hh>
#include <Cellerator/execution/launch_bindings.hh>
#include <Cellerator/execution/program/program_v2.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace cellerator::compute::operation::nf1 {

// Developmental semantic extension of operation core v2, not an allocator or
// backend registry. All views are borrowed through preparation/validation.
inline constexpr std::uint32_t contract_version = 1;
using identity = v2::stable_id;

enum class status : std::uint8_t {
    success, invalid_contract, invalid_identity, axis_mismatch,
    duplicate_output, unsupported_capability, invalid_effect,
    stale_generation, invalid_binding, unsupported_derivative
};

enum capability : std::uint32_t {
    forward = 1u, jvp = 2u, vjp = 4u, second_direction = 8u
};
inline constexpr std::uint32_t known_capabilities = forward | jvp | vjp | second_direction;

struct operand_signature {
    identity role{};
    std::span<const execution::persistent_axis_identity> axes{};
    std::uint64_t element_count = 0;
    execution::numeric_type storage = execution::numeric_type::invalid;
};

struct output_signature {
    operand_signature operand{};
    identity assembly_owner{};
    execution::output_effect_contract effect{
        execution::output_update_kind::overwrite, false, false, 0,
        execution::invalid_scalar_binding_id, execution::invalid_scalar_binding_id};
};

struct operation_contract {
    std::uint32_t version = contract_version;
    identity definition{};
    std::span<const operand_signature> arguments{};
    std::span<const output_signature> outputs{};
    v2::numerical_policy numeric{};
    v2::determinism_contract determinism{};
    std::uint32_t capabilities = forward;
    // All arguments read one launch snapshot; alias permission cannot change it.
    bool reads_single_snapshot = true;
};

inline bool same_axis(const execution::persistent_axis_identity& a,
                      const execution::persistent_axis_identity& b) noexcept {
    return execution::same_identity(a.domain, b.domain)
        && execution::same_identity(a.order, b.order)
        && execution::same_identity(a.geometry, b.geometry)
        && execution::same_identity(a.partition, b.partition);
}

inline bool valid_operand(const operand_signature& operand) noexcept {
    if (!v2::valid_stable_id(operand.role)
        || operand.storage == execution::numeric_type::invalid
        || operand.axes.size() > execution::biological_operand_max_axes) return false;
    for (const auto& axis : operand.axes)
        if (execution::validate_persistent_axis_identity(axis)
            != execution::biological_validation_code::ok) return false;
    return true;
}

inline status validate_operation(const operation_contract& contract) noexcept {
    if (contract.version != contract_version || !contract.reads_single_snapshot
        || contract.outputs.empty()) return status::invalid_contract;
    if (!v2::valid_stable_id(contract.definition)) return status::invalid_identity;
    if (!(contract.capabilities & forward)
        || (contract.capabilities & ~known_capabilities)) return status::unsupported_capability;
    if (!v2::validate_numerical_policy(contract.numeric)
        || !v2::validate_determinism_contract(contract.determinism, contract.numeric))
        return status::invalid_contract;
    for (const auto& input : contract.arguments)
        if (!valid_operand(input)) return status::invalid_identity;
    for (std::size_t i = 0; i < contract.outputs.size(); ++i) {
        const auto& output = contract.outputs[i];
        if (!valid_operand(output.operand) || !v2::valid_stable_id(output.assembly_owner))
            return status::invalid_identity;
        if (!execution::valid_output_effect_contract(output.effect)) return status::invalid_effect;
        for (std::size_t j = 0; j < i; ++j)
            if (v2::same_stable_id(output.operand.role, contract.outputs[j].operand.role))
                return status::duplicate_output;
    }
    return status::success;
}

inline status match_operand(const operand_signature& expected,
                            const operand_signature& actual) noexcept {
    if (!valid_operand(actual) || !v2::same_stable_id(expected.role, actual.role)
        || expected.storage != actual.storage || expected.element_count != actual.element_count
        || expected.axes.size() != actual.axes.size()) return status::invalid_binding;
    for (std::size_t i = 0; i < expected.axes.size(); ++i)
        if (!same_axis(expected.axes[i], actual.axes[i])) return status::axis_mismatch;
    return status::success;
}

// Definition, structure, values and result caching have independent identity.
struct prepared_identity {
    identity definition{};
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    execution::projection_id projection{};
};
struct generation_stamp {
    identity instance{};
    execution::value_generation generation{};
};
struct instance_binding {
    prepared_identity prepared{};
    generation_stamp state{};
    generation_stamp parameters{};
    // Nonzero only after an explicit caller parameter-sharing decision.
    identity parameter_tie_group{};
};

enum class reuse_kind : std::uint8_t {
    definition, structure, tied_parameters, memoized_result
};

inline bool valid_stamp(generation_stamp stamp, bool required = true) noexcept {
    if (!required && !v2::valid_stable_id(stamp.instance))
        return stamp.generation.value == 0;
    return v2::valid_stable_id(stamp.instance) && stamp.generation.value != 0;
}
inline bool same_stamp(generation_stamp a, generation_stamp b) noexcept {
    return v2::same_stable_id(a.instance, b.instance)
        && a.generation.value == b.generation.value;
}
inline bool same_preparation(const prepared_identity& a,
                             const prepared_identity& b) noexcept {
    return v2::same_stable_id(a.definition, b.definition)
        && execution::same_identity(a.structure, b.structure)
        && a.epoch.value == b.epoch.value
        && execution::same_identity(a.projection, b.projection);
}
inline status validate_instance(const prepared_identity& prepared,
                                const instance_binding& binding) noexcept {
    if (!v2::valid_stable_id(prepared.definition)
        || !execution::valid_identity(prepared.structure) || prepared.epoch.value == 0
        || !execution::valid_identity(prepared.projection)) return status::invalid_identity;
    if (!same_preparation(prepared, binding.prepared)) return status::invalid_binding;
    if (!valid_stamp(binding.state) || !valid_stamp(binding.parameters, false))
        return status::stale_generation;
    if (v2::valid_stable_id(binding.parameter_tie_group)
        && !valid_stamp(binding.parameters)) return status::invalid_binding;
    return status::success;
}
inline bool explicitly_tied(const instance_binding& a,
                            const instance_binding& b) noexcept {
    return v2::valid_stable_id(a.parameter_tie_group)
        && v2::same_stable_id(a.parameter_tie_group, b.parameter_tie_group);
}

enum class differentiated_object : std::uint8_t {
    vector_field, discrete_step, observation, implemented_rollout
};
enum class derivative_convention : std::uint8_t {
    mathematical_at_stored_values, through_rounding
};
struct primal_record {
    instance_binding instance{};
    generation_stamp forcing{}, context{}, activity{}, branches{};
};
struct derivative_request {
    capability action = jvp;
    differentiated_object object = differentiated_object::vector_field;
    derivative_convention convention = derivative_convention::mathematical_at_stored_values;
    primal_record primal{};
    operand_signature direction_domain{};
    operand_signature response_domain{};
    double direction_scale = 1.0;
    double response_scale = 1.0;
    bool smooth_at_primal = true;
};
inline bool same_primal(const primal_record& a, const primal_record& b) noexcept {
    return same_preparation(a.instance.prepared, b.instance.prepared)
        && same_stamp(a.instance.state, b.instance.state)
        && same_stamp(a.instance.parameters, b.instance.parameters)
        && same_stamp(a.forcing, b.forcing) && same_stamp(a.context, b.context)
        && same_stamp(a.activity, b.activity) && same_stamp(a.branches, b.branches);
}
inline bool valid_primal(const primal_record& primal) noexcept {
    return validate_instance(primal.instance.prepared, primal.instance) == status::success
        && valid_stamp(primal.forcing, false) && valid_stamp(primal.context, false)
        && valid_stamp(primal.activity, false) && valid_stamp(primal.branches, false);
}
inline status validate_derivative(const operation_contract& operation,
                                  const derivative_request& request,
                                  const primal_record& live_primal,
                                  const operand_signature& direction,
                                  const operand_signature& response) noexcept {
    auto state = validate_operation(operation);
    if (state != status::success) return state;
    if ((request.action != jvp && request.action != vjp && request.action != second_direction)
        || !(operation.capabilities & request.action)
        || !request.smooth_at_primal
        || request.convention != derivative_convention::mathematical_at_stored_values)
        return status::unsupported_derivative;
    if (request.object < differentiated_object::vector_field
        || request.object > differentiated_object::implemented_rollout)
        return status::invalid_contract;
    if (!v2::same_stable_id(operation.definition, request.primal.instance.prepared.definition)
        || !valid_primal(request.primal) || !valid_primal(live_primal)
        || !same_primal(request.primal, live_primal)) return status::stale_generation;
    if (!std::isfinite(request.direction_scale) || request.direction_scale <= 0
        || !std::isfinite(request.response_scale) || request.response_scale <= 0)
        return status::invalid_contract;
    if (match_operand(request.direction_domain, direction) != status::success
        || match_operand(request.response_domain, response) != status::success)
        return status::axis_mismatch;
    return status::success;
}

enum class contribution_kind : std::uint8_t {
    structural_absence, predicate_excluded, numerical_zero, active
};
enum class support_realization : std::uint8_t {
    persistent, compact_exact_active, approximate_drop
};
enum class approximation_kind : std::uint8_t {
    exact_declared_arithmetic, justified_bound, empirical, unassessed
};
struct support_contract {
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    generation_stamp activity{};
    support_realization realization = support_realization::persistent;
    approximation_kind accuracy = approximation_kind::exact_declared_arithmetic;
    bool response_requested = false;
    bool differentiate_predicate = false;
    bool preserve_nonfinite_exclusion = true;
    // Distinct budgets, not a single measurement-noise allowance.
    double storage_error_bound = 0, arithmetic_error_bound = 0, dropping_error_bound = 0;
};
enum invalidation : std::uint32_t {
    invalidate_nothing = 0, rebind_values = 1u, refresh_activity = 2u,
    rebuild_projection = 4u, rebuild_structure = 8u, invalidate_primal = 16u
};
inline status validate_support(const support_contract& support) noexcept {
    if (!execution::valid_identity(support.structure) || support.epoch.value == 0
        || !valid_stamp(support.activity, false)) return status::invalid_identity;
    if (support.realization < support_realization::persistent
        || support.realization > support_realization::approximate_drop
        || support.accuracy < approximation_kind::exact_declared_arithmetic
        || support.accuracy > approximation_kind::unassessed) return status::invalid_contract;
    if (support.differentiate_predicate) return status::unsupported_derivative;
    if (!support.preserve_nonfinite_exclusion) return status::invalid_contract;
    if (support.realization == support_realization::approximate_drop
        && support.accuracy == approximation_kind::exact_declared_arithmetic)
        return status::invalid_contract;
    for (auto bound : {support.storage_error_bound, support.arithmetic_error_bound,
                       support.dropping_error_bound})
        if (!std::isfinite(bound) || bound < 0) return status::invalid_contract;
    return status::success;
}
inline bool may_drop_response(contribution_kind contribution,
                              bool exact_response_zero_proved = false) noexcept {
    return contribution == contribution_kind::structural_absence
        || contribution == contribution_kind::predicate_excluded
        || exact_response_zero_proved;
}
inline std::uint32_t invalidation_for(const support_contract& before,
                                     const support_contract& after,
                                     bool values_changed) noexcept {
    if (!execution::same_identity(before.structure, after.structure)
        || before.epoch.value != after.epoch.value)
        return rebuild_structure | rebuild_projection | invalidate_primal;
    std::uint32_t result = values_changed ? rebind_values | invalidate_primal : invalidate_nothing;
    if (!same_stamp(before.activity, after.activity)) {
        result |= refresh_activity | invalidate_primal;
        if (after.realization != support_realization::persistent) result |= rebuild_projection;
    }
    if (before.realization != after.realization || before.accuracy != after.accuracy
        || before.storage_error_bound != after.storage_error_bound
        || before.arithmetic_error_bound != after.arithmetic_error_bound
        || before.dropping_error_bound != after.dropping_error_bound
        || before.response_requested != after.response_requested)
        result |= rebuild_projection | invalidate_primal;
    // Approximate value-dependent drops must be reconsidered after value edits.
    if (values_changed && after.realization == support_realization::approximate_drop)
        result |= rebuild_projection;
    return result;
}

struct dependency_effects {
    bool dependencies_known = false;
    bool writes_only_declared_outputs = false;
    bool deterministic = false;
    bool allocation_free_launch = false;
};
struct compiled_block {
    operation_contract contract{};
    dependency_effects effects{};
    // One invocation for a prepared group; never a per-edge virtual object.
    execution::program::stage_launch_v2 forward_launch = nullptr;
    execution::program::stage_launch_v2 jvp_launch = nullptr;
    execution::program::stage_launch_v2 vjp_launch = nullptr;
    execution::program::stage_launch_v2 second_launch = nullptr;
};
inline bool permits_fusion_or_memoization(const compiled_block& block) noexcept {
    return block.effects.dependencies_known && block.effects.writes_only_declared_outputs
        && block.effects.deterministic && block.effects.allocation_free_launch;
}
inline bool permits_result_reuse(const compiled_block& block,
                                 const primal_record& cached,
                                 const primal_record& requested) noexcept {
    return permits_fusion_or_memoization(block) && valid_primal(cached)
        && valid_primal(requested) && same_primal(cached, requested)
        && v2::same_stable_id(block.contract.definition, cached.instance.prepared.definition);
}
inline status validate_compiled_block(const compiled_block& block) noexcept {
    auto state = validate_operation(block.contract);
    if (state != status::success) return state;
    if (!block.forward_launch) return status::unsupported_capability;
    if (((block.contract.capabilities & jvp) != 0) != (block.jvp_launch != nullptr)
        || ((block.contract.capabilities & vjp) != 0) != (block.vjp_launch != nullptr)
        || ((block.contract.capabilities & second_direction) != 0) != (block.second_launch != nullptr))
        return status::unsupported_derivative;
    return status::success;
}
// The provider defines the typed payload behind existing launch_binding_v2.
// Prepared state and all payload buffers remain owned by the existing session
// or caller. This adapter creates no allocator, stream, graph or tensor owner.
inline status bind_compiled_stage(const compiled_block& block, capability action,
                                 const void* prepared_state,
                                 std::uint64_t stage_id, std::uint64_t candidate_id,
                                 std::uint32_t binding_index,
                                 execution::program::prepared_stage_v2& stage) noexcept {
    auto state = validate_compiled_block(block);
    if (state != status::success) return state;
    execution::program::stage_launch_v2 launch = nullptr;
    switch (action) {
    case forward: launch = block.forward_launch; break;
    case jvp: launch = block.jvp_launch; break;
    case vjp: launch = block.vjp_launch; break;
    case second_direction: launch = block.second_launch; break;
    default: return status::unsupported_capability;
    }
    if (!launch) return status::unsupported_derivative;
    if (!stage_id || !candidate_id) return status::invalid_identity;
    stage = {stage_id, candidate_id, prepared_state, launch, 0, 0, binding_index, 0};
    return status::success;
}

} // namespace cellerator::compute::operation::nf1
