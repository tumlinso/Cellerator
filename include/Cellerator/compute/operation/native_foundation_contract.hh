#pragma once

#include <Cellerator/compute/operation/operation_core_v2/schema.hh>
#include <Cellerator/execution/launch_bindings.hh>
#include <Cellerator/execution/program/program_v2.h>

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

} // namespace cellerator::compute::operation::nf1
