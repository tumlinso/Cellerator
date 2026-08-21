#include <Cellerator/compute/math/operation_core/operation_core.hh>

#include <cstdint>

namespace cellerator::compute::math::core {
namespace {

bool valid_stable_id(stable_id value) noexcept {
    return value.low != 0u || value.high != 0u;
}

bool valid_structure_key(const structure_key &key) noexcept {
    return execution::valid_identity(key.persistent)
        && execution::valid_handle(key.runtime)
        && key.epoch.value != 0u;
}

bool valid_projection_key(const projection_key &key) noexcept {
    return execution::valid_identity(key.persistent)
        && execution::valid_handle(key.runtime)
        && key.schema_version != 0u;
}

bool valid_numeric_type(execution::numeric_type type) noexcept {
    return type != execution::numeric_type::invalid;
}

} // namespace

operation_status validate_operation_problem(
    const operation_problem &problem,
    const structure_key &structure) noexcept {
    if (problem.schema_version != operation_core_schema_version)
        return {operation_status_code::unsupported_problem,
            execution::binding_validation_code::ok,
            "unsupported operation-core schema"};
    if (!valid_stable_id(problem.operation)
        || problem.input_count == 0u || problem.output_count == 0u
        || problem.logical_work_items == 0u)
        return {operation_status_code::invalid_argument,
            execution::binding_validation_code::ok,
            "operation problem is incomplete"};
    if (!valid_structure_key(structure))
        return {operation_status_code::stale_structure,
            execution::binding_validation_code::stale_structure,
            "structure key is invalid"};
    return {};
}

operation_status validate_numeric_policy(const numeric_policy &numeric) noexcept {
    if (!valid_numeric_type(numeric.sparse_storage)
        || !valid_numeric_type(numeric.dense_storage)
        || !valid_numeric_type(numeric.output_storage)
        || !valid_numeric_type(numeric.multiply)
        || !valid_numeric_type(numeric.accumulation)
        || !valid_numeric_type(numeric.scalar))
        return {operation_status_code::unsupported_numeric_policy,
            execution::binding_validation_code::ok,
            "numeric policy contains an invalid required type"};
    if (numeric.quantization == quantization_granularity::none
        && numeric.saturation == saturation_policy::saturate)
        return {operation_status_code::unsupported_numeric_policy,
            execution::binding_validation_code::ok,
            "saturation requires an explicit quantized policy"};
    return {};
}

operation_status validate_prepared_operation(
    const prepared_operation &prepared) noexcept {
    const operation_status problem =
        validate_operation_problem(prepared.problem, prepared.structure);
    if (!problem) return problem;
    const operation_status numeric = validate_numeric_policy(prepared.numeric);
    if (!numeric) return numeric;
    if (!valid_projection_key(prepared.projection))
        return {operation_status_code::unsupported_projection,
            execution::binding_validation_code::ok,
            "projection key is incomplete"};
    if (!valid_stable_id(prepared.kernel) || prepared.run == nullptr)
        return {operation_status_code::preparation_failed,
            execution::binding_validation_code::ok,
            "prepared operation has no direct dispatch"};
    if (!execution::same_handle(
            prepared.structure.runtime,
            prepared.binding_contract.structure)
        || prepared.structure.epoch.value
            != prepared.binding_contract.epoch.value)
        return {operation_status_code::stale_structure,
            execution::binding_validation_code::stale_structure,
            "prepared binding contract does not match structure key"};
    if (prepared.problem.input_count != prepared.binding_contract.input_count
        || prepared.problem.output_count != prepared.binding_contract.output_count)
        return {operation_status_code::preparation_failed,
            execution::binding_validation_code::operand_count_mismatch,
            "prepared binding arity does not match operation problem"};
    return {};
}

operation_status run_prepared_operation(
    const prepared_operation &prepared,
    const execution::launch_bindings &launch) noexcept {
    const operation_status valid = validate_prepared_operation(prepared);
    if (!valid) return valid;
    const execution::binding_validation_code binding =
        execution::validate_launch_bindings(prepared.binding_contract, launch);
    if (binding != execution::binding_validation_code::ok)
        return {binding == execution::binding_validation_code::stale_structure
                    ? operation_status_code::stale_structure
                    : operation_status_code::invalid_launch_bindings,
            binding,
            "launch bindings do not satisfy prepared contract"};
    return prepared.run(prepared, launch);
}

operation_status register_candidate(
    candidate_registry *registry,
    const operation_candidate &candidate) noexcept {
    if (registry == nullptr || !valid_stable_id(candidate.identity)
        || candidate.name == nullptr || candidate.supports_numeric == nullptr
        || candidate.prepare == nullptr)
        return {operation_status_code::invalid_argument,
            execution::binding_validation_code::ok,
            "candidate registration is incomplete"};
    if (find_candidate(*registry, candidate.identity) != nullptr)
        return {operation_status_code::duplicate_candidate,
            execution::binding_validation_code::ok,
            "candidate identity is already registered"};
    if (registry->size == operation_candidate_capacity)
        return {operation_status_code::registry_full,
            execution::binding_validation_code::ok,
            "candidate registry is full"};
    registry->candidates[registry->size++] = candidate;
    return {};
}

const operation_candidate *find_candidate(
    const candidate_registry &registry,
    stable_id identity) noexcept {
    for (std::uint32_t index = 0u; index < registry.size; ++index)
        if (same_stable_id(registry.candidates[index].identity, identity))
            return &registry.candidates[index];
    return nullptr;
}

operation_status prepare_candidate(
    const operation_candidate &candidate,
    const operation_problem &problem,
    const structure_key &structure,
    const projection_key &projection,
    const numeric_policy &numeric,
    const prepare_policy &policy,
    prepared_operation *prepared) noexcept {
    if (prepared == nullptr || candidate.prepare == nullptr
        || candidate.supports_numeric == nullptr)
        return {operation_status_code::invalid_argument,
            execution::binding_validation_code::ok,
            "prepare requires a complete candidate and output"};
    const operation_status problem_status =
        validate_operation_problem(problem, structure);
    if (!problem_status) return problem_status;
    const operation_status numeric_status = validate_numeric_policy(numeric);
    if (!numeric_status) return numeric_status;
    if (candidate.operation != problem.kind)
        return {operation_status_code::unsupported_problem,
            execution::binding_validation_code::ok,
            "candidate does not support operation kind"};
    if (!valid_projection_key(projection)
        || candidate.projection != projection.kind)
        return {operation_status_code::unsupported_projection,
            execution::binding_validation_code::ok,
            "candidate does not support projection"};
    if (!candidate.supports_numeric(numeric))
        return {operation_status_code::unsupported_numeric_policy,
            execution::binding_validation_code::ok,
            "candidate rejects numeric policy"};
    if (policy.deterministic
        && (candidate.capability_flags & candidate_deterministic) == 0u)
        return {operation_status_code::capability_rejected,
            execution::binding_validation_code::ok,
            "candidate is not deterministic"};
    if (policy.graph_capture_required
        && (candidate.capability_flags & candidate_graph_capture) == 0u)
        return {operation_status_code::capability_rejected,
            execution::binding_validation_code::ok,
            "candidate is not graph-capture compatible"};
    if (!policy.allow_persistent_preprocessing
        && (candidate.capability_flags
            & candidate_persistent_preprocessing) != 0u)
        return {operation_status_code::capability_rejected,
            execution::binding_validation_code::ok,
            "persistent preprocessing is disabled"};
    if (!policy.allow_composed_epilogue
        && (candidate.capability_flags & candidate_composed_epilogue) != 0u)
        return {operation_status_code::capability_rejected,
            execution::binding_validation_code::ok,
            "composed epilogue is disabled"};
    if ((policy.persistent_memory_limit != 0u
            && candidate.persistent_bytes > policy.persistent_memory_limit)
        || (policy.transient_memory_limit != 0u
            && candidate.transient_bytes > policy.transient_memory_limit))
        return {operation_status_code::capability_rejected,
            execution::binding_validation_code::ok,
            "candidate exceeds memory policy"};
    return candidate.prepare(
        candidate, problem, structure, projection, numeric, policy, prepared);
}

} // namespace cellerator::compute::math::core
