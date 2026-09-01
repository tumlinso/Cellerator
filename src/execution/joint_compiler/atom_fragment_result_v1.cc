#include <Cellerator/execution/joint_compiler/atom_fragment_result_v1.hh>

namespace cellerator::execution::joint_compiler {
namespace {

atom_fragment_result_validation_result_v1 failure(
    atom_fragment_result_validation_code_v1 code,
    std::uint64_t candidate = 0u,
    std::uint64_t element = 0u) noexcept {
    return {code, candidate, element};
}

bool valid_id(persistent_identity_v1 value) noexcept {
    return static_cast<bool>(validate_persistent_identity_v1(value));
}

bool zero_id(persistent_identity_v1 value) noexcept {
    return value.producer_namespace == 0u && value.local_identity == 0u;
}

bool less_id(persistent_identity_v1 lhs, persistent_identity_v1 rhs) noexcept {
    return lhs.producer_namespace < rhs.producer_namespace
        || (lhs.producer_namespace == rhs.producer_namespace
            && lhs.local_identity < rhs.local_identity);
}

atom_fragment_result_validation_result_v1 validate_ids(
    const persistent_identity_v1 *values,
    std::uint64_t count,
    std::uint64_t candidate,
    atom_fragment_result_validation_code_v1 missing,
    atom_fragment_result_validation_code_v1 invalid,
    atom_fragment_result_validation_code_v1 unordered) noexcept {
    if (count == 0u || values == nullptr) return failure(missing, candidate);
    for (std::uint64_t index = 0u; index < count; ++index) {
        if (!valid_id(values[index])) return failure(invalid, candidate, index);
        if (index != 0u && !less_id(values[index - 1u], values[index]))
            return failure(unordered, candidate, index);
    }
    return {};
}

}  // namespace

atom_fragment_result_validation_result_v1 validate_atom_fragment_result_v1(
    const atom_fragment_result_v1 &result) noexcept {
    if (result.schema_version != atom_fragment_result_schema_version_v1)
        return failure(
            atom_fragment_result_validation_code_v1::unsupported_schema);
    if (result.record_bytes != sizeof(atom_fragment_result_v1))
        return failure(
            atom_fragment_result_validation_code_v1::invalid_record_bytes);
    for (std::uint8_t value : result.reserved)
        if (value != 0u)
            return failure(
                atom_fragment_result_validation_code_v1::nonzero_reserved);
    if (!valid_id(result.result_identity))
        return failure(
            atom_fragment_result_validation_code_v1::invalid_result_identity);
    if (!valid_id(result.request_identity))
        return failure(
            atom_fragment_result_validation_code_v1::invalid_request_identity);
    if (result.candidate_capacity > maximum_atom_fragment_candidates_v1
        || result.candidate_count > result.candidate_capacity
        || (result.candidate_count != 0u && result.candidates == nullptr))
        return failure(
            atom_fragment_result_validation_code_v1::invalid_candidate_bounds);
    if (result.no_candidate_reason < no_candidate_reason_v1::none
        || result.no_candidate_reason > no_candidate_reason_v1::bounded_frontier_empty)
        return failure(atom_fragment_result_validation_code_v1::
            invalid_no_candidate_reason);
    if (result.candidate_count == 0u) {
        if (result.candidates != nullptr || result.candidate_capacity != 0u
            || result.no_candidate_reason == no_candidate_reason_v1::none
            || result.frontier_truncated)
            return failure(atom_fragment_result_validation_code_v1::
                inconsistent_empty_frontier);
        return {};
    }
    if (result.no_candidate_reason != no_candidate_reason_v1::none)
        return failure(atom_fragment_result_validation_code_v1::
            inconsistent_empty_frontier);
    if (result.frontier_truncated
        && result.candidate_count != result.candidate_capacity)
        return failure(atom_fragment_result_validation_code_v1::
            inconsistent_truncation);

    for (std::uint64_t index = 0u; index < result.candidate_count; ++index) {
        const atom_fragment_candidate_v1 &candidate = result.candidates[index];
        for (std::uint8_t value : candidate.reserved)
            if (value != 0u)
                return failure(atom_fragment_result_validation_code_v1::
                    nonzero_reserved, index);
        if (!valid_id(candidate.candidate_identity))
            return failure(atom_fragment_result_validation_code_v1::
                invalid_candidate_identity, index);
        if (index != 0u && !less_id(
                result.candidates[index - 1u].candidate_identity,
                candidate.candidate_identity))
            return failure(atom_fragment_result_validation_code_v1::
                duplicate_or_unordered_candidate, index);
        if (!valid_id(candidate.exact_local_coverage))
            return failure(atom_fragment_result_validation_code_v1::
                invalid_local_coverage, index);
        auto ids = validate_ids(candidate.required_atom_inputs,
            candidate.required_atom_input_count, index,
            atom_fragment_result_validation_code_v1::missing_atom_inputs,
            atom_fragment_result_validation_code_v1::invalid_atom_input,
            atom_fragment_result_validation_code_v1::
                duplicate_or_unordered_atom_input);
        if (!ids) return ids;
        if (!valid_id(candidate.program_recipe))
            return failure(atom_fragment_result_validation_code_v1::
                invalid_program_recipe, index);
        ids = validate_ids(candidate.projection_requirements,
            candidate.projection_requirement_count, index,
            atom_fragment_result_validation_code_v1::
                missing_projection_requirements,
            atom_fragment_result_validation_code_v1::
                invalid_projection_requirement,
            atom_fragment_result_validation_code_v1::
                duplicate_or_unordered_projection_requirement);
        if (!ids) return ids;
        if (!valid_id(candidate.output_affordance))
            return failure(atom_fragment_result_validation_code_v1::
                invalid_output_affordance, index);
        const bool partial =
            (candidate.flags & candidate_produces_partial_v1) != 0u;
        if (partial && !valid_id(candidate.partial_affordance))
            return failure(atom_fragment_result_validation_code_v1::
                invalid_partial_affordance, index);
        if (!partial && !zero_id(candidate.partial_affordance))
            return failure(atom_fragment_result_validation_code_v1::
                unexpected_partial_affordance, index);
        if (!valid_identity(candidate.input_order)
            || !valid_identity(candidate.output_order))
            return failure(
                atom_fragment_result_validation_code_v1::invalid_order, index);
        if (candidate.resources.launch_count == 0u
            || candidate.resources.extent_count == 0u)
            return failure(atom_fragment_result_validation_code_v1::
                invalid_resource, index);
        if (candidate.complete_cost.execution_ns == 0u
            || candidate.complete_cost.expected_reuse == 0u)
            return failure(atom_fragment_result_validation_code_v1::
                invalid_complete_cost, index);
        if (candidate.empirical_status < empirical_status_v1::analytical_only
            || candidate.empirical_status > empirical_status_v1::unavailable)
            return failure(atom_fragment_result_validation_code_v1::
                invalid_empirical_status, index);
        if ((candidate.flags & ~known_fragment_candidate_flags_v1) != 0u)
            return failure(
                atom_fragment_result_validation_code_v1::unknown_flag, index);
        if (!valid_id(candidate.validation_receipt))
            return failure(atom_fragment_result_validation_code_v1::
                invalid_validation_receipt, index);
    }
    return {};
}

}  // namespace cellerator::execution::joint_compiler
