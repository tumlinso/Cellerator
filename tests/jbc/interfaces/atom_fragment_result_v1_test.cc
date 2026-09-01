#include <Cellerator/execution/joint_compiler/atom_fragment_result_v1.hh>

#include <cassert>

namespace joint_compiler = cellerator::execution::joint_compiler;

int main() {
    const joint_compiler::persistent_identity_v1 atoms[] = {
        {1u, 1u}, {1u, 2u}};
    const joint_compiler::persistent_identity_v1 projections[] = {
        {2u, 1u}, {2u, 2u}};
    joint_compiler::atom_fragment_candidate_v1 candidates[2]{};
    for (std::uint64_t index = 0u; index < 2u; ++index) {
        auto &candidate = candidates[index];
        candidate.candidate_identity = {3u, index + 1u};
        candidate.exact_local_coverage = {4u, index + 1u};
        candidate.required_atom_inputs = atoms;
        candidate.required_atom_input_count = 2u;
        candidate.program_recipe = {5u, index + 1u};
        candidate.projection_requirements = projections;
        candidate.projection_requirement_count = 2u;
        candidate.output_affordance = {6u, index + 1u};
        candidate.input_order = {7u, index + 1u};
        candidate.output_order = {8u, index + 1u};
        candidate.resources.launch_count = 1u;
        candidate.resources.extent_count = 1u;
        candidate.complete_cost.execution_ns = 100u + index;
        candidate.complete_cost.expected_reuse = 4u;
        candidate.flags = joint_compiler::deterministic_candidate_v1;
        candidate.validation_receipt = {9u, index + 1u};
    }

    joint_compiler::atom_fragment_result_v1 result{};
    result.result_identity = {10u, 1u};
    result.request_identity = {10u, 2u};
    result.candidates = candidates;
    result.candidate_count = 2u;
    result.candidate_capacity = 2u;
    assert(joint_compiler::validate_atom_fragment_result_v1(result));

    auto malformed = result;
    malformed.candidate_capacity =
        joint_compiler::maximum_atom_fragment_candidates_v1 + 1u;
    assert(joint_compiler::validate_atom_fragment_result_v1(malformed).code
        == joint_compiler::atom_fragment_result_validation_code_v1::
            invalid_candidate_bounds);
    malformed = result;
    malformed.frontier_truncated = true;
    malformed.candidate_capacity = 3u;
    assert(joint_compiler::validate_atom_fragment_result_v1(malformed).code
        == joint_compiler::atom_fragment_result_validation_code_v1::
            inconsistent_truncation);

    candidates[0].flags |= joint_compiler::candidate_produces_partial_v1;
    assert(joint_compiler::validate_atom_fragment_result_v1(result).code
        == joint_compiler::atom_fragment_result_validation_code_v1::
            invalid_partial_affordance);
    candidates[0].partial_affordance = {11u, 1u};
    assert(joint_compiler::validate_atom_fragment_result_v1(result));
    candidates[0].flags &= ~joint_compiler::candidate_produces_partial_v1;
    assert(joint_compiler::validate_atom_fragment_result_v1(result).code
        == joint_compiler::atom_fragment_result_validation_code_v1::
            unexpected_partial_affordance);
    candidates[0].partial_affordance = {};

    const joint_compiler::persistent_identity_v1 duplicate_atoms[] = {
        {1u, 1u}, {1u, 1u}};
    candidates[0].required_atom_inputs = duplicate_atoms;
    assert(joint_compiler::validate_atom_fragment_result_v1(result).code
        == joint_compiler::atom_fragment_result_validation_code_v1::
            duplicate_or_unordered_atom_input);
    candidates[0].required_atom_inputs = atoms;

    // Empty frontier with an explicit bounded outcome is valid.
    result.candidates = nullptr;
    result.candidate_count = 0u;
    result.candidate_capacity = 0u;
    result.no_candidate_reason =
        joint_compiler::no_candidate_reason_v1::unmet_atom_requirement;
    assert(joint_compiler::validate_atom_fragment_result_v1(result));
    result.no_candidate_reason = joint_compiler::no_candidate_reason_v1::none;
    assert(joint_compiler::validate_atom_fragment_result_v1(result).code
        == joint_compiler::atom_fragment_result_validation_code_v1::
            inconsistent_empty_frontier);
    return 0;
}
