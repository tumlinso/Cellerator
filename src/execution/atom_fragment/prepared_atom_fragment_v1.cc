#include <Cellerator/execution/atom_fragment/prepared_atom_fragment_v1.hh>

namespace cellerator::execution::atom_fragment {
namespace {

bool valid(joint_compiler::persistent_identity_v1 value) noexcept {
    return static_cast<bool>(
        joint_compiler::validate_persistent_identity_v1(value));
}

} // namespace

prepared_atom_fragment_status_v1 prepare_atom_fragment_v1(
    const atom_bound_candidate_v1 &candidate,
    const program::prepared_program_v2 &program_value,
    order_id input_order,
    order_id output_order,
    prepared_atom_fragment_v1 *output) noexcept {
    using code = prepared_atom_fragment_status_code_v1;
    if (output == nullptr)
        return {code::null_output, 0u};
    *output = {};
    if (candidate.candidate_id == 0u || !valid(candidate.atom_identity)
        || !valid(candidate.requirement_identity)
        || !valid(candidate.affordance_identity))
        return {code::invalid_candidate, 0u};
    if (program::validate_prepared_program_v2(program_value)
        != program::program_status::success)
        return {code::invalid_program, 0u};
    if (program_value.stage_count == 0u)
        return {code::empty_program, 0u};
    if (!valid_identity(input_order) || !valid_identity(output_order))
        return {code::invalid_order, 0u};

    std::uint64_t binding_count = 0u;
    std::uint64_t maximum_workspace = 0u;
    for (std::uint64_t index = 0u; index < program_value.stage_count;
         ++index) {
        const auto &stage = program_value.stages[index];
        if (stage.candidate_id != candidate.candidate_id)
            return {code::foreign_candidate_stage, index};
        const std::uint64_t stage_binding_count =
            static_cast<std::uint64_t>(stage.binding_index) + 1u;
        if (binding_count < stage_binding_count)
            binding_count = stage_binding_count;
        if (maximum_workspace < stage.required_workspace_bytes)
            maximum_workspace = stage.required_workspace_bytes;
    }
    *output = {candidate, &program_value, input_order, output_order,
        binding_count, maximum_workspace};
    return {};
}

} // namespace cellerator::execution::atom_fragment
