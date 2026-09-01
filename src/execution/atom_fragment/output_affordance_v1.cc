#include <Cellerator/execution/atom_fragment/output_affordance_v1.hh>

namespace cellerator::execution::atom_fragment {
namespace {

bool valid(joint_compiler::persistent_identity_v1 value) noexcept {
    return static_cast<bool>(
        joint_compiler::validate_persistent_identity_v1(value));
}

} // namespace

output_affordance_status_v1 describe_fragment_output_affordances_v1(
    const prepared_atom_fragment_v1 &prepared,
    const compute::operation::v2::operation_problem &operation,
    const output_affordance_recipe_v1 &recipe,
    fragment_output_description_v1 *description) noexcept {
    using code = output_affordance_status_code_v1;
    if (description == nullptr)
        return {code::null_output};
    *description = {};
    if (prepared.program == nullptr || prepared.candidate.candidate_id == 0u)
        return {code::invalid_prepared_fragment};
    if (!compute::operation::v2::validate_operation_problem(operation))
        return {code::invalid_operation};
    if (!same_identity(prepared.output_order, operation.result_axis.order)
        && !operation.output.explicit_order_transform)
        return {code::inconsistent_output_order};
    if (!valid(recipe.output_atom_identity)
        || !valid(recipe.output_affordance_identity)
        || !valid(recipe.output_plane_identity)
        || !valid(recipe.exact_output_coverage)
        || recipe.output_generation.value == 0u)
        return {code::invalid_recipe};
    if (recipe.produces_partial
        && (!valid(recipe.partial_affordance_identity)
            || !valid(recipe.partial_plane_identity)
            || !valid(recipe.partial_algebra)))
        return {code::invalid_partial_recipe};
    if (!recipe.produces_partial
        && (valid(recipe.partial_affordance_identity)
            || valid(recipe.partial_plane_identity)
            || valid(recipe.partial_algebra)))
        return {code::invalid_partial_recipe};

    description->output = {recipe.output_atom_identity,
        recipe.output_affordance_identity, recipe.output_plane_identity,
        recipe.exact_output_coverage, prepared.output_order,
        operation.numeric.output_storage, operation.numeric.output_storage,
        recipe.output_generation, false, {}};
    if (recipe.produces_partial) {
        description->partial = {recipe.output_atom_identity,
            recipe.partial_affordance_identity, recipe.partial_plane_identity,
            recipe.exact_output_coverage, prepared.output_order,
            operation.numeric.accumulation, operation.numeric.accumulation,
            recipe.output_generation, true, recipe.partial_algebra};
        description->has_partial = true;
    }
    return {};
}

} // namespace cellerator::execution::atom_fragment
