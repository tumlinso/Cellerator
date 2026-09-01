#include <Cellerator/compute/decomposition/partial_result_algebra_v1.hh>

namespace cellerator::compute::decomposition {
namespace {

using execution::joint_compiler::persistent_identity_v1;
using execution::joint_compiler::validate_persistent_identity_v1;

bool zero_identity(persistent_identity_v1 identity) noexcept {
    return identity.producer_namespace == 0u && identity.local_identity == 0u;
}

bool valid_numeric(execution::numeric_type type) noexcept {
    return type >= execution::numeric_type::bit
        && type <= execution::numeric_type::f64;
}

bool valid_numerical_policy(
    const operation::v2::numerical_policy &policy) noexcept {
    return valid_numeric(policy.relation_storage)
        && valid_numeric(policy.state_storage)
        && valid_numeric(policy.multiply)
        && valid_numeric(policy.accumulation)
        && valid_numeric(policy.output_storage)
        && valid_numeric(policy.scalar)
        && policy.rounding >= operation::v2::rounding_policy::nearest_even
        && policy.rounding <= operation::v2::rounding_policy::stochastic
        && policy.saturation >= operation::v2::saturation_policy::none
        && policy.saturation <= operation::v2::saturation_policy::saturate
        && policy.nan >= operation::v2::nan_policy::propagate
        && policy.nan <= operation::v2::nan_policy::reject
        && policy.infinity >= operation::v2::infinity_policy::propagate
        && policy.infinity <= operation::v2::infinity_policy::saturate;
}

}  // namespace

partial_result_algebra_validation_result_v1
validate_partial_result_algebra_v1(
    const partial_result_algebra_v1 &algebra) noexcept {
    if (algebra.schema_version != partial_result_algebra_schema_version_v1)
        return {
            partial_result_algebra_validation_code_v1::unsupported_schema};
    if (algebra.record_bytes != sizeof(partial_result_algebra_v1))
        return {
            partial_result_algebra_validation_code_v1::invalid_record_bytes};
    if (algebra.reserved != 0u)
        return {partial_result_algebra_validation_code_v1::nonzero_reserved};
    if (!validate_persistent_identity_v1(algebra.algebra_identity))
        return {
            partial_result_algebra_validation_code_v1::invalid_algebra_identity};
    if (!validate_persistent_identity_v1(algebra.state_layout_identity))
        return {
            partial_result_algebra_validation_code_v1::invalid_state_layout};
    if (!validate_persistent_identity_v1(algebra.neutral_element_identity))
        return {
            partial_result_algebra_validation_code_v1::invalid_neutral_element};
    if (!validate_persistent_identity_v1(algebra.merge_operation_identity))
        return {
            partial_result_algebra_validation_code_v1::invalid_merge_operation};
    if (!validate_persistent_identity_v1(algebra.finalize_operation_identity))
        return {partial_result_algebra_validation_code_v1::
            invalid_finalize_operation};
    if (algebra.state_bytes == 0u)
        return {partial_result_algebra_validation_code_v1::invalid_state_size};
    if (algebra.state_alignment == 0u
        || (algebra.state_alignment & (algebra.state_alignment - 1u)) != 0u)
        return {
            partial_result_algebra_validation_code_v1::invalid_state_alignment};
    if ((algebra.flags & ~known_partial_result_algebra_flags_v1) != 0u)
        return {partial_result_algebra_validation_code_v1::unknown_flag};
    if ((algebra.flags & (associative_v1 | ordered_only_v1)) == 0u)
        return {partial_result_algebra_validation_code_v1::
            missing_reconstruction_rule};

    const bool ordered = (algebra.flags & ordered_only_v1) != 0u;
    if (ordered && !execution::valid_identity(algebra.required_merge_order))
        return {
            partial_result_algebra_validation_code_v1::invalid_order_constraint};
    if (!ordered && execution::valid_identity(algebra.required_merge_order))
        return {partial_result_algebra_validation_code_v1::
            unexpected_order_constraint};

    const bool fixed_tree =
        (algebra.flags & deterministic_tree_required_v1) != 0u;
    if (fixed_tree
        && !validate_persistent_identity_v1(algebra.deterministic_tree_identity))
        return {partial_result_algebra_validation_code_v1::
            missing_deterministic_tree};
    if (!fixed_tree && !zero_identity(algebra.deterministic_tree_identity))
        return {partial_result_algebra_validation_code_v1::
            unexpected_deterministic_tree};
    if (!valid_numerical_policy(algebra.numerical))
        return {partial_result_algebra_validation_code_v1::
            invalid_numerical_policy};
    return {};
}

}  // namespace cellerator::compute::decomposition
