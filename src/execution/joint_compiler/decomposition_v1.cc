#include <Cellerator/compute/decomposition/decomposition_v1.hh>

namespace cellerator::compute::decomposition {
namespace {

using identity = execution::joint_compiler::persistent_identity_v1;

decomposition_validation_result_v1 failure(
    decomposition_validation_code_v1 code,
    std::uint64_t alternative = 0u,
    std::uint64_t element = 0u) noexcept {
    return {code, alternative, element};
}

bool valid_id(identity value) noexcept {
    return static_cast<bool>(
        execution::joint_compiler::validate_persistent_identity_v1(value));
}

bool zero_id(identity value) noexcept {
    return value.producer_namespace == 0u && value.local_identity == 0u;
}

bool less_id(identity lhs, identity rhs) noexcept {
    return lhs.producer_namespace < rhs.producer_namespace
        || (lhs.producer_namespace == rhs.producer_namespace
            && lhs.local_identity < rhs.local_identity);
}

bool valid_numeric(execution::numeric_type type) noexcept {
    return type >= execution::numeric_type::bit
        && type <= execution::numeric_type::f64;
}

bool valid_numerical(
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

decomposition_validation_result_v1 validate_ids(
    const identity *values,
    std::uint64_t count,
    std::uint64_t alternative,
    decomposition_validation_code_v1 invalid,
    decomposition_validation_code_v1 unordered) noexcept {
    if (count == 0u || values == nullptr)
        return failure(invalid, alternative);
    for (std::uint64_t index = 0u; index < count; ++index) {
        if (!valid_id(values[index]))
            return failure(invalid, alternative, index);
        if (index != 0u && !less_id(values[index - 1u], values[index]))
            return failure(unordered, alternative, index);
    }
    return {};
}

}  // namespace

decomposition_validation_result_v1 validate_decomposition_portfolio_v1(
    const decomposition_portfolio_v1 &portfolio) noexcept {
    if (portfolio.schema_version != decomposition_schema_version_v1)
        return failure(decomposition_validation_code_v1::unsupported_schema);
    if (portfolio.record_bytes != sizeof(decomposition_portfolio_v1))
        return failure(decomposition_validation_code_v1::invalid_record_bytes);
    if (!valid_id(portfolio.portfolio_identity))
        return failure(
            decomposition_validation_code_v1::invalid_portfolio_identity);
    if (portfolio.alternative_count == 0u || portfolio.alternatives == nullptr)
        return failure(decomposition_validation_code_v1::missing_alternatives);

    bool found_fallback = false;
    for (std::uint64_t index = 0u; index < portfolio.alternative_count; ++index) {
        const decomposition_alternative_v1 &alternative =
            portfolio.alternatives[index];
        if (!valid_id(alternative.alternative_identity))
            return failure(decomposition_validation_code_v1::
                invalid_alternative_identity, index);
        if (index != 0u && !less_id(
                portfolio.alternatives[index - 1u].alternative_identity,
                alternative.alternative_identity))
            return failure(decomposition_validation_code_v1::
                duplicate_or_unordered_alternative, index);
        if (!valid_id(alternative.candidate_family))
            return failure(decomposition_validation_code_v1::
                invalid_candidate_family, index);
        if (alternative.split_axis < split_axis_v1::none
            || alternative.split_axis > split_axis_v1::extents)
            return failure(decomposition_validation_code_v1::
                invalid_split_axis, index);
        if (alternative.reserved0[0] != 0u || alternative.reserved0[1] != 0u
            || alternative.reserved0[2] != 0u)
            return failure(decomposition_validation_code_v1::
                nonzero_reserved, index);
        if ((alternative.flags & ~known_decomposition_flags_v1) != 0u)
            return failure(
                decomposition_validation_code_v1::unknown_flag, index);
        if ((alternative.flags & legal_alternative_v1) == 0u)
            return failure(decomposition_validation_code_v1::
                alternative_not_legal, index);

        const bool fallback =
            (alternative.flags & complete_unsplit_fallback_v1) != 0u;
        if (fallback) {
            if (found_fallback)
                return failure(decomposition_validation_code_v1::
                    duplicate_fallback, index);
            found_fallback = true;
            if (alternative.split_axis != split_axis_v1::none
                || (alternative.flags & (produces_partial_result_v1
                    | requires_replication_v1 | requires_halo_v1)) != 0u)
                return failure(
                    decomposition_validation_code_v1::invalid_fallback, index);
        } else if (alternative.split_axis == split_axis_v1::none) {
            return failure(
                decomposition_validation_code_v1::invalid_split_axis, index);
        }

        auto ids = validate_ids(alternative.required_input_coverages,
            alternative.required_input_coverage_count, index,
            decomposition_validation_code_v1::invalid_input_coverage,
            decomposition_validation_code_v1::
                duplicate_or_unordered_input_coverage);
        if (!ids) return ids;
        if (!valid_id(alternative.output_coverage))
            return failure(decomposition_validation_code_v1::
                invalid_output_coverage, index);

        const bool needs_replication =
            (alternative.flags & requires_replication_v1) != 0u;
        if (needs_replication) {
            ids = validate_ids(alternative.replication_coverages,
                alternative.replication_coverage_count, index,
                decomposition_validation_code_v1::invalid_replication_coverage,
                decomposition_validation_code_v1::invalid_replication_coverage);
            if (!ids) return ids;
        } else if (alternative.replication_coverage_count != 0u
            || alternative.replication_coverages != nullptr) {
            return failure(decomposition_validation_code_v1::
                invalid_replication_flag, index);
        }
        const bool needs_halo =
            (alternative.flags & requires_halo_v1) != 0u;
        if (needs_halo) {
            ids = validate_ids(alternative.halo_coverages,
                alternative.halo_coverage_count, index,
                decomposition_validation_code_v1::invalid_halo_coverage,
                decomposition_validation_code_v1::invalid_halo_coverage);
            if (!ids) return ids;
        } else if (alternative.halo_coverage_count != 0u
            || alternative.halo_coverages != nullptr) {
            return failure(
                decomposition_validation_code_v1::invalid_halo_flag, index);
        }
        if (!execution::valid_identity(alternative.input_order)
            || !execution::valid_identity(alternative.output_order))
            return failure(
                decomposition_validation_code_v1::invalid_order, index);
        const bool partial =
            (alternative.flags & produces_partial_result_v1) != 0u;
        if (partial && !valid_id(alternative.partial_algebra))
            return failure(decomposition_validation_code_v1::
                invalid_partial_algebra, index);
        if (!partial && !zero_id(alternative.partial_algebra))
            return failure(decomposition_validation_code_v1::
                unexpected_partial_algebra, index);
        if (!valid_numerical(alternative.numerical))
            return failure(decomposition_validation_code_v1::
                invalid_numerical_policy, index);
    }
    if (!found_fallback)
        return failure(decomposition_validation_code_v1::missing_fallback);
    return {};
}

}  // namespace cellerator::compute::decomposition
