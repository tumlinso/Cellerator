#include <Cellerator/compute/decomposition/relation_bundle_v1.hh>

#include <limits>

namespace cellerator::compute::decomposition {
namespace {

relation_bundle_validation_result_v1 failure(
    relation_bundle_validation_code_v1 code,
    std::uint64_t fragment_index = 0u,
    std::uint64_t relation_index = 0u) noexcept {
    return {code, fragment_index, relation_index};
}

bool all_zero(const std::uint8_t *values, std::uint64_t count) noexcept {
    for (std::uint64_t index = 0u; index < count; ++index) {
        if (values[index] != 0u)
            return false;
    }
    return true;
}

}  // namespace

relation_bundle_validation_result_v1
validate_relation_bundle_type_decomposition_v1(
    const relation_bundle_type_decomposition_v1 &decomposition) noexcept {
    using code = relation_bundle_validation_code_v1;

    if (decomposition.schema_version != relation_bundle_schema_version_v1)
        return failure(code::unsupported_schema);
    if (decomposition.reserved != 0u
        || !all_zero(decomposition.reserved2, sizeof(decomposition.reserved2)))
        return failure(code::nonzero_reserved);
    if (!operation::v2::valid_stable_id(decomposition.decomposition_identity))
        return failure(code::invalid_identity);
    if (decomposition.problem == nullptr)
        return failure(code::missing_problem);
    if (!operation::v2::validate_operation_problem(*decomposition.problem))
        return failure(code::invalid_problem);
    if (decomposition.problem->kind
        != operation::v2::operation_kind::relation_bundle_apply)
        return failure(code::unsupported_operation);
    if (decomposition.fragment_count == 0u)
        return failure(code::invalid_fragment_count);
    if (decomposition.fragments == nullptr)
        return failure(code::missing_fragments);
    if (decomposition.kind != decomposition_kind_v1::disjoint
        || decomposition.fragment_role != fragment_role_v1::owned)
        return failure(code::invalid_vocabulary);
    if (!decomposition.produces_partial_results
        || !decomposition.requires_partial_algebra)
        return failure(code::invalid_partial_result_contract);

    std::uint64_t expected_relation = 0u;
    const auto relation_count =
        decomposition.problem->relations.relation_count;
    for (std::uint64_t fragment_index = 0u;
         fragment_index < decomposition.fragment_count; ++fragment_index) {
        const auto &fragment = decomposition.fragments[fragment_index];
        if (fragment.relation_count == 0u)
            return failure(code::empty_fragment, fragment_index);
        if (fragment.first_relation != expected_relation)
            return failure(code::relation_offset_mismatch, fragment_index);
        if (fragment.relation_count
            > std::numeric_limits<std::uint64_t>::max()
                - fragment.first_relation)
            return failure(code::relation_range_overflow, fragment_index);
        const auto relation_end =
            fragment.first_relation + fragment.relation_count;
        if (relation_end > relation_count)
            return failure(code::relation_range_overflow, fragment_index);
        if (!execution::valid_identity(fragment.source_domain)
            || !execution::valid_identity(fragment.destination_domain))
            return failure(code::invalid_relation_type, fragment_index);

        for (std::uint64_t relation_index = fragment.first_relation;
             relation_index < relation_end; ++relation_index) {
            const auto &relation =
                decomposition.problem->relations.relations[relation_index];
            if (!execution::same_identity(
                    relation.source_axis.domain, fragment.source_domain)
                || !execution::same_identity(relation.destination_axis.domain,
                    fragment.destination_domain))
                return failure(code::relation_type_mismatch, fragment_index,
                    relation_index);
        }
        expected_relation = relation_end;
    }
    if (expected_relation != relation_count)
        return failure(code::incomplete_relation_partition,
            decomposition.fragment_count - 1u, expected_relation);
    return {};
}

}  // namespace cellerator::compute::decomposition
