#include <Cellerator/compute/decomposition/edge_component_v1.hh>

#include <limits>

namespace cellerator::compute::decomposition {
namespace {

edge_component_validation_result_v1 failure(
    edge_component_validation_code_v1 code,
    std::uint64_t fragment_index = 0u) noexcept {
    return {code, fragment_index};
}

bool all_zero(const std::uint8_t *values, std::uint64_t count) noexcept {
    for (std::uint64_t index = 0u; index < count; ++index) {
        if (values[index] != 0u)
            return false;
    }
    return true;
}

}  // namespace

edge_component_validation_result_v1 validate_edge_component_relation_apply_v1(
    const edge_component_relation_apply_v1 &decomposition,
    geometry::relation_cover_validation_workspace workspace) noexcept {
    using code = edge_component_validation_code_v1;

    if (decomposition.schema_version != edge_component_schema_version_v1)
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
            != operation::v2::operation_kind::relation_apply
        || decomposition.problem->orientation
            != operation::v2::relation_orientation::forward)
        return failure(code::unsupported_operation);
    if (decomposition.relation_index
        >= decomposition.problem->relations.relation_count)
        return failure(code::invalid_relation_index);
    if (decomposition.cover == nullptr)
        return failure(code::missing_cover);
    if (!geometry::validate_relation_cover(*decomposition.cover, workspace))
        return failure(code::invalid_cover);
    const auto &relation = decomposition.problem->relations.relations[
        decomposition.relation_index];
    if (relation.logical_edge_count != decomposition.cover->logical_edge_count)
        return failure(code::relation_edge_count_mismatch);
    if (decomposition.fragment_count == 0u)
        return failure(code::invalid_fragment_count);
    if (decomposition.fragments == nullptr)
        return failure(code::missing_fragments);
    if (decomposition.split_axis != split_axis_kind_v1::semantic_component
        || decomposition.kind != decomposition_kind_v1::disjoint
        || decomposition.fragment_role != fragment_role_v1::owned)
        return failure(code::invalid_vocabulary);
    if (!decomposition.produces_partial_results
        || !decomposition.requires_partial_algebra)
        return failure(code::invalid_partial_result_contract);

    std::uint32_t expected_component = 0u;
    for (std::uint64_t index = 0u; index < decomposition.fragment_count;
         ++index) {
        const auto &fragment = decomposition.fragments[index];
        if (fragment.component_count == 0u)
            return failure(code::empty_fragment, index);
        if (fragment.first_component != expected_component)
            return failure(code::component_offset_mismatch, index);
        if (fragment.component_count
            > std::numeric_limits<std::uint32_t>::max()
                - fragment.first_component)
            return failure(code::component_range_overflow, index);
        const auto end = fragment.first_component + fragment.component_count;
        if (end > decomposition.cover->component_count)
            return failure(code::component_range_overflow, index);

        const auto &first =
            decomposition.cover->components[fragment.first_component];
        const auto &last = decomposition.cover->components[end - 1u];
        const auto logical_end =
            last.logical_edge_offset + last.logical_edge_count;
        if (fragment.logical_edge_begin != first.logical_edge_offset)
            return failure(code::logical_edge_offset_mismatch, index);
        if (fragment.logical_edge_count
            != logical_end - first.logical_edge_offset)
            return failure(code::logical_edge_count_mismatch, index);
        expected_component = end;
    }
    if (expected_component != decomposition.cover->component_count)
        return failure(code::incomplete_component_partition,
            decomposition.fragment_count - 1u);
    return {};
}

}  // namespace cellerator::compute::decomposition
