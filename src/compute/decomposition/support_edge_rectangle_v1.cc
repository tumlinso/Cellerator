#include <Cellerator/compute/decomposition/support_edge_rectangle_v1.hh>

#include <limits>

namespace cellerator::compute::decomposition {
namespace {

support_edge_rectangle_validation_result_v1 failure(
    support_edge_rectangle_validation_code_v1 code,
    std::uint64_t fragment_index = 0u,
    std::uint32_t component_index = 0u) noexcept {
    return {code, fragment_index, component_index};
}

bool all_zero(const std::uint8_t *values, std::uint64_t count) noexcept {
    for (std::uint64_t index = 0u; index < count; ++index) {
        if (values[index] != 0u)
            return false;
    }
    return true;
}

}  // namespace

support_edge_rectangle_validation_result_v1
validate_support_edge_rectangle_decomposition_v1(
    const support_edge_rectangle_decomposition_v1 &decomposition,
    geometry::relation_cover_validation_workspace workspace) noexcept {
    using code = support_edge_rectangle_validation_code_v1;

    if (decomposition.schema_version
        != support_edge_rectangle_schema_version_v1)
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
            != operation::v2::operation_kind::contract_on_support
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
    if (decomposition.problem->relations.relations[decomposition.relation_index]
            .logical_edge_count != decomposition.cover->logical_edge_count)
        return failure(code::relation_edge_count_mismatch);
    if (decomposition.fragment_count == 0u)
        return failure(code::invalid_fragment_count);
    if (decomposition.fragments == nullptr)
        return failure(code::missing_fragments);
    if (decomposition.mode != support_edge_rectangle_mode_v1::logical_edge
        && decomposition.mode
            != support_edge_rectangle_mode_v1::semantic_rectangle)
        return failure(code::invalid_mode);
    if (decomposition.kind != decomposition_kind_v1::disjoint
        || decomposition.fragment_role != fragment_role_v1::owned)
        return failure(code::invalid_vocabulary);
    if (!decomposition.produces_partial_results
        || !decomposition.requires_partial_algebra)
        return failure(code::invalid_partial_result_contract);

    std::uint64_t expected_edge = 0u;
    std::uint32_t expected_component = 0u;
    for (std::uint64_t index = 0u; index < decomposition.fragment_count;
         ++index) {
        const auto &fragment = decomposition.fragments[index];
        if (fragment.logical_edge_count == 0u)
            return failure(code::empty_fragment, index);
        if (fragment.logical_edge_begin != expected_edge)
            return failure(code::edge_offset_mismatch, index);
        if (fragment.logical_edge_count
            > std::numeric_limits<std::uint64_t>::max()
                - fragment.logical_edge_begin)
            return failure(code::edge_range_overflow, index);
        expected_edge = fragment.logical_edge_begin
            + fragment.logical_edge_count;
        if (expected_edge > decomposition.cover->logical_edge_count)
            return failure(code::edge_range_overflow, index);

        if (decomposition.mode
            == support_edge_rectangle_mode_v1::semantic_rectangle) {
            if (fragment.component_count == 0u)
                return failure(code::empty_fragment, index);
            if (fragment.first_component != expected_component)
                return failure(code::component_offset_mismatch, index);
            if (fragment.component_count
                > std::numeric_limits<std::uint32_t>::max()
                    - fragment.first_component)
                return failure(code::component_range_overflow, index);
            const auto component_end =
                fragment.first_component + fragment.component_count;
            if (component_end > decomposition.cover->component_count)
                return failure(code::component_range_overflow, index);
            for (std::uint32_t component_index = fragment.first_component;
                 component_index < component_end; ++component_index) {
                if (decomposition.cover->components[component_index].kind
                    != geometry::semantic_component_kind::rectangular)
                    return failure(code::nonrectangular_component, index,
                        component_index);
            }
            const auto &first =
                decomposition.cover->components[fragment.first_component];
            const auto &last =
                decomposition.cover->components[component_end - 1u];
            const auto rectangle_edge_count = last.logical_edge_offset
                + last.logical_edge_count - first.logical_edge_offset;
            if (fragment.logical_edge_begin != first.logical_edge_offset
                || fragment.logical_edge_count != rectangle_edge_count)
                return failure(code::rectangle_edge_mismatch, index);
            expected_component = component_end;
        } else if (fragment.component_count != 0u
            || fragment.first_component != 0u) {
            return failure(code::invalid_vocabulary, index);
        }
    }
    if (expected_edge != decomposition.cover->logical_edge_count
        || (decomposition.mode
                == support_edge_rectangle_mode_v1::semantic_rectangle
            && expected_component != decomposition.cover->component_count))
        return failure(code::incomplete_partition,
            decomposition.fragment_count - 1u, expected_component);
    return {};
}

}  // namespace cellerator::compute::decomposition
