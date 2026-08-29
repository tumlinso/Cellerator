#include <Cellerator/geometry/relation_cover.hh>

namespace cellerator::geometry {
namespace {

constexpr relation_cover_validation_result failure(
    relation_cover_validation_code code,
    u32 component_index = invalid_semantic_component_index,
    u64 logical_edge_id = invalid_logical_edge_id) noexcept {
    return {code, component_index, logical_edge_id};
}

} // namespace

relation_cover_validation_result validate_relation_cover(
    const relation_cover_view_v1 &cover,
    relation_cover_validation_workspace workspace) noexcept {
    if (cover.schema_version != relation_cover_schema_version)
        return failure(
            relation_cover_validation_code::unsupported_version);
    if (cover.reserved != 0u || cover.reserved2 != 0u)
        return failure(relation_cover_validation_code::nonzero_reserved);
    if (!execution::valid_handle(cover.structure)
        || cover.structure_epoch.value == 0u)
        return failure(relation_cover_validation_code::invalid_structure);
    if (!execution::valid_axis_identity(cover.source_axis))
        return failure(relation_cover_validation_code::invalid_source_axis);
    if (!execution::valid_axis_identity(cover.destination_axis))
        return failure(
            relation_cover_validation_code::invalid_destination_axis);

    if (cover.logical_edge_count == 0u) {
        if (cover.component_count != 0u)
            return failure(
                relation_cover_validation_code::invalid_component_count);
        if (cover.components != nullptr || cover.logical_edge_ids != nullptr)
            return failure(relation_cover_validation_code::nonzero_reserved);
        return {};
    }
    if (cover.component_count == 0u
        || static_cast<u64>(cover.component_count) > cover.logical_edge_count)
        return failure(
            relation_cover_validation_code::invalid_component_count);
    if (cover.components == nullptr)
        return failure(relation_cover_validation_code::missing_components);
    if (cover.logical_edge_ids == nullptr)
        return failure(
            relation_cover_validation_code::missing_logical_edge_ids);
    if (workspace.edge_marks == nullptr)
        return failure(relation_cover_validation_code::missing_workspace);
    if (workspace.edge_mark_capacity < cover.logical_edge_count)
        return failure(
            relation_cover_validation_code::insufficient_workspace);

    for (u64 edge = 0u; edge < cover.logical_edge_count; ++edge)
        workspace.edge_marks[edge] = 0u;

    u64 expected_offset = 0u;
    for (u32 component_index = 0u;
         component_index < cover.component_count; ++component_index) {
        const semantic_component_v1 &component =
            cover.components[component_index];
        if (component.component_id == invalid_semantic_component_id)
            return failure(relation_cover_validation_code::invalid_component_id,
                component_index);
        if (!valid_semantic_component_kind(component.kind))
            return failure(
                relation_cover_validation_code::invalid_component_kind,
                component_index);
        if (component.reserved[0] != 0u || component.reserved[1] != 0u
            || component.reserved[2] != 0u)
            return failure(
                relation_cover_validation_code::nonzero_component_reserved,
                component_index);
        if (component.logical_edge_count == 0u)
            return failure(relation_cover_validation_code::empty_component,
                component_index);
        if (component.logical_edge_offset != expected_offset)
            return failure(
                relation_cover_validation_code::component_offset_mismatch,
                component_index);
        if (component.logical_edge_count
            > cover.logical_edge_count - expected_offset)
            return failure(
                relation_cover_validation_code::component_edge_range_overflow,
                component_index);
        for (u32 previous = 0u; previous < component_index; ++previous)
            if (cover.components[previous].component_id
                == component.component_id)
                return failure(
                    relation_cover_validation_code::duplicate_component_id,
                    component_index);

        const u64 component_end =
            expected_offset + component.logical_edge_count;
        for (u64 position = expected_offset; position < component_end;
             ++position) {
            const u64 logical_edge_id = cover.logical_edge_ids[position];
            if (logical_edge_id >= cover.logical_edge_count)
                return failure(
                    relation_cover_validation_code::logical_edge_out_of_bounds,
                    component_index, logical_edge_id);
            if (workspace.edge_marks[logical_edge_id] != 0u)
                return failure(
                    relation_cover_validation_code::duplicate_logical_edge,
                    component_index, logical_edge_id);
            workspace.edge_marks[logical_edge_id] = 1u;
        }
        expected_offset = component_end;
    }

    if (expected_offset != cover.logical_edge_count)
        return failure(
            relation_cover_validation_code::incomplete_component_partition);
    for (u64 logical_edge_id = 0u;
         logical_edge_id < cover.logical_edge_count; ++logical_edge_id)
        if (workspace.edge_marks[logical_edge_id] == 0u)
            return failure(relation_cover_validation_code::missing_logical_edge,
                invalid_semantic_component_index, logical_edge_id);
    return {};
}

} // namespace cellerator::geometry
