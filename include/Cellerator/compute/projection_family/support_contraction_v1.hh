#pragma once

#include <Cellerator/compute/projection_family/support_family_identity_v1.hh>
#include <Cellerator/geometry/relation_cover.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::projection_family {

struct support_contraction_workspace_v1 {
    std::uint8_t *logical_edge_marks = nullptr;
    std::uint64_t logical_edge_mark_capacity = 0;
    std::uint8_t *component_marks = nullptr;
    std::uint32_t component_mark_capacity = 0;
};

// Component-major physical organization for contraction on one exact support.
// Component metadata and edge IDs remain caller-owned immutable structure;
// numerical value and gradient planes are deliberately not bound here.
struct support_contraction_view_v1 {
    support_family_identity_v1 family{};
    execution::projection_id projection_identity{};
    execution::order_id physical_edge_order{};
    const geometry::semantic_component_v1 *components = nullptr;
    const std::uint64_t *logical_edge_ids = nullptr;
    std::uint32_t component_count = 0;
    std::uint32_t reserved = 0;
    std::uint64_t logical_edge_count = 0;
};

enum class support_contraction_code_v1 : std::uint32_t {
    built = 0,
    invalid_family,
    operation_not_supported,
    invalid_projection_identity,
    invalid_physical_order,
    unsupported_cover_schema,
    nonzero_cover_reserved,
    edge_count_mismatch,
    empty_component_set,
    missing_components,
    missing_logical_edge_ids,
    missing_workspace,
    insufficient_edge_mark_capacity,
    insufficient_component_mark_capacity,
    invalid_component_id,
    duplicate_component_id,
    invalid_component_kind,
    nonzero_component_reserved,
    empty_component,
    component_offset_mismatch,
    component_range_overflow,
    incomplete_component_partition,
    logical_edge_out_of_range,
    duplicate_logical_edge,
    missing_logical_edge,
    missing_output,
};

struct support_contraction_result_v1 {
    support_contraction_code_v1 code = support_contraction_code_v1::built;
    std::uint64_t item_index = 0;

    [[nodiscard]] constexpr bool built() const noexcept {
        return code == support_contraction_code_v1::built;
    }
};

[[nodiscard]] inline support_contraction_result_v1
build_support_contraction_view_v1(
    const support_family_descriptor_v1 &family,
    execution::projection_id projection_identity,
    execution::order_id physical_edge_order,
    const geometry::relation_cover_view_v1 &cover,
    support_contraction_workspace_v1 workspace,
    support_contraction_view_v1 *output) noexcept {
    const auto family_status = validate_support_family_descriptor_v1(family);
    if (!family_status.valid()) {
        return {support_contraction_code_v1::invalid_family,
                static_cast<std::uint64_t>(family_status.code)};
    }
    if (!support_family_supports_v1(family, support_contract_on_support_v1)) {
        return {support_contraction_code_v1::operation_not_supported};
    }
    if (!execution::valid_identity(projection_identity)) {
        return {support_contraction_code_v1::invalid_projection_identity};
    }
    if (!execution::valid_identity(physical_edge_order)) {
        return {support_contraction_code_v1::invalid_physical_order};
    }
    if (cover.schema_version != geometry::relation_cover_schema_version) {
        return {support_contraction_code_v1::unsupported_cover_schema};
    }
    if (cover.reserved != 0 || cover.reserved2 != 0) {
        return {support_contraction_code_v1::nonzero_cover_reserved};
    }
    if (cover.logical_edge_count != family.identity.logical_edge_count) {
        return {support_contraction_code_v1::edge_count_mismatch};
    }
    if (cover.component_count == 0) {
        return {support_contraction_code_v1::empty_component_set};
    }
    if (cover.components == nullptr) {
        return {support_contraction_code_v1::missing_components};
    }
    if (cover.logical_edge_ids == nullptr) {
        return {support_contraction_code_v1::missing_logical_edge_ids};
    }
    if (workspace.logical_edge_marks == nullptr
        || workspace.component_marks == nullptr) {
        return {support_contraction_code_v1::missing_workspace};
    }
    if (workspace.logical_edge_mark_capacity < cover.logical_edge_count) {
        return {support_contraction_code_v1::insufficient_edge_mark_capacity};
    }
    if (workspace.component_mark_capacity < cover.component_count) {
        return {support_contraction_code_v1::
                    insufficient_component_mark_capacity};
    }
    if (output == nullptr) {
        return {support_contraction_code_v1::missing_output};
    }
    *output = {};

    for (std::uint64_t edge = 0; edge < cover.logical_edge_count; ++edge) {
        workspace.logical_edge_marks[edge] = 0;
    }
    for (std::uint32_t component = 0;
         component < cover.component_count;
         ++component) {
        workspace.component_marks[component] = 0;
    }

    std::uint64_t expected_offset = 0;
    for (std::uint32_t index = 0; index < cover.component_count; ++index) {
        const auto &component = cover.components[index];
        if (component.component_id == geometry::invalid_semantic_component_id
            || component.component_id > cover.component_count) {
            return {support_contraction_code_v1::invalid_component_id, index};
        }
        const auto component_slot = component.component_id - 1;
        if (workspace.component_marks[component_slot] != 0) {
            return {support_contraction_code_v1::duplicate_component_id,
                    index};
        }
        workspace.component_marks[component_slot] = 1;
        if (!geometry::valid_semantic_component_kind(component.kind)) {
            return {support_contraction_code_v1::invalid_component_kind,
                    index};
        }
        if (component.reserved[0] != 0 || component.reserved[1] != 0
            || component.reserved[2] != 0) {
            return {support_contraction_code_v1::nonzero_component_reserved,
                    index};
        }
        if (component.logical_edge_count == 0) {
            return {support_contraction_code_v1::empty_component, index};
        }
        if (component.logical_edge_offset != expected_offset) {
            return {support_contraction_code_v1::component_offset_mismatch,
                    index};
        }
        if (component.logical_edge_count
            > cover.logical_edge_count - expected_offset) {
            return {support_contraction_code_v1::component_range_overflow,
                    index};
        }
        expected_offset += component.logical_edge_count;
    }
    if (expected_offset != cover.logical_edge_count) {
        return {support_contraction_code_v1::
                    incomplete_component_partition,
                expected_offset};
    }

    for (std::uint64_t index = 0; index < cover.logical_edge_count; ++index) {
        const auto edge = cover.logical_edge_ids[index];
        if (edge >= cover.logical_edge_count) {
            return {support_contraction_code_v1::logical_edge_out_of_range,
                    index};
        }
        if (workspace.logical_edge_marks[edge] != 0) {
            return {support_contraction_code_v1::duplicate_logical_edge,
                    index};
        }
        workspace.logical_edge_marks[edge] = 1;
    }
    for (std::uint64_t edge = 0; edge < cover.logical_edge_count; ++edge) {
        if (workspace.logical_edge_marks[edge] == 0) {
            return {support_contraction_code_v1::missing_logical_edge, edge};
        }
    }

    *output = {family.identity,
               projection_identity,
               physical_edge_order,
               cover.components,
               cover.logical_edge_ids,
               cover.component_count,
               0,
               cover.logical_edge_count};
    return {support_contraction_code_v1::built, cover.logical_edge_count};
}

static_assert(std::is_standard_layout_v<support_contraction_view_v1>);
static_assert(std::is_trivially_copyable_v<support_contraction_view_v1>);
static_assert(std::is_standard_layout_v<support_contraction_workspace_v1>);
static_assert(std::is_trivially_copyable_v<support_contraction_workspace_v1>);

} // namespace cellerator::compute::projection_family
