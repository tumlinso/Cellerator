#pragma once

#include <Cellerator/geometry/relation_cover.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::geometry {

inline constexpr u32 rectangular_support_schema_version = 1u;
inline constexpr u64 invalid_rectangular_support_index = ~u64{0u};

// A portable reference into separately owned support evidence. evidence_kind is
// interpreted by the originating evidence schema; this contract does not
// duplicate support statistics or require them for semantic-cover validity.
struct portable_support_reference_v1 {
    u64 evidence_identity = 0u;
    u32 evidence_kind = 0u;
    u32 reserved = 0u;
    u64 record_offset = 0u;
    u64 record_count = 0u;
};

// One record attaches arbitrary, unique source- and destination-axis positions
// to a rectangular semantic component. Spans index the enclosing flattened
// arrays. Counts describe semantic membership and have no tile-shape meaning.
struct rectangular_component_membership_v1 {
    u32 component_id = invalid_semantic_component_id;
    u32 reserved = 0u;
    u64 source_member_offset = 0u;
    u64 source_member_count = 0u;
    u64 destination_member_offset = 0u;
    u64 destination_member_count = 0u;
    u64 support_reference_offset = 0u;
    u64 support_reference_count = 0u;
};

struct rectangular_support_view_v1 {
    u32 schema_version = rectangular_support_schema_version;
    u32 reserved = 0u;
    execution::axis_identity source_axis{};
    execution::axis_identity destination_axis{};
    const rectangular_component_membership_v1 *memberships = nullptr;
    u64 membership_count = 0u;
    const u32 *source_members = nullptr;
    u64 source_member_count = 0u;
    const u32 *destination_members = nullptr;
    u64 destination_member_count = 0u;
    const portable_support_reference_v1 *support_references = nullptr;
    u64 support_reference_count = 0u;
};

enum class rectangular_support_validation_code : u8 {
    ok = 0u,
    unsupported_version = 1u,
    nonzero_reserved = 2u,
    axis_mismatch = 3u,
    missing_memberships = 4u,
    missing_source_members = 5u,
    missing_destination_members = 6u,
    missing_support_references = 7u,
    invalid_component = 8u,
    duplicate_component = 9u,
    missing_rectangular_component = 10u,
    empty_axis_membership = 11u,
    member_span_out_of_bounds = 12u,
    duplicate_axis_member = 13u,
    support_span_out_of_bounds = 14u,
    invalid_support_reference = 15u
};

struct rectangular_support_validation_result {
    rectangular_support_validation_code code =
        rectangular_support_validation_code::ok;
    u64 membership_index = invalid_rectangular_support_index;
    u64 element_index = invalid_rectangular_support_index;

    constexpr explicit operator bool() const noexcept {
        return code == rectangular_support_validation_code::ok;
    }
};

namespace detail {

constexpr bool span_is_bounded(u64 offset, u64 count, u64 extent) noexcept {
    return offset <= extent && count <= extent - offset;
}

constexpr rectangular_support_validation_result rectangular_failure(
    rectangular_support_validation_code code,
    u64 membership_index = invalid_rectangular_support_index,
    u64 element_index = invalid_rectangular_support_index) noexcept {
    return {code, membership_index, element_index};
}

} // namespace detail

// Contract validation is allocation-free and independent of the compiler that
// produced the records. Axis extents remain relation-structure input and are
// intentionally not inferred from shapes or maximum member values here.
constexpr rectangular_support_validation_result validate_rectangular_support(
    const relation_cover_view_v1 &cover,
    const rectangular_support_view_v1 &support) noexcept {
    if (support.schema_version != rectangular_support_schema_version)
        return detail::rectangular_failure(
            rectangular_support_validation_code::unsupported_version);
    if (support.reserved != 0u)
        return detail::rectangular_failure(
            rectangular_support_validation_code::nonzero_reserved);
    if (!execution::same_axis_identity(support.source_axis, cover.source_axis)
        || !execution::same_axis_identity(
            support.destination_axis, cover.destination_axis))
        return detail::rectangular_failure(
            rectangular_support_validation_code::axis_mismatch);
    if (support.membership_count != 0u && support.memberships == nullptr)
        return detail::rectangular_failure(
            rectangular_support_validation_code::missing_memberships);
    if (support.source_member_count != 0u && support.source_members == nullptr)
        return detail::rectangular_failure(
            rectangular_support_validation_code::missing_source_members);
    if (support.destination_member_count != 0u
        && support.destination_members == nullptr)
        return detail::rectangular_failure(
            rectangular_support_validation_code::missing_destination_members);
    if (support.support_reference_count != 0u
        && support.support_references == nullptr)
        return detail::rectangular_failure(
            rectangular_support_validation_code::missing_support_references);

    for (u64 membership_index = 0u;
         membership_index < support.membership_count; ++membership_index) {
        const rectangular_component_membership_v1 &membership =
            support.memberships[membership_index];
        if (membership.reserved != 0u)
            return detail::rectangular_failure(
                rectangular_support_validation_code::nonzero_reserved,
                membership_index);

        bool found_component = false;
        for (u32 component_index = 0u;
             component_index < cover.component_count; ++component_index)
            if (cover.components != nullptr
                && cover.components[component_index].component_id
                    == membership.component_id
                && cover.components[component_index].kind
                    == semantic_component_kind::rectangular)
                found_component = true;
        if (!found_component)
            return detail::rectangular_failure(
                rectangular_support_validation_code::invalid_component,
                membership_index);
        for (u64 previous = 0u; previous < membership_index; ++previous)
            if (support.memberships[previous].component_id
                == membership.component_id)
                return detail::rectangular_failure(
                    rectangular_support_validation_code::duplicate_component,
                    membership_index, previous);

        if (membership.source_member_count == 0u
            || membership.destination_member_count == 0u)
            return detail::rectangular_failure(
                rectangular_support_validation_code::empty_axis_membership,
                membership_index);
        if (!detail::span_is_bounded(membership.source_member_offset,
                membership.source_member_count, support.source_member_count)
            || !detail::span_is_bounded(membership.destination_member_offset,
                membership.destination_member_count,
                support.destination_member_count))
            return detail::rectangular_failure(
                rectangular_support_validation_code::member_span_out_of_bounds,
                membership_index);
        if (!detail::span_is_bounded(membership.support_reference_offset,
                membership.support_reference_count,
                support.support_reference_count))
            return detail::rectangular_failure(
                rectangular_support_validation_code::support_span_out_of_bounds,
                membership_index);

        for (u64 index = 0u; index < membership.source_member_count; ++index)
            for (u64 previous = 0u; previous < index; ++previous)
                if (support.source_members[membership.source_member_offset + index]
                    == support.source_members[
                        membership.source_member_offset + previous])
                    return detail::rectangular_failure(
                        rectangular_support_validation_code::duplicate_axis_member,
                        membership_index, index);
        for (u64 index = 0u; index < membership.destination_member_count;
             ++index)
            for (u64 previous = 0u; previous < index; ++previous)
                if (support.destination_members[
                        membership.destination_member_offset + index]
                    == support.destination_members[
                        membership.destination_member_offset + previous])
                    return detail::rectangular_failure(
                        rectangular_support_validation_code::duplicate_axis_member,
                        membership_index, index);

        for (u64 index = 0u; index < membership.support_reference_count;
             ++index) {
            const portable_support_reference_v1 &reference =
                support.support_references[
                    membership.support_reference_offset + index];
            if (reference.evidence_identity == 0u
                || reference.evidence_kind == 0u || reference.reserved != 0u
                || reference.record_count == 0u
                || reference.record_count
                    > ~u64{0u} - reference.record_offset)
                return detail::rectangular_failure(
                    rectangular_support_validation_code::invalid_support_reference,
                    membership_index, index);
        }
    }

    for (u32 component_index = 0u; component_index < cover.component_count;
         ++component_index) {
        if (cover.components == nullptr
            || cover.components[component_index].kind
                != semantic_component_kind::rectangular)
            continue;
        bool found_membership = false;
        for (u64 membership_index = 0u;
             membership_index < support.membership_count; ++membership_index)
            if (support.memberships[membership_index].component_id
                == cover.components[component_index].component_id)
                found_membership = true;
        if (!found_membership)
            return detail::rectangular_failure(
                rectangular_support_validation_code::missing_rectangular_component,
                component_index);
    }
    return {};
}

static_assert(
    std::is_trivially_copyable<portable_support_reference_v1>::value,
    "portable support references must remain pointer-copyable");
static_assert(
    std::is_trivially_copyable<rectangular_component_membership_v1>::value,
    "rectangular memberships must remain pointer-copyable");
static_assert(std::is_trivially_copyable<rectangular_support_view_v1>::value,
    "rectangular support views must remain pointer-copyable");

} // namespace cellerator::geometry
