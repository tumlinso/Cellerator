#include <Cellerator/execution/joint_compiler/logical_coverage_v1.hh>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::execution::joint_compiler {
namespace {

logical_coverage_validation_result_v1 failure(
    logical_coverage_validation_code_v1 code,
    std::uint64_t member_index = 0u) noexcept {
    return {code, member_index};
}

bool valid_kind(logical_coverage_kind_v1 kind) noexcept {
    return kind >= logical_coverage_kind_v1::canonical_intervals
        && kind <= logical_coverage_kind_v1::provider_defined;
}

std::uint64_t expected_member_bytes(logical_coverage_kind_v1 kind) noexcept {
    switch (kind) {
        case logical_coverage_kind_v1::canonical_intervals:
            return sizeof(canonical_interval_v1);
        case logical_coverage_kind_v1::explicit_ids:
        case logical_coverage_kind_v1::relation_edge_ids:
            return sizeof(std::uint64_t);
        case logical_coverage_kind_v1::semantic_components:
            return sizeof(semantic_component_reference_v1);
        case logical_coverage_kind_v1::segment_set:
            return sizeof(segment_reference_v1);
        case logical_coverage_kind_v1::coverage_union:
            return sizeof(coverage_union_reference_v1);
        case logical_coverage_kind_v1::provider_defined:
            return 0u;
    }
    return 0u;
}

std::size_t expected_alignment(logical_coverage_kind_v1 kind) noexcept {
    switch (kind) {
        case logical_coverage_kind_v1::canonical_intervals:
            return alignof(canonical_interval_v1);
        case logical_coverage_kind_v1::explicit_ids:
        case logical_coverage_kind_v1::relation_edge_ids:
            return alignof(std::uint64_t);
        case logical_coverage_kind_v1::semantic_components:
            return alignof(semantic_component_reference_v1);
        case logical_coverage_kind_v1::segment_set:
            return alignof(segment_reference_v1);
        case logical_coverage_kind_v1::coverage_union:
            return alignof(coverage_union_reference_v1);
        case logical_coverage_kind_v1::provider_defined:
            return 1u;
    }
    return 1u;
}

bool identity_is_zero(persistent_identity_v1 identity) noexcept {
    return identity.producer_namespace == 0u && identity.local_identity == 0u;
}

bool identity_less(
    persistent_identity_v1 lhs, persistent_identity_v1 rhs) noexcept {
    return lhs.producer_namespace < rhs.producer_namespace
        || (lhs.producer_namespace == rhs.producer_namespace
            && lhs.local_identity < rhs.local_identity);
}

}  // namespace

logical_coverage_validation_result_v1 validate_logical_coverage_v1(
    const logical_coverage_view_v1 &coverage) noexcept {
    if (coverage.schema_version != logical_coverage_schema_version_v1)
        return failure(
            logical_coverage_validation_code_v1::unsupported_schema);
    if (coverage.record_bytes != sizeof(logical_coverage_view_v1))
        return failure(
            logical_coverage_validation_code_v1::invalid_record_bytes);
    if (coverage.reserved != 0u)
        return failure(logical_coverage_validation_code_v1::nonzero_reserved);
    if (!validate_persistent_identity_v1(coverage.coverage_identity))
        return failure(
            logical_coverage_validation_code_v1::invalid_coverage_identity);
    if (!valid_kind(coverage.kind))
        return failure(logical_coverage_validation_code_v1::invalid_kind);
    if ((coverage.role_flags & certified_exact_coverage_role_v1) == 0u)
        return failure(logical_coverage_validation_code_v1::missing_exact_role);
    if (!validate_coverage_role_flags_v1(coverage.role_flags))
        return failure(
            logical_coverage_validation_code_v1::invalid_role_combination);
    if (!valid_identity(coverage.structure))
        return failure(logical_coverage_validation_code_v1::invalid_structure);
    if (coverage.epoch.value == 0u)
        return failure(
            logical_coverage_validation_code_v1::invalid_structure_epoch);
    if (validate_persistent_axis_identity(coverage.source_axis)
        != biological_validation_code::ok)
        return failure(logical_coverage_validation_code_v1::invalid_source_axis);
    if (validate_persistent_axis_identity(coverage.destination_axis)
        != biological_validation_code::ok)
        return failure(
            logical_coverage_validation_code_v1::invalid_destination_axis);
    if (coverage.logical_count == 0u || coverage.member_count == 0u)
        return failure(logical_coverage_validation_code_v1::empty_coverage);
    if (coverage.members == nullptr)
        return failure(logical_coverage_validation_code_v1::missing_members);
    if (coverage.member_bytes == 0u)
        return failure(
            logical_coverage_validation_code_v1::invalid_member_bytes);
    if (coverage.member_count
        > std::numeric_limits<std::uint64_t>::max() / coverage.member_bytes)
        return failure(
            logical_coverage_validation_code_v1::member_bytes_overflow);

    const std::uint64_t built_in_bytes = expected_member_bytes(coverage.kind);
    if (coverage.kind == logical_coverage_kind_v1::provider_defined) {
        if (!validate_persistent_identity_v1(coverage.payload_schema))
            return failure(
                logical_coverage_validation_code_v1::missing_payload_schema);
        return {};
    }
    if (!identity_is_zero(coverage.payload_schema))
        return failure(
            logical_coverage_validation_code_v1::unexpected_payload_schema);
    if (coverage.member_bytes != built_in_bytes)
        return failure(
            logical_coverage_validation_code_v1::invalid_member_bytes);
    if (reinterpret_cast<std::uintptr_t>(coverage.members)
        % expected_alignment(coverage.kind) != 0u)
        return failure(logical_coverage_validation_code_v1::misaligned_members);

    if (coverage.kind == logical_coverage_kind_v1::canonical_intervals) {
        const auto *members =
            static_cast<const canonical_interval_v1 *>(coverage.members);
        std::uint64_t logical_count = 0u;
        std::uint64_t previous_end = 0u;
        for (std::uint64_t index = 0u; index < coverage.member_count; ++index) {
            const canonical_interval_v1 member = members[index];
            if (member.count == 0u
                || member.begin > std::numeric_limits<std::uint64_t>::max()
                    - member.count)
                return failure(
                    logical_coverage_validation_code_v1::empty_member, index);
            if (index != 0u && member.begin < previous_end)
                return failure(logical_coverage_validation_code_v1::
                    unordered_or_overlapping_members, index);
            if (logical_count
                > std::numeric_limits<std::uint64_t>::max() - member.count)
                return failure(logical_coverage_validation_code_v1::
                    logical_count_mismatch, index);
            logical_count += member.count;
            previous_end = member.begin + member.count;
        }
        if (logical_count != coverage.logical_count)
            return failure(
                logical_coverage_validation_code_v1::logical_count_mismatch);
        return {};
    }

    if (coverage.kind == logical_coverage_kind_v1::explicit_ids
        || coverage.kind == logical_coverage_kind_v1::relation_edge_ids) {
        if (coverage.member_count != coverage.logical_count)
            return failure(
                logical_coverage_validation_code_v1::logical_count_mismatch);
        const auto *members = static_cast<const std::uint64_t *>(coverage.members);
        for (std::uint64_t index = 1u; index < coverage.member_count; ++index)
            if (members[index] <= members[index - 1u])
                return failure(
                    members[index] == members[index - 1u]
                        ? logical_coverage_validation_code_v1::duplicate_member
                        : logical_coverage_validation_code_v1::
                            unordered_or_overlapping_members,
                    index);
        return {};
    }

    if (coverage.kind == logical_coverage_kind_v1::semantic_components) {
        if (coverage.member_count != coverage.logical_count)
            return failure(
                logical_coverage_validation_code_v1::logical_count_mismatch);
        const auto *members = static_cast<
            const semantic_component_reference_v1 *>(coverage.members);
        for (std::uint64_t index = 0u; index < coverage.member_count; ++index) {
            if (!validate_persistent_identity_v1(members[index].cover_identity)
                || members[index].component_identity == 0u)
                return failure(logical_coverage_validation_code_v1::
                    invalid_member_identity, index);
            if (index != 0u) {
                const bool same_cover = same_persistent_identity_v1(
                    members[index].cover_identity,
                    members[index - 1u].cover_identity);
                if ((!same_cover
                        && !identity_less(members[index - 1u].cover_identity,
                            members[index].cover_identity))
                    || (same_cover && members[index].component_identity
                        <= members[index - 1u].component_identity))
                    return failure(logical_coverage_validation_code_v1::
                        unordered_or_overlapping_members, index);
            }
        }
        return {};
    }

    if (coverage.kind == logical_coverage_kind_v1::segment_set) {
        const auto *members =
            static_cast<const segment_reference_v1 *>(coverage.members);
        std::uint64_t logical_count = 0u;
        for (std::uint64_t index = 0u; index < coverage.member_count; ++index) {
            const segment_reference_v1 member = members[index];
            if (!validate_persistent_identity_v1(member.segment_space)
                || member.segment_count == 0u
                || member.first_segment
                    > std::numeric_limits<std::uint64_t>::max()
                        - member.segment_count)
                return failure(logical_coverage_validation_code_v1::
                    invalid_member_identity, index);
            if (logical_count > std::numeric_limits<std::uint64_t>::max()
                    - member.segment_count)
                return failure(logical_coverage_validation_code_v1::
                    logical_count_mismatch, index);
            logical_count += member.segment_count;
            if (index != 0u) {
                const segment_reference_v1 previous = members[index - 1u];
                const bool same_space = same_persistent_identity_v1(
                    member.segment_space, previous.segment_space);
                if ((!same_space
                        && !identity_less(previous.segment_space,
                            member.segment_space))
                    || (same_space && member.first_segment
                        < previous.first_segment + previous.segment_count))
                    return failure(logical_coverage_validation_code_v1::
                        unordered_or_overlapping_members, index);
            }
        }
        if (logical_count != coverage.logical_count)
            return failure(
                logical_coverage_validation_code_v1::logical_count_mismatch);
        return {};
    }

    const auto *members =
        static_cast<const coverage_union_reference_v1 *>(coverage.members);
    for (std::uint64_t index = 0u; index < coverage.member_count; ++index) {
        if (!validate_persistent_identity_v1(members[index].coverage_identity))
            return failure(
                logical_coverage_validation_code_v1::invalid_member_identity,
                index);
        if (same_persistent_identity_v1(
                members[index].coverage_identity, coverage.coverage_identity))
            return failure(
                logical_coverage_validation_code_v1::recursive_union, index);
        if (index != 0u) {
            if (same_persistent_identity_v1(members[index].coverage_identity,
                    members[index - 1u].coverage_identity))
                return failure(logical_coverage_validation_code_v1::
                    duplicate_member, index);
            if (!identity_less(members[index - 1u].coverage_identity,
                    members[index].coverage_identity))
                return failure(logical_coverage_validation_code_v1::
                    unordered_or_overlapping_members, index);
        }
    }
    return {};
}

}  // namespace cellerator::execution::joint_compiler
