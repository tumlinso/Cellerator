#pragma once

#include <Cellerator/execution/joint_compiler/persistent_identity_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::joint_compiler {

inline constexpr std::uint32_t logical_coverage_schema_version_v1 = 1u;
inline constexpr std::uint32_t certified_exact_coverage_role_v1 = 1u << 0u;

enum class logical_coverage_kind_v1 : std::uint16_t {
    canonical_intervals = 1u,
    explicit_ids = 2u,
    relation_edge_ids = 3u,
    semantic_components = 4u,
    segment_set = 5u,
    coverage_union = 6u,
    provider_defined = 7u
};

struct canonical_interval_v1 {
    std::uint64_t begin = 0u;
    std::uint64_t count = 0u;
};

struct semantic_component_reference_v1 {
    persistent_identity_v1 cover_identity{};
    std::uint64_t component_identity = 0u;
};

struct segment_reference_v1 {
    persistent_identity_v1 segment_space{};
    std::uint64_t first_segment = 0u;
    std::uint64_t segment_count = 0u;
};

struct coverage_union_reference_v1 {
    persistent_identity_v1 coverage_identity{};
};

// A cold, non-owning semantic view. Built-in membership arrays have the record
// type selected by kind. provider_defined arrays are interpreted only through
// payload_schema. member_bytes describes one record; no storage is allocated,
// copied, canonicalized, or inferred by validation.
struct logical_coverage_view_v1 {
    std::uint32_t schema_version = logical_coverage_schema_version_v1;
    std::uint32_t record_bytes = sizeof(logical_coverage_view_v1);
    persistent_identity_v1 coverage_identity{};
    logical_coverage_kind_v1 kind =
        logical_coverage_kind_v1::canonical_intervals;
    std::uint16_t reserved = 0u;
    std::uint32_t role_flags = certified_exact_coverage_role_v1;
    structure_id structure{};
    structure_epoch epoch{};
    persistent_axis_identity source_axis{};
    persistent_axis_identity destination_axis{};
    std::uint64_t logical_count = 0u;
    persistent_identity_v1 payload_schema{};
    const void *members = nullptr;
    std::uint64_t member_count = 0u;
    std::uint64_t member_bytes = 0u;
};

enum class logical_coverage_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema = 1u,
    invalid_record_bytes = 2u,
    nonzero_reserved = 3u,
    invalid_coverage_identity = 4u,
    invalid_kind = 5u,
    missing_exact_role = 6u,
    invalid_structure = 7u,
    invalid_structure_epoch = 8u,
    invalid_source_axis = 9u,
    invalid_destination_axis = 10u,
    empty_coverage = 11u,
    missing_members = 12u,
    invalid_member_bytes = 13u,
    member_bytes_overflow = 14u,
    misaligned_members = 15u,
    unexpected_payload_schema = 16u,
    missing_payload_schema = 17u,
    empty_member = 18u,
    unordered_or_overlapping_members = 19u,
    duplicate_member = 20u,
    logical_count_mismatch = 21u,
    invalid_member_identity = 22u,
    recursive_union = 23u
};

struct logical_coverage_validation_result_v1 {
    logical_coverage_validation_code_v1 code =
        logical_coverage_validation_code_v1::ok;
    std::uint64_t member_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == logical_coverage_validation_code_v1::ok;
    }
};

logical_coverage_validation_result_v1 validate_logical_coverage_v1(
    const logical_coverage_view_v1 &coverage) noexcept;

static_assert(std::is_standard_layout_v<canonical_interval_v1>);
static_assert(std::is_trivially_copyable_v<canonical_interval_v1>);
static_assert(std::is_standard_layout_v<semantic_component_reference_v1>);
static_assert(std::is_trivially_copyable_v<semantic_component_reference_v1>);
static_assert(std::is_standard_layout_v<segment_reference_v1>);
static_assert(std::is_trivially_copyable_v<segment_reference_v1>);
static_assert(std::is_standard_layout_v<coverage_union_reference_v1>);
static_assert(std::is_trivially_copyable_v<coverage_union_reference_v1>);
static_assert(std::is_standard_layout_v<logical_coverage_view_v1>);
static_assert(std::is_trivially_copyable_v<logical_coverage_view_v1>);

}  // namespace cellerator::execution::joint_compiler
