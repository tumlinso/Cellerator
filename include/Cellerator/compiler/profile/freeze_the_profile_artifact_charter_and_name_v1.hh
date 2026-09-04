#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::profile::v1 {

inline constexpr unsigned char profile_artifact_magic_v1[8] = {
    'C', 'E', 'L', 'L', 'P', 'R', 'F', '1'};
inline constexpr char profile_artifact_name_v1[] = "Cellerator Profile Artifact v1";
inline constexpr char profile_artifact_suffix_v1[] = ".ceprofile";
inline constexpr std::uint32_t profile_artifact_schema_version_v1 = 1u;
inline constexpr std::uint32_t profile_artifact_endian_marker_v1 = 0x01020304u;
inline constexpr std::uint32_t profile_artifact_alignment_v1 = 64u;
inline constexpr std::uint32_t profile_artifact_header_bytes_v1 = 112u;

enum profile_artifact_flags_v1 : std::uint64_t {
    profile_artifact_flag_none = 0u,
    profile_artifact_flag_data_derived = 1ull << 0u,
    profile_artifact_flag_external_evidence = 1ull << 1u,
    profile_artifact_flag_compressed_sections = 1ull << 2u
};

// Cold, pointer-free container identity. A profile records representative
// semantic state and its evidence. It never carries compiler policy, runtime
// pointers, or a claim that representative statistics establish correctness.
struct profile_artifact_header_v1 {
    unsigned char magic[8]{};
    std::uint32_t schema_version = profile_artifact_schema_version_v1;
    std::uint32_t header_bytes = profile_artifact_header_bytes_v1;
    std::uint32_t endian = profile_artifact_endian_marker_v1;
    std::uint32_t alignment = profile_artifact_alignment_v1;
    std::uint64_t image_bytes = profile_artifact_header_bytes_v1;
    std::uint64_t artifact_identity_low = 0u;
    std::uint64_t artifact_identity_high = 0u;
    std::uint64_t semantic_environment_identity_low = 0u;
    std::uint64_t semantic_environment_identity_high = 0u;
    std::uint64_t evidence_revision = 0u;
    std::uint64_t flags = profile_artifact_flag_data_derived;
    std::uint32_t section_count = 0u;
    std::uint32_t section_entry_bytes = 0u;
    std::uint64_t section_directory_offset = 0u;
    std::uint64_t content_checksum = 0u;
    std::uint64_t reserved = 0u;
};

enum class profile_artifact_charter_status_v1 : std::uint8_t {
    ok = 0u,
    invalid_magic,
    unsupported_schema,
    invalid_header_size,
    unsupported_endian,
    invalid_alignment,
    truncated_image,
    missing_artifact_identity,
    missing_environment_identity,
    not_data_derived,
    reserved_field_nonzero
};

profile_artifact_header_v1 make_profile_artifact_header_v1(
    std::uint64_t artifact_identity_low,
    std::uint64_t artifact_identity_high,
    std::uint64_t semantic_environment_identity_low,
    std::uint64_t semantic_environment_identity_high,
    std::uint64_t evidence_revision) noexcept;

profile_artifact_charter_status_v1 validate_profile_artifact_charter_v1(
    const profile_artifact_header_v1 &header) noexcept;

static_assert(sizeof(profile_artifact_header_v1) == profile_artifact_header_bytes_v1);
static_assert(std::is_standard_layout_v<profile_artifact_header_v1>);
static_assert(std::is_trivially_copyable_v<profile_artifact_header_v1>);

}  // namespace cellerator::compiler::profile::v1
