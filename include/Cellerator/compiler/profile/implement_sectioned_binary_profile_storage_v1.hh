#pragma once

#include <Cellerator/compiler/profile/freeze_the_profile_artifact_charter_and_name_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::profile::v1 {

inline constexpr std::uint32_t profile_section_entry_bytes_v1 = 64u;

enum class profile_section_kind_v1 : std::uint32_t {
    named_environment = 1u,
    structural_evidence = 2u,
    value_evidence = 3u,
    reuse_evidence = 4u,
    provenance = 5u,
    semantic_attachment = 6u,
    extension = 0x80000000u
};

enum profile_section_flags_v1 : std::uint32_t {
    profile_section_flag_none = 0u,
    profile_section_flag_optional = 1u << 0u
};

enum class profile_compression_v1 : std::uint32_t {
    none = 0u,
    zstd = 1u,
    application_defined = 0x80000000u
};

struct profile_section_entry_v1 {
    std::uint32_t kind = 0u;
    std::uint32_t schema_version = 0u;
    std::uint32_t flags = profile_section_flag_none;
    profile_compression_v1 compression = profile_compression_v1::none;
    std::uint64_t identity_low = 0u;
    std::uint64_t identity_high = 0u;
    std::uint64_t offset = 0u;
    std::uint64_t stored_bytes = 0u;
    std::uint64_t logical_bytes = 0u;
    std::uint64_t checksum = 0u;
};

struct profile_section_source_v1 {
    std::uint32_t kind = 0u;
    std::uint32_t schema_version = 0u;
    std::uint32_t flags = profile_section_flag_none;
    profile_compression_v1 compression = profile_compression_v1::none;
    std::uint64_t identity_low = 0u;
    std::uint64_t identity_high = 0u;
    const void *data = nullptr;
    std::uint64_t stored_bytes = 0u;
    std::uint64_t logical_bytes = 0u;
};

struct profile_artifact_build_request_v1 {
    profile_artifact_header_v1 header{};
    const profile_section_source_v1 *sections = nullptr;
    std::uint32_t section_count = 0u;
};

struct profile_artifact_requirements_v1 {
    std::uint64_t image_bytes = 0u;
    std::uint64_t directory_bytes = 0u;
    std::uint64_t payload_bytes = 0u;
    std::uint64_t padding_bytes = 0u;
};

struct profile_artifact_buffer_v1 {
    void *data = nullptr;
    std::uint64_t capacity_bytes = 0u;
};

// A view is a validated non-owning mapping. Offsets in the artifact remain
// pointer-free and relocation-safe; the view may refer directly to mmap data.
struct profile_artifact_view_v1 {
    const void *image_base = nullptr;
    std::uint64_t image_bytes = 0u;
    profile_artifact_header_v1 header{};
    const profile_section_entry_v1 *sections = nullptr;
};

struct profile_section_view_v1 {
    profile_section_entry_v1 entry{};
    const void *stored_data = nullptr;
};

enum class profile_storage_status_v1 : std::uint8_t {
    ok = 0u,
    invalid_argument,
    invalid_header,
    arithmetic_overflow,
    insufficient_capacity,
    misaligned_image,
    invalid_directory,
    duplicate_identity,
    unknown_required_section,
    invalid_compression,
    section_out_of_bounds,
    section_checksum_mismatch,
    image_checksum_mismatch,
    section_not_found
};

profile_storage_status_v1 query_profile_artifact_requirements_v1(
    const profile_artifact_build_request_v1 &request,
    profile_artifact_requirements_v1 *requirements) noexcept;

profile_storage_status_v1 build_profile_artifact_v1(
    const profile_artifact_build_request_v1 &request,
    profile_artifact_buffer_v1 buffer,
    profile_artifact_view_v1 *view) noexcept;

profile_storage_status_v1 validate_profile_artifact_v1(
    const void *image, std::uint64_t available_bytes,
    profile_artifact_view_v1 *view) noexcept;

profile_storage_status_v1 find_profile_section_v1(
    const profile_artifact_view_v1 &artifact, std::uint32_t kind,
    profile_section_view_v1 *section) noexcept;

std::uint64_t profile_storage_checksum_v1(
    const void *data, std::uint64_t bytes) noexcept;

static_assert(sizeof(profile_section_entry_v1) == profile_section_entry_bytes_v1);
static_assert(std::is_standard_layout_v<profile_section_entry_v1>);
static_assert(std::is_trivially_copyable_v<profile_section_entry_v1>);

}  // namespace cellerator::compiler::profile::v1
