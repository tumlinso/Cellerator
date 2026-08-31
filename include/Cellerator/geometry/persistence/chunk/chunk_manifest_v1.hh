#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <Cellerator/execution/index_space/hierarchical_index_space_v1.hh>

namespace cellerator::geometry::persistence {

inline constexpr std::uint64_t chunk_manifest_magic_v1 = 0x314b4e4843454343ull;
inline constexpr std::uint16_t chunk_manifest_version_v1 = 1u;

enum class chunk_payload_domain_v1 : std::uint8_t {
    semantic = 0u,
    physical = 1u,
};

// Persisted extension header.  Records immediately follow header_bytes from
// the start of the extension section; there are no persisted pointers.
struct chunk_manifest_header_v1 {
    std::uint64_t magic = chunk_manifest_magic_v1;
    std::uint64_t manifest_identity = 0u;
    std::uint64_t aggregate_element_count = 0u;
    std::uint64_t chunk_count = 0u;
    std::uint16_t version = chunk_manifest_version_v1;
    std::uint16_t header_bytes = sizeof(chunk_manifest_header_v1);
    std::uint16_t record_bytes = 0u;
    std::uint16_t flags = 0u;
};

// One independently bounded payload.  section_table_position indexes the
// caller's CSG1/CPE2 extension-directory view.  section_identity prevents a
// record from silently rebinding to a different section after relocation.
struct chunk_manifest_record_v1 {
    std::uint64_t chunk_identity = 0u;
    std::uint64_t component_identity = 0u;
    std::uint64_t aggregate_begin = 0u;
    std::uint64_t local_element_count = 0u;
    std::uint64_t section_identity = 0u;
    std::uint64_t section_byte_offset = 0u;
    std::uint64_t section_byte_count = 0u;
    std::uint64_t payload_checksum = 0u;
    std::uint32_t section_table_position = 0u;
    std::uint32_t element_stride = 0u;
    execution::local_index_width_v1 local_width =
        execution::local_index_width_v1::u32;
    chunk_payload_domain_v1 domain = chunk_payload_domain_v1::semantic;
    std::uint8_t alignment_log2 = 0u;
    std::uint8_t reserved = 0u;
};

// Relocated, non-owning directory entry supplied by the containing CSG1/CPE2
// image.  It is deliberately generic and does not define a new top-level image.
struct chunk_section_extent_v1 {
    std::uint64_t section_identity = 0u;
    std::uint64_t byte_count = 0u;
};

struct chunk_manifest_view_v1 {
    const chunk_manifest_header_v1 *header = nullptr;
    const chunk_manifest_record_v1 *records = nullptr;
};

enum class chunk_manifest_status_v1 : std::uint32_t {
    valid = 0u,
    null_pointer,
    truncated,
    invalid_magic,
    invalid_version,
    invalid_record_size,
    arithmetic_overflow,
    chunk_order,
    aggregate_discontinuity,
    aggregate_extent_mismatch,
    invalid_width,
    invalid_domain,
    invalid_alignment,
    invalid_stride,
    section_out_of_range,
    section_identity_mismatch,
    section_bounds,
};

struct chunk_manifest_validation_v1 {
    chunk_manifest_status_v1 status = chunk_manifest_status_v1::valid;
    std::uint32_t reserved = 0u;
    std::uint64_t chunk = 0u;
    std::uint64_t operations = 0u;
};

bool chunk_manifest_required_bytes_v1(std::uint64_t chunk_count,
                                      std::uint64_t *out) noexcept;

chunk_manifest_status_v1 bind_chunk_manifest_v1(
    const void *data, std::uint64_t bytes,
    chunk_manifest_view_v1 *out) noexcept;

chunk_manifest_validation_v1 validate_chunk_manifest_v1(
    const chunk_manifest_view_v1 &manifest,
    const chunk_section_extent_v1 *sections,
    std::uint64_t section_count) noexcept;

static_assert(std::is_trivially_copyable_v<chunk_manifest_header_v1>);
static_assert(std::is_standard_layout_v<chunk_manifest_header_v1>);
static_assert(std::is_trivially_copyable_v<chunk_manifest_record_v1>);
static_assert(std::is_standard_layout_v<chunk_manifest_record_v1>);

}  // namespace cellerator::geometry::persistence
