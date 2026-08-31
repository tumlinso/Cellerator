#include <Cellerator/geometry/persistence/chunk/chunk_manifest_v1.hh>

#include <limits>

namespace cellerator::geometry::persistence {
namespace {

chunk_manifest_validation_v1 failure(chunk_manifest_status_v1 status,
                                     std::uint64_t chunk,
                                     std::uint64_t operations) noexcept {
    chunk_manifest_validation_v1 result{};
    result.status = status;
    result.chunk = chunk;
    result.operations = operations;
    return result;
}

bool checked_add(std::uint64_t left, std::uint64_t right,
                 std::uint64_t *out) noexcept {
    if (out == nullptr || right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    *out = left + right;
    return true;
}

bool checked_multiply(std::uint64_t left, std::uint64_t right,
                      std::uint64_t *out) noexcept {
    if (out == nullptr
        || (left != 0u && right > std::numeric_limits<std::uint64_t>::max() / left)) {
        return false;
    }
    *out = left * right;
    return true;
}

bool width_valid(execution::local_index_width_v1 width,
                 std::uint64_t extent) noexcept {
    switch (width) {
        case execution::local_index_width_v1::u16:
            return extent <= (std::uint64_t{1} << 16u);
        case execution::local_index_width_v1::u32:
            return extent <= (std::uint64_t{1} << 32u);
        case execution::local_index_width_v1::u64:
            return true;
    }
    return false;
}

}  // namespace

bool chunk_manifest_required_bytes_v1(std::uint64_t chunk_count,
                                      std::uint64_t *out) noexcept {
    std::uint64_t records = 0u;
    return checked_multiply(chunk_count, sizeof(chunk_manifest_record_v1), &records)
        && checked_add(sizeof(chunk_manifest_header_v1), records, out);
}

chunk_manifest_status_v1 bind_chunk_manifest_v1(
    const void *data, std::uint64_t bytes,
    chunk_manifest_view_v1 *out) noexcept {
    if (data == nullptr || out == nullptr) {
        return chunk_manifest_status_v1::null_pointer;
    }
    if (bytes < sizeof(chunk_manifest_header_v1)) {
        return chunk_manifest_status_v1::truncated;
    }
    const auto *header = static_cast<const chunk_manifest_header_v1 *>(data);
    if (header->magic != chunk_manifest_magic_v1) {
        return chunk_manifest_status_v1::invalid_magic;
    }
    if (header->version != chunk_manifest_version_v1) {
        return chunk_manifest_status_v1::invalid_version;
    }
    if (header->header_bytes < sizeof(chunk_manifest_header_v1)
        || header->record_bytes != sizeof(chunk_manifest_record_v1)) {
        return chunk_manifest_status_v1::invalid_record_size;
    }
    if (header->header_bytes % alignof(chunk_manifest_record_v1) != 0u) {
        return chunk_manifest_status_v1::invalid_alignment;
    }
    std::uint64_t records_bytes = 0u;
    std::uint64_t required = 0u;
    if (!checked_multiply(header->chunk_count, header->record_bytes, &records_bytes)
        || !checked_add(header->header_bytes, records_bytes, &required)) {
        return chunk_manifest_status_v1::arithmetic_overflow;
    }
    if (required > bytes) {
        return chunk_manifest_status_v1::truncated;
    }
    const auto *base = static_cast<const std::uint8_t *>(data);
    out->header = header;
    out->records = reinterpret_cast<const chunk_manifest_record_v1 *>(
        base + header->header_bytes);
    return chunk_manifest_status_v1::valid;
}

chunk_manifest_validation_v1 validate_chunk_manifest_v1(
    const chunk_manifest_view_v1 &manifest,
    const chunk_section_extent_v1 *sections,
    std::uint64_t section_count) noexcept {
    std::uint64_t operations = 0u;
    if (manifest.header == nullptr
        || (manifest.header->chunk_count != 0u && manifest.records == nullptr)
        || (section_count != 0u && sections == nullptr)) {
        return failure(chunk_manifest_status_v1::null_pointer, 0u, operations);
    }
    const auto &header = *manifest.header;
    if (header.magic != chunk_manifest_magic_v1) {
        return failure(chunk_manifest_status_v1::invalid_magic, 0u, operations);
    }
    if (header.version != chunk_manifest_version_v1) {
        return failure(chunk_manifest_status_v1::invalid_version, 0u, operations);
    }
    if (header.record_bytes != sizeof(chunk_manifest_record_v1)) {
        return failure(chunk_manifest_status_v1::invalid_record_size, 0u, operations);
    }

    std::uint64_t aggregate = 0u;
    std::uint64_t previous_chunk_identity = 0u;
    for (std::uint64_t chunk = 0u; chunk < header.chunk_count; ++chunk) {
        const auto &record = manifest.records[chunk];
        ++operations;
        if (chunk != 0u && record.chunk_identity <= previous_chunk_identity) {
            return failure(chunk_manifest_status_v1::chunk_order, chunk, operations);
        }
        if (record.aggregate_begin != aggregate) {
            return failure(chunk_manifest_status_v1::aggregate_discontinuity,
                           chunk, operations);
        }
        if (!width_valid(record.local_width, record.local_element_count)) {
            return failure(chunk_manifest_status_v1::invalid_width, chunk, operations);
        }
        if (record.domain != chunk_payload_domain_v1::semantic
            && record.domain != chunk_payload_domain_v1::physical) {
            return failure(chunk_manifest_status_v1::invalid_domain, chunk, operations);
        }
        if (record.alignment_log2 >= 64u) {
            return failure(chunk_manifest_status_v1::invalid_alignment, chunk, operations);
        }
        const std::uint64_t alignment = std::uint64_t{1} << record.alignment_log2;
        if ((record.section_byte_offset & (alignment - 1u)) != 0u) {
            return failure(chunk_manifest_status_v1::invalid_alignment, chunk, operations);
        }
        if (record.local_element_count != 0u && record.element_stride == 0u) {
            return failure(chunk_manifest_status_v1::invalid_stride, chunk, operations);
        }
        std::uint64_t payload_bytes = 0u;
        if (!checked_multiply(record.local_element_count, record.element_stride,
                              &payload_bytes)
            || payload_bytes > record.section_byte_count) {
            return failure(chunk_manifest_status_v1::invalid_stride, chunk, operations);
        }
        if (record.section_table_position >= section_count) {
            return failure(chunk_manifest_status_v1::section_out_of_range,
                           chunk, operations);
        }
        const auto &section = sections[record.section_table_position];
        if (section.section_identity != record.section_identity) {
            return failure(chunk_manifest_status_v1::section_identity_mismatch,
                           chunk, operations);
        }
        std::uint64_t section_end = 0u;
        if (!checked_add(record.section_byte_offset, record.section_byte_count,
                         &section_end)
            || section_end > section.byte_count) {
            return failure(chunk_manifest_status_v1::section_bounds, chunk, operations);
        }
        if (!checked_add(aggregate, record.local_element_count, &aggregate)) {
            return failure(chunk_manifest_status_v1::arithmetic_overflow,
                           chunk, operations);
        }
        previous_chunk_identity = record.chunk_identity;
    }
    if (aggregate != header.aggregate_element_count) {
        return failure(chunk_manifest_status_v1::aggregate_extent_mismatch,
                       header.chunk_count, operations);
    }
    chunk_manifest_validation_v1 result{};
    result.operations = operations;
    return result;
}

}  // namespace cellerator::geometry::persistence
