#include "CellPack/persistence/execution_image_v2.hh"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>

namespace cellpack::persistence {
namespace {

constexpr unsigned char image_magic[8] = {'C','E','L','L','E','X','0','2'};
constexpr u64 fnv1a_offset = 1469598103934665603ull;
constexpr u64 fnv1a_prime = 1099511628211ull;

validation_result invalid(const char *message) noexcept {
    return validation_error(validation_code::invalid_matrix_view, invalid_id, message);
}

bool add_size(std::size_t left, std::size_t right, std::size_t *out) noexcept {
    if (right > std::numeric_limits<std::size_t>::max() - left) return false;
    *out = left + right;
    return true;
}

bool multiply_size(std::size_t left, std::size_t right, std::size_t *out) noexcept {
    if (left != 0u && right > std::numeric_limits<std::size_t>::max() / left)
        return false;
    *out = left * right;
    return true;
}

bool power_of_two(u32 value) noexcept {
    return value != 0u && (value & (value - 1u)) == 0u;
}

bool align_size(std::size_t cursor, u32 alignment, std::size_t *out) noexcept {
    if (!power_of_two(alignment)) return false;
    const std::size_t mask = static_cast<std::size_t>(alignment - 1u);
    if (cursor > std::numeric_limits<std::size_t>::max() - mask) return false;
    *out = (cursor + mask) & ~mask;
    return true;
}

u64 hash_bytes(u64 hash, const void *data, std::size_t bytes) noexcept {
    const auto *cursor = static_cast<const unsigned char *>(data);
    for (std::size_t index = 0u; index < bytes; ++index) {
        hash ^= cursor[index];
        hash *= fnv1a_prime;
    }
    return hash;
}

u64 nonzero_hash(const void *data, std::size_t bytes) noexcept {
    const u64 hash = hash_bytes(fnv1a_offset, data, bytes);
    return hash == 0u ? 1u : hash;
}

u64 image_identity(const void *image, std::size_t bytes) noexcept {
    constexpr std::size_t identity_offset = offsetof(execution_image_v2_header,
        image_identity);
    const auto *base = static_cast<const unsigned char *>(image);
    const u64 zero = 0u;
    u64 hash = hash_bytes(fnv1a_offset, base, identity_offset);
    hash = hash_bytes(hash, &zero, sizeof(zero));
    hash = hash_bytes(hash, base + identity_offset + sizeof(zero),
        bytes - identity_offset - sizeof(zero));
    return hash == 0u ? 1u : hash;
}

bool valid_section_kind(execution_section_kind kind) noexcept {
    const u32 value = static_cast<u32>(kind);
    return value >= static_cast<u32>(execution_section_kind::domain_table)
        && value <= static_cast<u32>(execution_section_kind::cpk1_v1_compatibility);
}

bool valid_projection_kind(execution_projection_kind kind) noexcept {
    const u32 value = static_cast<u32>(kind);
    return value >= static_cast<u32>(execution_projection_kind::native_row_masked)
        && value <= static_cast<u32>(execution_projection_kind::architecture_specific);
}

bool identity_present(u64 low, u64 high) noexcept {
    return low != 0u || high != 0u;
}

bool same_identity(u64 left_low, u64 left_high,
    u64 right_low, u64 right_high) noexcept {
    return left_low == right_low && left_high == right_high;
}

execution_axis_record_v1 axis_record(
    const cellerator::execution::persistent_axis_identity &axis) noexcept {
    return {axis.domain.low, axis.domain.high, axis.order.low, axis.order.high,
        axis.geometry.low, axis.geometry.high, axis.partition.low,
        axis.partition.high};
}

bool section_count(const execution_image_v2_build_request &request,
    execution_section_kind kind, u32 *out) noexcept {
    u32 count = 0u;
    for (u32 index = 0u; index < request.section_count; ++index)
        if (request.sections[index].kind == kind) ++count;
    *out = count;
    return true;
}

bool source_section_reference(const execution_image_v2_build_request &request,
    u32 index, execution_section_kind expected) noexcept {
    return index < request.section_count && request.sections[index].kind == expected;
}

bool source_payload_reference(const execution_image_v2_build_request &request,
    u32 index) noexcept {
    return index < request.section_count
        && (request.sections[index].kind == execution_section_kind::projection_payload
            || request.sections[index].kind
                == execution_section_kind::cpk1_v1_compatibility);
}

validation_result validate_request(
    const execution_image_v2_build_request &request) noexcept {
    using namespace cellerator::execution;
    if (!valid_identity(request.structure_identity)
        || request.structure_epoch == 0u
        || !valid_identity(request.semantic_geometry_identity)
        || !valid_identity(request.projection_catalog_identity))
        return invalid("execution image semantic identity is invalid");
    if (validate_persistent_axis_identity(request.source_axis)
            != biological_validation_code::ok
        || validate_persistent_axis_identity(request.destination_axis)
            != biological_validation_code::ok)
        return invalid("execution image axis identity is invalid");
    if (request.sections == nullptr || request.section_count == 0u
        || request.projections == nullptr || request.projection_count == 0u)
        return invalid("execution image requires section and projection directories");

    u32 domain_count = 0u, order_count = 0u, relation_count = 0u,
        geometry_count = 0u, initial_value_count = 0u;
    section_count(request, execution_section_kind::domain_table, &domain_count);
    section_count(request, execution_section_kind::order_partition_table, &order_count);
    section_count(request, execution_section_kind::relation_structure, &relation_count);
    section_count(request, execution_section_kind::semantic_geometry, &geometry_count);
    section_count(request, execution_section_kind::initial_values, &initial_value_count);
    if (domain_count != 1u || order_count != 1u || relation_count != 1u
        || geometry_count != 1u || initial_value_count > 1u)
        return invalid("execution image foundational sections are incomplete or duplicated");
    if ((initial_value_count == 0u) != (request.initial_value_generation == 0u))
        return invalid("initial values and value generation must be present together");

    for (u32 index = 0u; index < request.section_count; ++index) {
        const execution_section_source &section = request.sections[index];
        if ((!valid_section_kind(section.kind)
                && (section.flags & directory_optional) == 0u)
            || section.schema_version == 0u
            || !power_of_two(section.alignment) || section.alignment > 4096u
            || !identity_present(section.identity_low, section.identity_high)
            || section.data == nullptr || section.bytes == 0u)
            return invalid("execution image section source is invalid");
        if ((section.element_count == 0u) != (section.element_bytes == 0u))
            return invalid("execution image section element shape is partial");
        if (section.element_count != 0u) {
            std::size_t shaped_bytes = 0u;
            if (!multiply_size(section.element_count, section.element_bytes,
                    &shaped_bytes) || shaped_bytes != section.bytes)
                return invalid("execution image section element shape mismatches bytes");
        }
        for (u32 prior = 0u; prior < index; ++prior) {
            if (same_identity(section.identity_low, section.identity_high,
                    request.sections[prior].identity_low,
                    request.sections[prior].identity_high))
                return invalid("execution image section identity is duplicated");
        }
    }

    for (u32 index = 0u; index < request.projection_count; ++index) {
        const execution_projection_entry_v1 &entry = request.projections[index].entry;
        if (!identity_present(entry.identity_low, entry.identity_high)
            || entry.schema_version == 0u
            || (!valid_projection_kind(entry.kind)
                && (entry.flags & directory_optional) == 0u))
            return invalid("execution image projection descriptor is invalid");
        const bool has_payload = entry.payload_section != invalid_directory_index;
        if (has_payload != source_payload_reference(request, entry.payload_section))
            return invalid("execution image projection payload reference is invalid");
        if (!has_payload && (entry.flags & projection_lazy_constructible) == 0u)
            return invalid("projection without bytes must be lazy-constructible");
        if (entry.forward_map_section != invalid_directory_index
            && !source_section_reference(request, entry.forward_map_section,
                execution_section_kind::forward_value_map))
            return invalid("projection forward map reference is invalid");
        if (entry.transpose_map_section != invalid_directory_index
            && !source_section_reference(request, entry.transpose_map_section,
                execution_section_kind::transpose_value_map))
            return invalid("projection transpose map reference is invalid");
        if (entry.scheduling_summary_section != invalid_directory_index
            && !source_section_reference(request, entry.scheduling_summary_section,
                execution_section_kind::scheduling_summary))
            return invalid("projection scheduling summary reference is invalid");
        if (entry.capability_section != invalid_directory_index
            && entry.capability_section >= request.section_count)
            return invalid("projection capability reference is invalid");
        for (u32 prior = 0u; prior < index; ++prior) {
            const execution_projection_entry_v1 &other =
                request.projections[prior].entry;
            if (same_identity(entry.identity_low, entry.identity_high,
                    other.identity_low, other.identity_high))
                return invalid("execution image projection identity is duplicated");
        }
    }
    return validation_ok();
}

validation_result compute_layout(const execution_image_v2_build_request &request,
    execution_image_v2_requirements *out) noexcept {
    std::size_t section_directory_bytes = 0u, projection_directory_bytes = 0u;
    if (!multiply_size(request.section_count, sizeof(execution_section_entry_v1),
            &section_directory_bytes)
        || !multiply_size(request.projection_count,
            sizeof(execution_projection_entry_v1), &projection_directory_bytes))
        return validation_error(validation_code::integer_overflow, invalid_id,
            "execution image directory byte count overflows");
    std::size_t cursor = sizeof(execution_image_v2_header), aligned = 0u;
    if (!add_size(cursor, section_directory_bytes, &cursor)
        || !align_size(cursor, execution_image_v2_alignment, &cursor)
        || !add_size(cursor, projection_directory_bytes, &cursor)
        || !align_size(cursor, execution_image_v2_alignment, &cursor))
        return validation_error(validation_code::integer_overflow, invalid_id,
            "execution image directory layout overflows");
    const std::size_t directory_end = cursor;
    std::size_t section_bytes = 0u, padding_bytes =
        directory_end - sizeof(execution_image_v2_header)
        - section_directory_bytes - projection_directory_bytes;
    for (u32 index = 0u; index < request.section_count; ++index) {
        if (!align_size(cursor, request.sections[index].alignment, &aligned))
            return validation_error(validation_code::integer_overflow, invalid_id,
                "execution image section alignment overflows");
        padding_bytes += aligned - cursor;
        cursor = aligned;
        if (!add_size(section_bytes, request.sections[index].bytes, &section_bytes)
            || !add_size(cursor, request.sections[index].bytes, &cursor))
            return validation_error(validation_code::integer_overflow, invalid_id,
                "execution image section byte count overflows");
    }
    if (!align_size(cursor, execution_image_v2_alignment, &aligned))
        return validation_error(validation_code::integer_overflow, invalid_id,
            "execution image terminal alignment overflows");
    padding_bytes += aligned - cursor;
    out->image_bytes = aligned;
    out->directory_bytes = section_directory_bytes + projection_directory_bytes;
    out->section_bytes = section_bytes;
    out->alignment_padding_bytes = padding_bytes;
    return validation_ok();
}

template<typename T>
const T *pointer_at(const void *base, u64 offset) noexcept {
    return reinterpret_cast<const T *>(
        static_cast<const unsigned char *>(base) + offset);
}

bool range_valid(u64 offset, u64 bytes, std::size_t image_bytes) noexcept {
    return offset <= image_bytes && bytes <= image_bytes - offset;
}

bool section_reference(const execution_image_v2_view &view, u32 index,
    execution_section_kind expected) noexcept {
    return index < view.header.section_count && view.sections[index].kind == expected;
}

bool payload_reference(const execution_image_v2_view &view, u32 index) noexcept {
    return index < view.header.section_count
        && (view.sections[index].kind == execution_section_kind::projection_payload
            || view.sections[index].kind
                == execution_section_kind::cpk1_v1_compatibility);
}

bool relocated_pointer(const void *base, u64 offset, const void **out) noexcept {
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(base);
    if (offset > std::numeric_limits<std::uintptr_t>::max() - address)
        return false;
    *out = reinterpret_cast<const void *>(
        address + static_cast<std::uintptr_t>(offset));
    return true;
}

} // namespace

validation_result query_execution_image_v2_requirements_host(
    const execution_image_v2_build_request &request,
    execution_image_v2_requirements *out) noexcept {
    if (out == nullptr)
        return validation_error(validation_code::null_pointer, invalid_id,
            "execution image requirements output is null");
    validation_result status = validate_request(request);
    if (!status) return status;
    execution_image_v2_requirements result;
    status = compute_layout(request, &result);
    if (status) *out = result;
    return status;
}

validation_result build_execution_image_v2_host(
    const execution_image_v2_build_request &request,
    const execution_image_v2_buffer &buffer,
    execution_image_v2_view *out) noexcept {
    if (buffer.image == nullptr || out == nullptr)
        return validation_error(validation_code::null_pointer, invalid_id,
            "execution image buffer or output is null");
    execution_image_v2_requirements required;
    validation_result status = query_execution_image_v2_requirements_host(
        request, &required);
    if (!status) return status;
    if (buffer.capacity_bytes < required.image_bytes)
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "execution image buffer is too small");

    std::memset(buffer.image, 0, required.image_bytes);
    execution_image_v2_header header{};
    std::memcpy(header.magic, image_magic, sizeof(image_magic));
    header.schema_version = execution_image_v2_schema_version;
    header.header_bytes = sizeof(header);
    header.endian = execution_image_v2_endian_marker;
    header.alignment = execution_image_v2_alignment;
    header.image_bytes = required.image_bytes;
    header.structure_identity_low = request.structure_identity.low;
    header.structure_identity_high = request.structure_identity.high;
    header.structure_epoch = request.structure_epoch;
    header.semantic_geometry_identity_low = request.semantic_geometry_identity.low;
    header.semantic_geometry_identity_high = request.semantic_geometry_identity.high;
    header.projection_catalog_identity_low = request.projection_catalog_identity.low;
    header.projection_catalog_identity_high = request.projection_catalog_identity.high;
    header.initial_value_generation = request.initial_value_generation;
    header.source_axis = axis_record(request.source_axis);
    header.destination_axis = axis_record(request.destination_axis);
    header.section_count = request.section_count;
    header.projection_count = request.projection_count;
    header.section_directory_offset = sizeof(header);
    std::size_t projection_offset = sizeof(header)
        + static_cast<std::size_t>(request.section_count)
            * sizeof(execution_section_entry_v1);
    align_size(projection_offset, execution_image_v2_alignment, &projection_offset);
    header.projection_directory_offset = projection_offset;
    std::memcpy(buffer.image, &header, sizeof(header));

    auto *base = static_cast<unsigned char *>(buffer.image);
    std::size_t cursor = projection_offset
        + static_cast<std::size_t>(request.projection_count)
            * sizeof(execution_projection_entry_v1);
    align_size(cursor, execution_image_v2_alignment, &cursor);
    for (u32 index = 0u; index < request.section_count; ++index) {
        const execution_section_source &source = request.sections[index];
        align_size(cursor, source.alignment, &cursor);
        execution_section_entry_v1 entry{source.kind, source.schema_version,
            source.flags, source.alignment, source.identity_low,
            source.identity_high, static_cast<u64>(cursor),
            static_cast<u64>(source.bytes), nonzero_hash(source.data, source.bytes),
            source.element_count, source.element_bytes};
        std::memcpy(base + header.section_directory_offset
                + static_cast<std::size_t>(index) * sizeof(entry),
            &entry, sizeof(entry));
        std::memcpy(base + cursor, source.data, source.bytes);
        cursor += source.bytes;
    }
    for (u32 index = 0u; index < request.projection_count; ++index) {
        const execution_projection_entry_v1 &entry = request.projections[index].entry;
        std::memcpy(base + header.projection_directory_offset
                + static_cast<std::size_t>(index) * sizeof(entry),
            &entry, sizeof(entry));
    }
    header.image_identity = image_identity(buffer.image, required.image_bytes);
    std::memcpy(buffer.image, &header, sizeof(header));
    const execution_image_v2_expected expected{request.structure_identity,
        request.structure_epoch, request.semantic_geometry_identity,
        request.projection_catalog_identity, header.image_identity};
    return validate_execution_image_v2_host(buffer.image, required.image_bytes,
        expected, out);
}

validation_result validate_execution_image_v2_host(
    const void *image,
    std::size_t image_bytes,
    const execution_image_v2_expected &expected,
    execution_image_v2_view *out) noexcept {
    using namespace cellerator::execution;
    if (image == nullptr || out == nullptr)
        return validation_error(validation_code::null_pointer, invalid_id,
            "execution image or output is null");
    if (image_bytes < sizeof(execution_image_v2_header))
        return invalid("execution image is truncated");
    execution_image_v2_header header;
    std::memcpy(&header, image, sizeof(header));
    if (std::memcmp(header.magic, image_magic, sizeof(image_magic)) != 0
        || header.schema_version != execution_image_v2_schema_version
        || header.header_bytes != sizeof(header)
        || header.endian != execution_image_v2_endian_marker
        || header.alignment != execution_image_v2_alignment
        || header.image_bytes != image_bytes || header.image_identity == 0u
        || header.image_identity != image_identity(image, image_bytes))
        return invalid("execution image header or checksum is invalid");
    if (!valid_identity(expected.structure_identity)
        || expected.structure_epoch == 0u
        || !valid_identity(expected.semantic_geometry_identity)
        || !valid_identity(expected.projection_catalog_identity)
        || !same_identity(header.structure_identity_low,
            header.structure_identity_high, expected.structure_identity.low,
            expected.structure_identity.high)
        || header.structure_epoch != expected.structure_epoch
        || !same_identity(header.semantic_geometry_identity_low,
            header.semantic_geometry_identity_high,
            expected.semantic_geometry_identity.low,
            expected.semantic_geometry_identity.high)
        || !same_identity(header.projection_catalog_identity_low,
            header.projection_catalog_identity_high,
            expected.projection_catalog_identity.low,
            expected.projection_catalog_identity.high)
        || (expected.image_identity != 0u
            && header.image_identity != expected.image_identity))
        return validation_error(validation_code::invalid_signature, invalid_id,
            "execution image compatibility identity mismatches");
    if (header.section_count == 0u || header.projection_count == 0u
        || header.section_directory_offset != sizeof(header)
        || header.projection_directory_offset % execution_image_v2_alignment != 0u)
        return invalid("execution image directory metadata is invalid");
    std::size_t section_directory_bytes = 0u, projection_directory_bytes = 0u;
    if (!multiply_size(header.section_count, sizeof(execution_section_entry_v1),
            &section_directory_bytes)
        || !multiply_size(header.projection_count,
            sizeof(execution_projection_entry_v1), &projection_directory_bytes)
        || !range_valid(header.section_directory_offset, section_directory_bytes,
            image_bytes)
        || !range_valid(header.projection_directory_offset,
            projection_directory_bytes, image_bytes))
        return invalid("execution image directory range is invalid");
    const auto *sections = pointer_at<execution_section_entry_v1>(image,
        header.section_directory_offset);
    const auto *projections = pointer_at<execution_projection_entry_v1>(image,
        header.projection_directory_offset);
    execution_image_v2_view view{header, image, image_bytes, sections, projections};

    u32 domain_count = 0u, order_count = 0u, relation_count = 0u,
        geometry_count = 0u, initial_value_count = 0u;
    const u64 directory_end = std::max(
        header.section_directory_offset + section_directory_bytes,
        header.projection_directory_offset + projection_directory_bytes);
    for (u32 index = 0u; index < header.section_count; ++index) {
        const execution_section_entry_v1 &section = sections[index];
        if ((!valid_section_kind(section.kind)
                && (section.flags & directory_optional) == 0u)
            || section.schema_version == 0u || !power_of_two(section.alignment)
            || section.alignment > 4096u
            || !identity_present(section.identity_low, section.identity_high)
            || section.bytes == 0u || section.offset < directory_end
            || section.offset % section.alignment != 0u
            || !range_valid(section.offset, section.bytes, image_bytes)
            || section.checksum != nonzero_hash(
                static_cast<const unsigned char *>(image) + section.offset,
                static_cast<std::size_t>(section.bytes)))
            return invalid("execution image section directory is invalid");
        if ((section.element_count == 0u) != (section.element_bytes == 0u)
            || (section.element_count != 0u
                && static_cast<u64>(section.element_count) * section.element_bytes
                    != section.bytes))
            return invalid("execution image section element shape is invalid");
        for (u32 prior = 0u; prior < index; ++prior) {
            const execution_section_entry_v1 &other = sections[prior];
            const bool overlap = section.offset < other.offset + other.bytes
                && other.offset < section.offset + section.bytes;
            if (overlap || same_identity(section.identity_low, section.identity_high,
                    other.identity_low, other.identity_high))
                return invalid("execution image sections overlap or duplicate identity");
        }
        domain_count += section.kind == execution_section_kind::domain_table;
        order_count += section.kind == execution_section_kind::order_partition_table;
        relation_count += section.kind == execution_section_kind::relation_structure;
        geometry_count += section.kind == execution_section_kind::semantic_geometry;
        initial_value_count += section.kind == execution_section_kind::initial_values;
    }
    if (domain_count != 1u || order_count != 1u || relation_count != 1u
        || geometry_count != 1u || initial_value_count > 1u
        || ((initial_value_count == 0u) != (header.initial_value_generation == 0u)))
        return invalid("execution image foundational section set is invalid");

    for (u32 index = 0u; index < header.projection_count; ++index) {
        const execution_projection_entry_v1 &entry = projections[index];
        if (!identity_present(entry.identity_low, entry.identity_high)
            || entry.schema_version == 0u
            || (!valid_projection_kind(entry.kind)
                && (entry.flags & directory_optional) == 0u))
            return invalid("execution image projection directory is invalid");
        const bool has_payload = entry.payload_section != invalid_directory_index;
        if (has_payload != payload_reference(view, entry.payload_section)
            || (!has_payload
                && (entry.flags & projection_lazy_constructible) == 0u))
            return invalid("execution image projection payload is invalid");
        if (entry.forward_map_section != invalid_directory_index
            && !section_reference(view, entry.forward_map_section,
                execution_section_kind::forward_value_map))
            return invalid("execution image forward map is invalid");
        if (entry.transpose_map_section != invalid_directory_index
            && !section_reference(view, entry.transpose_map_section,
                execution_section_kind::transpose_value_map))
            return invalid("execution image transpose map is invalid");
        if (entry.scheduling_summary_section != invalid_directory_index
            && !section_reference(view, entry.scheduling_summary_section,
                execution_section_kind::scheduling_summary))
            return invalid("execution image scheduling summary is invalid");
        if (entry.capability_section != invalid_directory_index
            && entry.capability_section >= header.section_count)
            return invalid("execution image projection capability is invalid");
        for (u32 prior = 0u; prior < index; ++prior) {
            if (same_identity(entry.identity_low, entry.identity_high,
                    projections[prior].identity_low, projections[prior].identity_high))
                return invalid("execution image projection identity is duplicated");
        }
    }
    *out = view;
    return validation_ok();
}

validation_result rebind_execution_image_v2(
    const execution_image_v2_view &validated_host_view,
    const void *new_image_base,
    std::size_t new_image_bytes,
    execution_image_v2_view *out) noexcept {
    if (validated_host_view.image_base == nullptr || new_image_base == nullptr
        || out == nullptr)
        return validation_error(validation_code::null_pointer, invalid_id,
            "execution image rebind pointer is null");
    if (new_image_bytes != validated_host_view.image_bytes
        || validated_host_view.header.image_bytes != new_image_bytes)
        return invalid("execution image rebind size differs");
    execution_image_v2_view result = validated_host_view;
    result.image_base = new_image_base;
    result.image_bytes = new_image_bytes;
    result.sections = pointer_at<execution_section_entry_v1>(new_image_base,
        result.header.section_directory_offset);
    result.projections = pointer_at<execution_projection_entry_v1>(new_image_base,
        result.header.projection_directory_offset);
    *out = result;
    return validation_ok();
}

validation_result prebind_execution_projection_host(
    const execution_image_v2_view &validated_host_view,
    u32 projection_index,
    prebound_projection_view_v1 *out) noexcept {
    return prebind_execution_projection_for_base_host(validated_host_view,
        projection_index, validated_host_view.image_base,
        validated_host_view.image_bytes, out);
}

validation_result prebind_execution_projection_for_base_host(
    const execution_image_v2_view &validated_host_view,
    u32 projection_index,
    const void *destination_image_base,
    std::size_t destination_image_bytes,
    prebound_projection_view_v1 *out) noexcept {
    if (validated_host_view.image_base == nullptr
        || validated_host_view.sections == nullptr
        || validated_host_view.projections == nullptr
        || destination_image_base == nullptr || out == nullptr)
        return validation_error(validation_code::null_pointer, invalid_id,
            "projection prebind image or output is null");
    if (destination_image_bytes != validated_host_view.image_bytes
        || validated_host_view.header.image_bytes != destination_image_bytes)
        return invalid("projection prebind destination size differs");
    if (projection_index >= validated_host_view.header.projection_count)
        return invalid("projection prebind index is out of range");
    const execution_projection_entry_v1 &entry =
        validated_host_view.projections[projection_index];
    prebound_projection_view_v1 result{};
    result.descriptor = entry;
    bool valid_offsets = true;
    auto bind = [&](u32 section_index, const void **data, std::size_t *bytes) {
        if (section_index == invalid_directory_index || !valid_offsets) return;
        valid_offsets = relocated_pointer(destination_image_base,
            validated_host_view.sections[section_index].offset, data);
        *bytes = static_cast<std::size_t>(
            validated_host_view.sections[section_index].bytes);
    };
    bind(entry.payload_section, &result.payload, &result.payload_bytes);
    bind(entry.forward_map_section, &result.forward_map, &result.forward_map_bytes);
    bind(entry.transpose_map_section, &result.transpose_map,
        &result.transpose_map_bytes);
    bind(entry.scheduling_summary_section, &result.scheduling_summary,
        &result.scheduling_summary_bytes);
    if (!valid_offsets)
        return invalid("projection prebind destination pointer overflows");
    *out = result;
    return validation_ok();
}

} // namespace cellpack::persistence
