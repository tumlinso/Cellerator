#include <Cellerator/geometry/persistence/semantic_geometry_image_v1.hh>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace cellerator::geometry::persistence {
namespace {

constexpr u8 image_magic[8] = {'C', 'E', 'L', 'L', 'C', 'S', 'G', '1'};
constexpr u32 endian_marker = 0x01020304u;
constexpr u64 fnv1a_offset = 1469598103934665603ull;
constexpr u64 fnv1a_prime = 1099511628211ull;
constexpr u64 geometry_high_seed = 1099511628211ull ^ 0x4353473153454d41ull;

constexpr u64 header_kind_offset = 8u;
constexpr u64 header_schema_offset = 12u;
constexpr u64 header_bytes_offset = 16u;
constexpr u64 header_endian_offset = 20u;
constexpr u64 header_alignment_offset = 24u;
constexpr u64 header_reserved_offset = 28u;
constexpr u64 header_image_bytes_offset = 32u;
constexpr u64 header_checksum_offset = 40u;
constexpr u64 header_geometry_low_offset = 48u;
constexpr u64 header_geometry_high_offset = 56u;
constexpr u64 header_relation_low_offset = 64u;
constexpr u64 header_relation_high_offset = 72u;
constexpr u64 header_structure_low_offset = 80u;
constexpr u64 header_structure_high_offset = 88u;
constexpr u64 header_structure_epoch_offset = 96u;
constexpr u64 header_work_window_low_offset = 104u;
constexpr u64 header_work_window_high_offset = 112u;
constexpr u64 header_source_axis_offset = 120u;
constexpr u64 header_destination_axis_offset = 184u;
constexpr u64 header_logical_edge_count_offset = 248u;
constexpr u64 header_work_count_offset = 256u;
constexpr u64 header_component_count_offset = 260u;
constexpr u64 header_section_count_offset = 264u;
constexpr u64 header_section_entry_bytes_offset = 268u;
constexpr u64 header_directory_offset_offset = 272u;
constexpr u64 header_directory_bytes_offset = 280u;
constexpr u64 header_flags_offset = 288u;
constexpr u64 header_trailing_reserved_offset = 296u;

constexpr u64 section_kind_offset = 0u;
constexpr u64 section_schema_offset = 4u;
constexpr u64 section_flags_offset = 8u;
constexpr u64 section_data_offset = 16u;
constexpr u64 section_data_bytes_offset = 24u;
constexpr u64 section_element_count_offset = 32u;
constexpr u64 section_element_bytes_offset = 40u;
constexpr u64 section_alignment_offset = 44u;
constexpr u64 section_checksum_offset = 48u;
constexpr u64 section_reserved_offset = 56u;

struct section_metadata {
    u32 kind = 0u;
    u32 schema_version = 0u;
    u64 flags = 0u;
    u64 data_offset = 0u;
    u64 data_bytes = 0u;
    u64 element_count = 0u;
    u32 element_bytes = 0u;
    u32 alignment = 0u;
    u64 checksum = 0u;
};

bool checked_add(u64 lhs, u64 rhs, u64 *out) noexcept {
    if (rhs > std::numeric_limits<u64>::max() - lhs)
        return false;
    *out = lhs + rhs;
    return true;
}

bool checked_multiply(u64 lhs, u64 rhs, u64 *out) noexcept {
    if (lhs != 0u && rhs > std::numeric_limits<u64>::max() / lhs)
        return false;
    *out = lhs * rhs;
    return true;
}

bool power_of_two(u64 value) noexcept {
    return value != 0u && (value & (value - 1u)) == 0u;
}

bool align_up(u64 value, u64 alignment, u64 *out) noexcept {
    if (!power_of_two(alignment))
        return false;
    const u64 mask = alignment - 1u;
    if (value > std::numeric_limits<u64>::max() - mask)
        return false;
    *out = (value + mask) & ~mask;
    return true;
}

void write_u32(u8 *base, u64 offset, u32 value) noexcept {
    for (u32 byte = 0u; byte < 4u; ++byte)
        base[offset + byte] = static_cast<u8>(value >> (byte * 8u));
}

void write_u64(u8 *base, u64 offset, u64 value) noexcept {
    for (u32 byte = 0u; byte < 8u; ++byte)
        base[offset + byte] = static_cast<u8>(value >> (byte * 8u));
}

u32 read_u32(const u8 *base, u64 offset) noexcept {
    u32 value = 0u;
    for (u32 byte = 0u; byte < 4u; ++byte)
        value |= static_cast<u32>(base[offset + byte]) << (byte * 8u);
    return value;
}

u64 read_u64(const u8 *base, u64 offset) noexcept {
    u64 value = 0u;
    for (u32 byte = 0u; byte < 8u; ++byte)
        value |= static_cast<u64>(base[offset + byte]) << (byte * 8u);
    return value;
}

u64 hash_bytes(const u8 *data, u64 bytes, u64 seed) noexcept {
    u64 hash = seed;
    for (u64 index = 0u; index < bytes; ++index) {
        hash ^= data[index];
        hash *= fnv1a_prime;
    }
    return hash == 0u ? 1u : hash;
}

u64 hash_image_with_zeros(
    const u8 *image,
    u64 bytes,
    u64 seed,
    bool zero_geometry) noexcept {
    u64 hash = seed;
    for (u64 index = 0u; index < bytes; ++index) {
        const bool checksum_byte =
            index >= header_checksum_offset && index < header_checksum_offset + 8u;
        const bool geometry_byte = zero_geometry
            && index >= header_geometry_low_offset
            && index < header_geometry_high_offset + 8u;
        hash ^= checksum_byte || geometry_byte ? 0u : image[index];
        hash *= fnv1a_prime;
    }
    return hash == 0u ? 1u : hash;
}

bool valid_persistent_axis(
    const execution::persistent_axis_identity &axis) noexcept {
    return execution::validate_persistent_axis_identity(axis)
            == execution::biological_validation_code::ok
        && execution::valid_identity(axis.domain)
        && execution::valid_identity(axis.order)
        && execution::valid_identity(axis.geometry)
        && execution::valid_identity(axis.partition);
}

void write_identity(u8 *base, u64 offset, u64 low, u64 high) noexcept {
    write_u64(base, offset, low);
    write_u64(base, offset + 8u, high);
}

void write_axis(
    u8 *base,
    u64 offset,
    const execution::persistent_axis_identity &axis) noexcept {
    write_identity(base, offset, axis.domain.low, axis.domain.high);
    write_identity(base, offset + 16u, axis.order.low, axis.order.high);
    write_identity(base, offset + 32u, axis.geometry.low, axis.geometry.high);
    write_identity(base, offset + 48u, axis.partition.low, axis.partition.high);
}

execution::persistent_axis_identity read_axis(
    const u8 *base,
    u64 offset) noexcept {
    execution::persistent_axis_identity axis{};
    axis.header.schema_version = execution::biological_abi_version;
    axis.header.kind = execution::serialized_record_kind::persistent_axis_identity;
    axis.header.byte_count = sizeof(execution::persistent_axis_identity);
    axis.domain = {read_u64(base, offset), read_u64(base, offset + 8u)};
    axis.order = {read_u64(base, offset + 16u), read_u64(base, offset + 24u)};
    axis.geometry = {read_u64(base, offset + 32u), read_u64(base, offset + 40u)};
    axis.partition = {read_u64(base, offset + 48u), read_u64(base, offset + 56u)};
    return axis;
}

bool valid_optional_sections(
    const semantic_geometry_image_build_request_v1 &request) noexcept {
    if (request.optional_section_count != 0u
        && request.optional_sections == nullptr)
        return false;
    for (u32 index = 0u; index < request.optional_section_count; ++index) {
        const semantic_geometry_optional_section_v1 &section =
            request.optional_sections[index];
        if (section.kind < semantic_geometry_first_optional_section_kind_v1
            || section.schema_version == 0u || section.data == nullptr
            || section.data_bytes == 0u || !power_of_two(section.alignment)
            || section.alignment > 4096u)
            return false;
        for (u32 previous = 0u; previous < index; ++previous)
            if (request.optional_sections[previous].kind == section.kind)
                return false;
    }
    return true;
}

semantic_geometry_image_status_v1 calculate_requirements(
    const semantic_geometry_image_build_request_v1 &request,
    semantic_geometry_image_requirements_v1 *requirements) noexcept {
    if (requirements == nullptr || !valid_optional_sections(request)
        || request.optional_section_count
            > std::numeric_limits<u32>::max()
                - semantic_geometry_mandatory_section_count_v1)
        return semantic_geometry_image_status_v1::invalid_argument;
    const u32 section_count = semantic_geometry_mandatory_section_count_v1
        + request.optional_section_count;
    u64 directory_bytes = 0u;
    u64 cursor = 0u;
    if (!checked_multiply(section_count,
            semantic_geometry_section_entry_bytes_v1, &directory_bytes)
        || !checked_add(semantic_geometry_image_header_bytes_v1,
            directory_bytes, &cursor)
        || !align_up(cursor, semantic_geometry_image_alignment_v1, &cursor))
        return semantic_geometry_image_status_v1::arithmetic_overflow;

    const u64 mandatory_counts[] = {
        request.work_window.member_count,
        request.work_layout.work_count,
        request.work_layout.work_count,
        request.relation_cover.component_count,
        request.relation_cover.logical_edge_count
    };
    const u64 mandatory_widths[] = {4u, 4u, 4u, 24u, 8u};
    for (u32 index = 0u;
         index < semantic_geometry_mandatory_section_count_v1; ++index) {
        u64 bytes = 0u;
        if (!checked_multiply(
                mandatory_counts[index], mandatory_widths[index], &bytes)
            || !checked_add(cursor, bytes, &cursor)
            || !align_up(cursor, semantic_geometry_image_alignment_v1, &cursor))
            return semantic_geometry_image_status_v1::arithmetic_overflow;
    }
    for (u32 index = 0u; index < request.optional_section_count; ++index) {
        const u64 alignment = request.optional_sections[index].alignment
                > semantic_geometry_image_alignment_v1
            ? request.optional_sections[index].alignment
            : semantic_geometry_image_alignment_v1;
        if (!align_up(cursor, alignment, &cursor)
            || !checked_add(
                cursor, request.optional_sections[index].data_bytes, &cursor)
            || !align_up(cursor, semantic_geometry_image_alignment_v1, &cursor))
            return semantic_geometry_image_status_v1::arithmetic_overflow;
    }

    semantic_geometry_image_requirements_v1 result{};
    result.image_bytes = cursor;
    result.section_count = section_count;
    result.validation_workspace_bytes =
        request.relation_cover.logical_edge_count;
    *requirements = result;
    return semantic_geometry_image_status_v1::ok;
}

void write_section_entry(
    u8 *image,
    u32 section_index,
    const section_metadata &section) noexcept {
    const u64 entry = semantic_geometry_image_header_bytes_v1
        + static_cast<u64>(section_index)
            * semantic_geometry_section_entry_bytes_v1;
    write_u32(image, entry + section_kind_offset, section.kind);
    write_u32(image, entry + section_schema_offset, section.schema_version);
    write_u64(image, entry + section_flags_offset, section.flags);
    write_u64(image, entry + section_data_offset, section.data_offset);
    write_u64(image, entry + section_data_bytes_offset, section.data_bytes);
    write_u64(image, entry + section_element_count_offset,
        section.element_count);
    write_u32(image, entry + section_element_bytes_offset,
        section.element_bytes);
    write_u32(image, entry + section_alignment_offset, section.alignment);
    write_u64(image, entry + section_checksum_offset, section.checksum);
    write_u64(image, entry + section_reserved_offset, 0u);
}

section_metadata read_section_entry(
    const u8 *image,
    u32 section_index) noexcept {
    const u64 entry = semantic_geometry_image_header_bytes_v1
        + static_cast<u64>(section_index)
            * semantic_geometry_section_entry_bytes_v1;
    section_metadata section{};
    section.kind = read_u32(image, entry + section_kind_offset);
    section.schema_version = read_u32(image, entry + section_schema_offset);
    section.flags = read_u64(image, entry + section_flags_offset);
    section.data_offset = read_u64(image, entry + section_data_offset);
    section.data_bytes = read_u64(image, entry + section_data_bytes_offset);
    section.element_count =
        read_u64(image, entry + section_element_count_offset);
    section.element_bytes =
        read_u32(image, entry + section_element_bytes_offset);
    section.alignment = read_u32(image, entry + section_alignment_offset);
    section.checksum = read_u64(image, entry + section_checksum_offset);
    return section;
}

void write_u32_section(
    u8 *image,
    const section_metadata &section,
    const u32 *values) noexcept {
    for (u64 index = 0u; index < section.element_count; ++index)
        write_u32(image, section.data_offset + index * 4u, values[index]);
}

void write_component_section(
    u8 *image,
    const section_metadata &section,
    const semantic_component_v1 *components) noexcept {
    for (u64 index = 0u; index < section.element_count; ++index) {
        const u64 offset = section.data_offset + index * 24u;
        write_u32(image, offset, components[index].component_id);
        write_u32(image, offset + 4u,
            static_cast<u32>(components[index].kind));
        write_u64(image, offset + 8u, components[index].logical_edge_offset);
        write_u64(image, offset + 16u, components[index].logical_edge_count);
    }
}

void write_u64_section(
    u8 *image,
    const section_metadata &section,
    const u64 *values) noexcept {
    for (u64 index = 0u; index < section.element_count; ++index)
        write_u64(image, section.data_offset + index * 8u, values[index]);
}

bool section_matches(
    const section_metadata &section,
    u32 kind,
    u64 count,
    u32 width) noexcept {
    u64 expected_bytes = 0u;
    return checked_multiply(count, width, &expected_bytes)
        && section.kind == kind && section.schema_version == 1u
        && section.flags == 0u && section.element_count == count
        && section.element_bytes == width
        && section.data_bytes == expected_bytes;
}

semantic_geometry_image_status_v1 validate_semantic_sections(
    const u8 *image,
    const semantic_geometry_image_view_v1 &view,
    semantic_geometry_image_validation_workspace_v1 workspace) noexcept {
    section_metadata required[semantic_geometry_mandatory_section_count_v1]{};
    bool found[semantic_geometry_mandatory_section_count_v1]{};
    for (u32 section_index = 0u; section_index < view.section_count;
         ++section_index) {
        const section_metadata section =
            read_section_entry(image, section_index);
        if (section.kind >= 1u
            && section.kind <= semantic_geometry_mandatory_section_count_v1) {
            const u32 required_index = section.kind - 1u;
            required[required_index] = section;
            found[required_index] = true;
        }
    }
    for (bool present : found)
        if (!present)
            return semantic_geometry_image_status_v1::missing_mandatory_section;

    if (!section_matches(required[0], 1u, view.work_count, 4u)
        || !section_matches(required[1], 2u, view.work_count, 4u)
        || !section_matches(required[2], 3u, view.work_count, 4u)
        || !section_matches(required[3], 4u, view.component_count, 24u)
        || !section_matches(required[4], 5u, view.logical_edge_count, 8u))
        return semantic_geometry_image_status_v1::invalid_semantic_data;

    for (u32 index = 0u; index < view.work_count; ++index) {
        const u32 member = read_u32(image, required[0].data_offset + index * 4u);
        for (u32 previous = 0u; previous < index; ++previous)
            if (read_u32(image,
                    required[0].data_offset + previous * 4u) == member)
                return semantic_geometry_image_status_v1::invalid_semantic_data;

        const u32 window_index =
            read_u32(image, required[1].data_offset + index * 4u);
        if (window_index >= view.work_count)
            return semantic_geometry_image_status_v1::invalid_semantic_data;
        for (u32 previous = 0u; previous < index; ++previous)
            if (read_u32(image,
                    required[1].data_offset + previous * 4u) == window_index)
                return semantic_geometry_image_status_v1::invalid_semantic_data;
        const u32 inverse =
            read_u32(image, required[2].data_offset + window_index * 4u);
        if (inverse != index)
            return semantic_geometry_image_status_v1::invalid_semantic_data;
    }

    u64 expected_edge_offset = 0u;
    for (u32 component_index = 0u; component_index < view.component_count;
         ++component_index) {
        const u64 offset = required[3].data_offset
            + static_cast<u64>(component_index) * 24u;
        const u32 component_id = read_u32(image, offset);
        const auto kind = static_cast<semantic_component_kind>(
            read_u32(image, offset + 4u));
        const u64 edge_offset = read_u64(image, offset + 8u);
        const u64 edge_count = read_u64(image, offset + 16u);
        if (component_id == invalid_semantic_component_id
            || !valid_semantic_component_kind(kind) || edge_count == 0u
            || edge_offset != expected_edge_offset
            || edge_count > view.logical_edge_count - expected_edge_offset)
            return semantic_geometry_image_status_v1::invalid_semantic_data;
        for (u32 previous = 0u; previous < component_index; ++previous)
            if (read_u32(image, required[3].data_offset
                    + static_cast<u64>(previous) * 24u) == component_id)
                return semantic_geometry_image_status_v1::invalid_semantic_data;
        expected_edge_offset += edge_count;
    }
    if (expected_edge_offset != view.logical_edge_count)
        return semantic_geometry_image_status_v1::invalid_semantic_data;
    if (view.logical_edge_count != 0u
        && (workspace.edge_marks == nullptr
            || workspace.edge_mark_capacity < view.logical_edge_count))
        return semantic_geometry_image_status_v1::insufficient_validation_workspace;
    for (u64 edge = 0u; edge < view.logical_edge_count; ++edge)
        workspace.edge_marks[edge] = 0u;
    for (u64 position = 0u; position < view.logical_edge_count; ++position) {
        const u64 edge =
            read_u64(image, required[4].data_offset + position * 8u);
        if (edge >= view.logical_edge_count || workspace.edge_marks[edge] != 0u)
            return semantic_geometry_image_status_v1::invalid_semantic_data;
        workspace.edge_marks[edge] = 1u;
    }
    return semantic_geometry_image_status_v1::ok;
}

} // namespace

semantic_geometry_image_status_v1
query_semantic_geometry_image_requirements_v1(
    const semantic_geometry_image_build_request_v1 &request,
    semantic_geometry_image_requirements_v1 *requirements) noexcept {
    if (!execution::valid_identity(request.relation)
        || !execution::valid_identity(request.structure)
        || request.structure_epoch.value == 0u
        || !valid_persistent_axis(request.source_axis)
        || !valid_persistent_axis(request.destination_axis))
        return semantic_geometry_image_status_v1::invalid_argument;
    return calculate_requirements(request, requirements);
}

semantic_geometry_image_status_v1 build_semantic_geometry_image_v1(
    const semantic_geometry_image_build_request_v1 &request,
    semantic_geometry_image_buffer_v1 buffer,
    semantic_geometry_image_validation_workspace_v1 validation_workspace,
    semantic_geometry_image_view_v1 *view) noexcept {
    if (view == nullptr || !validate_work_window(request.work_window))
        return semantic_geometry_image_status_v1::invalid_work_window;
    if (!validate_work_layout(request.work_window, request.work_layout))
        return semantic_geometry_image_status_v1::invalid_work_layout;
    if (!validate_relation_cover(request.relation_cover,
            {validation_workspace.edge_marks,
                validation_workspace.edge_mark_capacity}))
        return semantic_geometry_image_status_v1::invalid_relation_cover;
    semantic_geometry_image_requirements_v1 requirements{};
    const semantic_geometry_image_status_v1 requirement_status =
        query_semantic_geometry_image_requirements_v1(request, &requirements);
    if (requirement_status != semantic_geometry_image_status_v1::ok)
        return requirement_status;
    if (buffer.image == nullptr || buffer.image_capacity < requirements.image_bytes)
        return semantic_geometry_image_status_v1::insufficient_capacity;
    if (reinterpret_cast<std::uintptr_t>(buffer.image)
        % semantic_geometry_image_alignment_v1 != 0u)
        return semantic_geometry_image_status_v1::misaligned_image;
    if (requirements.image_bytes > std::numeric_limits<std::size_t>::max())
        return semantic_geometry_image_status_v1::arithmetic_overflow;

    auto *image = static_cast<u8 *>(buffer.image);
    std::memset(image, 0, static_cast<std::size_t>(requirements.image_bytes));
    std::memcpy(image, image_magic, sizeof(image_magic));
    write_u32(image, header_kind_offset, semantic_geometry_image_kind_v1);
    write_u32(image, header_schema_offset,
        semantic_geometry_image_schema_version_v1);
    write_u32(image, header_bytes_offset,
        semantic_geometry_image_header_bytes_v1);
    write_u32(image, header_endian_offset, endian_marker);
    write_u32(image, header_alignment_offset,
        semantic_geometry_image_alignment_v1);
    write_u32(image, header_reserved_offset, 0u);
    write_u64(image, header_image_bytes_offset, requirements.image_bytes);
    write_identity(image, header_relation_low_offset,
        request.relation.low, request.relation.high);
    write_identity(image, header_structure_low_offset,
        request.structure.low, request.structure.high);
    write_u64(image, header_structure_epoch_offset,
        request.structure_epoch.value);
    write_identity(image, header_work_window_low_offset,
        request.work_window.identity.low, request.work_window.identity.high);
    write_axis(image, header_source_axis_offset, request.source_axis);
    write_axis(image, header_destination_axis_offset, request.destination_axis);
    write_u64(image, header_logical_edge_count_offset,
        request.relation_cover.logical_edge_count);
    write_u32(image, header_work_count_offset,
        request.work_layout.work_count);
    write_u32(image, header_component_count_offset,
        request.relation_cover.component_count);
    write_u32(image, header_section_count_offset, requirements.section_count);
    write_u32(image, header_section_entry_bytes_offset,
        semantic_geometry_section_entry_bytes_v1);
    write_u64(image, header_directory_offset_offset,
        semantic_geometry_image_header_bytes_v1);
    write_u64(image, header_directory_bytes_offset,
        static_cast<u64>(requirements.section_count)
            * semantic_geometry_section_entry_bytes_v1);
    write_u64(image, header_flags_offset,
        request.optional_section_count == 0u ? 0u : 1u);

    u64 cursor = semantic_geometry_image_header_bytes_v1
        + static_cast<u64>(requirements.section_count)
            * semantic_geometry_section_entry_bytes_v1;
    align_up(cursor, semantic_geometry_image_alignment_v1, &cursor);
    const u64 counts[] = {
        request.work_window.member_count,
        request.work_layout.work_count,
        request.work_layout.work_count,
        request.relation_cover.component_count,
        request.relation_cover.logical_edge_count
    };
    const u32 widths[] = {4u, 4u, 4u, 24u, 8u};
    const u32 schemas[] = {work_window_schema_version,
        work_layout_schema_version, work_layout_schema_version,
        relation_cover_schema_version, relation_cover_schema_version};
    for (u32 index = 0u;
         index < semantic_geometry_mandatory_section_count_v1; ++index) {
        section_metadata section{};
        section.kind = index + 1u;
        section.schema_version = schemas[index];
        section.data_offset = cursor;
        section.element_count = counts[index];
        section.element_bytes = widths[index];
        section.alignment = semantic_geometry_image_alignment_v1;
        checked_multiply(section.element_count, section.element_bytes,
            &section.data_bytes);
        if (index == 0u)
            write_u32_section(image, section, request.work_window.members);
        else if (index == 1u)
            write_u32_section(
                image, section, request.work_layout.execution_to_window);
        else if (index == 2u)
            write_u32_section(
                image, section, request.work_layout.window_to_execution);
        else if (index == 3u)
            write_component_section(
                image, section, request.relation_cover.components);
        else
            write_u64_section(
                image, section, request.relation_cover.logical_edge_ids);
        section.checksum = hash_bytes(image + section.data_offset,
            section.data_bytes, fnv1a_offset);
        write_section_entry(image, index, section);
        checked_add(cursor, section.data_bytes, &cursor);
        align_up(cursor, semantic_geometry_image_alignment_v1, &cursor);
    }
    for (u32 optional_index = 0u;
         optional_index < request.optional_section_count; ++optional_index) {
        const semantic_geometry_optional_section_v1 &input =
            request.optional_sections[optional_index];
        const u64 alignment = input.alignment
                > semantic_geometry_image_alignment_v1
            ? input.alignment
            : semantic_geometry_image_alignment_v1;
        align_up(cursor, alignment, &cursor);
        section_metadata section{};
        section.kind = input.kind;
        section.schema_version = input.schema_version;
        section.flags = input.flags;
        section.data_offset = cursor;
        section.data_bytes = input.data_bytes;
        section.element_count = input.data_bytes;
        section.element_bytes = 1u;
        section.alignment = static_cast<u32>(alignment);
        std::memcpy(image + cursor, input.data,
            static_cast<std::size_t>(input.data_bytes));
        section.checksum = hash_bytes(
            image + cursor, input.data_bytes, fnv1a_offset);
        write_section_entry(image,
            semantic_geometry_mandatory_section_count_v1 + optional_index,
            section);
        checked_add(cursor, input.data_bytes, &cursor);
        align_up(cursor, semantic_geometry_image_alignment_v1, &cursor);
    }

    const u64 identity_low = hash_image_with_zeros(
        image, requirements.image_bytes, fnv1a_offset, true);
    const u64 identity_high = hash_image_with_zeros(
        image, requirements.image_bytes, geometry_high_seed, true);
    write_u64(image, header_geometry_low_offset, identity_low);
    write_u64(image, header_geometry_high_offset, identity_high);
    const u64 checksum = hash_image_with_zeros(
        image, requirements.image_bytes, fnv1a_offset, false);
    write_u64(image, header_checksum_offset, checksum);
    return validate_semantic_geometry_image_v1(buffer.image,
        requirements.image_bytes, validation_workspace, view);
}

semantic_geometry_image_status_v1 validate_semantic_geometry_image_v1(
    const void *image_pointer,
    u64 image_bytes,
    semantic_geometry_image_validation_workspace_v1 validation_workspace,
    semantic_geometry_image_view_v1 *view) noexcept {
    if (image_pointer == nullptr || view == nullptr
        || image_bytes < semantic_geometry_image_header_bytes_v1
        || image_bytes > std::numeric_limits<std::size_t>::max())
        return semantic_geometry_image_status_v1::invalid_argument;
    const auto *image = static_cast<const u8 *>(image_pointer);
    if (std::memcmp(image, image_magic, sizeof(image_magic)) != 0
        || read_u32(image, header_kind_offset)
            != semantic_geometry_image_kind_v1
        || read_u32(image, header_schema_offset)
            != semantic_geometry_image_schema_version_v1
        || read_u32(image, header_bytes_offset)
            != semantic_geometry_image_header_bytes_v1
        || read_u32(image, header_endian_offset) != endian_marker
        || read_u32(image, header_alignment_offset)
            != semantic_geometry_image_alignment_v1
        || read_u32(image, header_reserved_offset) != 0u
        || read_u64(image, header_image_bytes_offset) != image_bytes
        || read_u32(image, header_section_entry_bytes_offset)
            != semantic_geometry_section_entry_bytes_v1
        || read_u64(image, header_directory_offset_offset)
            != semantic_geometry_image_header_bytes_v1
        || read_u64(image, header_flags_offset) > 1u)
        return semantic_geometry_image_status_v1::invalid_format;
    for (u64 offset = header_trailing_reserved_offset;
         offset < semantic_geometry_image_header_bytes_v1; ++offset)
        if (image[offset] != 0u)
            return semantic_geometry_image_status_v1::invalid_format;

    const u32 section_count = read_u32(image, header_section_count_offset);
    const bool has_optional_sections =
        section_count > semantic_geometry_mandatory_section_count_v1;
    if ((read_u64(image, header_flags_offset) != 0u)
        != has_optional_sections)
        return semantic_geometry_image_status_v1::invalid_format;
    const u64 directory_bytes = read_u64(image, header_directory_bytes_offset);
    u64 expected_directory_bytes = 0u;
    u64 data_begin = 0u;
    if (section_count < semantic_geometry_mandatory_section_count_v1
        || !checked_multiply(section_count,
            semantic_geometry_section_entry_bytes_v1,
            &expected_directory_bytes)
        || directory_bytes != expected_directory_bytes
        || !checked_add(semantic_geometry_image_header_bytes_v1,
            directory_bytes, &data_begin)
        || !align_up(data_begin, semantic_geometry_image_alignment_v1,
            &data_begin)
        || data_begin > image_bytes)
        return semantic_geometry_image_status_v1::invalid_section_directory;

    u64 previous_end = data_begin;
    for (u32 section_index = 0u; section_index < section_count;
         ++section_index) {
        const section_metadata section =
            read_section_entry(image, section_index);
        const u64 entry = semantic_geometry_image_header_bytes_v1
            + static_cast<u64>(section_index)
                * semantic_geometry_section_entry_bytes_v1;
        if (read_u64(image, entry + section_reserved_offset) != 0u
            || section.kind == 0u || section.schema_version == 0u
            || !power_of_two(section.alignment)
            || section.alignment < semantic_geometry_image_alignment_v1
            || section.data_offset % section.alignment != 0u
            || section.data_offset < previous_end
            || section.data_offset > image_bytes
            || section.data_bytes > image_bytes - section.data_offset)
            return semantic_geometry_image_status_v1::section_out_of_bounds;
        if (section.element_bytes == 0u
            || section.element_count
                > std::numeric_limits<u64>::max() / section.element_bytes
            || section.element_count * section.element_bytes
                != section.data_bytes)
            return semantic_geometry_image_status_v1::invalid_section_directory;
        for (u32 previous = 0u; previous < section_index; ++previous)
            if (read_section_entry(image, previous).kind == section.kind)
                return semantic_geometry_image_status_v1::duplicate_section;
        if (section.kind <= semantic_geometry_mandatory_section_count_v1) {
            if (section.flags != 0u)
                return semantic_geometry_image_status_v1::invalid_section_directory;
        } else if (section.kind
            < semantic_geometry_first_optional_section_kind_v1)
            return semantic_geometry_image_status_v1::invalid_section_directory;
        if (hash_bytes(image + section.data_offset,
                section.data_bytes, fnv1a_offset) != section.checksum)
            return semantic_geometry_image_status_v1::section_checksum_mismatch;
        previous_end = section.data_offset + section.data_bytes;
    }
    if (hash_image_with_zeros(image, image_bytes, fnv1a_offset, false)
        != read_u64(image, header_checksum_offset))
        return semantic_geometry_image_status_v1::image_checksum_mismatch;
    const u64 expected_identity_low = hash_image_with_zeros(
        image, image_bytes, fnv1a_offset, true);
    const u64 expected_identity_high = hash_image_with_zeros(
        image, image_bytes, geometry_high_seed, true);
    if (expected_identity_low != read_u64(image, header_geometry_low_offset)
        || expected_identity_high
            != read_u64(image, header_geometry_high_offset))
        return semantic_geometry_image_status_v1::geometry_identity_mismatch;

    semantic_geometry_image_view_v1 result{};
    result.image_base = image_pointer;
    result.image_bytes = image_bytes;
    result.geometry_identity = {
        read_u64(image, header_geometry_low_offset),
        read_u64(image, header_geometry_high_offset)};
    result.relation = {read_u64(image, header_relation_low_offset),
        read_u64(image, header_relation_high_offset)};
    result.structure = {read_u64(image, header_structure_low_offset),
        read_u64(image, header_structure_high_offset)};
    result.structure_epoch = {read_u64(image, header_structure_epoch_offset)};
    result.work_window = {read_u64(image, header_work_window_low_offset),
        read_u64(image, header_work_window_high_offset)};
    result.source_axis = read_axis(image, header_source_axis_offset);
    result.destination_axis = read_axis(image, header_destination_axis_offset);
    result.logical_edge_count =
        read_u64(image, header_logical_edge_count_offset);
    result.work_count = read_u32(image, header_work_count_offset);
    result.component_count = read_u32(image, header_component_count_offset);
    result.section_count = section_count;
    if (!execution::valid_identity(result.geometry_identity)
        || !execution::valid_identity(result.relation)
        || !execution::valid_identity(result.structure)
        || result.structure_epoch.value == 0u
        || !execution::valid_identity(result.work_window)
        || !valid_persistent_axis(result.source_axis)
        || !valid_persistent_axis(result.destination_axis))
        return semantic_geometry_image_status_v1::invalid_semantic_data;
    const semantic_geometry_image_status_v1 semantic_status =
        validate_semantic_sections(image, result, validation_workspace);
    if (semantic_status != semantic_geometry_image_status_v1::ok)
        return semantic_status;
    *view = result;
    return semantic_geometry_image_status_v1::ok;
}

semantic_geometry_image_status_v1 rebind_semantic_geometry_image_v1(
    const semantic_geometry_image_view_v1 &validated_view,
    const void *new_image_base,
    u64 new_image_bytes,
    semantic_geometry_image_view_v1 *rebound_view) noexcept {
    if (validated_view.image_base == nullptr || validated_view.image_bytes == 0u
        || new_image_base == nullptr || rebound_view == nullptr
        || new_image_bytes != validated_view.image_bytes)
        return semantic_geometry_image_status_v1::incompatible_relocation;
    if (reinterpret_cast<std::uintptr_t>(new_image_base)
            % semantic_geometry_image_alignment_v1 != 0u
        || new_image_bytes < semantic_geometry_image_header_bytes_v1)
        return semantic_geometry_image_status_v1::incompatible_relocation;
    const auto *image = static_cast<const u8 *>(new_image_base);
    if (std::memcmp(image, image_magic, sizeof(image_magic)) != 0
        || read_u32(image, header_kind_offset)
            != semantic_geometry_image_kind_v1
        || read_u32(image, header_schema_offset)
            != semantic_geometry_image_schema_version_v1
        || read_u64(image, header_image_bytes_offset) != new_image_bytes
        || read_u64(image, header_geometry_low_offset)
            != validated_view.geometry_identity.low
        || read_u64(image, header_geometry_high_offset)
            != validated_view.geometry_identity.high
        || hash_image_with_zeros(image, new_image_bytes, fnv1a_offset, false)
            != read_u64(image, header_checksum_offset)
        || hash_image_with_zeros(image, new_image_bytes, fnv1a_offset, true)
            != validated_view.geometry_identity.low
        || hash_image_with_zeros(image, new_image_bytes, geometry_high_seed,
                true) != validated_view.geometry_identity.high)
        return semantic_geometry_image_status_v1::incompatible_relocation;
    semantic_geometry_image_view_v1 result = validated_view;
    result.image_base = new_image_base;
    result.image_bytes = new_image_bytes;
    *rebound_view = result;
    return semantic_geometry_image_status_v1::ok;
}

semantic_geometry_image_status_v1 find_semantic_geometry_section_v1(
    const semantic_geometry_image_view_v1 &validated_view,
    u32 section_kind,
    semantic_geometry_section_view_v1 *section_view) noexcept {
    if (validated_view.image_base == nullptr || section_view == nullptr
        || section_kind == 0u)
        return semantic_geometry_image_status_v1::invalid_argument;
    const auto *image = static_cast<const u8 *>(validated_view.image_base);
    for (u32 index = 0u; index < validated_view.section_count; ++index) {
        const section_metadata section = read_section_entry(image, index);
        if (section.kind != section_kind)
            continue;
        semantic_geometry_section_view_v1 result{};
        result.kind = section.kind;
        result.schema_version = section.schema_version;
        result.flags = section.flags;
        result.data = image + section.data_offset;
        result.data_bytes = section.data_bytes;
        result.element_count = section.element_count;
        result.element_bytes = section.element_bytes;
        result.alignment = section.alignment;
        result.checksum = section.checksum;
        *section_view = result;
        return semantic_geometry_image_status_v1::ok;
    }
    return semantic_geometry_image_status_v1::section_not_found;
}

} // namespace cellerator::geometry::persistence
