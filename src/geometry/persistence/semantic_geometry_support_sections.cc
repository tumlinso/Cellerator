#include <Cellerator/geometry/persistence/semantic_geometry_image_v1.hh>
#include <Cellerator/geometry/support_atlas.hh>

#include <cstdint>
#include <cstring>
#include <limits>

namespace cellerator::geometry::persistence {

inline constexpr u32 semantic_geometry_support_atlas_section_kind_v1 =
    semantic_geometry_first_optional_section_kind_v1;
inline constexpr u32 semantic_geometry_support_reference_section_kind_v1 =
    semantic_geometry_first_optional_section_kind_v1 + 1u;
inline constexpr u32 support_atlas_reference_schema_version_v1 = 1u;

enum class support_section_status_v1 : u8 {
    success = 0u,
    invalid_argument = 1u,
    invalid_atlas = 2u,
    arithmetic_overflow = 3u,
    insufficient_capacity = 4u,
    invalid_section = 5u
};

struct support_atlas_section_requirements_v1 {
    u64 section_bytes = 0u;
    u64 alignment = semantic_geometry_image_alignment_v1;
};

// Pointer-free reference to evidence stored and transported by an external
// owner. Cellerator records identity and byte extent but no path or transport
// policy. content_identity authenticates the referenced support section.
struct support_atlas_external_reference_v1 {
    u32 schema_version = support_atlas_reference_schema_version_v1;
    u32 record_bytes = sizeof(support_atlas_external_reference_v1);
    u64 evidence_identity = 0u;
    u64 relation_identity = 0u;
    u64 structure_identity = 0u;
    u64 structure_epoch = 0u;
    u64 source_axis_identity = 0u;
    u64 destination_axis_identity = 0u;
    u64 object_identity_low = 0u;
    u64 object_identity_high = 0u;
    u64 content_identity = 0u;
    u64 byte_offset = 0u;
    u64 byte_count = 0u;
    u64 reserved[3]{};
};

namespace {

bool checked_add(u64 left, u64 right, u64 *out) noexcept {
    if (right > std::numeric_limits<u64>::max() - left)
        return false;
    *out = left + right;
    return true;
}

bool checked_multiply(u64 left, u64 right, u64 *out) noexcept {
    if (left != 0u && right > std::numeric_limits<u64>::max() / left)
        return false;
    *out = left * right;
    return true;
}

bool align_up(u64 value, u64 alignment, u64 *out) noexcept {
    const u64 mask = alignment - 1u;
    return value <= std::numeric_limits<u64>::max() - mask
        && ((*out = (value + mask) & ~mask), true);
}

bool valid_pointer_count(const void *pointer, u64 count) noexcept {
    return count == 0u || pointer != nullptr;
}

bool valid_atlas(const support_atlas_view_v1 &atlas) noexcept {
    if (atlas.schema_version != support_atlas_schema_version_v1
        || atlas.reserved != 0u || atlas.evidence_identity == 0u
        || atlas.relation_identity == 0u || atlas.structure_identity == 0u
        || atlas.structure_epoch == 0u || atlas.source_axis_identity == 0u
        || atlas.destination_axis_identity == 0u
        || atlas.provenance.schema_version != support_atlas_schema_version_v1
        || atlas.provenance.sampling_algorithm_version
            != support_sampling_algorithm_version_v1
        || atlas.provenance.normalization_algorithm_version
            != support_normalization_algorithm_version_v1
        || atlas.provenance.reserved != 0u)
        return false;
    return valid_pointer_count(atlas.prevalence, atlas.prevalence_count)
        && valid_pointer_count(atlas.destination_degrees,
            atlas.destination_degree_count)
        && valid_pointer_count(atlas.co_support, atlas.co_support_count)
        && valid_pointer_count(atlas.affinity, atlas.affinity_count)
        && valid_pointer_count(atlas.communities, atlas.community_count)
        && valid_pointer_count(atlas.work_signatures,
            atlas.work_signature_count)
        && valid_pointer_count(atlas.strata, atlas.stratum_count)
        && valid_pointer_count(atlas.stability, atlas.stability_count)
        && valid_pointer_count(atlas.exact_rescans, atlas.exact_rescan_count)
        && valid_pointer_count(atlas.validation_summaries,
            atlas.validation_summary_count);
}

bool append_requirement(u64 count, u64 width, u64 *cursor) noexcept {
    u64 bytes = 0u;
    return align_up(*cursor, 8u, cursor)
        && checked_multiply(count, width, &bytes)
        && checked_add(*cursor, bytes, cursor);
}

void append_array(u8 *section, u64 *cursor, const void *data, u64 count,
                  u64 width, support_atlas_section_span_v1 *span) noexcept {
    align_up(*cursor, 8u, cursor);
    span->byte_offset = count == 0u ? 0u : *cursor;
    span->element_count = count;
    if (count != 0u) {
        const u64 bytes = count * width;
        std::memcpy(section + *cursor, data, static_cast<std::size_t>(bytes));
        *cursor += bytes;
    }
}

bool valid_span(const support_atlas_section_header_v1 &header,
                support_atlas_section_span_v1 span, u64 width) noexcept {
    if (span.element_count == 0u)
        return span.byte_offset == 0u;
    u64 bytes = 0u;
    return span.byte_offset >= header.header_bytes
        && span.byte_offset % 8u == 0u
        && checked_multiply(span.element_count, width, &bytes)
        && span.byte_offset <= header.section_bytes
        && bytes <= header.section_bytes - span.byte_offset;
}

template<typename T>
const T *span_pointer(const u8 *section,
                      support_atlas_section_span_v1 span) noexcept {
    return span.element_count == 0u
        ? nullptr : reinterpret_cast<const T *>(section + span.byte_offset);
}

} // namespace

support_section_status_v1 query_support_atlas_section_requirements_v1(
    const support_atlas_view_v1 &atlas,
    support_atlas_section_requirements_v1 *out) noexcept {
    if (out == nullptr)
        return support_section_status_v1::invalid_argument;
    *out = {};
    if (!valid_atlas(atlas))
        return support_section_status_v1::invalid_atlas;
    u64 cursor = sizeof(support_atlas_section_header_v1);
    const u64 counts[] = {atlas.prevalence_count,
        atlas.destination_degree_count, atlas.co_support_count,
        atlas.affinity_count, atlas.community_count,
        atlas.work_signature_count, atlas.stratum_count,
        atlas.stability_count, atlas.exact_rescan_count,
        atlas.validation_summary_count};
    const u64 widths[] = {sizeof(source_prevalence_v1),
        sizeof(destination_degree_v1), sizeof(co_support_record_v1),
        sizeof(source_affinity_record_v1), sizeof(community_assignment_v1),
        sizeof(work_signature_v1), sizeof(biological_stratum_v1),
        sizeof(resampling_stability_v1), sizeof(exact_rescan_summary_v1),
        sizeof(support_validation_summary_v1)};
    for (u32 index = 0u; index < 10u; ++index)
        if (!append_requirement(counts[index], widths[index], &cursor))
            return support_section_status_v1::arithmetic_overflow;
    if (!align_up(cursor, 8u, &cursor))
        return support_section_status_v1::arithmetic_overflow;
    out->section_bytes = cursor;
    return support_section_status_v1::success;
}

support_section_status_v1 build_support_atlas_optional_section_v1(
    const support_atlas_view_v1 &atlas, void *storage, u64 storage_capacity,
    semantic_geometry_optional_section_v1 *out) noexcept {
    if (out == nullptr || storage == nullptr
        || reinterpret_cast<std::uintptr_t>(storage) % 8u != 0u)
        return support_section_status_v1::invalid_argument;
    *out = {};
    support_atlas_section_requirements_v1 required{};
    const support_section_status_v1 status =
        query_support_atlas_section_requirements_v1(atlas, &required);
    if (status != support_section_status_v1::success)
        return status;
    if (storage_capacity < required.section_bytes)
        return support_section_status_v1::insufficient_capacity;
    auto *section = static_cast<u8 *>(storage);
    std::memset(section, 0, static_cast<std::size_t>(required.section_bytes));
    support_atlas_section_header_v1 header{};
    header.section_bytes = required.section_bytes;
    header.flags = atlas.flags;
    header.evidence_identity = atlas.evidence_identity;
    header.relation_identity = atlas.relation_identity;
    header.structure_identity = atlas.structure_identity;
    header.structure_epoch = atlas.structure_epoch;
    header.source_axis_identity = atlas.source_axis_identity;
    header.destination_axis_identity = atlas.destination_axis_identity;
    header.source_count = atlas.source_count;
    header.destination_count = atlas.destination_count;
    header.provenance = atlas.provenance;
    u64 cursor = sizeof(header);
    append_array(section, &cursor, atlas.prevalence, atlas.prevalence_count,
        sizeof(source_prevalence_v1), &header.prevalence);
    append_array(section, &cursor, atlas.destination_degrees,
        atlas.destination_degree_count, sizeof(destination_degree_v1),
        &header.destination_degrees);
    append_array(section, &cursor, atlas.co_support, atlas.co_support_count,
        sizeof(co_support_record_v1), &header.co_support);
    append_array(section, &cursor, atlas.affinity, atlas.affinity_count,
        sizeof(source_affinity_record_v1), &header.affinity);
    append_array(section, &cursor, atlas.communities, atlas.community_count,
        sizeof(community_assignment_v1), &header.communities);
    append_array(section, &cursor, atlas.work_signatures,
        atlas.work_signature_count, sizeof(work_signature_v1),
        &header.work_signatures);
    append_array(section, &cursor, atlas.strata, atlas.stratum_count,
        sizeof(biological_stratum_v1), &header.strata);
    append_array(section, &cursor, atlas.stability, atlas.stability_count,
        sizeof(resampling_stability_v1), &header.stability);
    append_array(section, &cursor, atlas.exact_rescans,
        atlas.exact_rescan_count, sizeof(exact_rescan_summary_v1),
        &header.exact_rescans);
    append_array(section, &cursor, atlas.validation_summaries,
        atlas.validation_summary_count, sizeof(support_validation_summary_v1),
        &header.validation_summaries);
    std::memcpy(section, &header, sizeof(header));
    *out = {semantic_geometry_support_atlas_section_kind_v1,
        support_atlas_section_schema_version_v1, atlas.flags,
        semantic_geometry_image_alignment_v1, storage, required.section_bytes};
    return support_section_status_v1::success;
}

support_section_status_v1 rebind_support_atlas_section_v1(
    const semantic_geometry_section_view_v1 &section_view,
    support_atlas_view_v1 *out) noexcept {
    if (out == nullptr || section_view.data == nullptr)
        return support_section_status_v1::invalid_argument;
    *out = {};
    if (section_view.kind != semantic_geometry_support_atlas_section_kind_v1
        || section_view.schema_version
            != support_atlas_section_schema_version_v1
        || section_view.data_bytes < sizeof(support_atlas_section_header_v1)
        || reinterpret_cast<std::uintptr_t>(section_view.data) % 8u != 0u)
        return support_section_status_v1::invalid_section;
    const auto *section = static_cast<const u8 *>(section_view.data);
    support_atlas_section_header_v1 header{};
    std::memcpy(&header, section, sizeof(header));
    if (header.schema_version != support_atlas_section_schema_version_v1
        || header.header_bytes != sizeof(header)
        || header.section_bytes != section_view.data_bytes
        || header.flags != section_view.flags || header.evidence_identity == 0u
        || header.relation_identity == 0u || header.structure_identity == 0u
        || header.structure_epoch == 0u || header.source_axis_identity == 0u
        || header.destination_axis_identity == 0u
        || header.provenance.schema_version != support_atlas_schema_version_v1
        || header.provenance.reserved != 0u)
        return support_section_status_v1::invalid_section;
    const support_atlas_section_span_v1 spans[] = {header.prevalence,
        header.destination_degrees, header.co_support, header.affinity,
        header.communities, header.work_signatures, header.strata,
        header.stability, header.exact_rescans, header.validation_summaries};
    const u64 widths[] = {sizeof(source_prevalence_v1),
        sizeof(destination_degree_v1), sizeof(co_support_record_v1),
        sizeof(source_affinity_record_v1), sizeof(community_assignment_v1),
        sizeof(work_signature_v1), sizeof(biological_stratum_v1),
        sizeof(resampling_stability_v1), sizeof(exact_rescan_summary_v1),
        sizeof(support_validation_summary_v1)};
    u64 previous_end = header.header_bytes;
    for (u32 index = 0u; index < 10u; ++index) {
        if (!valid_span(header, spans[index], widths[index]))
            return support_section_status_v1::invalid_section;
        if (spans[index].element_count == 0u)
            continue;
        const u64 bytes = spans[index].element_count * widths[index];
        if (spans[index].byte_offset < previous_end)
            return support_section_status_v1::invalid_section;
        previous_end = spans[index].byte_offset + bytes;
    }
    support_atlas_view_v1 result{};
    result.flags = header.flags;
    result.evidence_identity = header.evidence_identity;
    result.relation_identity = header.relation_identity;
    result.structure_identity = header.structure_identity;
    result.structure_epoch = header.structure_epoch;
    result.source_axis_identity = header.source_axis_identity;
    result.destination_axis_identity = header.destination_axis_identity;
    result.source_count = header.source_count;
    result.destination_count = header.destination_count;
    result.provenance = header.provenance;
    result.prevalence = span_pointer<source_prevalence_v1>(section, header.prevalence);
    result.prevalence_count = header.prevalence.element_count;
    result.destination_degrees = span_pointer<destination_degree_v1>(section,
        header.destination_degrees);
    result.destination_degree_count = header.destination_degrees.element_count;
    result.co_support = span_pointer<co_support_record_v1>(section, header.co_support);
    result.co_support_count = header.co_support.element_count;
    result.affinity = span_pointer<source_affinity_record_v1>(section, header.affinity);
    result.affinity_count = header.affinity.element_count;
    result.communities = span_pointer<community_assignment_v1>(section, header.communities);
    result.community_count = header.communities.element_count;
    result.work_signatures = span_pointer<work_signature_v1>(section,
        header.work_signatures);
    result.work_signature_count = header.work_signatures.element_count;
    result.strata = span_pointer<biological_stratum_v1>(section, header.strata);
    result.stratum_count = header.strata.element_count;
    result.stability = span_pointer<resampling_stability_v1>(section, header.stability);
    result.stability_count = header.stability.element_count;
    result.exact_rescans = span_pointer<exact_rescan_summary_v1>(section,
        header.exact_rescans);
    result.exact_rescan_count = header.exact_rescans.element_count;
    result.validation_summaries = span_pointer<support_validation_summary_v1>(
        section, header.validation_summaries);
    result.validation_summary_count = header.validation_summaries.element_count;
    if (!valid_atlas(result))
        return support_section_status_v1::invalid_section;
    *out = result;
    return support_section_status_v1::success;
}

support_section_status_v1 make_support_atlas_reference_section_v1(
    const support_atlas_external_reference_v1 &reference,
    semantic_geometry_optional_section_v1 *out) noexcept {
    if (out == nullptr)
        return support_section_status_v1::invalid_argument;
    *out = {};
    if (reference.schema_version != support_atlas_reference_schema_version_v1
        || reference.record_bytes != sizeof(reference)
        || reference.evidence_identity == 0u || reference.relation_identity == 0u
        || reference.structure_identity == 0u || reference.structure_epoch == 0u
        || reference.source_axis_identity == 0u
        || reference.destination_axis_identity == 0u
        || (reference.object_identity_low == 0u
            && reference.object_identity_high == 0u)
        || reference.content_identity == 0u || reference.byte_count == 0u
        || reference.byte_count
            > std::numeric_limits<u64>::max() - reference.byte_offset
        || reference.reserved[0] != 0u || reference.reserved[1] != 0u
        || reference.reserved[2] != 0u)
        return support_section_status_v1::invalid_argument;
    *out = {semantic_geometry_support_reference_section_kind_v1,
        support_atlas_reference_schema_version_v1, 0u,
        semantic_geometry_image_alignment_v1, &reference, sizeof(reference)};
    return support_section_status_v1::success;
}

} // namespace cellerator::geometry::persistence
