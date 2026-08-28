#include "Cellerator/geometry/persistence/execution_image_v2.hh"

namespace cellpack::persistence {

validation_result load_cpk1_v1_compatibility_host(
    const execution_image_v2_view &validated_host_view,
    u32 projection_index,
    const persistent_packing_payload_compatibility &expected,
    persistent_packing_payload_view *out) noexcept {
    if (out == nullptr)
        return validation_error(validation_code::null_pointer, invalid_id,
            "CPK1 compatibility output is null");
    if (projection_index >= validated_host_view.header.projection_count)
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "CPK1 compatibility projection index is out of range");
    const execution_projection_entry_v1 &projection =
        validated_host_view.projections[projection_index];
    if (projection.kind != execution_projection_kind::native_row_masked
        || projection.payload_section == invalid_directory_index
        || projection.payload_section >= validated_host_view.header.section_count
        || validated_host_view.sections[projection.payload_section].kind
            != execution_section_kind::cpk1_v1_compatibility)
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "projection does not contain a CPK1 compatibility section");
    const execution_section_entry_v1 &section =
        validated_host_view.sections[projection.payload_section];
    const auto *section_bytes = static_cast<const unsigned char *>(
        validated_host_view.image_base) + section.offset;
    return validate_persistent_packing_payload_host(section_bytes,
        static_cast<std::size_t>(section.bytes), expected, out);
}

} // namespace cellpack::persistence
