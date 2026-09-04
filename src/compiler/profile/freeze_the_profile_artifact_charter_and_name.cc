#include <Cellerator/compiler/profile/freeze_the_profile_artifact_charter_and_name_v1.hh>

#include <cstring>

namespace cellerator::compiler::profile::v1 {

profile_artifact_header_v1 make_profile_artifact_header_v1(
    std::uint64_t artifact_identity_low,
    std::uint64_t artifact_identity_high,
    std::uint64_t semantic_environment_identity_low,
    std::uint64_t semantic_environment_identity_high,
    std::uint64_t evidence_revision) noexcept {
    profile_artifact_header_v1 header{};
    std::memcpy(header.magic, profile_artifact_magic_v1,
                sizeof(profile_artifact_magic_v1));
    header.artifact_identity_low = artifact_identity_low;
    header.artifact_identity_high = artifact_identity_high;
    header.semantic_environment_identity_low = semantic_environment_identity_low;
    header.semantic_environment_identity_high = semantic_environment_identity_high;
    header.evidence_revision = evidence_revision;
    return header;
}

profile_artifact_charter_status_v1 validate_profile_artifact_charter_v1(
    const profile_artifact_header_v1 &header) noexcept {
    if (std::memcmp(header.magic, profile_artifact_magic_v1,
                    sizeof(profile_artifact_magic_v1)) != 0)
        return profile_artifact_charter_status_v1::invalid_magic;
    if (header.schema_version != profile_artifact_schema_version_v1)
        return profile_artifact_charter_status_v1::unsupported_schema;
    if (header.header_bytes != profile_artifact_header_bytes_v1)
        return profile_artifact_charter_status_v1::invalid_header_size;
    if (header.endian != profile_artifact_endian_marker_v1)
        return profile_artifact_charter_status_v1::unsupported_endian;
    if (header.alignment != profile_artifact_alignment_v1)
        return profile_artifact_charter_status_v1::invalid_alignment;
    if (header.image_bytes < header.header_bytes)
        return profile_artifact_charter_status_v1::truncated_image;
    if (header.artifact_identity_low == 0u && header.artifact_identity_high == 0u)
        return profile_artifact_charter_status_v1::missing_artifact_identity;
    if (header.semantic_environment_identity_low == 0u
        && header.semantic_environment_identity_high == 0u)
        return profile_artifact_charter_status_v1::missing_environment_identity;
    if ((header.flags & profile_artifact_flag_data_derived) == 0u)
        return profile_artifact_charter_status_v1::not_data_derived;
    if (header.reserved != 0u)
        return profile_artifact_charter_status_v1::reserved_field_nonzero;
    return profile_artifact_charter_status_v1::ok;
}

}  // namespace cellerator::compiler::profile::v1
