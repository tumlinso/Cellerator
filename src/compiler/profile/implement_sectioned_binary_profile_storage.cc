#include <Cellerator/compiler/profile/implement_sectioned_binary_profile_storage_v1.hh>

#include <cstring>
#include <limits>

namespace cellerator::compiler::profile::v1 {
namespace {

constexpr std::uint64_t checksum_basis = 1469598103934665603ull;
constexpr std::uint64_t checksum_prime = 1099511628211ull;

bool add_overflow(std::uint64_t a, std::uint64_t b,
                  std::uint64_t *result) noexcept {
    if (b > std::numeric_limits<std::uint64_t>::max() - a)
        return true;
    *result = a + b;
    return false;
}

bool align_up(std::uint64_t value, std::uint64_t alignment,
              std::uint64_t *result) noexcept {
    const auto mask = alignment - 1u;
    std::uint64_t added = 0u;
    if (add_overflow(value, mask, &added))
        return false;
    *result = added & ~mask;
    return true;
}

std::uint64_t checksum_update(std::uint64_t value,
                              const unsigned char *data,
                              std::uint64_t bytes) noexcept {
    for (std::uint64_t i = 0; i < bytes; ++i) {
        value ^= data[i];
        value *= checksum_prime;
    }
    return value;
}

std::uint64_t image_checksum(const unsigned char *image,
                             std::uint64_t bytes) noexcept {
    constexpr auto checksum_offset = offsetof(profile_artifact_header_v1,
                                               content_checksum);
    auto value = checksum_basis;
    for (std::uint64_t i = 0; i < bytes; ++i) {
        const auto byte = i >= checksum_offset
                && i < checksum_offset + sizeof(std::uint64_t)
            ? 0u : image[i];
        value ^= byte;
        value *= checksum_prime;
    }
    return value;
}

bool known_kind(std::uint32_t kind) noexcept {
    return kind >= static_cast<std::uint32_t>(profile_section_kind_v1::named_environment)
        && kind <= static_cast<std::uint32_t>(profile_section_kind_v1::semantic_attachment);
}

bool valid_source(const profile_section_source_v1 &source) noexcept {
    if (source.kind == 0u || source.schema_version == 0u
        || (source.identity_low == 0u && source.identity_high == 0u)
        || (source.stored_bytes != 0u && source.data == nullptr))
        return false;
    if (!known_kind(source.kind)
        && (source.flags & profile_section_flag_optional) == 0u)
        return false;
    if (source.compression == profile_compression_v1::none)
        return source.logical_bytes == source.stored_bytes;
    return source.logical_bytes >= source.stored_bytes;
}

}  // namespace

std::uint64_t profile_storage_checksum_v1(
    const void *data, std::uint64_t bytes) noexcept {
    if (data == nullptr && bytes != 0u)
        return 0u;
    return checksum_update(checksum_basis,
        static_cast<const unsigned char *>(data), bytes);
}

profile_storage_status_v1 query_profile_artifact_requirements_v1(
    const profile_artifact_build_request_v1 &request,
    profile_artifact_requirements_v1 *requirements) noexcept {
    if (requirements == nullptr
        || (request.section_count != 0u && request.sections == nullptr))
        return profile_storage_status_v1::invalid_argument;
    if (validate_profile_artifact_charter_v1(request.header)
        != profile_artifact_charter_status_v1::ok)
        return profile_storage_status_v1::invalid_header;

    profile_artifact_requirements_v1 result{};
    result.directory_bytes = static_cast<std::uint64_t>(request.section_count)
        * profile_section_entry_bytes_v1;
    std::uint64_t cursor = 0u;
    if (!align_up(profile_artifact_header_bytes_v1,
                  profile_artifact_alignment_v1, &cursor)
        || add_overflow(cursor, result.directory_bytes, &cursor))
        return profile_storage_status_v1::arithmetic_overflow;
    const auto unaligned_directory_end = cursor;
    if (!align_up(cursor, profile_artifact_alignment_v1, &cursor))
        return profile_storage_status_v1::arithmetic_overflow;
    result.padding_bytes = cursor - unaligned_directory_end;

    for (std::uint32_t i = 0; i < request.section_count; ++i) {
        const auto &source = request.sections[i];
        if (!valid_source(source))
            return !known_kind(source.kind)
                    && (source.flags & profile_section_flag_optional) == 0u
                ? profile_storage_status_v1::unknown_required_section
                : profile_storage_status_v1::invalid_compression;
        for (std::uint32_t j = 0; j < i; ++j)
            if (source.identity_low == request.sections[j].identity_low
                && source.identity_high == request.sections[j].identity_high)
                return profile_storage_status_v1::duplicate_identity;
        const auto before = cursor;
        if (!align_up(cursor, profile_artifact_alignment_v1, &cursor)
            || add_overflow(cursor, source.stored_bytes, &cursor))
            return profile_storage_status_v1::arithmetic_overflow;
        result.padding_bytes += cursor - source.stored_bytes - before;
        if (add_overflow(result.payload_bytes, source.stored_bytes,
                         &result.payload_bytes))
            return profile_storage_status_v1::arithmetic_overflow;
    }
    result.image_bytes = cursor;
    *requirements = result;
    return profile_storage_status_v1::ok;
}

profile_storage_status_v1 build_profile_artifact_v1(
    const profile_artifact_build_request_v1 &request,
    profile_artifact_buffer_v1 buffer,
    profile_artifact_view_v1 *view) noexcept {
    if (buffer.data == nullptr || view == nullptr)
        return profile_storage_status_v1::invalid_argument;
    if (reinterpret_cast<std::uintptr_t>(buffer.data)
        % profile_artifact_alignment_v1 != 0u)
        return profile_storage_status_v1::misaligned_image;
    profile_artifact_requirements_v1 requirements{};
    const auto queried = query_profile_artifact_requirements_v1(request, &requirements);
    if (queried != profile_storage_status_v1::ok)
        return queried;
    if (buffer.capacity_bytes < requirements.image_bytes)
        return profile_storage_status_v1::insufficient_capacity;

    auto *image = static_cast<unsigned char *>(buffer.data);
    std::memset(image, 0, static_cast<std::size_t>(requirements.image_bytes));
    auto header = request.header;
    header.image_bytes = requirements.image_bytes;
    header.section_count = request.section_count;
    header.section_entry_bytes = profile_section_entry_bytes_v1;
    align_up(profile_artifact_header_bytes_v1, profile_artifact_alignment_v1,
             &header.section_directory_offset);
    header.content_checksum = 0u;
    std::memcpy(image, &header, sizeof(header));
    auto *entries = reinterpret_cast<profile_section_entry_v1 *>(
        image + header.section_directory_offset);
    std::uint64_t cursor = header.section_directory_offset
        + requirements.directory_bytes;
    align_up(cursor, profile_artifact_alignment_v1, &cursor);
    for (std::uint32_t i = 0; i < request.section_count; ++i) {
        const auto &source = request.sections[i];
        align_up(cursor, profile_artifact_alignment_v1, &cursor);
        entries[i] = {source.kind, source.schema_version, source.flags,
            source.compression, source.identity_low, source.identity_high,
            cursor, source.stored_bytes, source.logical_bytes,
            profile_storage_checksum_v1(source.data, source.stored_bytes)};
        if (source.stored_bytes != 0u)
            std::memcpy(image + cursor, source.data,
                        static_cast<std::size_t>(source.stored_bytes));
        cursor += source.stored_bytes;
    }
    header.content_checksum = image_checksum(image, header.image_bytes);
    std::memcpy(image, &header, sizeof(header));
    return validate_profile_artifact_v1(image, header.image_bytes, view);
}

profile_storage_status_v1 validate_profile_artifact_v1(
    const void *image_data, std::uint64_t available_bytes,
    profile_artifact_view_v1 *view) noexcept {
    if (image_data == nullptr || view == nullptr)
        return profile_storage_status_v1::invalid_argument;
    if (reinterpret_cast<std::uintptr_t>(image_data)
        % profile_artifact_alignment_v1 != 0u)
        return profile_storage_status_v1::misaligned_image;
    if (available_bytes < sizeof(profile_artifact_header_v1))
        return profile_storage_status_v1::invalid_header;
    const auto *image = static_cast<const unsigned char *>(image_data);
    profile_artifact_header_v1 header{};
    std::memcpy(&header, image, sizeof(header));
    if (validate_profile_artifact_charter_v1(header)
        != profile_artifact_charter_status_v1::ok)
        return profile_storage_status_v1::invalid_header;
    if (header.image_bytes > available_bytes)
        return profile_storage_status_v1::section_out_of_bounds;
    if (header.section_entry_bytes != profile_section_entry_bytes_v1
        || header.section_directory_offset % profile_artifact_alignment_v1 != 0u)
        return profile_storage_status_v1::invalid_directory;
    const auto directory_bytes = static_cast<std::uint64_t>(header.section_count)
        * profile_section_entry_bytes_v1;
    if (header.section_directory_offset > header.image_bytes
        || directory_bytes > header.image_bytes - header.section_directory_offset)
        return profile_storage_status_v1::invalid_directory;
    const auto *entries = reinterpret_cast<const profile_section_entry_v1 *>(
        image + header.section_directory_offset);
    for (std::uint32_t i = 0; i < header.section_count; ++i) {
        const auto &entry = entries[i];
        if (!known_kind(entry.kind)
            && (entry.flags & profile_section_flag_optional) == 0u)
            return profile_storage_status_v1::unknown_required_section;
        if (entry.offset % profile_artifact_alignment_v1 != 0u
            || entry.offset > header.image_bytes
            || entry.stored_bytes > header.image_bytes - entry.offset)
            return profile_storage_status_v1::section_out_of_bounds;
        if ((entry.compression == profile_compression_v1::none
             && entry.logical_bytes != entry.stored_bytes)
            || (entry.compression != profile_compression_v1::none
                && entry.logical_bytes < entry.stored_bytes))
            return profile_storage_status_v1::invalid_compression;
        for (std::uint32_t j = 0; j < i; ++j)
            if (entry.identity_low == entries[j].identity_low
                && entry.identity_high == entries[j].identity_high)
                return profile_storage_status_v1::duplicate_identity;
        if (profile_storage_checksum_v1(image + entry.offset, entry.stored_bytes)
            != entry.checksum)
            return profile_storage_status_v1::section_checksum_mismatch;
    }
    if (image_checksum(image, header.image_bytes) != header.content_checksum)
        return profile_storage_status_v1::image_checksum_mismatch;
    *view = {image, header.image_bytes, header, entries};
    return profile_storage_status_v1::ok;
}

profile_storage_status_v1 find_profile_section_v1(
    const profile_artifact_view_v1 &artifact, std::uint32_t kind,
    profile_section_view_v1 *section) noexcept {
    if (artifact.image_base == nullptr || artifact.sections == nullptr
        || section == nullptr)
        return profile_storage_status_v1::invalid_argument;
    const auto *base = static_cast<const unsigned char *>(artifact.image_base);
    for (std::uint32_t i = 0; i < artifact.header.section_count; ++i) {
        if (artifact.sections[i].kind == kind) {
            *section = {artifact.sections[i], base + artifact.sections[i].offset};
            return profile_storage_status_v1::ok;
        }
    }
    return profile_storage_status_v1::section_not_found;
}

}  // namespace cellerator::compiler::profile::v1
