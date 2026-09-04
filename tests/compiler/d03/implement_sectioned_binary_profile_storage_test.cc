#include <Cellerator/compiler/profile/implement_sectioned_binary_profile_storage_v1.hh>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace profile = cellerator::compiler::profile::v1;

int main() {
    const unsigned char environment[] = {1u, 2u, 3u, 4u};
    const unsigned char compressed[] = {9u, 8u};
    const unsigned char extension[] = {7u};
    const profile::profile_section_source_v1 sections[] = {
        {static_cast<std::uint32_t>(profile::profile_section_kind_v1::named_environment),
         1u, profile::profile_section_flag_none, profile::profile_compression_v1::none,
         10u, 11u, environment, sizeof(environment), sizeof(environment)},
        {static_cast<std::uint32_t>(profile::profile_section_kind_v1::value_evidence),
         1u, profile::profile_section_flag_optional, profile::profile_compression_v1::zstd,
         12u, 13u, compressed, sizeof(compressed), 16u},
        {0x90000001u, 1u, profile::profile_section_flag_optional,
         profile::profile_compression_v1::application_defined,
         14u, 15u, extension, sizeof(extension), 4u}};
    profile::profile_artifact_build_request_v1 request{};
    request.header = profile::make_profile_artifact_header_v1(1u, 2u, 3u, 4u, 5u);
    request.header.flags |= profile::profile_artifact_flag_compressed_sections;
    request.sections = sections;
    request.section_count = 3u;

    profile::profile_artifact_requirements_v1 requirements{};
    assert(profile::query_profile_artifact_requirements_v1(request, &requirements)
           == profile::profile_storage_status_v1::ok);
    const auto allocation_bytes = (requirements.image_bytes + 63u) & ~63ull;
    auto *storage = static_cast<unsigned char *>(std::aligned_alloc(64u, allocation_bytes));
    assert(storage != nullptr);
    profile::profile_artifact_view_v1 built{};
    assert(profile::build_profile_artifact_v1(request,
               {storage, allocation_bytes}, &built)
           == profile::profile_storage_status_v1::ok);
    assert(built.header.section_count == 3u);
    profile::profile_section_view_v1 found{};
    assert(profile::find_profile_section_v1(built, 0x90000001u, &found)
           == profile::profile_storage_status_v1::ok);
    assert(found.entry.compression == profile::profile_compression_v1::application_defined);
    assert(*static_cast<const unsigned char *>(found.stored_data) == 7u);

    char path[] = "/tmp/cellerator-profile-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    assert(write(fd, storage, requirements.image_bytes)
           == static_cast<ssize_t>(requirements.image_bytes));
    void *mapping = mmap(nullptr, requirements.image_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(mapping != MAP_FAILED);
    profile::profile_artifact_view_v1 mapped{};
    assert(profile::validate_profile_artifact_v1(mapping, requirements.image_bytes, &mapped)
           == profile::profile_storage_status_v1::ok);
    assert(profile::find_profile_section_v1(mapped,
               static_cast<std::uint32_t>(profile::profile_section_kind_v1::named_environment),
               &found) == profile::profile_storage_status_v1::ok);
    assert(std::memcmp(found.stored_data, environment, sizeof(environment)) == 0);
    munmap(mapping, requirements.image_bytes);
    close(fd);
    unlink(path);

    auto *corrupt = static_cast<unsigned char *>(std::aligned_alloc(64u, allocation_bytes));
    assert(corrupt != nullptr);
    std::memcpy(corrupt, storage, requirements.image_bytes);
    corrupt[built.sections[0].offset] ^= 1u;
    assert(profile::validate_profile_artifact_v1(corrupt, requirements.image_bytes, &mapped)
           == profile::profile_storage_status_v1::section_checksum_mismatch);
    std::memcpy(corrupt, storage, requirements.image_bytes);
    reinterpret_cast<profile::profile_artifact_header_v1 *>(corrupt)->endian = 0u;
    assert(profile::validate_profile_artifact_v1(corrupt, requirements.image_bytes, &mapped)
           == profile::profile_storage_status_v1::invalid_header);
    assert(profile::validate_profile_artifact_v1(storage, requirements.image_bytes - 1u, &mapped)
           == profile::profile_storage_status_v1::section_out_of_bounds);

    auto required_unknown = sections[2];
    required_unknown.flags = profile::profile_section_flag_none;
    request.sections = &required_unknown;
    request.section_count = 1u;
    assert(profile::query_profile_artifact_requirements_v1(request, &requirements)
           == profile::profile_storage_status_v1::unknown_required_section);
    std::free(corrupt);
    std::free(storage);
}
