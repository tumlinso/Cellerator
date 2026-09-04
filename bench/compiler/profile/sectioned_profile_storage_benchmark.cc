#include <Cellerator/compiler/profile/implement_sectioned_binary_profile_storage_v1.hh>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace profile = cellerator::compiler::profile::v1;

int main() {
    constexpr std::uint32_t section_count = 8u;
    constexpr std::uint64_t section_bytes = 1u << 20u;
    std::vector<unsigned char> payload(section_bytes, 0x5au);
    profile::profile_section_source_v1 sections[section_count]{};
    for (std::uint32_t i = 0; i < section_count; ++i)
        sections[i] = {0x80000100u + i, 1u, profile::profile_section_flag_optional,
            profile::profile_compression_v1::none, i + 1u, i + 101u,
            payload.data(), payload.size(), payload.size()};
    profile::profile_artifact_build_request_v1 request{};
    request.header = profile::make_profile_artifact_header_v1(1u, 2u, 3u, 4u, 1u);
    request.sections = sections;
    request.section_count = section_count;
    profile::profile_artifact_requirements_v1 requirements{};
    if (profile::query_profile_artifact_requirements_v1(request, &requirements)
        != profile::profile_storage_status_v1::ok)
        return 1;
    const auto allocation_bytes = (requirements.image_bytes + 63u) & ~63ull;
    auto *storage = std::aligned_alloc(64u, allocation_bytes);
    profile::profile_artifact_view_v1 view{};
    if (storage == nullptr || profile::build_profile_artifact_v1(
            request, {storage, allocation_bytes}, &view)
            != profile::profile_storage_status_v1::ok)
        return 2;

    char path[] = "/tmp/cellerator-profile-bench-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0 || write(fd, storage, requirements.image_bytes)
        != static_cast<ssize_t>(requirements.image_bytes))
        return 3;
    void *mapping = mmap(nullptr, requirements.image_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED)
        return 4;
    constexpr std::uint32_t repeats = 32u;
    const auto begin = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < repeats; ++i)
        if (profile::validate_profile_artifact_v1(mapping, requirements.image_bytes, &view)
            != profile::profile_storage_status_v1::ok)
            return 5;
    const auto end = std::chrono::steady_clock::now();
    profile::profile_section_view_v1 section{};
    const auto access_begin = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < repeats * 1000u; ++i)
        profile::find_profile_section_v1(view, 0x80000107u, &section);
    const auto access_end = std::chrono::steady_clock::now();
    const auto validation_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - begin).count() / repeats;
    const auto access_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        access_end - access_begin).count() / (repeats * 1000u);
    const auto metadata_bytes = profile::profile_artifact_header_bytes_v1
        + section_count * profile::profile_section_entry_bytes_v1;
    std::cout << "image_bytes=" << requirements.image_bytes
              << " metadata_bytes=" << metadata_bytes
              << " mapped_validation_ns=" << validation_ns
              << " mapped_section_lookup_ns=" << access_ns
              << " validation_read_amplification=2.0\n";
    munmap(mapping, requirements.image_bytes);
    close(fd);
    unlink(path);
    std::free(storage);
}
