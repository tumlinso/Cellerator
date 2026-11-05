#pragma once

#include <Cellerator/execution/preparation/atom_fragment_request_v1.hh>
#include <Cellerator/execution/program/program_v2.h>

namespace cellerator::execution::atom_fragment {

enum class fragment_compile_status_code_v1 : std::uint8_t {
    success = 0u,
    unsupported,
    invalid_request,
    insufficient_workspace,
    preparation_failed,
};

using source_linked_fragment_compile_v1 = fragment_compile_status_code_v1 (*)(
    const void *source_context,
    const preparation::atom_fragment_request_v1 &request,
    std::uint64_t candidate_id,
    program::prepared_program_v2 *output) noexcept;

struct provider_entry_v1 {
    preparation::persistent_identity_v1 source_identity{};
    std::uint64_t candidate_id = 0u;
    preparation::persistent_identity_v1 provider_identity{};
    const void *source_context = nullptr;
    source_linked_fragment_compile_v1 compile = nullptr;
};

struct provider_registry_v1 {
    const provider_entry_v1 *entries = nullptr;
    std::uint64_t entry_count = 0u;
};

enum class provider_registry_status_code_v1 : std::uint8_t {
    success = 0u,
    missing_entries,
    invalid_source_identity,
    invalid_candidate,
    invalid_provider_identity,
    missing_source_context,
    missing_compile_function,
    duplicate_or_unordered_entry,
};

struct provider_registry_status_v1 {
    provider_registry_status_code_v1 code =
        provider_registry_status_code_v1::success;
    std::uint64_t index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == provider_registry_status_code_v1::success;
    }
};

provider_registry_status_v1 validate_provider_registry_v1(
    const provider_registry_v1 &registry) noexcept;

const provider_entry_v1 *find_provider_v1(
    const provider_registry_v1 &registry,
    preparation::persistent_identity_v1 source_identity,
    std::uint64_t candidate_id) noexcept;

} // namespace cellerator::execution::atom_fragment
