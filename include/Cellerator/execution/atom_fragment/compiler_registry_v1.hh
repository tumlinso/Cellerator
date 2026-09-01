#pragma once

#include <Cellerator/execution/joint_compiler/atom_fragment_request_v1.hh>
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
    const joint_compiler::atom_fragment_request_v1 &request,
    std::uint64_t candidate_id,
    program::prepared_program_v2 *output) noexcept;

struct fragment_compiler_entry_v1 {
    joint_compiler::persistent_identity_v1 source_identity{};
    std::uint64_t candidate_id = 0u;
    joint_compiler::persistent_identity_v1 compiler_identity{};
    const void *source_context = nullptr;
    source_linked_fragment_compile_v1 compile = nullptr;
};

struct fragment_compiler_registry_v1 {
    const fragment_compiler_entry_v1 *entries = nullptr;
    std::uint64_t entry_count = 0u;
};

enum class fragment_compiler_registry_status_code_v1 : std::uint8_t {
    success = 0u,
    missing_entries,
    invalid_source_identity,
    invalid_candidate,
    invalid_compiler_identity,
    missing_source_context,
    missing_compile_function,
    duplicate_or_unordered_entry,
};

struct fragment_compiler_registry_status_v1 {
    fragment_compiler_registry_status_code_v1 code =
        fragment_compiler_registry_status_code_v1::success;
    std::uint64_t index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == fragment_compiler_registry_status_code_v1::success;
    }
};

fragment_compiler_registry_status_v1 validate_fragment_compiler_registry_v1(
    const fragment_compiler_registry_v1 &registry) noexcept;

const fragment_compiler_entry_v1 *find_fragment_compiler_v1(
    const fragment_compiler_registry_v1 &registry,
    joint_compiler::persistent_identity_v1 source_identity,
    std::uint64_t candidate_id) noexcept;

} // namespace cellerator::execution::atom_fragment
