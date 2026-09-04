#pragma once

#include <Cellerator/compiler/profile/implement_sectioned_binary_profile_storage_v1.hh>
#include <Cellerator/compiler/profile/represent_evidence_provenance_and_revision_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::profile::v1 {

inline constexpr std::uint32_t profile_artifact_contract_version_v1 = 1u;

// Stable compiler-facing identity for a CELLPRF1 artifact. Storage remains
// sectioned and pointer-free; this descriptor does not own mapped bytes.
struct profile_artifact_v1 {
    std::uint32_t contract_version = profile_artifact_contract_version_v1;
    std::uint32_t flags = 0u;
    profile_identity_v1 artifact{};
    profile_identity_v1 semantic_subject{};
    profile_identity_v1 environment{};
    profile_revision_v1 revision{};
    std::uint64_t byte_count = 0u;
    std::uint64_t section_count = 0u;
    std::uint64_t content_fingerprint = 0u;
};

static_assert(std::is_standard_layout_v<profile_artifact_v1>);
static_assert(std::is_trivially_copyable_v<profile_artifact_v1>);

}  // namespace cellerator::compiler::profile::v1
