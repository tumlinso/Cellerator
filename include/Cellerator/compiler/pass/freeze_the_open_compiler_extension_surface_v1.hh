#pragma once

#include <Cellerator/compiler/pass/extension_v1.hh>
#include <Cellerator/compiler/pass/pass_v1.hh>
#include <Cellerator/compiler/pass/self_transform_v1.hh>

#include <cstdint>

namespace cellerator::compiler::pass::v1 {

inline constexpr std::uint32_t open_compiler_extension_abi_version_v1 = 1;
enum open_compiler_extension_capability_v1 : std::uint64_t {
    pass_management_v1 = 1ULL << 0U,
    pipeline_configuration_v1 = 1ULL << 1U,
    semantic_rewrite_v1 = 1ULL << 2U,
    planning_rewrite_v1 = 1ULL << 3U,
    realization_rewrite_v1 = 1ULL << 4U,
    stage_replacement_v1 = 1ULL << 5U,
    extension_registration_v1 = 1ULL << 6U,
    opaque_forwarding_v1 = 1ULL << 7U,
    capability_negotiation_v1 = 1ULL << 8U,
    same_compilation_staging_v1 = 1ULL << 9U,
    compiled_transform_cache_v1 = 1ULL << 10U,
    explicit_trust_policy_v1 = 1ULL << 11U,
    cold_provenance_v1 = 1ULL << 12U,
};

struct open_compiler_extension_surface_v1 {
    std::uint32_t abi_version = open_compiler_extension_abi_version_v1;
    std::uint32_t structure_size = sizeof(open_compiler_extension_surface_v1);
    std::uint64_t capabilities = 0;
    std::uint32_t pipeline_phase_count = 0;
    std::uint32_t pipeline_stage_count = 0;
};

[[nodiscard]] open_compiler_extension_surface_v1
open_compiler_extension_surface_descriptor_v1() noexcept;

}  // namespace cellerator::compiler::pass::v1
