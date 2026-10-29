#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::planning {

struct named_profile_plan_v1 {
    std::uint64_t profile_identity = 0u;
    std::string profile_name;
    std::uint64_t semantic_program_identity = 0u;
    std::uint64_t selected_candidate_identity = 0u;
    std::uint64_t artifact_compatibility_identity = 0u;
    std::uint64_t runtime_predicate_identity = 0u;
};

struct profile_plan_variant_v1 {
    std::uint64_t profile_identity = 0u;
    std::uint64_t selected_candidate_identity = 0u;
    std::uint32_t shared_artifact_index = 0u;
    std::uint64_t runtime_predicate_identity = 0u;
};

struct shared_profile_artifact_v1 {
    std::uint64_t compatibility_identity = 0u;
    std::uint64_t semantic_program_identity = 0u;
    std::uint64_t reuse_count = 0u;
};

struct profile_family_plan_v1 {
    std::uint64_t semantic_program_identity = 0u;
    std::vector<std::string> profile_names;
    std::vector<profile_plan_variant_v1> variants;
    std::vector<shared_profile_artifact_v1> shared_artifacts;
    std::uint64_t runtime_selection_limit = 0u;
    std::uint64_t shared_artifact_reuses = 0u;
};

enum class profile_family_plan_code_v1 : std::uint8_t {
    ok = 0u,
    invalid_alternative,
    duplicate_profile,
    semantic_program_mismatch,
    runtime_variant_limit_exceeded,
};

struct profile_family_plan_result_v1 {
    profile_family_plan_code_v1 code = profile_family_plan_code_v1::invalid_alternative;
    std::uint64_t alternative_index = 0u;
    profile_family_plan_v1 plan{};

    constexpr explicit operator bool() const noexcept {
        return code == profile_family_plan_code_v1::ok;
    }
};

[[nodiscard]] profile_family_plan_result_v1 implement_profile_family_plan_variants_v1(
    const std::vector<named_profile_plan_v1>& alternatives,
    std::uint64_t maximum_runtime_variants);

}  // namespace Cellerator::compiler::planning
