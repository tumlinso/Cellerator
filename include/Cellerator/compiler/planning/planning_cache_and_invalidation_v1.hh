#pragma once

#include <cstdint>

namespace Cellerator::compiler::planning {

struct planning_cache_key_v1 {
    std::uint64_t semantic_fingerprint_high = 0u;
    std::uint64_t semantic_fingerprint_low = 0u;
    std::uint64_t profile_identity = 0u;
    std::uint64_t evidence_revision = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t order_identity = 0u;
    std::uint64_t target_class_identity = 0u;
    std::uint64_t toolchain_identity = 0u;
    std::uint64_t constraints_fingerprint = 0u;
    std::uint64_t planner_revision = 0u;
};

enum class planning_resumption_point_v1 : std::uint8_t {
    semantic_lowering = 1u,
    profile_evidence,
    structure_planning,
    order_transitions,
    target_candidates,
    constraint_filtering,
    planner_selection,
    complete_plan,
};

enum class planning_cache_validation_code_v1 : std::uint8_t {
    reusable = 0u,
    invalid_key,
    invalidated,
};

struct planning_cache_validation_v1 {
    planning_cache_validation_code_v1 code =
        planning_cache_validation_code_v1::invalid_key;
    planning_resumption_point_v1 resume_at =
        planning_resumption_point_v1::semantic_lowering;

    constexpr explicit operator bool() const noexcept {
        return code == planning_cache_validation_code_v1::reusable;
    }
};

[[nodiscard]] planning_cache_validation_v1 validate_planning_cache_key_v1(
    const planning_cache_key_v1& cached,
    const planning_cache_key_v1& current) noexcept;

}  // namespace Cellerator::compiler::planning
