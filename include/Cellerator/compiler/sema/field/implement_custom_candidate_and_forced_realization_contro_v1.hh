#pragma once

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::sema::field {

enum class realization_control_kind_v1 : std::uint8_t {
    offer_candidate = 1,
    force_candidate,
    force_decomposition,
    force_realization,
};

struct controllable_realization_candidate_v1 {
    std::uint64_t candidate_identity = 0;
    std::uint64_t decomposition_identity = 0;
    std::uint64_t realization_identity = 0;
    double estimated_total_cost = 0.0;
    bool legal = true;
};

struct custom_candidate_or_forced_realization_v1 {
    realization_control_kind_v1 kind = realization_control_kind_v1::offer_candidate;
    std::uint64_t selected_identity = 0;
    bool explicitly_unsafe = false;
};

struct resolved_realization_control_v1 {
    std::vector<std::uint64_t> considered_candidates;
    std::uint64_t selected_candidate_identity = 0;
    bool custom_candidate_won = false;
    bool forced = false;
    bool unsafe = false;
};

enum class realization_control_status_v1 : std::uint8_t {
    success = 0,
    invalid_output,
    invalid_candidate,
    selected_object_unavailable,
    selected_object_illegal,
    no_legal_candidate,
};

[[nodiscard]] realization_control_status_v1
implement_custom_candidate_and_forced_realization_contro_v1(
    const std::vector<controllable_realization_candidate_v1>& candidates,
    const custom_candidate_or_forced_realization_v1& control,
    resolved_realization_control_v1* resolved) noexcept;

}  // namespace Cellerator::compiler::sema::field
