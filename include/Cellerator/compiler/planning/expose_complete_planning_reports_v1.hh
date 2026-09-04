#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::planning {

enum class planning_report_rejection_v1 : std::uint8_t {
    none = 0u,
    invalid,
    incorrect,
    resource_limit,
    unavailable,
};

enum class planning_report_selection_source_v1 : std::uint8_t {
    none = 0u,
    automatic,
    source_edit,
    pipeline_edit,
    user_edit,
    external,
    cache,
    fallback,
};

struct planning_candidate_report_v1 {
    std::uint64_t candidate_identity = 0u;
    bool exact_coverage = false;
    std::uint64_t complete_cost_nanoseconds = 0u;
    std::uint64_t evidence_revision = 0u;
    std::uint64_t current_evidence_revision = 0u;
    planning_report_rejection_v1 rejection = planning_report_rejection_v1::none;
    bool dominated = false;
    bool selected = false;
    planning_report_selection_source_v1 selected_source =
        planning_report_selection_source_v1::none;
    bool forced_edit = false;
    bool fallback = false;
};

enum class complete_planning_report_code_v1 : std::uint8_t {
    ok = 0u,
    invalid_candidate,
    duplicate_candidate,
    invalid_selection,
};

struct complete_planning_report_v1 {
    complete_planning_report_code_v1 code =
        complete_planning_report_code_v1::invalid_candidate;
    std::string snapshot;

    constexpr explicit operator bool() const noexcept {
        return code == complete_planning_report_code_v1::ok;
    }
};

[[nodiscard]] complete_planning_report_v1 expose_complete_planning_report_v1(
    const std::vector<planning_candidate_report_v1>& candidates);

}  // namespace Cellerator::compiler::planning
