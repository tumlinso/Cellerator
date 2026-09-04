#pragma once

#include <Cellerator/compiler/ir/planning/planning_ir_v1.hh>
#include <Cellerator/compiler/ir/semantic/semantic_ir_v1.hh>
#include <Cellerator/compiler/planning/adapt_decomposition_portfolios_to_planning_ir_v1.hh>
#include <Cellerator/compiler/planning/implement_complete_cost_normalization_v1.hh>
#include <Cellerator/compiler/profile/profile_environment_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Cellerator::compiler::planning {

struct source_to_selected_plan_request_v1 {
    std::string source;
    const cellerator::compiler::profile::v1::profile_compile_state_v1* profile = nullptr;
    std::vector<complete_cost_evidence_v1> conventional_cost;
    std::vector<complete_cost_evidence_v1> data_dependent_cost;
    std::uint32_t required_cost_phases =
        cost_phase_preparation_v1 | cost_phase_movement_v1 |
        cost_phase_execution_v1 | cost_phase_synchronization_v1 |
        cost_phase_output_transform_v1;
};

enum class vertical_slice_candidate_kind_v1 : std::uint8_t {
    conventional_fallback = 0u,
    data_dependent = 1u,
};

struct vertical_slice_candidate_v1 {
    cellerator::compiler::ir::planning::v1::planning_identity_v1 candidate{};
    cellerator::compiler::ir::planning::v1::planning_identity_v1 provider{};
    vertical_slice_candidate_kind_v1 kind =
        vertical_slice_candidate_kind_v1::conventional_fallback;
    normalized_complete_cost_v1 complete_cost{};
    bool exact_coverage = true;
    bool profile_admissible = false;
};

struct source_to_selected_plan_result_v1 {
    Cellerator::compiler::ir::semantic::source_linked_semantic_module_v1 semantic;
    Cellerator::compiler::ir::semantic::semantic_source_receipt_v1 source_receipt{};
    std::vector<cellerator::compiler::ir::planning::v1::semantic_operation_scope_v1>
        operation_scopes;
    cellerator::compiler::ir::planning::v1::planning_problem_v1 problem{};
    decomposition_provider_kind_v1 decomposition = decomposition_provider_kind_v1::greedy;
    std::vector<vertical_slice_candidate_v1> candidates;
    std::vector<cellerator::compiler::ir::planning::v1::decision_record_v1> decisions;
    cellerator::compiler::ir::planning::v1::planning_ir_module_v1 planning_module{};
    cellerator::compiler::ir::planning::v1::planning_identity_v1 selected_candidate{};
    std::string portable_ruleset;

    source_to_selected_plan_result_v1() noexcept = default;
    source_to_selected_plan_result_v1(const source_to_selected_plan_result_v1& other);
    source_to_selected_plan_result_v1& operator=(
        const source_to_selected_plan_result_v1& other);
    source_to_selected_plan_result_v1(source_to_selected_plan_result_v1&& other) noexcept;
    source_to_selected_plan_result_v1& operator=(
        source_to_selected_plan_result_v1&& other) noexcept;
    void refresh_views() noexcept;
};

enum class source_to_selected_plan_status_v1 : std::uint8_t {
    success = 0u,
    invalid_source,
    invalid_profile,
    wrong_operation_count,
    invalid_cost,
    unavailable_decomposition,
    invalid_planning_ir,
};

[[nodiscard]] std::optional<source_to_selected_plan_result_v1>
deliver_source_to_selected_plan_vertical_slice_v1(
    const source_to_selected_plan_request_v1& request,
    source_to_selected_plan_status_v1* status = nullptr) noexcept;

}  // namespace Cellerator::compiler::planning
