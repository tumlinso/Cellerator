#pragma once

#include <Cellerator/compiler/ir/planning/implement_complete_cost_vectors_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_planning_problems_and_operation_scopes_v1.hh>
#include <Cellerator/compiler/ir/planning/implement_rejection_and_dominance_explanations_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cellerator::compiler::ir::planning::v1 {

enum class first_search_candidate_kind_v1 : std::uint8_t {
    conventional_fallback = 0u,
    structure_dependent,
};

struct first_search_candidate_input_v1 {
    planning_identity_v1 candidate{};
    planning_identity_v1 provider{};
    complete_cost_vector_v1 cost{};
};

struct first_search_space_input_v1 {
    const planning_problem_v1* problem = nullptr;
    first_search_candidate_input_v1 conventional{};
    first_search_candidate_input_v1 structure_dependent{};
    planning_identity_v1 profile_evidence{};
    std::uint64_t profiled_support_count = 0u;
    double profile_confidence = 0.0;
};

struct inspectable_candidate_entry_v1 {
    planning_identity_v1 candidate{};
    planning_identity_v1 provider{};
    first_search_candidate_kind_v1 kind =
        first_search_candidate_kind_v1::conventional_fallback;
    complete_cost_vector_v1 cost{};
    bool profile_admissible = false;
};

struct inspectable_candidate_search_space_v1 {
    planning_identity_v1 problem{};
    planning_identity_v1 profile_family{};
    planning_identity_v1 profile_evidence{};
    std::vector<inspectable_candidate_entry_v1> candidates;
    std::vector<decision_record_v1> decisions;
    std::vector<removal_explanation_v1> explanations;
    planning_ir_module_v1 module{};

    inspectable_candidate_search_space_v1() noexcept = default;
    inspectable_candidate_search_space_v1(
        const inspectable_candidate_search_space_v1& other);
    inspectable_candidate_search_space_v1& operator=(
        const inspectable_candidate_search_space_v1& other);
    inspectable_candidate_search_space_v1(
        inspectable_candidate_search_space_v1&& other) noexcept;
    inspectable_candidate_search_space_v1& operator=(
        inspectable_candidate_search_space_v1&& other) noexcept;
    void refresh_views() noexcept;
};

enum class first_search_space_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_problem,
    invalid_identity,
    duplicate_candidate,
    invalid_profile,
    invalid_cost,
    invalid_module,
    no_selected_candidate,
};

[[nodiscard]] std::optional<inspectable_candidate_search_space_v1>
build_first_inspectable_candidate_search_space_v1(
    const first_search_space_input_v1& input,
    first_search_space_status_v1* status = nullptr) noexcept;

[[nodiscard]] std::optional<std::string> compile_selected_plan_dump_v1(
    const inspectable_candidate_search_space_v1& search_space,
    first_search_space_status_v1* status = nullptr) noexcept;

}  // namespace cellerator::compiler::ir::planning::v1
