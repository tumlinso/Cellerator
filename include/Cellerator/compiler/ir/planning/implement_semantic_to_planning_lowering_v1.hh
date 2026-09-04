#pragma once

#include <Cellerator/compiler/ir/planning/implement_planning_problems_and_operation_scopes_v1.hh>
#include <Cellerator/compiler/ir/semantic/semantic_ir_v1.hh>
#include <Cellerator/compiler/profile/profile_environment_v1.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cellerator::compiler::ir::planning::v1 {

struct semantic_planning_input_v1 {
    planning_identity_v1 semantic_module{};
    planning_identity_v1 semantic_fingerprint{};
    const Cellerator::compiler::ir::semantic::source_linked_semantic_operation_v1*
        source_operations = nullptr;
    const Cellerator::compiler::ir::semantic::semantic_canonical_record_v1*
        canonical_operations = nullptr;
    const Cellerator::compiler::ir::semantic::semantic_lifetime_state_v1*
        lifetime_states = nullptr;
    std::uint32_t operation_count = 0u;
    const Cellerator::compiler::ir::semantic::execution_field_region_ir_v1* fields = nullptr;
    std::uint32_t field_count = 0u;
};

struct semantic_planning_profile_v1 {
    const cellerator::compiler::profile::v1::named_profile_environment_v1* environment = nullptr;
    const cellerator::compiler::profile::v1::profile_compile_state_v1* state = nullptr;
};

struct semantic_to_planning_options_v1 {
    planning_target_class_v1 target = planning_target_class_v1::portable_host;
    std::uint32_t objectives = planning_objective_latency_v1;
};

// The original spelling and value remain available even when a constraint has
// a direct Planning IR flag. Unknown constraints are therefore not erased or
// silently interpreted by the compiler.
struct lowered_explicit_constraint_v1 {
    planning_identity_v1 field{};
    std::string name;
    std::string value;
    bool hard = true;
};

struct lowered_semantic_operation_v1 {
    semantic_operation_scope_v1 scope{};
    cellerator::compute::operation::v2::operation_kind kind =
        cellerator::compute::operation::v2::operation_kind::relation_apply;
    Cellerator::compiler::ir::semantic::numeric_tuple_ir_v1 numeric{};
    std::uint64_t structure_epoch = 0u;
    std::uint64_t value_generation = 0u;
    std::uint64_t support_generation = 0u;
    std::uint64_t order_generation = 0u;
    std::uint32_t first_constraint = 0u;
    std::uint32_t constraint_count = 0u;
    cellerator::compute::operation::v2::operation_problem planner_request{};
};

struct semantic_to_planning_result_v1 {
    planning_identity_v1 profile_environment{};
    planning_identity_v1 profile_state{};
    std::vector<semantic_operation_scope_v1> operation_scopes;
    std::vector<lowered_semantic_operation_v1> operations;
    std::vector<lowered_explicit_constraint_v1> constraints;
    std::vector<planning_problem_v1> problems;

    semantic_to_planning_result_v1() noexcept = default;
    semantic_to_planning_result_v1(const semantic_to_planning_result_v1& other);
    semantic_to_planning_result_v1& operator=(const semantic_to_planning_result_v1& other);
    semantic_to_planning_result_v1(semantic_to_planning_result_v1&& other) noexcept;
    semantic_to_planning_result_v1& operator=(semantic_to_planning_result_v1&& other) noexcept;
    void refresh_views() noexcept;
};

enum class semantic_to_planning_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_module,
    invalid_profile_environment,
    profile_state_not_found,
    profile_state_mismatch,
    invalid_operation,
    invalid_canonical_operation,
    invalid_field,
    invalid_generation,
    invalid_constraint,
    invalid_planning_problem,
};

[[nodiscard]] std::optional<semantic_to_planning_result_v1>
lower_semantic_to_planning_v1(
    const semantic_planning_input_v1& semantic,
    const semantic_planning_profile_v1& profile,
    semantic_to_planning_options_v1 options = {},
    semantic_to_planning_status_v1* status = nullptr) noexcept;

}  // namespace cellerator::compiler::ir::planning::v1
