#pragma once

#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::ir::planning::v1 {

enum class planning_scope_kind_v1 : std::uint8_t {
    field = 0u, operation, bundle, chain, program, profile_family
};

enum class planning_target_class_v1 : std::uint8_t {
    portable_host = 0u, cpu, nvidia_gpu, external_backend
};

enum planning_constraint_v1 : std::uint32_t {
    planning_constraint_none_v1 = 0u,
    planning_constraint_exact_numerics_v1 = 1u << 0u,
    planning_constraint_deterministic_v1 = 1u << 1u,
    planning_constraint_memory_bounded_v1 = 1u << 2u,
    planning_constraint_graph_capture_v1 = 1u << 3u
};

enum planning_objective_v1 : std::uint32_t {
    planning_objective_latency_v1 = 1u << 0u,
    planning_objective_throughput_v1 = 1u << 1u,
    planning_objective_memory_v1 = 1u << 2u,
    planning_objective_communication_v1 = 1u << 3u
};

struct semantic_operation_scope_v1 {
    planning_identity_v1 operation{};
    planning_identity_v1 field{};
    std::uint32_t ordinal = 0u;
    std::uint32_t reserved = 0u;
};

struct planning_problem_v1 {
    planning_identity_v1 problem{};
    planning_identity_v1 semantic_module{};
    planning_identity_v1 semantic_fingerprint{};
    planning_identity_v1 field{};
    planning_identity_v1 profile_family{};
    const semantic_operation_scope_v1 *operations = nullptr;
    std::uint32_t operation_count = 0u;
    std::uint32_t first_operation = 0u;
    planning_scope_kind_v1 scope = planning_scope_kind_v1::field;
    planning_target_class_v1 target = planning_target_class_v1::portable_host;
    std::uint16_t reserved16 = 0u;
    std::uint32_t constraints = planning_constraint_none_v1;
    std::uint32_t objectives = planning_objective_latency_v1;
};

enum class planning_problem_status_v1 : std::uint8_t {
    ok = 0u, invalid_argument, invalid_identity, invalid_scope,
    invalid_target, invalid_constraint, invalid_objective,
    invalid_operation_range, operation_field_mismatch, unordered_operation
};

planning_problem_status_v1 validate_planning_problem_v1(
    const planning_problem_v1 &problem) noexcept;

static_assert(std::is_trivially_copyable_v<semantic_operation_scope_v1>);
static_assert(std::is_trivially_copyable_v<planning_problem_v1>);

}  // namespace cellerator::compiler::ir::planning::v1
