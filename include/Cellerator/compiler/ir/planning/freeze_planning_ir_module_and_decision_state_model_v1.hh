#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::ir::planning::v1 {

inline constexpr std::uint32_t planning_ir_schema_version_v1 = 1u;

struct planning_identity_v1 {
    std::uint64_t low = 0u;
    std::uint64_t high = 0u;
};

enum class decision_state_v1 : std::uint8_t {
    unresolved = 0u,
    offered,
    admissible,
    rejected,
    dominated,
    selected,
    forced,
    externally_selected,
    fallback
};

enum decision_flags_v1 : std::uint32_t {
    decision_flag_none_v1 = 0u,
    decision_flag_correct_v1 = 1u << 0u,
    decision_flag_measured_v1 = 1u << 1u,
    decision_flag_user_authored_v1 = 1u << 2u
};

// Every alternative remains in one inspectable record throughout planning.
// A state transition changes only decision metadata; it never substitutes an
// opaque planner-owned object for the original candidate identity.
struct decision_record_v1 {
    planning_identity_v1 decision{};
    planning_identity_v1 candidate{};
    planning_identity_v1 source_operation{};
    decision_state_v1 state = decision_state_v1::unresolved;
    std::uint8_t reason = 0u;
    std::uint16_t reserved16 = 0u;
    std::uint32_t flags = decision_flag_none_v1;
    std::uint64_t evidence_revision = 0u;
};

struct planning_ir_module_v1 {
    std::uint32_t schema_version = planning_ir_schema_version_v1;
    std::uint32_t reserved = 0u;
    planning_identity_v1 module{};
    const decision_record_v1 *decisions = nullptr;
    std::uint32_t decision_count = 0u;
    std::uint32_t reserved_count = 0u;
};

enum class planning_ir_status_v1 : std::uint8_t {
    ok = 0u,
    invalid_argument,
    unsupported_schema,
    invalid_identity,
    invalid_state,
    nonzero_reserved,
    duplicate_decision,
    insufficient_capacity,
    malformed_binary
};

planning_ir_status_v1 validate_planning_ir_module_v1(
    const planning_ir_module_v1 &module) noexcept;
planning_ir_status_v1 serialize_planning_decisions_v1(
    const planning_ir_module_v1 &module, void *destination,
    std::size_t capacity, std::size_t *written) noexcept;
planning_ir_status_v1 deserialize_planning_decisions_v1(
    const void *source, std::size_t source_bytes, decision_record_v1 *decisions,
    std::uint32_t capacity, planning_ir_module_v1 *module) noexcept;

static_assert(std::is_standard_layout_v<decision_record_v1>);
static_assert(std::is_trivially_copyable_v<decision_record_v1>);

}  // namespace cellerator::compiler::ir::planning::v1
