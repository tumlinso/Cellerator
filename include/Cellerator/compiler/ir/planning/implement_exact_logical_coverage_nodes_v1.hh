#pragma once

#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::ir::planning::v1 {

enum class logical_member_kind_v1 : std::uint8_t { member = 0u, edge = 1u };
enum class logical_ownership_role_v1 : std::uint8_t {
    exact_read = 0u, exclusive_output, partial_contribution, read_only_halo, replica
};

struct logical_coverage_member_v1 {
    std::uint64_t logical_id = 0u;
    std::uint64_t canonical_id = 0u;
    logical_member_kind_v1 kind = logical_member_kind_v1::member;
    logical_ownership_role_v1 role = logical_ownership_role_v1::exact_read;
    std::uint16_t reserved16 = 0u;
    std::uint32_t owner = 0u;
};

struct exact_coverage_equation_v1 {
    std::uint64_t universe_members = 0u;
    std::uint64_t covered_members = 0u;
    std::uint64_t universe_edges = 0u;
    std::uint64_t covered_edges = 0u;
};

struct exact_logical_coverage_node_v1 {
    planning_identity_v1 coverage{};
    planning_identity_v1 semantic_subject{};
    planning_identity_v1 certification_receipt{};
    const logical_coverage_member_v1 *members = nullptr;
    std::uint32_t member_count = 0u;
    std::uint32_t reserved = 0u;
    exact_coverage_equation_v1 equation{};
    planning_identity_v1 approximate_proposal_evidence{};
};

enum class exact_coverage_status_v1 : std::uint8_t {
    ok = 0u, invalid_argument, invalid_identity, nonzero_reserved,
    invalid_kind, invalid_role, duplicate_logical_id, duplicate_canonical_id,
    omitted_member, omitted_edge, wrong_role
};

exact_coverage_status_v1 validate_exact_logical_coverage_node_v1(
    const exact_logical_coverage_node_v1 &coverage) noexcept;

static_assert(std::is_trivially_copyable_v<logical_coverage_member_v1>);
static_assert(std::is_trivially_copyable_v<exact_logical_coverage_node_v1>);

}  // namespace cellerator::compiler::ir::planning::v1
