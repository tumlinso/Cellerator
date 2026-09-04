#include <Cellerator/compiler/ir/planning/implement_exact_logical_coverage_nodes_v1.hh>

namespace cellerator::compiler::ir::planning::v1 {
namespace {
bool zero(planning_identity_v1 value) noexcept {
    return value.low == 0u && value.high == 0u;
}
bool output_role(logical_ownership_role_v1 role) noexcept {
    return role == logical_ownership_role_v1::exclusive_output ||
           role == logical_ownership_role_v1::partial_contribution;
}
}  // namespace

exact_coverage_status_v1 validate_exact_logical_coverage_node_v1(
    const exact_logical_coverage_node_v1 &coverage) noexcept {
    if (coverage.member_count == 0u || coverage.members == nullptr) {
        return exact_coverage_status_v1::invalid_argument;
    }
    if (zero(coverage.coverage) || zero(coverage.semantic_subject) ||
        zero(coverage.certification_receipt)) {
        return exact_coverage_status_v1::invalid_identity;
    }
    if (coverage.reserved != 0u) {
        return exact_coverage_status_v1::nonzero_reserved;
    }
    std::uint64_t members = 0u;
    std::uint64_t edges = 0u;
    for (std::uint32_t index = 0u; index != coverage.member_count; ++index) {
        const auto &item = coverage.members[index];
        if (item.logical_id == 0u || item.canonical_id == 0u) {
            return exact_coverage_status_v1::invalid_identity;
        }
        if (item.reserved16 != 0u) {
            return exact_coverage_status_v1::nonzero_reserved;
        }
        if (static_cast<std::uint8_t>(item.kind) >
            static_cast<std::uint8_t>(logical_member_kind_v1::edge)) {
            return exact_coverage_status_v1::invalid_kind;
        }
        if (static_cast<std::uint8_t>(item.role) >
            static_cast<std::uint8_t>(logical_ownership_role_v1::replica)) {
            return exact_coverage_status_v1::invalid_role;
        }
        if (item.kind == logical_member_kind_v1::edge && !output_role(item.role)) {
            return exact_coverage_status_v1::wrong_role;
        }
        for (std::uint32_t other = 0u; other != index; ++other) {
            if (item.logical_id == coverage.members[other].logical_id) {
                return exact_coverage_status_v1::duplicate_logical_id;
            }
            if (item.canonical_id == coverage.members[other].canonical_id &&
                item.kind == coverage.members[other].kind) {
                return exact_coverage_status_v1::duplicate_canonical_id;
            }
        }
        if (item.kind == logical_member_kind_v1::member) {
            ++members;
        } else {
            ++edges;
        }
    }
    if (members != coverage.equation.covered_members ||
        coverage.equation.covered_members != coverage.equation.universe_members) {
        return exact_coverage_status_v1::omitted_member;
    }
    if (edges != coverage.equation.covered_edges ||
        coverage.equation.covered_edges != coverage.equation.universe_edges) {
        return exact_coverage_status_v1::omitted_edge;
    }
    return exact_coverage_status_v1::ok;
}

}  // namespace cellerator::compiler::ir::planning::v1
