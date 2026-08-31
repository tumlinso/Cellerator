#pragma once

#include <cstddef>
#include <cstdint>

namespace cellerator::geometry::optimizer::overlap {

using source_id = std::uint64_t;
using source_group_id = std::uint64_t;
using logical_contribution_id = std::uint64_t;

struct source_group_dictionary_view {
    const std::uint64_t *group_offsets = nullptr;
    const source_id *source_ids = nullptr;
    std::uint64_t group_count = 0;
    std::uint64_t membership_count = 0;
    std::uint64_t source_count = 0;
};

struct logical_contribution_owner {
    logical_contribution_id contribution = 0;
    source_group_id owner_group = 0;
};

struct logical_contribution_ownership_view {
    const logical_contribution_owner *owners = nullptr;
    std::uint64_t owner_count = 0;
    std::uint64_t contribution_count = 0;
};

struct replication_unit_cost {
    std::uint64_t source_state = 0;
    std::uint64_t dense_input_movement = 0;
    std::uint64_t value_maps = 0;
    std::uint64_t gradient_reconciliation = 0;
    std::uint64_t persistent_bytes = 0;
    std::uint64_t construction = 0;
    std::uint64_t canonical_recovery = 0;
};

struct replication_cost {
    replication_unit_cost repeated{};
    std::uint64_t replicated_memberships = 0;
    std::uint64_t replicated_sources = 0;
    std::uint64_t total = 0;
};

struct replication_workspace_view {
    std::uint64_t *source_use_counts = nullptr;
    std::uint64_t source_capacity = 0;
};

enum class contract_error : std::uint8_t {
    none = 0,
    null_pointer,
    invalid_offset,
    source_out_of_range,
    duplicate_source_in_group,
    empty_group,
    contribution_out_of_range,
    owner_group_out_of_range,
    duplicate_contribution_owner,
    missing_contribution_owner,
    insufficient_workspace,
    integer_overflow
};

struct contract_status {
    contract_error error = contract_error::none;
    std::uint64_t index = 0;

    constexpr explicit operator bool() const noexcept {
        return error == contract_error::none;
    }
};

contract_status validate_source_group_dictionary(source_group_dictionary_view dictionary) noexcept;

contract_status validate_logical_contribution_ownership(
    logical_contribution_ownership_view ownership,
    std::uint64_t group_count) noexcept;

contract_status evaluate_replication_cost(
    source_group_dictionary_view dictionary,
    replication_unit_cost unit_cost,
    replication_workspace_view workspace,
    replication_cost *result) noexcept;

contract_status query_is_disjoint(
    source_group_dictionary_view dictionary,
    replication_workspace_view workspace,
    bool *result) noexcept;

}  // namespace cellerator::geometry::optimizer::overlap
