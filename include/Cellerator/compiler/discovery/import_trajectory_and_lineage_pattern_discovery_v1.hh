#pragma once

#include <Cellerator/compiler/discovery/import_the_common_jbc_atom_identity_adapters_v1.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::discovery {

struct trajectory_state_observation_v1 {
    persistent_atom_identity_v1 trajectory_identity{};
    persistent_atom_identity_v1 lineage_identity{};
    persistent_atom_identity_v1 parent_lineage_identity{};
    persistent_atom_identity_v1 state_identity{};
    std::uint64_t time_tick = 0;
    std::uint64_t mutation_generation = 0;
};

enum class trajectory_pattern_kind_v1 : std::uint8_t {
    recurring_prefix = 1,
    branch_local_delta,
    state_neighborhood,
};

struct trajectory_pattern_evidence_v1 {
    trajectory_pattern_kind_v1 kind = trajectory_pattern_kind_v1::recurring_prefix;
    persistent_atom_identity_v1 trajectory_identity{};
    persistent_atom_identity_v1 lineage_identity{};
    persistent_atom_identity_v1 related_lineage_identity{};
    persistent_atom_identity_v1 first_state_identity{};
    persistent_atom_identity_v1 second_state_identity{};
    std::uint64_t observation_count = 0;
    std::uint64_t mutation_horizon_ticks = 0;
    std::uint64_t mutation_horizon_generations = 0;
};

struct trajectory_discovery_limits_v1 {
    std::uint64_t minimum_prefix_states = 2;
    std::uint64_t minimum_neighborhood_occurrences = 2;
    std::uint64_t maximum_observations = 0;
    std::uint64_t maximum_proposals = 0;
};

enum class trajectory_discovery_status_v1 : std::uint8_t {
    success = 0,
    invalid_limits,
    invalid_observation,
    unordered_observations,
    inconsistent_parent,
    proposal_bound_exceeded,
    allocation_failure,
};

[[nodiscard]] trajectory_discovery_status_v1 discover_trajectory_and_lineage_patterns_v1(
    const std::vector<trajectory_state_observation_v1>& observations,
    trajectory_discovery_limits_v1 limits,
    std::vector<trajectory_pattern_evidence_v1>* output) noexcept;

[[nodiscard]] constexpr bool authorizes_execution(
    const trajectory_pattern_evidence_v1&) noexcept {
    return false;
}

}  // namespace Cellerator::compiler::discovery
