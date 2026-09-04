#include <Cellerator/compiler/discovery/import_trajectory_and_lineage_pattern_discovery_v1.hh>

#include <algorithm>
#include <map>
#include <utility>

namespace Cellerator::compiler::discovery {
namespace {

struct lineage_span_v1 {
    std::size_t begin = 0;
    std::size_t end = 0;
};

struct identity_less_v1 {
    bool operator()(persistent_atom_identity_v1 left,
                    persistent_atom_identity_v1 right) const noexcept {
        return persistent_atom_identity_less_v1(left, right);
    }
};

struct neighborhood_key_v1 {
    persistent_atom_identity_v1 trajectory{};
    persistent_atom_identity_v1 first{};
    persistent_atom_identity_v1 second{};
};

struct neighborhood_key_less_v1 {
    bool operator()(const neighborhood_key_v1& left,
                    const neighborhood_key_v1& right) const noexcept {
        if (left.trajectory != right.trajectory) {
            return persistent_atom_identity_less_v1(left.trajectory, right.trajectory);
        }
        if (left.first != right.first) {
            return persistent_atom_identity_less_v1(left.first, right.first);
        }
        return persistent_atom_identity_less_v1(left.second, right.second);
    }
};

struct neighborhood_summary_v1 {
    std::uint64_t count = 0;
    std::uint64_t maximum_horizon = 0;
    std::uint64_t maximum_generation_horizon = 0;
};

bool empty_identity_v1(persistent_atom_identity_v1 identity) noexcept {
    return identity.producer_namespace == 0 && identity.local_identity == 0;
}

bool observation_less_v1(const trajectory_state_observation_v1& left,
                         const trajectory_state_observation_v1& right) noexcept {
    if (left.trajectory_identity != right.trajectory_identity) {
        return persistent_atom_identity_less_v1(
            left.trajectory_identity, right.trajectory_identity);
    }
    if (left.lineage_identity != right.lineage_identity) {
        return persistent_atom_identity_less_v1(left.lineage_identity,
                                                right.lineage_identity);
    }
    return left.time_tick < right.time_tick;
}

bool proposal_less_v1(const trajectory_pattern_evidence_v1& left,
                      const trajectory_pattern_evidence_v1& right) noexcept {
    if (left.kind != right.kind) {
        return left.kind < right.kind;
    }
    if (left.trajectory_identity != right.trajectory_identity) {
        return persistent_atom_identity_less_v1(
            left.trajectory_identity, right.trajectory_identity);
    }
    if (left.lineage_identity != right.lineage_identity) {
        return persistent_atom_identity_less_v1(left.lineage_identity,
                                                right.lineage_identity);
    }
    if (left.related_lineage_identity != right.related_lineage_identity) {
        return persistent_atom_identity_less_v1(left.related_lineage_identity,
                                                right.related_lineage_identity);
    }
    if (left.first_state_identity != right.first_state_identity) {
        return persistent_atom_identity_less_v1(left.first_state_identity,
                                                right.first_state_identity);
    }
    return persistent_atom_identity_less_v1(left.second_state_identity,
                                            right.second_state_identity);
}

}  // namespace

trajectory_discovery_status_v1 discover_trajectory_and_lineage_patterns_v1(
    const std::vector<trajectory_state_observation_v1>& observations,
    trajectory_discovery_limits_v1 limits,
    std::vector<trajectory_pattern_evidence_v1>* output) noexcept {
    if (output == nullptr || limits.minimum_prefix_states == 0 ||
        limits.minimum_neighborhood_occurrences == 0 ||
        limits.maximum_observations == 0 || limits.maximum_proposals == 0) {
        return trajectory_discovery_status_v1::invalid_limits;
    }
    if (observations.size() > limits.maximum_observations) {
        return trajectory_discovery_status_v1::proposal_bound_exceeded;
    }

    try {
        std::vector<lineage_span_v1> lineages;
        for (std::size_t index = 0; index < observations.size(); ++index) {
            const auto& observation = observations[index];
            if (!valid_persistent_atom_identity_v1(observation.trajectory_identity) ||
                !valid_persistent_atom_identity_v1(observation.lineage_identity) ||
                !valid_persistent_atom_identity_v1(observation.state_identity) ||
                (!empty_identity_v1(observation.parent_lineage_identity) &&
                 !valid_persistent_atom_identity_v1(
                     observation.parent_lineage_identity))) {
                return trajectory_discovery_status_v1::invalid_observation;
            }
            if (index != 0 && observation_less_v1(observation, observations[index - 1])) {
                return trajectory_discovery_status_v1::unordered_observations;
            }
            if (index != 0 &&
                observation.trajectory_identity ==
                    observations[index - 1].trajectory_identity &&
                observation.lineage_identity ==
                    observations[index - 1].lineage_identity) {
                if (observation.time_tick <= observations[index - 1].time_tick ||
                    observation.mutation_generation <
                        observations[index - 1].mutation_generation) {
                    return trajectory_discovery_status_v1::unordered_observations;
                }
                if (observation.parent_lineage_identity !=
                    observations[index - 1].parent_lineage_identity) {
                    return trajectory_discovery_status_v1::inconsistent_parent;
                }
            } else {
                lineages.push_back({index, index});
            }
            lineages.back().end = index + 1;
        }

        std::vector<trajectory_pattern_evidence_v1> proposals;
        std::map<persistent_atom_identity_v1, lineage_span_v1, identity_less_v1>
            lineage_by_identity;
        for (const auto span : lineages) {
            lineage_by_identity.emplace(
                observations[span.begin].lineage_identity, span);
        }

        for (std::size_t first = 0; first < lineages.size(); ++first) {
            const auto first_span = lineages[first];
            for (std::size_t second = first + 1; second < lineages.size(); ++second) {
                const auto second_span = lineages[second];
                if (observations[first_span.begin].trajectory_identity !=
                    observations[second_span.begin].trajectory_identity) {
                    break;
                }
                const auto available = std::min(first_span.end - first_span.begin,
                                                second_span.end - second_span.begin);
                std::size_t common = 0;
                while (common < available &&
                       observations[first_span.begin + common].state_identity ==
                           observations[second_span.begin + common].state_identity) {
                    ++common;
                }
                if (common >= limits.minimum_prefix_states) {
                    const auto& begin = observations[first_span.begin];
                    const auto& end = observations[first_span.begin + common - 1];
                    proposals.push_back({
                        trajectory_pattern_kind_v1::recurring_prefix,
                        begin.trajectory_identity,
                        begin.lineage_identity,
                        observations[second_span.begin].lineage_identity,
                        begin.state_identity,
                        end.state_identity,
                        2,
                        end.time_tick - begin.time_tick,
                        end.mutation_generation - begin.mutation_generation,
                    });
                }
            }
        }

        for (const auto child_span : lineages) {
            const auto& child = observations[child_span.begin];
            if (empty_identity_v1(child.parent_lineage_identity)) {
                continue;
            }
            const auto parent_position =
                lineage_by_identity.find(child.parent_lineage_identity);
            if (parent_position == lineage_by_identity.end()) {
                continue;
            }
            const auto parent_span = parent_position->second;
            for (auto child_index = child_span.begin; child_index < child_span.end;
                 ++child_index) {
                const auto& child_state = observations[child_index];
                const trajectory_state_observation_v1* parent_state = nullptr;
                for (auto index = parent_span.begin; index < parent_span.end; ++index) {
                    if (observations[index].time_tick > child_state.time_tick) {
                        break;
                    }
                    parent_state = &observations[index];
                }
                if (parent_state != nullptr &&
                    parent_state->trajectory_identity == child.trajectory_identity &&
                    parent_state->state_identity != child_state.state_identity) {
                    proposals.push_back({
                        trajectory_pattern_kind_v1::branch_local_delta,
                        child.trajectory_identity,
                        child.lineage_identity,
                        child.parent_lineage_identity,
                        parent_state->state_identity,
                        child_state.state_identity,
                        1,
                        child_state.time_tick - parent_state->time_tick,
                        child_state.mutation_generation >= parent_state->mutation_generation
                            ? child_state.mutation_generation -
                                  parent_state->mutation_generation
                            : 0,
                    });
                    break;
                }
            }
        }

        std::map<neighborhood_key_v1, neighborhood_summary_v1,
                 neighborhood_key_less_v1>
            neighborhoods;
        for (const auto span : lineages) {
            for (auto index = span.begin + 1; index < span.end; ++index) {
                const auto& previous = observations[index - 1];
                const auto& current = observations[index];
                auto& summary = neighborhoods[{
                    current.trajectory_identity,
                    previous.state_identity,
                    current.state_identity,
                }];
                ++summary.count;
                summary.maximum_horizon = std::max(
                    summary.maximum_horizon,
                    current.time_tick - previous.time_tick);
                summary.maximum_generation_horizon = std::max(
                    summary.maximum_generation_horizon,
                    current.mutation_generation - previous.mutation_generation);
            }
        }
        for (const auto& entry : neighborhoods) {
            if (entry.second.count < limits.minimum_neighborhood_occurrences) {
                continue;
            }
            proposals.push_back({
                trajectory_pattern_kind_v1::state_neighborhood,
                entry.first.trajectory,
                {},
                {},
                entry.first.first,
                entry.first.second,
                entry.second.count,
                entry.second.maximum_horizon,
                entry.second.maximum_generation_horizon,
            });
        }

        if (proposals.size() > limits.maximum_proposals) {
            return trajectory_discovery_status_v1::proposal_bound_exceeded;
        }
        std::sort(proposals.begin(), proposals.end(), proposal_less_v1);
        *output = std::move(proposals);
        return trajectory_discovery_status_v1::success;
    } catch (...) {
        return trajectory_discovery_status_v1::allocation_failure;
    }
}

}  // namespace Cellerator::compiler::discovery
