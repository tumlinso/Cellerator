#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cgs {

enum class PredicateOp : std::uint8_t { feature, all, any, logical_not };
// Edge-root reuse over one shared condition-node bank.
enum class ProgramPattern : std::uint8_t { coherent, diverse };
enum class ActivityPattern : std::uint8_t { sparse, dense };
enum class OverlapPattern : std::uint8_t { low, high };
enum class ParameterPattern : std::uint8_t { frozen, update_every_step };
enum class WorkloadRole : std::uint8_t { complete_consumer, producer_only_diagnostic };

struct PredicateNode final {
    PredicateOp op = PredicateOp::feature;
    std::uint64_t feature = 0;
    // Children occupy [first_child, first_child+child_count) and always have
    // lower node IDs than their parent. This maps directly to NodeV1.
    std::uint32_t first_child = 0;
    std::uint32_t child_count = 0;
};

struct ConditionDAG final {
    std::vector<PredicateNode> nodes;
    // Node IDs in topological evaluation order, grouped by dependency level.
    std::vector<std::vector<std::uint32_t>> levels;
    // Candidate output nodes; edges bind independently to nodes in this shared
    // bank. root remains the final legacy node for scalar helper callers.
    std::vector<std::uint32_t> roots;
    std::uint32_t root = 0;
};

struct WorkloadSpec final {
    std::string id;
    std::uint64_t seed = 0;
    std::uint32_t dag_nodes = 7;
    ProgramPattern program_pattern = ProgramPattern::coherent;
    ActivityPattern activity_pattern = ActivityPattern::sparse;
    OverlapPattern overlap_pattern = OverlapPattern::low;
    ParameterPattern parameter_pattern = ParameterPattern::frozen;
    std::uint32_t reuse_horizon = 1;
    std::uint8_t predicate_change_percent = 0;
    std::uint32_t cells = 256;
    std::uint32_t features = 4096;
    std::uint32_t groups = 64;
    // Condition roots bound for this workload. Complete-consumer cases bind
    // one numerical group to every declared edge.
    std::uint32_t condition_edges = 2176;
    // Select the historical edge-to-root map from shared_condition_probe.
    bool legacy_root_map = false;
    // Producer-only fixtures preserve historical producer coverage while
    // intentionally leaving most roots without numerical consumers.
    WorkloadRole role = WorkloadRole::producer_only_diagnostic;
};

struct WorkloadFixture final {
    WorkloadSpec spec;
    // One shared DAG bank. Different condition edges bind roots in this bank.
    std::vector<ConditionDAG> programs;
    std::vector<std::uint32_t> edge_roots;
    // Unique demanded roots and their shared-bank ancestor closure, sorted by
    // original node ID. demanded_levels preserves topological level order and
    // can drive a demand-pruned producer/control without changing node IDs.
    std::vector<std::uint32_t> demanded_roots;
    std::vector<std::uint32_t> demanded_nodes;
    std::vector<std::vector<std::uint32_t>> demanded_levels;
    // Number of predicate leaves used by the bank. Accessibility retains its
    // full [cell, spec.features] row stride; leaves use the prefix.
    std::uint32_t predicate_features = 0;
    // Row-major [cell, feature] binary accessibility inputs.
    std::vector<std::uint8_t> accessibility;
    std::vector<std::uint8_t> accessibility_after;
    // Row-major [cell, edge] actual root decisions from each bank pass.
    std::vector<std::uint8_t> predicates_before;
    std::vector<std::uint8_t> predicates_after;
    // Input changes are cell-local feature predicate observations, recorded
    // as feature IDs in source order. Their count follows the requested
    // fraction exactly (rounded to nearest integer). Root decision changes
    // are an observed result and are not forced to match that input fraction.
    std::vector<std::uint64_t> changed_input_offsets;
    std::vector<std::uint32_t> changed_input_features;
    std::uint64_t root_decision_changes = 0;
    // One shared-DAG dependency closure per cell. Entries are sorted
    // topological node IDs, with no replication for multiple edge roots.
    std::vector<std::uint64_t> dirty_node_offsets;
    std::vector<std::uint32_t> dirty_nodes;
    // Local program and feature neighborhoods. Distal features are explicit
    // IDs in the upper half of the declared feature axis.
    std::vector<std::uint32_t> group_program;
    std::vector<std::uint64_t> group_feature_offsets;
    std::vector<std::uint64_t> group_features;
    std::vector<std::uint64_t> group_output_offsets;
    std::vector<std::uint64_t> group_outputs;
    std::uint64_t distal_memberships = 0;
};

const std::vector<WorkloadSpec>& workload_manifest();
ConditionDAG make_condition_dag(std::uint32_t node_count,
                                std::uint32_t feature_count,
                                std::uint64_t seed);
bool evaluate_condition(const ConditionDAG&, const std::uint8_t* accessibility,
                        std::uint64_t feature_count);
std::vector<std::uint8_t> evaluate_condition_nodes(
    const ConditionDAG&, const std::uint8_t* accessibility,
    std::uint64_t feature_count);
WorkloadFixture materialize_workload(const WorkloadSpec&);

// Stable digest-like workload identity used only for repeatable fixture seeds,
// not for source integrity or cryptographic provenance.
std::uint64_t workload_mix(std::uint64_t value) noexcept;

bool run_reference_checks(std::string* error = nullptr);

} // namespace cgs
