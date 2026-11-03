#include "workloads.hh"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cgs {
namespace {

std::uint64_t next_random(std::uint64_t& state) noexcept {
    state += 0x9e3779b97f4a7c15ull;
    return workload_mix(state);
}

std::uint32_t bounded(std::uint64_t& state, std::uint32_t limit) noexcept {
    return limit == 0 ? 0 : static_cast<std::uint32_t>(next_random(state) % limit);
}

void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(message);
}

WorkloadSpec spec(std::string id, std::uint64_t seed, std::uint32_t dag_nodes,
                  ProgramPattern programs, ActivityPattern activity,
                  OverlapPattern overlap, ParameterPattern parameters,
                  std::uint32_t reuse, std::uint8_t change,
                  std::uint32_t groups, bool legacy_roots,
                  WorkloadRole role, std::uint32_t edges = 2176) {
    WorkloadSpec result;
    result.id = std::move(id);
    result.seed = seed;
    result.dag_nodes = dag_nodes;
    result.program_pattern = programs;
    result.activity_pattern = activity;
    result.overlap_pattern = overlap;
    result.parameter_pattern = parameters;
    result.reuse_horizon = reuse;
    result.predicate_change_percent = change;
    result.groups = groups;
    result.condition_edges = edges;
    result.legacy_root_map = legacy_roots;
    result.role = role;
    return result;
}

std::uint32_t predicate_input_count(std::uint32_t node_count) {
    if (node_count == 7u) return 4u;
    if (node_count == 2304u) return 128u;
    throw std::invalid_argument("frozen workload supports seven- or 2,304-node DAGs");
}

} // namespace

std::uint64_t workload_mix(std::uint64_t value) noexcept {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

const std::vector<WorkloadSpec>& workload_manifest() {
    static const std::vector<WorkloadSpec> manifest{
        spec("dag7-legacy-producer-only-sparse-low-frozen-r1-change0",
             0x8b27d419u, 7, ProgramPattern::diverse, ActivityPattern::sparse,
             OverlapPattern::low, ParameterPattern::frozen, 1, 0, 64, true,
             WorkloadRole::producer_only_diagnostic),
        spec("dag7-coherent-producer-only-dense-high-update-r16-change1",
             0x4e15a2b3u, 7, ProgramPattern::coherent, ActivityPattern::dense,
             OverlapPattern::high, ParameterPattern::update_every_step, 16, 1,
             64, false, WorkloadRole::producer_only_diagnostic),
        spec("dag7-diverse-producer-only-sparse-high-update-r256-change10",
             0x9330c17du, 7, ProgramPattern::diverse, ActivityPattern::sparse,
             OverlapPattern::high, ParameterPattern::update_every_step, 256, 10,
             64, false, WorkloadRole::producer_only_diagnostic),
        spec("dag7-legacy-complete-consumer-sparse-low-frozen-r1-change0",
             0x17ad496bu, 7, ProgramPattern::diverse, ActivityPattern::sparse,
             OverlapPattern::low, ParameterPattern::frozen, 1, 0, 2176, true,
             WorkloadRole::complete_consumer),
        spec("dag7-64root-complete-consumer-coherent-dense-high-update-r16-change1",
             0x8e61ca35u, 7, ProgramPattern::coherent, ActivityPattern::dense,
             OverlapPattern::high, ParameterPattern::update_every_step, 16, 1,
             64, false, WorkloadRole::complete_consumer, 64),
        spec("dag2304-legacy-producer-only-sparse-high-frozen-r256-change10",
             0xc64a98e1u, 2304, ProgramPattern::diverse, ActivityPattern::sparse,
             OverlapPattern::high, ParameterPattern::frozen, 256, 10, 64, true,
             WorkloadRole::producer_only_diagnostic),
        spec("dag2304-diverse-producer-only-dense-low-update-r1-change100",
             0x721f0a5bu, 2304, ProgramPattern::diverse, ActivityPattern::dense,
             OverlapPattern::low, ParameterPattern::update_every_step, 1, 100,
             64, false, WorkloadRole::producer_only_diagnostic),
        spec("dag2304-coherent-producer-only-dense-low-frozen-r16-change1",
             0x0a59de33u, 2304, ProgramPattern::coherent, ActivityPattern::dense,
             OverlapPattern::low, ParameterPattern::frozen, 16, 1, 64, false,
             WorkloadRole::producer_only_diagnostic),
        spec("dag2304-legacy-complete-consumer-sparse-high-frozen-r256-change10",
             0x325acf17u, 2304, ProgramPattern::diverse, ActivityPattern::sparse,
             OverlapPattern::high, ParameterPattern::frozen, 256, 10, 2176, true,
             WorkloadRole::complete_consumer),
        spec("dag2304-64root-complete-consumer-diverse-dense-low-update-r1-change100",
             0x731b9d45u, 2304, ProgramPattern::diverse, ActivityPattern::dense,
             OverlapPattern::low, ParameterPattern::update_every_step, 1, 100,
             64, false, WorkloadRole::complete_consumer, 64),
    };
    return manifest;
}

ConditionDAG make_condition_dag(std::uint32_t count,
                                std::uint32_t feature_count,
                                std::uint64_t seed) {
    (void) seed; // The two frozen historical banks have stable identities.
    const std::uint32_t predicate_features = predicate_input_count(count);
    if (feature_count < predicate_features)
        throw std::invalid_argument("condition DAG lacks its declared predicate features");

    ConditionDAG dag;
    dag.nodes.reserve(count);
    if (count == 7u) {
        for (std::uint32_t feature = 0; feature < 4; ++feature)
            dag.nodes.push_back({PredicateOp::feature, feature, 0, 0});
        dag.nodes.push_back({PredicateOp::all, 0, 0, 2});
        dag.nodes.push_back({PredicateOp::any, 0, 2, 2});
        dag.nodes.push_back({PredicateOp::all, 0, 0, 3});
        dag.roots = {0, 1, 2, 3, 4, 5, 6};
    } else {
        for (std::uint32_t feature = 0; feature < 128; ++feature)
            dag.nodes.push_back({PredicateOp::feature, feature, 0, 0});
        std::uint32_t start = 0;
        std::uint32_t width = 2;
        while (dag.nodes.size() < count) {
            if (width > predicate_features)
                throw std::logic_error("legacy window bank exhausted candidate roots");
            const auto root = static_cast<std::uint32_t>(dag.nodes.size());
            dag.nodes.push_back({PredicateOp::all, 0, start, width});
            dag.roots.push_back(root);
            ++start;
            if (start + width > predicate_features) {
                ++width;
                start = 0;
            }
        }
    }

    std::vector<std::uint32_t> depth(dag.nodes.size(), 0u);
    for (std::uint32_t node_id = 0; node_id < dag.nodes.size(); ++node_id) {
        const auto& node = dag.nodes[node_id];
        if (node.op == PredicateOp::all || node.op == PredicateOp::any) {
            for (std::uint32_t child = node.first_child;
                 child < node.first_child + node.child_count; ++child)
                depth[node_id] = std::max(depth[node_id], depth[child] + 1u);
        } else if (node.op == PredicateOp::logical_not) {
            depth[node_id] = depth[node.first_child] + 1u;
        }
    }
    const auto maximum_depth = *std::max_element(depth.begin(), depth.end());
    dag.levels.resize(static_cast<std::size_t>(maximum_depth) + 1u);
    for (std::uint32_t node_id = 0; node_id < dag.nodes.size(); ++node_id)
        dag.levels[depth[node_id]].push_back(node_id);
    if (dag.roots.empty()) throw std::logic_error("condition DAG has no roots");
    dag.root = dag.roots.back();
    return dag;
}

std::vector<std::uint8_t> evaluate_condition_nodes(
    const ConditionDAG& dag, const std::uint8_t* accessibility,
    std::uint64_t feature_count) {
    if (dag.nodes.empty()) throw std::invalid_argument("invalid condition DAG");
    if (feature_count != 0 && accessibility == nullptr)
        throw std::invalid_argument("condition evaluation has null accessibility");
    std::vector<std::uint8_t> values(dag.nodes.size(), 0u);
    for (std::size_t index = 0; index < dag.nodes.size(); ++index) {
        const auto& node = dag.nodes[index];
        switch (node.op) {
        case PredicateOp::feature:
            if (node.feature >= feature_count)
                throw std::invalid_argument("condition feature is out of range");
            values[index] = accessibility[node.feature] != 0;
            break;
        case PredicateOp::all:
        case PredicateOp::any:
            if (node.child_count == 0u ||
                static_cast<std::uint64_t>(node.first_child) + node.child_count > index)
                throw std::invalid_argument("condition DAG is not topologically ordered");
            values[index] = node.op == PredicateOp::all;
            for (std::uint32_t child = node.first_child;
                 child < node.first_child + node.child_count; ++child) {
                if (node.op == PredicateOp::all)
                    values[index] = values[index] && values[child];
                else
                    values[index] = values[index] || values[child];
            }
            break;
        case PredicateOp::logical_not:
            if (node.child_count != 1u || node.first_child >= index)
                throw std::invalid_argument("condition DAG is not topologically ordered");
            values[index] = !values[node.first_child];
            break;
        }
    }
    return values;
}

bool evaluate_condition(const ConditionDAG& dag,
                        const std::uint8_t* accessibility,
                        std::uint64_t feature_count) {
    if (dag.root >= dag.nodes.size())
        throw std::invalid_argument("invalid condition DAG root");
    const auto values = evaluate_condition_nodes(dag, accessibility, feature_count);
    return values[dag.root] != 0;
}

WorkloadFixture materialize_workload(const WorkloadSpec& spec) {
    const auto predicate_features = predicate_input_count(spec.dag_nodes);
    require(spec.features >= std::max<std::uint32_t>(16u, predicate_features),
            "workload needs room for predicates and distal memberships");
    require(spec.cells > 0 && spec.groups > 0 && spec.condition_edges > 0,
            "workload needs cells, groups, and condition edges");
    require(spec.predicate_change_percent <= 100,
            "predicate change percentage exceeds 100");
    require(spec.reuse_horizon == 1 || spec.reuse_horizon == 16 ||
            spec.reuse_horizon == 256, "unsupported workload reuse horizon");
    if (spec.legacy_root_map && spec.dag_nodes == 2304u)
        require(spec.condition_edges == 2176u,
                "legacy 2,304-node map has exactly 2,176 roots");
    if (spec.role == WorkloadRole::complete_consumer)
        require(spec.groups == spec.condition_edges,
                "complete consumer case must demand every condition edge exactly once");

    WorkloadFixture fixture;
    fixture.spec = spec;
    fixture.predicate_features = predicate_features;
    fixture.programs.push_back(make_condition_dag(spec.dag_nodes, spec.features,
                                                   spec.seed));
    const auto& bank = fixture.programs.front();
    fixture.edge_roots.resize(spec.condition_edges);
    if (spec.legacy_root_map) {
        if (spec.dag_nodes == 7u) {
            const auto dirty_prefix = std::max<std::uint32_t>(
                1u, (spec.condition_edges + 15u) / 16u);
            constexpr std::uint32_t clean_roots[]{1u, 2u, 3u, 5u};
            for (std::uint32_t edge = 0; edge < spec.condition_edges; ++edge) {
                if (edge < dirty_prefix) {
                    const auto modulo = edge % 3u;
                    fixture.edge_roots[edge] = modulo == 0u ? 0u :
                        modulo == 1u ? 4u : 6u;
                } else {
                    fixture.edge_roots[edge] = clean_roots[
                        (edge - dirty_prefix) % std::size(clean_roots)];
                }
            }
        } else {
            require(bank.roots.size() == spec.condition_edges,
                    "legacy 2,304-node map must bind every historical root");
            fixture.edge_roots = bank.roots;
        }
    } else if (spec.program_pattern == ProgramPattern::coherent) {
        std::fill(fixture.edge_roots.begin(), fixture.edge_roots.end(), bank.root);
    } else if (spec.dag_nodes == 7u) {
        const auto start = static_cast<std::uint32_t>(spec.seed % bank.roots.size());
        for (std::uint32_t edge = 0; edge < spec.condition_edges; ++edge)
            fixture.edge_roots[edge] = bank.roots[(start + 3u * edge) % bank.roots.size()];
    } else {
        const auto root_count = static_cast<std::uint32_t>(bank.roots.size());
        const auto start = static_cast<std::uint32_t>(spec.seed % root_count);
        constexpr std::uint32_t step = 37u; // coprime to 2,176: a full permutation
        for (std::uint32_t edge = 0; edge < spec.condition_edges; ++edge)
            fixture.edge_roots[edge] = bank.roots[(start + step * edge) % root_count];
    }

    std::vector<std::uint8_t> demanded(bank.nodes.size(), 0u);
    std::vector<std::uint8_t> unique_roots(bank.nodes.size(), 0u);
    for (const auto root : fixture.edge_roots) {
        if (root >= bank.nodes.size())
            throw std::logic_error("condition edge references a missing root node");
        unique_roots[root] = 1u;
        demanded[root] = 1u;
    }
    for (std::size_t reverse = bank.nodes.size(); reverse != 0; --reverse) {
        const auto node_id = reverse - 1u;
        if (!demanded[node_id]) continue;
        const auto& node = bank.nodes[node_id];
        if (node.op == PredicateOp::all || node.op == PredicateOp::any) {
            for (std::uint32_t child = node.first_child;
                 child < node.first_child + node.child_count; ++child)
                demanded[child] = 1u;
        } else if (node.op == PredicateOp::logical_not) {
            demanded[node.first_child] = 1u;
        }
    }
    fixture.demanded_levels.resize(bank.levels.size());
    for (std::size_t level = 0; level < bank.levels.size(); ++level)
        for (const auto node_id : bank.levels[level])
            if (demanded[node_id]) fixture.demanded_levels[level].push_back(node_id);
    for (std::uint32_t node_id = 0; node_id < bank.nodes.size(); ++node_id) {
        if (unique_roots[node_id]) fixture.demanded_roots.push_back(node_id);
        if (demanded[node_id]) fixture.demanded_nodes.push_back(node_id);
    }

    const std::uint64_t state_count =
        static_cast<std::uint64_t>(spec.cells) * spec.features;
    if (state_count > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("workload accessibility extent overflows");
    fixture.accessibility.resize(static_cast<std::size_t>(state_count));
    std::uint64_t rng = spec.seed;
    const std::uint64_t cutoff = spec.activity_pattern == ActivityPattern::sparse
        ? (std::uint64_t{1} << 32) / 10u
        : (std::uint64_t{1} << 32) * 9u / 10u;
    for (auto& value : fixture.accessibility)
        value = static_cast<std::uint32_t>(next_random(rng)) < cutoff ? 1u : 0u;

    fixture.accessibility_after = fixture.accessibility;
    const std::uint64_t predicate_state_count =
        static_cast<std::uint64_t>(spec.cells) * predicate_features;
    const std::uint64_t changes = (predicate_state_count *
        spec.predicate_change_percent + 50u) / 100u;
    std::vector<std::uint64_t> positions(static_cast<std::size_t>(predicate_state_count));
    std::iota(positions.begin(), positions.end(), std::uint64_t{0});
    std::vector<std::uint64_t> changed_positions;
    changed_positions.reserve(static_cast<std::size_t>(changes));
    for (std::uint64_t index = 0; index < changes; ++index) {
        const auto selected = index + next_random(rng) % (predicate_state_count - index);
        std::swap(positions[static_cast<std::size_t>(index)],
                  positions[static_cast<std::size_t>(selected)]);
        const auto flat = positions[static_cast<std::size_t>(index)];
        const auto cell = flat / predicate_features;
        const auto feature = flat % predicate_features;
        fixture.accessibility_after[static_cast<std::size_t>(
            cell * spec.features + feature)] ^= 1u;
        changed_positions.push_back(flat);
    }
    std::sort(changed_positions.begin(), changed_positions.end());
    fixture.changed_input_offsets.reserve(static_cast<std::size_t>(spec.cells) + 1u);
    fixture.changed_input_offsets.push_back(0u);
    std::size_t changed_cursor = 0;
    for (std::uint32_t cell = 0; cell < spec.cells; ++cell) {
        while (changed_cursor < changed_positions.size() &&
               changed_positions[changed_cursor] / predicate_features == cell) {
            fixture.changed_input_features.push_back(static_cast<std::uint32_t>(
                changed_positions[changed_cursor] % predicate_features));
            ++changed_cursor;
        }
        fixture.changed_input_offsets.push_back(changed_cursor);
    }

    const std::uint64_t decision_count =
        static_cast<std::uint64_t>(spec.cells) * spec.condition_edges;
    fixture.predicates_before.resize(static_cast<std::size_t>(decision_count));
    fixture.predicates_after.resize(static_cast<std::size_t>(decision_count));
    fixture.dirty_node_offsets.reserve(static_cast<std::size_t>(spec.cells) + 1u);
    fixture.dirty_node_offsets.push_back(0u);
    for (std::uint32_t cell = 0; cell < spec.cells; ++cell) {
        const auto* before = fixture.accessibility.data() +
                             static_cast<std::size_t>(cell) * spec.features;
        const auto* after = fixture.accessibility_after.data() +
                            static_cast<std::size_t>(cell) * spec.features;
        const auto before_nodes = evaluate_condition_nodes(bank, before, spec.features);
        const auto after_nodes = evaluate_condition_nodes(bank, after, spec.features);
        for (std::uint32_t edge = 0; edge < spec.condition_edges; ++edge) {
            const auto at = static_cast<std::size_t>(cell) * spec.condition_edges + edge;
            const auto root = fixture.edge_roots[edge];
            fixture.predicates_before[at] = before_nodes[root];
            fixture.predicates_after[at] = after_nodes[root];
            fixture.root_decision_changes += before_nodes[root] != after_nodes[root];
        }
        std::vector<std::uint8_t> dirty(bank.nodes.size(), 0u);
        for (std::size_t node_index = 0; node_index < bank.nodes.size(); ++node_index) {
            const auto& node = bank.nodes[node_index];
            if (node.op == PredicateOp::feature) {
                dirty[node_index] = before[node.feature] != after[node.feature];
            } else if (node.op == PredicateOp::logical_not) {
                dirty[node_index] = dirty[node.first_child];
            } else {
                for (std::uint32_t child = node.first_child;
                     child < node.first_child + node.child_count; ++child)
                    dirty[node_index] = dirty[node_index] || dirty[child];
            }
            if (dirty[node_index])
                fixture.dirty_nodes.push_back(static_cast<std::uint32_t>(node_index));
        }
        fixture.dirty_node_offsets.push_back(fixture.dirty_nodes.size());
    }

    fixture.group_program.resize(spec.groups);
    fixture.group_feature_offsets.reserve(static_cast<std::size_t>(spec.groups) + 1u);
    fixture.group_output_offsets.reserve(static_cast<std::size_t>(spec.groups) + 1u);
    fixture.group_feature_offsets.push_back(0u);
    fixture.group_output_offsets.push_back(0u);
    for (std::uint32_t group = 0; group < spec.groups; ++group) {
        fixture.group_program[group] = spec.role == WorkloadRole::complete_consumer
            ? group : group % spec.condition_edges;
        const std::uint64_t local_base = spec.overlap_pattern == OverlapPattern::high
            ? group % 8u : static_cast<std::uint64_t>(group) * 12u;
        for (std::uint32_t slot = 0; slot < 4; ++slot)
            fixture.group_features.push_back((local_base + slot * 3u) %
                                               (spec.features / 2u));
        for (std::uint32_t slot = 0; slot < 4; ++slot) {
            const std::uint64_t key = spec.overlap_pattern == OverlapPattern::high
                ? (group % 8u) * 17u + slot * 5u
                : static_cast<std::uint64_t>(group) * 29u + slot * 13u;
            const std::uint64_t distal = spec.features / 2u +
                key % (spec.features - spec.features / 2u);
            fixture.group_features.push_back(distal);
            ++fixture.distal_memberships;
        }
        fixture.group_feature_offsets.push_back(fixture.group_features.size());
        if (spec.overlap_pattern == OverlapPattern::high) {
            fixture.group_outputs.push_back(group % 8u);
            fixture.group_outputs.push_back((group + 3u) % 8u);
        } else {
            const auto first = (static_cast<std::uint64_t>(group) * 2u) % spec.features;
            fixture.group_outputs.push_back(first);
            fixture.group_outputs.push_back((first + 1u) % spec.features);
        }
        fixture.group_output_offsets.push_back(fixture.group_outputs.size());
    }
    return fixture;
}

} // namespace cgs
