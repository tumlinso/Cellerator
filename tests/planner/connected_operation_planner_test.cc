#include <Cellerator/planner/end_to_end_planner.hh>

#include <cassert>
#include <cmath>
#include <cstdint>

namespace core = cellerator::compute::math::core;
namespace execution = cellerator::execution;
namespace planner = cellerator::planner;

namespace {

bool numeric_support(const core::numeric_policy &) noexcept { return true; }

core::operation_status unused_prepare(
    const core::operation_candidate &,
    const core::operation_problem &,
    const core::structure_set_key &,
    const core::projection_key &,
    const core::numeric_policy &,
    const core::prepare_policy &,
    core::prepared_operation *) noexcept { return {}; }

planner::phase_costs kernel_cost(double nanoseconds) {
    planner::phase_costs cost{};
    cost.kernel_ns = nanoseconds;
    return cost;
}

planner::phase_costs boundary_cost(double order, double conversion,
    double prepare = 0.0, double communication = 0.0) {
    planner::phase_costs cost{};
    cost.order_transform_ns = order;
    cost.dynamic_input_pack_ns = conversion;
    cost.backend_prepare_ns = prepare;
    cost.communication_ns = communication;
    return cost;
}

planner::planning_keys keys(core::stable_id problem,
    std::uint64_t base) {
    planner::planning_keys value{};
    value.problem.identity = problem;
    core::structure_set_key live{};
    live.count = 1u;
    live.structures[0] = {{base + 1u, base + 2u},
        {static_cast<std::uint32_t>(base + 3u), 1u}, {base + 4u}};
    assert(planner::make_persistent_structure_set_key(live, &value.structures));
    value.geometry = {{base + 5u, 1u}, {base + 6u, 1u},
        {base + 7u, 1u}, {base + 8u, 1u}, {base + 9u, 1u},
        {base + 10u, 1u}};
    value.device = {1u, 7u, 0u, 700u};
    value.build = {100u, 200u, 300u, 400u};
    value.policy = {8u, 4u, 2u, 1u, 1u, 1u, 1u};
    return value;
}

struct fixture {
    core::operation_candidate operations[4]{};
    planner::planner_candidate candidates[4]{};
    planner::connected_operation_stage stages[2]{};
    planner::connected_transition_cost transitions[4]{};
    planner::connected_planner_request request{};

    fixture() {
        const char *names[]{"stage0-fast", "stage0-shared",
            "stage1-fast", "stage1-preserve"};
        for (std::uint32_t index = 0u; index < 4u; ++index) {
            const core::stable_id identity{index + 1u, 86u};
            operations[index].identity = identity;
            operations[index].name = names[index];
            operations[index].operation =
                core::operation_kind::weighted_relation_reduce;
            operations[index].projection =
                core::projection_kind::native_row_masked;
            operations[index].backend = core::backend_kind::native_direct;
            operations[index].capability_flags = core::candidate_deterministic
                | core::candidate_graph_capture;
            operations[index].supports_numeric = numeric_support;
            operations[index].prepare = unused_prepare;
            candidates[index].identity = identity;
            candidates[index].name = names[index];
            candidates[index].operation = &operations[index];
            candidates[index].projection = {{100u + index, 200u + index},
                {300u + index, 1u},
                core::projection_kind::native_row_masked, 1u, index + 1u};
            candidates[index].flags = planner::planner_candidate_correct
                | planner::planner_candidate_deterministic
                | planner::planner_candidate_graph_capture;
        }
        candidates[0].analytical = kernel_cost(50.0);
        candidates[1].analytical = kernel_cost(80.0);
        candidates[2].analytical = kernel_cost(50.0);
        candidates[3].analytical = kernel_cost(80.0);
        const core::stable_id problems[]{{10u, 86u}, {20u, 86u}};
        for (std::uint32_t stage = 0u; stage < 2u; ++stage) {
            stages[stage].problem.kind =
                core::operation_kind::weighted_relation_reduce;
            stages[stage].problem.operation = problems[stage];
            stages[stage].problem.input_count = 1u;
            stages[stage].problem.output_count = 1u;
            stages[stage].problem.logical_work_items = 8192u;
            stages[stage].keys = keys(problems[stage], 1000u + stage * 100u);
            stages[stage].policy.deterministic = true;
            stages[stage].policy.graph_capture_required = true;
            stages[stage].candidates = candidates + stage * 2u;
            stages[stage].candidate_count = 2u;
        }
        transitions[0] = {0u, candidates[0].identity,
            candidates[2].identity, execution::order_transition_kind::transform,
            true, true, 0u, {1000u, 86u},
            boundary_cost(120.0, 80.0)};
        transitions[1] = {0u, candidates[0].identity,
            candidates[3].identity, execution::order_transition_kind::preserve,
            false, true, 0u, {}, boundary_cost(0.0, 0.0)};
        transitions[2] = {0u, candidates[1].identity,
            candidates[2].identity, execution::order_transition_kind::preserve,
            false, true, 0u, {}, boundary_cost(0.0, 0.0)};
        transitions[3] = {0u, candidates[1].identity,
            candidates[3].identity, execution::order_transition_kind::transform,
            true, true, 0u, {1001u, 86u},
            boundary_cost(100.0, 60.0, 40.0, 20.0)};
        request.graph_identity = {860u, 861u};
        request.hierarchy = {862u, 863u};
        request.stages = stages;
        request.stage_count = 2u;
        request.transitions = transitions;
        request.transition_count = 4u;
        request.shortlist_size = 2u;
        request.maximum_measurements = 0u;
        request.current_evidence_revision = 86u;
    }
};

void test_boundary_cost_changes_winner() {
    fixture value;
    planner::connected_planner_result result{};
    assert(planner::plan_connected_operations(value.request, &result));
    assert(result.source == planner::selection_source::analytical);
    assert(result.legal_path_count == 4u && result.shortlist_count == 2u);
    assert(result.winner.candidates[0].low == 1u
        && result.winner.candidates[1].low == 4u);
    assert(std::fabs(result.analytical_total_ns - 130.0) < 1e-9);
    assert(result.stages[0].candidate == &value.candidates[0]
        && result.stages[1].candidate == &value.candidates[3]);

    value.transitions[0].order = execution::order_transition_kind::preserve;
    value.transitions[0].format_conversion = false;
    value.transitions[0].conversion = {};
    value.transitions[0].phases = boundary_cost(0.0, 0.0);
    assert(planner::plan_connected_operations(value.request, &result));
    assert(result.winner.candidates[0].low == 1u
        && result.winner.candidates[1].low == 3u);
    assert(std::fabs(result.analytical_total_ns - 100.0) < 1e-9);
}

struct measurement_fixture {
    std::uint32_t calls = 0u;
};

bool measure(void *context, const planner::connected_plan_path &path,
    planner::measured_connected_plan *out) noexcept {
    auto *fixture = static_cast<measurement_fixture *>(context);
    ++fixture->calls;
    *out = {};
    out->correct = true;
    out->sample_count = 7u;
    out->spread_percent = 1.0;
    out->amortized_total_ns = path.candidates[0].low == 2u
        ? 105.0 : 125.0;
    return true;
}

struct cache_fixture {
    bool found = false;
    bool stored = false;
    planner::connected_plan_cache_entry entry{};
};

bool lookup(void *context, const planner::connected_planning_keys &,
    planner::connected_plan_cache_entry *out) noexcept {
    const auto *cache = static_cast<const cache_fixture *>(context);
    if (!cache->found) return false;
    *out = cache->entry;
    return true;
}

bool store(void *context,
    const planner::connected_plan_cache_entry &entry) noexcept {
    auto *cache = static_cast<cache_fixture *>(context);
    cache->stored = true;
    cache->entry = entry;
    return true;
}

void test_bounded_measurement_and_durable_cache() {
    fixture value;
    measurement_fixture measurement{};
    cache_fixture cache{};
    value.request.maximum_measurements = 2u;
    value.request.measurement = {&measurement, measure};
    value.request.cache = {&cache, lookup, store};
    planner::connected_planner_result result{};
    assert(planner::plan_connected_operations(value.request, &result));
    assert(result.source == planner::selection_source::empirical);
    assert(result.measurement_count == 2u && measurement.calls == 2u);
    assert(result.winner.candidates[0].low == 2u
        && result.winner.candidates[1].low == 3u);
    assert(cache.stored && result.confidence >= 0.8);

    cache.found = true;
    value.request.measurement = {};
    assert(planner::plan_connected_operations(value.request, &result));
    assert(result.source == planner::selection_source::cache
        && result.cache == planner::cache_state::hit);
    value.stages[1].keys.build.kernel_build += 1u;
    assert(planner::plan_connected_operations(value.request, &result));
    assert(result.cache == planner::cache_state::stale
        && result.source == planner::selection_source::analytical);
}

void test_uncertainty_and_contract_rejection() {
    fixture value;
    value.request.force_empirical = true;
    planner::connected_planner_result result{};
    assert(!planner::plan_connected_operations(value.request, &result));
    assert(result.status.code
        == planner::planner_status_code::no_correct_measurement);

    value.request.force_empirical = false;
    value.transitions[0].conversion = {};
    assert(!planner::plan_connected_operations(value.request, &result));
    assert(result.status.code == planner::planner_status_code::invalid_argument);

    value.transitions[0].conversion = {1000u, 86u};
    value.transitions[0].legal = false;
    value.transitions[1].legal = false;
    value.transitions[2].legal = false;
    value.transitions[3].legal = false;
    assert(!planner::plan_connected_operations(value.request, &result));
    assert(result.status.code
        == planner::planner_status_code::no_legal_candidate);
}

} // namespace

int main() {
    test_boundary_cost_changes_winner();
    test_bounded_measurement_and_durable_cache();
    test_uncertainty_and_contract_rejection();
    return 0;
}
