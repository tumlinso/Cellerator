#include <Cellerator/planner/end_to_end_planner.hh>

#include <cassert>
#include <cmath>
#include <cstdint>

namespace core = cellerator::compute::math::core;
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
    core::prepared_operation *) noexcept {
    return {};
}

struct fixture {
    core::operation_candidate operations[3]{};
    planner::planner_candidate candidates[3]{};
};

planner::phase_costs phases(double pack, double projection, double prepare,
    double kernel, double order = 0.0) {
    planner::phase_costs value{};
    value.host_preparation_ns = 10.0;
    value.semantic_packing_ns = pack;
    value.projection_construction_ns = projection;
    value.backend_prepare_ns = prepare;
    value.h2d_ns = 20.0;
    value.dynamic_input_pack_ns = 5.0;
    value.kernel_ns = kernel;
    value.epilogue_ns = 3.0;
    value.order_transform_ns = order;
    value.synchronization_ns = 2.0;
    value.communication_ns = 0.0;
    value.d2h_ns = 1.0;
    value.persistent_bytes = 1024u;
    value.transient_bytes = 256u;
    return value;
}

fixture candidates_fixture() {
    fixture value{};
    const char *names[] = {"native", "vendor-csr", "dense"};
    const core::backend_kind backends[] = {core::backend_kind::native_direct,
        core::backend_kind::vendor_library,
        core::backend_kind::vendor_library};
    const core::projection_kind projections[] = {
        core::projection_kind::native_row_masked,
        core::projection_kind::csr,
        core::projection_kind::dense_fragment};
    for (std::uint32_t index = 0u; index < 3u; ++index) {
        const core::stable_id identity{index + 1u, 9u};
        value.operations[index].identity = identity;
        value.operations[index].name = names[index];
        value.operations[index].operation =
            core::operation_kind::weighted_relation_reduce;
        value.operations[index].projection = projections[index];
        value.operations[index].backend = backends[index];
        value.operations[index].capability_flags =
            core::candidate_deterministic | core::candidate_graph_capture;
        value.operations[index].supports_numeric = numeric_support;
        value.operations[index].prepare = unused_prepare;
        value.candidates[index].identity = identity;
        value.candidates[index].name = names[index];
        value.candidates[index].operation = &value.operations[index];
        value.candidates[index].projection = {{100u + index, 200u + index},
            {300u + index, 1u}, projections[index], 1u, index};
        value.candidates[index].flags = planner::planner_candidate_correct
            | planner::planner_candidate_deterministic
            | planner::planner_candidate_graph_capture;
        if (index != 0u)
            value.candidates[index].flags |=
                planner::planner_candidate_conventional;
    }
    value.candidates[0].analytical = phases(1000.0, 500.0, 200.0, 30.0);
    value.candidates[1].analytical = phases(100.0, 100.0, 100.0, 70.0);
    value.candidates[2].analytical = phases(1000.0, 2000.0, 100.0, 15.0);
    return value;
}

planner::planning_keys keys(std::uint64_t reuse = 8u) {
    planner::planning_keys value{};
    value.problem.identity = {71u, 72u};
    core::structure_set_key live{};
    live.count = 2u;
    live.structures[0] = {{21u, 22u}, {12u, 1u}, {4u}};
    live.structures[1] = {{1u, 2u}, {11u, 1u}, {3u}};
    assert(planner::make_persistent_structure_set_key(live, &value.structures));
    value.geometry = {{2u, 3u}, {3u, 4u}, {4u, 5u},
        {6u, 7u}, {8u, 9u}, {10u, 11u}};
    value.device = {1u, 7u, 0u, 700u};
    value.build = {100u, 200u, 300u, 400u};
    value.policy = {reuse, reuse, 1u, 1u, 1u, 1u};
    return value;
}

planner::planner_request request(
    const fixture &value, std::uint64_t reuse = 8u) {
    planner::planner_request result{};
    result.problem.kind = core::operation_kind::weighted_relation_reduce;
    result.problem.operation = {71u, 72u};
    result.problem.input_count = 1u;
    result.problem.output_count = 1u;
    result.problem.logical_work_items = 8192u;
    result.keys = keys(reuse);
    result.candidates = value.candidates;
    result.candidate_count = 3u;
    result.policy.shortlist_size = 3u;
    result.policy.maximum_measurements = 3u;
    result.policy.minimum_tuning_work_items = 4096u;
    result.policy.practical_tolerance_percent = 2.0;
    result.policy.maximum_spread_percent = 5.0;
    result.policy.minimum_cache_confidence = 0.8;
    result.current_evidence_revision = 44u;
    return result;
}

struct measurement_fixture {
    std::uint32_t calls = 0u;
    bool contaminate_all = false;
};

bool measure(void *context, const planner::planner_candidate &candidate,
    planner::measured_candidate *out) noexcept {
    auto *state = static_cast<measurement_fixture *>(context);
    ++state->calls;
    *out = {};
    out->correct = true;
    out->contaminated = state->contaminate_all;
    out->sample_count = 7u;
    out->spread_percent = 1.0;
    if (candidate.identity.low == 1u)
        out->phases = phases(1000.0, 500.0, 200.0, 100.0, 80.0);
    else if (candidate.identity.low == 2u)
        out->phases = phases(100.0, 100.0, 100.0, 40.0, 0.0);
    else
        out->phases = phases(1000.0, 2000.0, 100.0, 20.0, 0.0);
    return true;
}

struct cache_fixture {
    bool found = false;
    bool stored = false;
    bool store_succeeds = true;
    planner::plan_cache_entry entry{};
};

bool lookup(void *context, const planner::planning_keys &,
    planner::plan_cache_entry *out) noexcept {
    const auto *cache = static_cast<const cache_fixture *>(context);
    if (!cache->found) return false;
    *out = cache->entry;
    return true;
}

bool store(void *context, const planner::plan_cache_entry &entry) noexcept {
    auto *cache = static_cast<cache_fixture *>(context);
    cache->stored = true;
    cache->entry = entry;
    return cache->store_succeeds;
}

void test_cost_accounting() {
    planner::total_cost cost{};
    const planner::phase_costs value = phases(800.0, 400.0, 200.0, 50.0, 25.0);
    assert(planner::compute_total_cost(value, 8u, 4u, &cost));
    const double expected = 10.0 + 100.0 + 100.0 + 50.0
        + 20.0 + 5.0 + 50.0 + 3.0 + 25.0 + 2.0 + 0.0 + 1.0;
    assert(std::fabs(cost.amortized_total_ns - expected) < 1e-9);
}

void test_measured_conventional_winner_and_cache() {
    fixture value = candidates_fixture();
    planner::planner_request plan_request = request(value);
    measurement_fixture measurements{};
    cache_fixture cache{};
    plan_request.measurement = {&measurements, measure};
    plan_request.cache = {&cache, lookup, store};
    planner::planner_result result{};
    assert(planner::plan_end_to_end(plan_request, &result));
    assert(result.source == planner::selection_source::empirical);
    assert(result.winner.low == 2u && result.conventional_winner);
    assert(result.measurement_count == 3u && measurements.calls == 3u);
    assert(cache.stored && cache.entry.winner.low == 2u);
    assert(result.confidence >= plan_request.policy.minimum_cache_confidence);
    assert(result.reason != nullptr);

    cache.found = true;
    plan_request.measurement = {};
    for (planner::planner_candidate &candidate : value.candidates)
        ++candidate.projection.runtime.generation;
    assert(planner::plan_end_to_end(plan_request, &result));
    assert(result.source == planner::selection_source::cache);
    assert(result.cache == planner::cache_state::hit);
    assert(result.winner.low == 2u);
    assert(result.selected->projection.runtime.generation == 2u);

    cache.entry.keys.build.kernel_build += 1u;
    assert(planner::plan_end_to_end(plan_request, &result));
    assert(result.cache == planner::cache_state::stale);
    assert(result.source == planner::selection_source::analytical);

    cache.entry.keys.build.kernel_build = plan_request.keys.build.kernel_build;
    plan_request.keys.structures.structures[0].identity.low += 1u;
    assert(planner::plan_end_to_end(plan_request, &result));
    assert(result.cache == planner::cache_state::stale);
    assert(result.source == planner::selection_source::analytical);

    cache.entry = {};
    cache.found = false;
    cache.store_succeeds = false;
    plan_request = request(value);
    plan_request.measurement = {&measurements, measure};
    plan_request.cache = {&cache, lookup, store};
    assert(planner::plan_end_to_end(plan_request, &result));
    assert(result.cache_store_failed && result.source
        == planner::selection_source::empirical);
}

void test_bounded_and_one_shot_tuning() {
    fixture value = candidates_fixture();
    planner::planner_request plan_request = request(value);
    measurement_fixture measurements{};
    plan_request.measurement = {&measurements, measure};
    plan_request.policy.shortlist_size = 2u;
    plan_request.policy.maximum_measurements = 1u;
    planner::planner_result result{};
    assert(planner::plan_end_to_end(plan_request, &result));
    assert(result.measurement_count == 1u && measurements.calls == 1u);

    plan_request = request(value, 1u);
    plan_request.measurement = {&measurements, measure};
    measurements.calls = 0u;
    assert(planner::plan_end_to_end(plan_request, &result));
    assert(result.tuning_skipped && result.measurement_count == 0u);
    assert(measurements.calls == 0u);
}

void test_policy_rejection() {
    fixture value = candidates_fixture();
    value.candidates[0].flags &= ~planner::planner_candidate_deterministic;
    planner::planner_request plan_request = request(value);
    plan_request.policy.deterministic = true;
    plan_request.policy.maximum_persistent_bytes = 1024u;
    planner::planner_result result{};
    assert(planner::plan_end_to_end(plan_request, &result));
    assert(result.diagnostics[0].rejection
        == planner::candidate_rejection::nondeterministic);
}

void test_persistent_structure_keys_and_measurement_fallback() {
    fixture value = candidates_fixture();
    planner::planner_request plan_request = request(value);
    const planner::planning_keys original = plan_request.keys;
    plan_request.keys.structures.structures[0].identity.low += 1u;
    assert(!planner::same_planning_keys(original, plan_request.keys));

    plan_request = request(value);
    measurement_fixture measurements{};
    measurements.contaminate_all = true;
    plan_request.measurement = {&measurements, measure};
    planner::planner_result result{};
    assert(planner::plan_end_to_end(plan_request, &result));
    assert(result.source == planner::selection_source::analytical);
    assert(result.reason != nullptr && result.measurement_count == 3u);

    plan_request.policy.allow_analytical_fallback_after_measurement_failure =
        false;
    assert(!planner::plan_end_to_end(plan_request, &result));
    assert(result.status.code
        == planner::planner_status_code::no_correct_measurement);
}

void test_objective_v2() {
    planner::objective_v2_statistics statistics{};
    statistics.useful_edges = 1000u;
    statistics.metadata_bytes = 4000u;
    statistics.value_bytes = 2000u;
    statistics.partial_block_slots = 100u;
    statistics.cross_partition_edges = 20u;
    statistics.feature_reuse = 0.5;
    statistics.row_imbalance = 0.1;
    statistics.module_overlap = 0.2;
    statistics.module_activation_frequency = 0.5;
    statistics.transpose_locality = 0.8;
    statistics.quantization_outlier_fraction = 0.01;
    planner::objective_v2_context context{};
    context.operation = core::operation_kind::weighted_relation_reduce;
    context.dense_width = 32u;
    context.registers_per_thread = 48u;
    context.shared_bytes_per_block = 1024u;
    context.expected_reuse = 10u;
    context.transpose_required = true;
    context.canonical_output_required = true;
    context.quantized = true;
    planner::objective_v2_weights weights{};
    weights.module_credit = 0.5;
    planner::objective_v2_result result{};
    assert(planner::evaluate_objective_v2(
        statistics, context, weights, &result));
    assert(result.schema_version == 2u);
    assert(result.storage > 0.0 && result.execution > 0.0);
    assert(result.order_and_transpose > 0.0 && result.communication > 0.0);
    assert(result.score != result.storage);
}

} // namespace

int main() {
    test_cost_accounting();
    test_measured_conventional_winner_and_cache();
    test_bounded_and_one_shot_tuning();
    test_policy_rejection();
    test_persistent_structure_keys_and_measurement_fallback();
    test_objective_v2();
    return 0;
}
