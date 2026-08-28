#include "Cellerator/geometry/alternating_refinement.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using cellpack::u32;
using cellpack::u64;

constexpr u64 dataset_identity = 0x4350313044415441ull;
constexpr u64 feature_axis_identity = 0x4350313046454154ull;
constexpr u64 row_domain_identity = 0x43503130524f5753ull;
constexpr u64 split_identity = 0x4350313053504c54ull;

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "cellPackAlternatingRefinementTest: " << message << '\n';
    std::exit(1);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

void require_status(const cellpack::validation_result &status, const char *context) {
    if (!status) fail(std::string(context) + ": " + status.message);
}

cellpack::frozen_packing_plan make_plan(
    const std::vector<std::vector<u32>> &blocks,
    u64 evaluation_identity) {
    constexpr u32 feature_count = 6u;
    std::vector<u32> permutation, inverse(feature_count), offsets(1u, 0u);
    std::vector<u32> feature_to_block(feature_count), feature_to_local(feature_count);
    for (u32 block = 0u; block < blocks.size(); ++block) {
        for (u32 local = 0u; local < blocks[block].size(); ++local) {
            const u32 feature = blocks[block][local];
            require(feature < feature_count, "plan fixture feature out of range");
            feature_to_block[feature] = block;
            feature_to_local[feature] = local;
            inverse[feature] = static_cast<u32>(permutation.size());
            permutation.push_back(feature);
        }
        offsets.push_back(static_cast<u32>(permutation.size()));
    }
    require(permutation.size() == feature_count, "plan fixture does not cover feature axis");
    const u32 row_offsets[] = {0u, 4u, 8u};
    cellpack::frozen_packing_plan_build_view build;
    build.row_count = 8u;
    build.feature_count = feature_count;
    build.feature_permutation = permutation.data();
    build.inverse_feature_permutation = inverse.data();
    build.feature_block_count = static_cast<u32>(blocks.size());
    build.feature_block_offsets = offsets.data();
    build.feature_to_block = feature_to_block.data();
    build.feature_to_local = feature_to_local.data();
    build.row_group_count = 2u;
    build.row_group_offsets = row_offsets;
    build.maximum_feature_block_width = 3u;
    build.row_group_width = 4u;
    build.identity.feature_axis_fingerprint = feature_axis_identity;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind = cellpack::packing_row_domain_kind::full_dataset_identity;
    build.identity.row_domain_identity = row_domain_identity;
    build.identity.evaluation_source_identity = evaluation_identity;
    build.objective_kind = cellpack::packing_exact_objective_kind::row_active_block_references;
    build.cost_policy_identity = 0x43503130434f5354ull;
    cellpack::frozen_packing_plan result;
    require_status(cellpack::freeze_packing_plan(build, &result), "freeze plan fixture");
    return result;
}

cellpack::packing_validation_metrics metrics(
    u64 rows,
    u64 nnz,
    u64 encoded_bytes,
    u64 metadata_bytes,
    u64 active_blocks,
    u64 tile_unions,
    u64 padding,
    u64 runtime_nanoseconds,
    u64 preprocessing_nanoseconds) {
    cellpack::packing_validation_metrics result;
    result.available = cellpack::packing_validation_metric_storage
        | cellpack::packing_validation_metric_records
        | cellpack::packing_validation_metric_tiles
        | cellpack::packing_validation_metric_preprocessing
        | cellpack::packing_validation_metric_runtime
        | cellpack::packing_validation_metric_correctness;
    result.dataset_identity = dataset_identity;
    result.feature_axis_identity = feature_axis_identity;
    result.row_domain_identity = row_domain_identity;
    result.split_identity = split_identity;
    result.row_count = rows;
    result.feature_count = 6u;
    result.nnz_count = nnz;
    result.encoded_bytes = encoded_bytes;
    result.metadata_bytes = metadata_bytes;
    result.baseline_bytes = 4u * nnz + 8u * rows + 4u * (rows + 1u);
    result.active_block_references = active_blocks;
    result.tile_count = 2u;
    result.tile_block_union_references = tile_unions;
    result.padding_slots = padding;
    result.preprocessing_input_nnz = nnz;
    result.preprocessing_elapsed_nanoseconds = preprocessing_nanoseconds;
    result.preprocessing_repeat_count = 2u;
    result.runtime_input_nnz = nnz;
    result.runtime_bytes = encoded_bytes;
    result.runtime_elapsed_nanoseconds = runtime_nanoseconds;
    result.runtime_repeat_count = 4u;
    result.correctness_items = nnz;
    return result;
}

cellpack::alternating_refinement_observation observation(
    const cellpack::frozen_packing_plan *plan,
    cellpack::alternating_refinement_phase phase,
    u32 iteration,
    u64 candidate_identity,
    u64 parent_identity,
    u64 held_out_bytes,
    u64 held_out_runtime,
    bool succeeded = true) {
    cellpack::alternating_refinement_observation result;
    result.phase = phase;
    result.iteration = iteration;
    result.candidate_identity = candidate_identity;
    result.parent_plan_identity = parent_identity;
    result.plan = plan;
    result.training = metrics(6u, 24u, held_out_bytes - 10u, 30u, 18u, 8u,
        2u, held_out_runtime - 40u, 200u);
    result.held_out = metrics(2u, 8u, held_out_bytes, 14u, 8u, 5u,
        1u, held_out_runtime, 80u);
    result.evaluation_succeeded = succeeded;
    return result;
}

cellpack::alternating_refinement_config config() {
    cellpack::alternating_refinement_config result;
    result.maximum_iterations = 8u;
    result.maximum_evaluations = 8u;
    result.maximum_consecutive_rejections = 4u;
    result.dataset_identity = dataset_identity;
    result.feature_axis_identity = feature_axis_identity;
    result.feature_axis_identity_version = 1u;
    result.row_domain_identity = row_domain_identity;
    result.split_identity = split_identity;
    result.seed = 0x123456789abcdef0ull;
    result.absolute_improvement_tolerance = 0.5;
    result.relative_improvement_tolerance = 0.0;
    result.weights.encoded_bytes = 1.0;
    result.weights.runtime_mean_nanoseconds = 0.1;
    return result;
}

void add_workload_profile(cellpack::packing_validation_metrics *metrics,
    u64 forward_mean, u64 transpose_mean, u64 active_interactions,
    u64 partition_cut_edges, u64 bootstrap_median, u64 bootstrap_mad) {
    metrics->available |= cellpack::packing_validation_metric_workload_profile;
    metrics->workload_profile_identity = 0x877001u;
    metrics->workload_evidence_revision = 0x877002u;
    metrics->forward_repeat_count = 4u;
    metrics->transpose_repeat_count = 4u;
    metrics->forward_elapsed_nanoseconds = forward_mean * 4u;
    metrics->transpose_elapsed_nanoseconds = transpose_mean * 4u;
    metrics->active_interactions = active_interactions;
    metrics->partition_cut_edges = partition_cut_edges;
    metrics->bootstrap_sample_count = 32u;
    metrics->bootstrap_median_total_nanoseconds = bootstrap_median;
    metrics->bootstrap_mad_nanoseconds = bootstrap_mad;
}

void test_acceptance_rollback_and_reproducibility() {
    cellpack::frozen_packing_plan baseline_plan = make_plan(
        {{0u}, {1u}, {2u}, {3u}, {4u}, {5u}}, 0x1001u);
    cellpack::frozen_packing_plan gene_plan = make_plan(
        {{0u, 1u}, {2u, 3u}, {4u}, {5u}}, 0x1002u);
    cellpack::frozen_packing_plan rejected_plan = make_plan(
        {{0u, 2u}, {1u, 3u}, {4u}, {5u}}, 0x1003u);
    cellpack::frozen_packing_plan final_plan = make_plan(
        {{0u, 1u}, {2u, 3u}, {4u, 5u}}, 0x1004u);
    cellpack::alternating_refinement_observation baseline = observation(
        &baseline_plan, cellpack::alternating_refinement_phase::baseline,
        0u, 0x2000u, 0u, 120u, 800u);
    const u64 baseline_id = cellpack::alternating_refinement_plan_identity(baseline_plan);
    const u64 gene_id = cellpack::alternating_refinement_plan_identity(gene_plan);
    std::vector<cellpack::alternating_refinement_observation> candidates;
    candidates.push_back(observation(&gene_plan,
        cellpack::alternating_refinement_phase::gene_blocks,
        1u, 0x2001u, baseline_id, 100u, 640u));
    // Better training bytes but worse held-out objective: must roll back.
    candidates.push_back(observation(&rejected_plan,
        cellpack::alternating_refinement_phase::cell_order_and_tiles,
        2u, 0x2002u, gene_id, 110u, 720u));
    candidates.back().training.encoded_bytes = 60u;
    candidates.push_back(observation(nullptr,
        cellpack::alternating_refinement_phase::gene_blocks,
        3u, 0x2003u, gene_id, 100u, 640u, false));
    candidates.push_back(observation(&final_plan,
        cellpack::alternating_refinement_phase::cell_order_and_tiles,
        4u, 0x2004u, gene_id, 82u, 560u));

    std::vector<cellpack::alternating_refinement_event> events(candidates.size());
    cellpack::alternating_refinement_result first;
    require_status(cellpack::run_alternating_refinement(
        baseline, candidates.data(), candidates.size(), config(),
        {events.size(), events.data()}, &first), "run alternating refinement");
    require(first.accepted_candidates == 2u && first.rejected_candidates == 2u,
        "accept/reject counts mismatch");
    require(first.evaluation_errors == 1u && first.best_plan == &final_plan,
        "evaluation rollback or best-plan checkpoint mismatch");
    require(events[0].outcome == cellpack::alternating_refinement_outcome::accepted
        && events[1].outcome
            == cellpack::alternating_refinement_outcome::rejected_no_improvement
        && events[2].outcome
            == cellpack::alternating_refinement_outcome::rejected_evaluation_error
        && events[3].outcome == cellpack::alternating_refinement_outcome::accepted,
        "event outcome sequence mismatch");
    require(first.best_held_out_objective < events[0].held_out_objective,
        "final accepted objective did not improve monotonically");

    std::vector<cellpack::alternating_refinement_event> repeat_events(candidates.size());
    cellpack::alternating_refinement_result second;
    require_status(cellpack::run_alternating_refinement(
        baseline, candidates.data(), candidates.size(), config(),
        {repeat_events.size(), repeat_events.data()}, &second),
        "repeat alternating refinement");
    require(first.controller_identity == second.controller_identity
        && first.best_plan_identity == second.best_plan_identity
        && first.best_held_out_objective == second.best_held_out_objective,
        "fixed inputs did not reproduce controller result");
}

void test_tolerance_caps_empty_and_identity_rejection() {
    cellpack::frozen_packing_plan baseline_plan = make_plan(
        {{0u}, {1u}, {2u}, {3u}, {4u}, {5u}}, 0x3001u);
    cellpack::frozen_packing_plan candidate_plan = make_plan(
        {{0u, 1u}, {2u}, {3u}, {4u}, {5u}}, 0x3002u);
    cellpack::alternating_refinement_observation baseline = observation(
        &baseline_plan, cellpack::alternating_refinement_phase::baseline,
        0u, 0x4000u, 0u, 100u, 400u);
    const u64 baseline_id = cellpack::alternating_refinement_plan_identity(baseline_plan);
    cellpack::alternating_refinement_observation tie = observation(
        &candidate_plan, cellpack::alternating_refinement_phase::gene_blocks,
        1u, 0x4001u, baseline_id, 100u, 398u);
    cellpack::alternating_refinement_config settings = config();
    settings.absolute_improvement_tolerance = 1.0;
    cellpack::alternating_refinement_event event;
    cellpack::alternating_refinement_result result;
    require_status(cellpack::run_alternating_refinement(
        baseline, &tie, 1u, settings, {1u, &event}, &result),
        "run tolerance tie");
    require(result.accepted_candidates == 0u && result.best_plan == &baseline_plan,
        "tolerance tie changed best plan");

    require_status(cellpack::run_alternating_refinement(
        baseline, nullptr, 0u, settings, {}, &result), "run empty candidate sequence");
    require(result.event_count == 0u && result.best_plan == &baseline_plan,
        "empty sequence did not preserve baseline");

    std::vector<cellpack::alternating_refinement_observation> capped(2u, tie);
    capped[1].phase = cellpack::alternating_refinement_phase::cell_order_and_tiles;
    capped[1].iteration = 2u;
    settings.maximum_iterations = 1u;
    require_status(cellpack::run_alternating_refinement(
        baseline, capped.data(), capped.size(), settings, {1u, &event}, &result),
        "run iteration cap");
    require(result.stop_reason == cellpack::alternating_refinement_stop_reason::iteration_cap
        && result.event_count == 1u, "iteration cap did not terminate exactly");

    tie.held_out.split_identity ^= 1u;
    settings.maximum_iterations = 8u;
    require(!cellpack::run_alternating_refinement(
        baseline, &tie, 1u, settings, {1u, &event}, &result),
        "tampered held-out identity was accepted");

    auto incomplete = baseline.held_out;
    incomplete.available &= ~cellpack::packing_validation_metric_storage;
    double objective = 0.0;
    require(!cellpack::evaluate_alternating_refinement_objective(
        incomplete, settings.weights, &objective),
        "storage-weighted objective accepted unavailable storage metrics");
}

void test_workload_profiles_held_out_and_bootstrap_stability() {
    cellpack::frozen_packing_plan baseline_plan = make_plan(
        {{0u}, {1u}, {2u}, {3u}, {4u}, {5u}}, 0x8701u);
    cellpack::frozen_packing_plan unstable_plan = make_plan(
        {{0u, 1u}, {2u}, {3u}, {4u}, {5u}}, 0x8702u);
    cellpack::frozen_packing_plan stable_plan = make_plan(
        {{0u, 1u}, {2u, 3u}, {4u}, {5u}}, 0x8703u);
    auto baseline = observation(&baseline_plan,
        cellpack::alternating_refinement_phase::baseline,
        0u, 0x8710u, 0u, 120u, 800u);
    add_workload_profile(&baseline.training,
        950u, 580u, 2900u, 280u, 900u, 20u);
    add_workload_profile(&baseline.held_out,
        1000u, 600u, 1000u, 100u, 940u, 20u);
    const u64 baseline_id =
        cellpack::alternating_refinement_plan_identity(baseline_plan);
    auto unstable = observation(&unstable_plan,
        cellpack::alternating_refinement_phase::gene_blocks,
        1u, 0x8711u, baseline_id, 100u, 600u);
    add_workload_profile(&unstable.training,
        650u, 400u, 2500u, 200u, 600u, 5u);
    // Attractive mean but unstable on held-out bootstrap draws.
    add_workload_profile(&unstable.held_out,
        800u, 500u, 900u, 80u, 742u, 300u);
    auto stable = observation(&stable_plan,
        cellpack::alternating_refinement_phase::cell_order_and_tiles,
        2u, 0x8712u, baseline_id, 105u, 650u);
    add_workload_profile(&stable.training,
        820u, 500u, 2700u, 230u, 750u, 8u);
    add_workload_profile(&stable.held_out,
        850u, 520u, 950u, 90u, 790u, 10u);
    cellpack::alternating_refinement_observation candidates[]{unstable, stable};

    auto settings = config();
    settings.workload_profile_identity = 0x877001u;
    settings.workload_evidence_revision = 0x877002u;
    settings.minimum_bootstrap_samples = 32u;
    settings.weights = cellpack::alternating_refinement_objective_weights{};
    settings.weights.encoded_bytes = 0.0;
    settings.weights.runtime_mean_nanoseconds = 0.0;
    settings.weights.preprocessing_mean_nanoseconds = 0.125;
    settings.weights.forward_mean_nanoseconds = 0.75;
    settings.weights.transpose_mean_nanoseconds = 0.25;
    settings.weights.active_interaction_nanoseconds = 0.01;
    settings.weights.partition_cut_edge_nanoseconds = 0.1;
    settings.weights.bootstrap_mad_nanoseconds = 1.0;
    double exact_surrogate = 0.0;
    require_status(cellpack::evaluate_alternating_refinement_objective(
        baseline.held_out, settings.weights, &exact_surrogate),
        "evaluate workload exact surrogate");
    const double expected = 0.125 * 40.0 + 0.75 * 1000.0
        + 0.25 * 600.0 + 0.01 * 1000.0 + 0.1 * 100.0 + 20.0;
    require(std::fabs(exact_surrogate - expected) < 1.0e-9,
        "workload exact surrogate accounting mismatch");

    cellpack::alternating_refinement_event events[2]{};
    cellpack::alternating_refinement_result result{};
    require_status(cellpack::run_alternating_refinement(baseline,
        candidates, 2u, settings, {2u, events}, &result),
        "run workload-profile refinement");
    require(events[0].outcome
            == cellpack::alternating_refinement_outcome::rejected_no_improvement
        && events[1].outcome
            == cellpack::alternating_refinement_outcome::accepted
        && result.best_plan == &stable_plan,
        "held-out/bootstrap stability did not control workload refinement");

    candidates[1].held_out.workload_evidence_revision ^= 1u;
    require(!cellpack::run_alternating_refinement(baseline,
        candidates, 2u, settings, {2u, events}, &result),
        "stale workload evidence was accepted");
    candidates[1].held_out.workload_evidence_revision ^= 1u;
    candidates[1].held_out.bootstrap_sample_count = 8u;
    require(!cellpack::run_alternating_refinement(baseline,
        candidates, 2u, settings, {2u, events}, &result),
        "under-sampled bootstrap evidence was accepted");
}

} // namespace

int main() {
    test_acceptance_rollback_and_reproducibility();
    test_tolerance_caps_empty_and_identity_rejection();
    test_workload_profiles_held_out_and_bootstrap_stability();
    return 0;
}
