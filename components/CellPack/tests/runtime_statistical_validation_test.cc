#include "CellPack/runtime_statistical_validation.hh"
#include "CellPack/optimizer.hh"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

using cellpack::u32;
using cellpack::u64;

constexpr u64 dataset_identity = 0x4350313152544441ull;
constexpr u64 feature_axis_identity = 0x4350313152544641ull;
constexpr u64 row_domain_identity = 0x435031315254524full;
constexpr u64 split_identity = 0x4350313152545350ull;

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "cellPackRuntimeStatisticalValidationTest: " << message << '\n';
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
            inverse[feature] = static_cast<u32>(permutation.size());
            feature_to_block[feature] = block;
            feature_to_local[feature] = local;
            permutation.push_back(feature);
        }
        offsets.push_back(static_cast<u32>(permutation.size()));
    }
    require(permutation.size() == feature_count, "plan fixture feature coverage mismatch");
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
    build.cost_policy_identity = 0x435031315254434full;
    cellpack::frozen_packing_plan result;
    require_status(cellpack::freeze_packing_plan(build, &result), "freeze runtime plan");
    return result;
}

cellpack::packing_validation_metrics metrics(u64 bytes, u64 runtime_ns, u64 preprocessing_ns) {
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
    result.row_count = 8u;
    result.feature_count = 6u;
    result.nnz_count = 24u;
    result.encoded_bytes = bytes;
    result.metadata_bytes = 24u;
    result.baseline_bytes = 256u;
    result.active_block_references = 16u;
    result.tile_count = 2u;
    result.tile_block_union_references = 8u;
    result.padding_slots = 4u;
    result.preprocessing_input_nnz = 24u;
    result.preprocessing_elapsed_nanoseconds = preprocessing_ns;
    result.preprocessing_repeat_count = 2u;
    result.runtime_input_nnz = 24u;
    result.runtime_bytes = bytes;
    result.runtime_elapsed_nanoseconds = runtime_ns;
    result.runtime_repeat_count = 4u;
    result.correctness_items = 24u;
    return result;
}

cellpack::alternating_refinement_config controller_config(u64 seed) {
    cellpack::alternating_refinement_config result;
    result.maximum_iterations = 2u;
    result.maximum_evaluations = 2u;
    result.maximum_consecutive_rejections = 2u;
    result.dataset_identity = dataset_identity;
    result.feature_axis_identity = feature_axis_identity;
    result.feature_axis_identity_version = 1u;
    result.row_domain_identity = row_domain_identity;
    result.split_identity = split_identity;
    result.seed = seed;
    result.weights.encoded_bytes = 1.0;
    result.weights.runtime_mean_nanoseconds = 0.01;
    return result;
}

struct controller_fixture {
    std::vector<cellpack::alternating_refinement_event> events{1u};
    cellpack::alternating_refinement_result result{};
};

controller_fixture run_controller(
    const cellpack::frozen_packing_plan &baseline_plan,
    const cellpack::frozen_packing_plan &candidate_plan,
    u64 seed,
    u64 candidate_identity,
    u64 bytes,
    u64 runtime_ns) {
    cellpack::alternating_refinement_observation baseline;
    baseline.phase = cellpack::alternating_refinement_phase::baseline;
    baseline.candidate_identity = candidate_identity - 1u;
    baseline.plan = &baseline_plan;
    baseline.training = metrics(150u, 800u, 200u);
    baseline.held_out = metrics(160u, 840u, 220u);
    cellpack::alternating_refinement_observation candidate;
    candidate.phase = cellpack::alternating_refinement_phase::gene_blocks;
    candidate.iteration = 1u;
    candidate.candidate_identity = candidate_identity;
    candidate.parent_plan_identity =
        cellpack::alternating_refinement_plan_identity(baseline_plan);
    candidate.plan = &candidate_plan;
    candidate.training = metrics(bytes - 10u, runtime_ns - 40u, 180u);
    candidate.held_out = metrics(bytes, runtime_ns, 200u);
    controller_fixture output;
    require_status(cellpack::run_alternating_refinement(
        baseline, &candidate, 1u, controller_config(seed),
        {output.events.size(), output.events.data()}, &output.result),
        "run controller for runtime validation");
    require(output.result.accepted_candidates == 1u
        && output.result.best_plan == &candidate_plan,
        "controller did not emit the relearned candidate plan");
    return output;
}

struct bootstrap_fixture {
    std::vector<u32> multiplicities;
    cellpack::validation_bootstrap_provenance provenance{};
};

bootstrap_fixture make_bootstrap(
    const cellpack::validation_identity_view &identities,
    u64 seed) {
    bootstrap_fixture result;
    result.multiplicities.resize(identities.row_count);
    cellpack::validation_bootstrap_config config;
    config.seed = seed;
    config.unit_draw_count = identities.group_identities == nullptr
        ? identities.row_count : 4u;
    require_status(cellpack::build_validation_bootstrap(
        identities, config, {result.multiplicities.size(), result.multiplicities.data()},
        &result.provenance), "build runtime bootstrap");
    return result;
}

struct optimizer_support_fixture {
    std::vector<u32> row_offsets{0u};
    std::vector<u32> feature_ids;
    std::vector<u32> support_words;
    std::vector<u32> support_counts;
    std::vector<u64> row_mapping;
    cellpack::prepared_csr_support prepared{};

    explicit optimizer_support_fixture(
        const std::vector<std::vector<u32>> &feature_rows) {
        constexpr u32 row_count = 8u;
        constexpr u32 feature_count = 6u;
        std::vector<std::vector<u32>> rows(row_count);
        support_words.assign(feature_count, 0u);
        support_counts.assign(feature_count, 0u);
        row_mapping.resize(row_count);
        std::iota(row_mapping.begin(), row_mapping.end(), 0u);
        for (u32 feature = 0u; feature < feature_count; ++feature) {
            for (u32 row : feature_rows[feature]) {
                require(row < row_count, "optimizer fixture row out of range");
                rows[row].push_back(feature);
                support_words[feature] |= u32{1u} << row;
                ++support_counts[feature];
            }
        }
        for (auto &row : rows) {
            std::sort(row.begin(), row.end());
            feature_ids.insert(feature_ids.end(), row.begin(), row.end());
            row_offsets.push_back(static_cast<u32>(feature_ids.size()));
        }
        cellpack::csr_support_view source;
        source.row_count = row_count;
        source.feature_count = feature_count;
        source.nnz_count = static_cast<u32>(feature_ids.size());
        source.row_offsets = row_offsets.data();
        source.feature_ids = feature_ids.data();
        require_status(cellpack::prepare_csr_support(source, &prepared),
            "prepare optimizer runtime fixture");
    }

    cellpack::sampled_feature_support_view support() const {
        return {8u, 6u, 1u, support_words.data(), support_counts.data(),
            row_mapping.data()};
    }
};

cellpack::packing_optimizer_result optimize_bootstrap_plan(
    const std::vector<std::vector<u32>> &feature_rows,
    const std::vector<std::pair<u32, u32>> &pairs,
    u64 evaluation_identity) {
    optimizer_support_fixture fixture(feature_rows);
    std::vector<cellpack::candidate_relation> candidates;
    for (const auto &pair : pairs) {
        cellpack::candidate_relation relation;
        relation.feature_a = pair.first;
        relation.feature_b = pair.second;
        relation.score_numerator = 1;
        relation.score_denominator = 1u;
        relation.score_kind = cellpack::candidate_score_kind::support_intersection;
        relation.evidence_flags = cellpack::candidate_evidence_exact;
        candidates.push_back(relation);
    }
    cellpack::packing_optimizer_config config;
    config.maximum_feature_block_width = 2u;
    config.row_group_width = 4u;
    config.maximum_coarsening_passes = 8u;
    config.maximum_refinement_passes = 0u;
    config.maximum_oracle_evaluations = 32u;
    config.enable_feature_moves = false;
    config.enable_feature_swaps = false;
    config.plan_identity.feature_axis_fingerprint = feature_axis_identity;
    config.plan_identity.feature_axis_fingerprint_version = 1u;
    config.plan_identity.row_domain_kind =
        cellpack::packing_row_domain_kind::sampled_rows_identity;
    config.plan_identity.row_domain_identity = row_domain_identity;
    config.plan_identity.evaluation_source_identity = evaluation_identity;
    config.plan_identity.sampling_provenance_identity = evaluation_identity ^ 0x55u;
    config.objective_kind =
        cellpack::packing_exact_objective_kind::row_active_block_references;
    config.cost_policy_identity = 0x435031315254434full;
    cellpack::packing_optimizer_workspace_requirements requirements;
    require_status(cellpack::query_packing_optimizer_workspace_requirements(
        fixture.prepared, config.row_group_width, &requirements),
        "query optimizer runtime workspace");
    std::vector<cellpack::packing_evaluation_entry> entries(
        requirements.evaluator.workspace_entry_capacity);
    std::vector<cellpack::occupied_tile_occupancy> tiles(
        requirements.evaluator.occupied_tile_capacity);
    std::vector<u32> row_active(requirements.evaluator.execution_row_capacity);
    std::vector<cellpack::row_group_occupancy> row_groups(
        requirements.evaluator.row_group_capacity);
    cellpack::packing_optimizer_workspace_view workspace{
        {entries.data(), static_cast<u32>(entries.size())},
        {tiles.data(), static_cast<u32>(tiles.size()), row_active.data(),
            static_cast<u32>(row_active.size()), row_groups.data(),
            static_cast<u32>(row_groups.size())}};
    cellpack::packing_optimizer_result result;
    require_status(cellpack::optimize_packing_plan(
        fixture.prepared, fixture.support(),
        {candidates.data(), static_cast<u64>(candidates.size())}, config,
        workspace, &result), "optimize bootstrap runtime plan");
    return result;
}

cellpack::relearned_plan_runtime_observation runtime_observation(
    const controller_fixture &controller,
    const bootstrap_fixture &bootstrap,
    u64 elapsed_ns,
    u64 ordering,
    u64 tile) {
    cellpack::relearned_plan_runtime_observation result;
    result.controller_identity = controller.result.controller_identity;
    result.plan_identity = controller.result.best_plan_identity;
    result.bootstrap_identity = bootstrap.provenance.bootstrap_identity;
    result.split_identity = split_identity;
    result.dataset_identity = dataset_identity;
    result.feature_axis_identity = feature_axis_identity;
    result.row_domain_identity = row_domain_identity;
    result.ordering_identity = ordering;
    result.tile_identity = tile;
    result.operation_identity = 0x4350313152544f50ull;
    result.feature_weight_identity = 0x4350313152545747ull;
    result.hardware_identity = 0x56313030534d3730ull;
    result.toolchain_identity = 0x4355444131323900ull;
    result.input_nnz = 24u;
    result.input_bytes = controller.result.best_held_out.encoded_bytes;
    result.elapsed_nanoseconds = elapsed_ns;
    result.correctness_items = 8u;
    result.warmup_count = 3u;
    result.repeat_count = 11u;
    result.launches_per_repeat = 1u;
    return result;
}

void test_actual_controller_outputs_and_label_invariant_mapping() {
    const u64 rows[] = {10u, 11u, 12u, 13u, 14u, 15u, 16u, 17u};
    const u64 groups[] = {100u, 100u, 200u, 200u, 300u, 300u, 400u, 400u};
    const cellpack::validation_identity_view identities{8u, rows, groups};
    cellpack::frozen_packing_plan baseline = make_plan(
        {{0u}, {1u}, {2u}, {3u}, {4u}, {5u}}, 0x5000u);
    cellpack::frozen_packing_plan reference = make_plan(
        {{0u, 1u}, {2u, 3u}, {4u, 5u}}, 0x5001u);
    // Same canonical memberships, deliberately renumbered block labels.
    cellpack::frozen_packing_plan relabelled = make_plan(
        {{4u, 5u}, {0u, 1u}, {2u, 3u}}, 0x5002u);
    cellpack::frozen_packing_plan changed = make_plan(
        {{0u, 2u}, {1u, 3u}, {4u, 5u}}, 0x5003u);
    std::vector<controller_fixture> controllers;
    controllers.push_back(run_controller(baseline, reference, 0x6001u, 0x7001u, 100u, 600u));
    controllers.push_back(run_controller(baseline, relabelled, 0x6002u, 0x7002u, 104u, 620u));
    controllers.push_back(run_controller(baseline, changed, 0x6003u, 0x7003u, 108u, 660u));
    std::vector<bootstrap_fixture> bootstraps;
    bootstraps.push_back(make_bootstrap(identities, 0x8001u));
    bootstraps.push_back(make_bootstrap(identities, 0x8002u));
    bootstraps.push_back(make_bootstrap(identities, 0x8003u));
    std::vector<cellpack::relearned_plan_runtime_input> inputs(3u);
    for (u32 index = 0u; index < inputs.size(); ++index) {
        inputs[index].bootstrap_provenance = &bootstraps[index].provenance;
        inputs[index].row_multiplicities = bootstraps[index].multiplicities.data();
        inputs[index].refinement = &controllers[index].result;
        inputs[index].runtime = runtime_observation(controllers[index], bootstraps[index],
            550u + index * 55u, 0x9000u + index, 0xa000u + index);
    }
    std::vector<cellpack::relearned_plan_runtime_replicate> packets(inputs.size());
    cellpack::relearned_plan_runtime_stability_summary summary;
    require_status(cellpack::evaluate_relearned_plan_runtime_stability(
        identities, inputs.data(), inputs.size(), {packets.size(), packets.data()},
        &summary), "evaluate real controller runtime stability");
    require(summary.repeat_count == 3u && summary.exact_mapping_count == 2u,
        "label-invariant mapping count mismatch");
    require(packets[1].exact_label_invariant_mapping
        && packets[1].co_membership_disagreements == 0u
        && !packets[2].exact_label_invariant_mapping
        && packets[2].co_membership_disagreements != 0u,
        "canonical co-membership comparison used arbitrary block labels");
    require(summary.claims_group_generalization
        && summary.unit_kind == cellpack::validation_unit_kind::caller_group_identity,
        "group-aware bootstrap scope was lost");
    require(summary.runtime_mean_nanoseconds.observation_count == 3u
        && summary.runtime_nnz_per_second.observation_count == 3u
        && summary.co_membership_agreement_fraction.mean < 1.0,
        "runtime/mapping distribution summary mismatch");

    auto tampered = inputs;
    tampered[1].runtime.toolchain_identity ^= 1u;
    require(!cellpack::evaluate_relearned_plan_runtime_stability(
        identities, tampered.data(), tampered.size(),
        {packets.size(), packets.data()}, &summary),
        "mixed toolchain identity was accepted");
    tampered = inputs;
    tampered[1].bootstrap_provenance = tampered[0].bootstrap_provenance;
    tampered[1].row_multiplicities = tampered[0].row_multiplicities;
    tampered[1].runtime.bootstrap_identity = tampered[0].runtime.bootstrap_identity;
    require(!cellpack::evaluate_relearned_plan_runtime_stability(
        identities, tampered.data(), tampered.size(),
        {packets.size(), packets.data()}, &summary),
        "duplicate bootstrap identity was accepted");
}

void test_cell_level_scope_and_zero_runtime_observation() {
    const u64 rows[] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
    const cellpack::validation_identity_view identities{8u, rows, nullptr};
    cellpack::frozen_packing_plan baseline = make_plan(
        {{0u}, {1u}, {2u}, {3u}, {4u}, {5u}}, 0xb001u);
    cellpack::frozen_packing_plan candidate = make_plan(
        {{0u, 1u}, {2u, 3u}, {4u, 5u}}, 0xb002u);
    controller_fixture controller = run_controller(
        baseline, candidate, 0xc001u, 0xd001u, 100u, 600u);
    bootstrap_fixture bootstrap = make_bootstrap(identities, 0xe001u);
    cellpack::relearned_plan_runtime_input input;
    input.bootstrap_provenance = &bootstrap.provenance;
    input.row_multiplicities = bootstrap.multiplicities.data();
    input.refinement = &controller.result;
    input.runtime = runtime_observation(controller, bootstrap, 550u, 0xf001u, 0xf002u);
    input.runtime.observed = false;
    input.runtime.input_nnz = 0u;
    input.runtime.input_bytes = 0u;
    input.runtime.elapsed_nanoseconds = 0u;
    input.runtime.correctness_items = 0u;
    input.runtime.warmup_count = 0u;
    input.runtime.repeat_count = 0u;
    input.runtime.launches_per_repeat = 0u;
    cellpack::relearned_plan_runtime_replicate packet;
    cellpack::relearned_plan_runtime_stability_summary summary;
    require_status(cellpack::evaluate_relearned_plan_runtime_stability(
        identities, &input, 1u, {1u, &packet}, &summary),
        "evaluate zero-observation cell-level stability");
    require(!summary.claims_group_generalization
        && summary.unit_kind == cellpack::validation_unit_kind::row_identity
        && summary.runtime_mean_nanoseconds.observation_count == 0u
        && summary.encoded_bytes.observation_count == 1u,
        "zero runtime observation or cell-level scope was fabricated");

    input.runtime.observed = true;
    require(!cellpack::evaluate_relearned_plan_runtime_stability(
        identities, &input, 1u, {1u, &packet}, &summary),
        "observed runtime accepted zero denominators");
}

void test_optimizer_controller_bootstrap_chain() {
    const std::vector<std::vector<u32>> support_a{
        {0u, 1u, 2u}, {0u, 1u, 2u}, {3u, 4u, 5u}, {3u, 4u, 5u},
        {6u, 7u}, {6u, 7u}};
    const std::vector<std::vector<u32>> support_b{
        {0u, 1u, 2u}, {3u, 4u, 5u}, {0u, 1u, 2u}, {3u, 4u, 5u},
        {6u, 7u}, {6u, 7u}};
    auto baseline = optimize_bootstrap_plan(support_a, {}, 0x1100u);
    auto learned_a = optimize_bootstrap_plan(
        support_a, {{0u, 1u}, {2u, 3u}, {4u, 5u}}, 0x1101u);
    auto learned_b = optimize_bootstrap_plan(
        support_b, {{0u, 2u}, {1u, 3u}, {4u, 5u}}, 0x1102u);
    require(learned_a.plan.feature_block_count() < baseline.plan.feature_block_count()
        && learned_b.plan.feature_block_count() < baseline.plan.feature_block_count(),
        "bootstrap plans were not learned by the optimizer");
    auto controller_a = run_controller(
        baseline.plan, learned_a.plan, 0x1201u, 0x1301u, 100u, 600u);
    auto controller_b = run_controller(
        baseline.plan, learned_b.plan, 0x1202u, 0x1302u, 104u, 620u);
    const u64 rows[] = {21u, 22u, 23u, 24u, 25u, 26u, 27u, 28u};
    const u64 groups[] = {1u, 1u, 2u, 2u, 3u, 3u, 4u, 4u};
    const cellpack::validation_identity_view identities{8u, rows, groups};
    auto bootstrap_a = make_bootstrap(identities, 0x1401u);
    auto bootstrap_b = make_bootstrap(identities, 0x1402u);
    cellpack::relearned_plan_runtime_input inputs[2];
    inputs[0] = {&bootstrap_a.provenance, bootstrap_a.multiplicities.data(),
        &controller_a.result, runtime_observation(
            controller_a, bootstrap_a, 600u, 0x1501u, 0x1601u)};
    inputs[1] = {&bootstrap_b.provenance, bootstrap_b.multiplicities.data(),
        &controller_b.result, runtime_observation(
            controller_b, bootstrap_b, 620u, 0x1502u, 0x1602u)};
    cellpack::relearned_plan_runtime_replicate packets[2];
    cellpack::relearned_plan_runtime_stability_summary summary;
    require_status(cellpack::evaluate_relearned_plan_runtime_stability(
        identities, inputs, 2u, {2u, packets}, &summary),
        "evaluate optimizer-controller-bootstrap chain");
    require(summary.repeat_count == 2u && summary.exact_mapping_count == 1u,
        "real relearning chain did not expose mapping variability");
}

} // namespace

int main() {
    test_actual_controller_outputs_and_label_invariant_mapping();
    test_cell_level_scope_and_zero_runtime_observation();
    test_optimizer_controller_bootstrap_chain();
    return 0;
}
