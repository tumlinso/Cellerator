#include <Cellerator/geometry/optimizer.hh>

#include <Cellerator/geometry/gene_support_bitset.hh>

#include "optimizer_state.hh"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

struct support_fixture {
    cellpack::u32 rows = 0u;
    cellpack::u32 features = 0u;
    cellpack::u32 words_per_feature = 0u;
    std::vector<cellpack::u32> words;
    std::vector<cellpack::u32> counts;
    std::vector<cellpack::u64> row_mapping;

    cellpack::sampled_feature_support_view view() const {
        return {rows, features, words_per_feature,
            words.empty() ? nullptr : words.data(),
            counts.empty() ? nullptr : counts.data(),
            row_mapping.empty() ? nullptr : row_mapping.data()};
    }
};

support_fixture make_support(cellpack::u32 rows, const std::vector<std::vector<cellpack::u32>> &feature_rows) {
    support_fixture result;
    result.rows = rows;
    result.features = static_cast<cellpack::u32>(feature_rows.size());
    result.words_per_feature = rows == 0u ? 0u : 1u + ((rows - 1u) / 32u);
    result.words.assign(static_cast<std::size_t>(result.features) * result.words_per_feature, 0u);
    result.counts.assign(result.features, 0u);
    result.row_mapping.resize(rows);
    std::iota(result.row_mapping.begin(), result.row_mapping.end(), 0u);
    for (cellpack::u32 feature = 0u; feature < result.features; ++feature) {
        std::vector<cellpack::u32> unique = feature_rows[feature];
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        for (cellpack::u32 row : unique) {
            require(row < rows, "support fixture row out of range");
            result.words[static_cast<std::size_t>(feature) * result.words_per_feature + row / 32u]
                |= cellpack::u32{1u} << (row % 32u);
        }
        result.counts[feature] = static_cast<cellpack::u32>(unique.size());
    }
    return result;
}

struct csr_fixture {
    cellpack::u32 rows = 0u;
    cellpack::u32 features = 0u;
    std::vector<cellpack::u32> offsets;
    std::vector<cellpack::u32> feature_ids;
    cellpack::prepared_csr_support prepared{};

    explicit csr_fixture(cellpack::u32 feature_count, const std::vector<std::vector<cellpack::u32>> &row_features)
        : rows(static_cast<cellpack::u32>(row_features.size())), features(feature_count), offsets(1u, 0u) {
        for (std::vector<cellpack::u32> row : row_features) {
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            feature_ids.insert(feature_ids.end(), row.begin(), row.end());
            offsets.push_back(static_cast<cellpack::u32>(feature_ids.size()));
        }
        const cellpack::csr_support_view view{rows, features,
            static_cast<cellpack::u32>(feature_ids.size()), offsets.data(),
            feature_ids.empty() ? nullptr : feature_ids.data()};
        const cellpack::validation_result status = cellpack::prepare_csr_support(view, &prepared);
        require(static_cast<bool>(status), status.message);
    }
};

struct workspace_fixture {
    std::vector<cellpack::packing_evaluation_entry> entries;
    std::vector<cellpack::occupied_tile_occupancy> tiles;
    std::vector<cellpack::u32> row_active;
    std::vector<cellpack::row_group_occupancy> row_groups;

    workspace_fixture(const cellpack::prepared_csr_support &source, cellpack::u32 row_group_width) {
        cellpack::packing_optimizer_workspace_requirements requirements;
        const cellpack::validation_result status = cellpack::query_packing_optimizer_workspace_requirements(
            source, row_group_width, &requirements);
        require(static_cast<bool>(status), status.message);
        entries.resize(requirements.evaluator.workspace_entry_capacity);
        tiles.resize(requirements.evaluator.occupied_tile_capacity);
        row_active.resize(requirements.evaluator.execution_row_capacity);
        row_groups.resize(requirements.evaluator.row_group_capacity);
    }

    cellpack::packing_optimizer_workspace_view view() {
        return {
            {entries.empty() ? nullptr : entries.data(), static_cast<cellpack::u32>(entries.size())},
            {tiles.empty() ? nullptr : tiles.data(), static_cast<cellpack::u32>(tiles.size()),
             row_active.empty() ? nullptr : row_active.data(), static_cast<cellpack::u32>(row_active.size()),
             row_groups.empty() ? nullptr : row_groups.data(), static_cast<cellpack::u32>(row_groups.size())}
        };
    }
};

cellpack::candidate_relation relation(
    cellpack::u32 a, cellpack::u32 b, std::int64_t score,
    cellpack::candidate_score_kind kind = cellpack::candidate_score_kind::support_intersection,
    cellpack::u32 flags = cellpack::candidate_evidence_exact) {
    cellpack::candidate_relation result;
    result.feature_a = a;
    result.feature_b = b;
    result.score_numerator = score;
    result.score_denominator = 1u;
    result.score_kind = kind;
    result.evidence_flags = flags;
    return result;
}

cellpack::packing_optimizer_config config(cellpack::u32 width, cellpack::u32 row_width) {
    cellpack::packing_optimizer_config result;
    result.maximum_feature_block_width = width;
    result.row_group_width = row_width;
    result.plan_identity.feature_axis_fingerprint = 0x12345678u;
    result.plan_identity.feature_axis_fingerprint_version = 1u;
    result.plan_identity.row_domain_kind = cellpack::packing_row_domain_kind::sampled_rows_identity;
    result.plan_identity.row_domain_identity = 0xabcdu;
    result.plan_identity.evaluation_source_identity = 0x777u;
    result.plan_identity.sampling_provenance_identity = 0x888u;
    result.objective_kind = cellpack::packing_exact_objective_kind::row_active_block_references;
    result.cost_policy_identity = 0x435042503034u;
    result.maximum_coarsening_passes = 16u;
    result.maximum_refinement_passes = 8u;
    result.maximum_oracle_evaluations = 128u;
    return result;
}

cellpack::packing_optimizer_result optimize(
    csr_fixture &source,
    const support_fixture &support,
    const std::vector<cellpack::candidate_relation> &candidates,
    const cellpack::packing_optimizer_config &settings) {
    workspace_fixture workspace(source.prepared, settings.row_group_width);
    cellpack::packing_optimizer_result result;
    const cellpack::validation_result status = cellpack::optimize_packing_plan(
        source.prepared, support.view(),
        {candidates.empty() ? nullptr : candidates.data(), static_cast<cellpack::u64>(candidates.size())},
        settings, workspace.view(), &result);
    require(static_cast<bool>(status), status.message);
    return result;
}

void require_round_trips(const cellpack::frozen_packing_plan &plan) {
    const cellpack::packing_plan_view view = plan.view();
    for (cellpack::u32 canonical = 0u; canonical < plan.feature_count(); ++canonical) {
        const cellpack::u32 execution = view.inverse_feature_permutation[canonical];
        require(view.feature_permutation[execution] == canonical, "frozen feature round trip failed");
        const cellpack::u32 block = plan.feature_to_block()[canonical];
        const cellpack::u32 local = plan.feature_to_local()[canonical];
        require(view.feature_block_offsets[block] + local == execution, "frozen block/local lookup failed");
    }
    require(static_cast<bool>(plan.validate()), "frozen plan validation failed");
}

void test_candidate_normalization() {
    std::vector<cellpack::candidate_relation> input{
        relation(2u, 1u, 5), relation(1u, 2u, 5), relation(1u, 2u, 3, cellpack::candidate_score_kind::jaccard,
            cellpack::candidate_evidence_approximate), relation(1u, 1u, 9),
        relation(0u, 3u, 7, cellpack::candidate_score_kind::jaccard, cellpack::candidate_evidence_approximate)
    };
    cellpack::normalized_candidate_relations normalized;
    cellpack::validation_result status = cellpack::normalize_candidate_relations(
        {input.data(), input.size()}, 4u, &normalized);
    require(static_cast<bool>(status), status.message);
    require(normalized.view().relation_count == 3u, "candidate normalization output count mismatch");
    require(normalized.statistics().self_edges_discarded == 1u, "self edge accounting mismatch");
    require(normalized.statistics().duplicates_collapsed == 1u, "duplicate accounting mismatch");
    require(normalized.view().relations[1].feature_a == 1u && normalized.view().relations[1].feature_b == 2u
        && normalized.view().relations[2].feature_a == 1u && normalized.view().relations[2].feature_b == 2u
        && normalized.view().relations[1].score_kind != normalized.view().relations[2].score_kind,
        "candidate endpoints were not canonicalized");

    std::reverse(input.begin(), input.end());
    cellpack::normalized_candidate_relations reversed;
    status = cellpack::normalize_candidate_relations({input.data(), input.size()}, 4u, &reversed);
    require(static_cast<bool>(status), status.message);
    require(reversed.view().relation_count == normalized.view().relation_count, "input order changed normalization count");
    for (cellpack::u64 i = 0u; i < normalized.view().relation_count; ++i) {
        require(reversed.view().relations[i].feature_a == normalized.view().relations[i].feature_a
            && reversed.view().relations[i].feature_b == normalized.view().relations[i].feature_b
            && reversed.view().relations[i].score_kind == normalized.view().relations[i].score_kind,
            "input order changed normalized relation sequence");
    }

    std::vector<cellpack::candidate_relation> conflict{relation(0u, 1u, 2), relation(1u, 0u, 3)};
    status = cellpack::normalize_candidate_relations({conflict.data(), conflict.size()}, 2u, &normalized);
    require(!static_cast<bool>(status), "conflicting exact duplicate was accepted");
    cellpack::candidate_relation invalid = relation(0u, 4u, 1);
    status = cellpack::normalize_candidate_relations({&invalid, 1u}, 4u, &normalized);
    require(!static_cast<bool>(status), "invalid endpoint was accepted");
    invalid = relation(0u, 1u, 1);
    invalid.score_denominator = 0u;
    status = cellpack::normalize_candidate_relations({&invalid, 1u}, 4u, &normalized);
    require(!static_cast<bool>(status), "zero score denominator was accepted");
}

void test_zero_copy_gene_support_adapter() {
    support_fixture fixture = make_support(3u, {{0u, 2u}, {1u}});
    cellerator::compute::gene_support::gene_support_bitset_view source;
    source.layout.sampled_cell_count = fixture.rows;
    source.layout.gene_count = fixture.features;
    source.layout.words_per_gene = fixture.words_per_feature;
    source.layout.support_word_count = fixture.words.size();
    source.layout.support_bytes = fixture.words.size() * sizeof(cellpack::u32);
    source.gene_support = fixture.words.data();
    source.detected_cell_counts = fixture.counts.data();
    source.sampled_position_to_global_row = fixture.row_mapping.data();
    cellpack::sampled_feature_support_view adapted;
    const cellpack::validation_result status = cellpack::make_sampled_feature_support_view(source, &adapted);
    require(static_cast<bool>(status), status.message);
    require(adapted.support_words == fixture.words.data()
        && adapted.detected_row_counts == fixture.counts.data()
        && adapted.sampled_position_to_global_row == fixture.row_mapping.data(),
        "gene support adapter copied or changed CP-BP-01 storage pointers");
}

void test_mutable_plan_and_proxy_formulas() {
    support_fixture support = make_support(3u, {{0u, 1u}, {1u, 2u}, {1u, 2u}, {0u}});
    cellpack::detail::optimizer_state state;
    cellpack::validation_result status = state.initialize(3u, support.view(), 3u, 2u);
    require(static_cast<bool>(status), status.message);
    require(state.merge_proxy_gain(0u, 1u) == 1, "merge proxy intersection mismatch");
    status = state.materialize_execution_geometry();
    require(static_cast<bool>(status), status.message);
    cellpack::packing_plan_view clean_view;
    require(static_cast<bool>(state.view(&clean_view)), "clean mutable view failed");
    status = state.merge_blocks(0u, 1u);
    require(static_cast<bool>(status), status.message);
    require(!static_cast<bool>(state.view(&clean_view)), "dirty mutable view was exposed");
    require(state.move_proxy_gain(2u, state.block_slot_for_feature(0u)) == 2,
        "move proxy did not account for source deletion");
    status = state.move_feature(2u, state.block_slot_for_feature(0u));
    require(static_cast<bool>(status), status.message);
    require(state.active_block_count() == 2u, "move did not delete emptied source block");
    status = state.materialize_execution_geometry();
    require(static_cast<bool>(status), status.message);
    require(static_cast<bool>(state.view(&clean_view)), "rematerialized view failed");
    for (cellpack::u32 canonical = 0u; canonical < clean_view.feature_count; ++canonical) {
        require(clean_view.feature_permutation[clean_view.inverse_feature_permutation[canonical]] == canonical,
            "mutable canonical round trip failed");
    }

    support_fixture swap_support = make_support(3u, {{0u}, {1u}, {0u}, {2u}});
    cellpack::detail::optimizer_state swap_state;
    require(static_cast<bool>(swap_state.initialize(3u, swap_support.view(), 2u, 2u)), "swap state initialize failed");
    require(static_cast<bool>(swap_state.merge_blocks(0u, 1u)), "first swap block merge failed");
    require(static_cast<bool>(swap_state.merge_blocks(2u, 3u)), "second swap block merge failed");
    require(swap_state.swap_proxy_gain(1u, 2u) == 1, "swap proxy mismatch");
    require(static_cast<bool>(swap_state.swap_features(1u, 2u)), "swap mutation failed");
    require(static_cast<bool>(swap_state.validate()), "swap state invariant failed");
    require(!static_cast<bool>(swap_state.move_feature(0u, swap_state.block_slot_for_feature(2u))),
        "over-capacity move was accepted");
}

void test_tail_mask_and_pathological_supports() {
    support_fixture support = make_support(33u, {{32u}, {32u}, {}, {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u,
        8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u, 19u, 20u, 21u, 22u, 23u,
        24u, 25u, 26u, 27u, 28u, 29u, 30u, 31u, 32u}});
    support.words[1u] = 0xffffffffu;
    support.words[3u] = 0xffffffffu;
    cellpack::detail::optimizer_state state;
    require(static_cast<bool>(state.initialize(33u, support.view(), 32u, 8u)), "tail state initialize failed");
    require(state.merge_proxy_gain(0u, 1u) == 1, "tail support bits escaped sampled row domain");
    require(state.merge_proxy_gain(0u, 2u) == 0, "empty support gained overlap");
    require(state.merge_proxy_gain(0u, 3u) == 1, "subset/ubiquitous proxy mismatch");

    for (cellpack::u32 maximum_width : {8u, 16u, 32u}) {
        std::vector<std::vector<cellpack::u32>> identical(maximum_width + 1u, {0u});
        support_fixture width_support = make_support(1u, identical);
        cellpack::detail::optimizer_state width_state;
        require(static_cast<bool>(width_state.initialize(1u, width_support.view(), maximum_width, 1u)),
            "configured-width state initialize failed");
        for (cellpack::u32 feature = 1u; feature < maximum_width; ++feature) {
            require(static_cast<bool>(width_state.merge_blocks(
                width_state.block_slot_for_feature(0u), width_state.block_slot_for_feature(feature))),
                "legal configured-width merge failed");
        }
        require(!static_cast<bool>(width_state.merge_blocks(
            width_state.block_slot_for_feature(0u), width_state.block_slot_for_feature(maximum_width))),
            "configured maximum width was exceeded");
    }
}

void test_no_edge_identity_and_uneven_rows() {
    csr_fixture source(4u, {{0u, 2u}, {}, {1u}, {3u}, {0u}});
    support_fixture support = make_support(5u, {{0u, 4u}, {2u}, {0u}, {3u}});
    const cellpack::packing_optimizer_result result = optimize(source, support, {}, config(8u, 2u));
    require(result.plan.feature_block_count() == 4u, "no-edge optimizer changed singleton plan");
    require(result.plan.row_group_count() == 3u, "uneven final row group was not materialized");
    require(result.plan.row_group_offsets()[3u] == 5u, "row boundary does not cover final row");
    require(result.diagnostics.final.objective == result.diagnostics.baseline.objective,
        "no-edge exact objective changed");
    require_round_trips(result.plan);
}

void test_obvious_merge_and_width_constraint() {
    csr_fixture source(3u, {{0u, 1u}, {0u, 1u}, {2u}});
    support_fixture support = make_support(3u, {{0u, 1u}, {0u, 1u}, {2u}});
    std::vector<cellpack::candidate_relation> edges{
        relation(0u, 1u, 2),
        relation(0u, 2u, 100, cellpack::candidate_score_kind::minhash_similarity,
            cellpack::candidate_evidence_approximate)};
    cellpack::packing_optimizer_result result = optimize(source, support, edges, config(2u, 2u));
    require(result.plan.feature_block_count() == 2u, "obvious merge was not accepted");
    require(result.plan.feature_to_block()[0u] != result.plan.feature_to_block()[2u],
        "high approximate evidence overrode zero exact support gain");
    require(result.diagnostics.merge_oracle_accepted == 1u, "merge acceptance diagnostic mismatch");
    require(result.diagnostics.final.objective < result.diagnostics.baseline.objective,
        "obvious merge did not improve exact objective");
    require_round_trips(result.plan);
    cellpack::packing_evaluation_requirements requirements;
    cellpack::validation_result status = cellpack::query_packing_evaluation_requirements(
        source.prepared, result.plan.view(), &requirements);
    require(static_cast<bool>(status), status.message);
    std::vector<cellpack::packing_evaluation_entry> entries(requirements.workspace_entry_capacity);
    std::vector<cellpack::occupied_tile_occupancy> tiles(requirements.occupied_tile_capacity);
    std::vector<cellpack::u32> rows(requirements.execution_row_capacity);
    std::vector<cellpack::row_group_occupancy> groups(requirements.row_group_capacity);
    cellpack::packing_occupancy_result frozen_occupancy;
    status = cellpack::evaluate_packing_plan(source.prepared, result.plan.view(),
        {entries.data(), static_cast<cellpack::u32>(entries.size())},
        {tiles.data(), static_cast<cellpack::u32>(tiles.size()), rows.data(), static_cast<cellpack::u32>(rows.size()),
         groups.data(), static_cast<cellpack::u32>(groups.size())}, &frozen_occupancy);
    require(static_cast<bool>(status), status.message);
    require(frozen_occupancy.totals.total_nnz == result.plan.final_evaluation().occupancy.total_nnz
        && frozen_occupancy.totals.row_active_block_references
            == result.plan.final_evaluation().occupancy.row_active_block_references,
        "mutable/frozen plan evaluator results differ");

    result = optimize(source, support, edges, config(1u, 2u));
    require(result.plan.feature_block_count() == 3u, "maximum width one allowed a merge");
}

void test_rollback_batch_shrink_and_blacklist() {
    csr_fixture source(4u, {{0u, 1u}, {2u, 3u}, {0u}, {}, {0u}, {}});
    support_fixture support = make_support(6u, {{0u, 2u, 4u}, {0u}, {1u}, {1u}});
    std::vector<cellpack::candidate_relation> edges{relation(0u, 1u, 1), relation(2u, 3u, 1)};
    cellpack::packing_optimizer_config settings = config(2u, 2u);
    settings.objective_kind = cellpack::packing_exact_objective_kind::total_bytes;
    settings.cost_model.value_bytes = 1u;
    settings.cost_model.dense_values_within_occupied_tiles = true;
    settings.cost_model.row_active_block_metadata_bytes = 1u;
    settings.initial_oracle_batch_size = 2u;
    const cellpack::packing_optimizer_result result = optimize(source, support, edges, settings);
    require(result.diagnostics.oracle_rollbacks >= 2u, "oracle rollback path was not exercised");
    require(result.diagnostics.oracle_batch_reductions >= 1u, "oracle batch shrink path was not exercised");
    require(result.diagnostics.blacklisted_mutations >= 1u, "single-mutation blacklist path was not exercised");
    require(result.diagnostics.merge_oracle_accepted == 1u, "good merge was not recovered after batch rollback");
    require(result.plan.feature_to_block()[2u] == result.plan.feature_to_block()[3u], "expected good merge missing");
    require(result.plan.feature_to_block()[0u] != result.plan.feature_to_block()[1u], "exact-worse merge escaped rollback");
    require(result.diagnostics.final.objective <= result.diagnostics.baseline.objective,
        "rollback fixture regressed final exact objective");
}

void test_move_and_swap_refinement() {
    csr_fixture move_source(3u, {{0u, 1u, 2u}, {0u, 1u, 2u}, {1u, 2u}});
    support_fixture move_support = make_support(3u, {{0u, 1u}, {0u, 1u, 2u}, {0u, 1u, 2u}});
    std::vector<cellpack::candidate_relation> move_edges{relation(1u, 2u, 1)};
    cellpack::packing_optimizer_config move_config = config(3u, 2u);
    move_config.candidate_fanout = 1u;
    move_config.maximum_coarsening_passes = 0u;
    cellpack::packing_optimizer_result move_result = optimize(move_source, move_support, move_edges, move_config);
    require(move_result.diagnostics.move_oracle_accepted >= 1u, "bounded move refinement was not accepted");
    require(move_result.plan.feature_block_count() == 2u
        && move_result.plan.feature_to_block()[1u] == move_result.plan.feature_to_block()[2u],
        "move did not delete singleton source block");

    csr_fixture swap_source(4u, {{0u, 2u}, {0u, 1u, 2u, 3u}, {1u, 3u}});
    support_fixture swap_support = make_support(3u, {{0u, 1u}, {1u, 2u}, {0u, 1u}, {1u, 2u}});
    std::vector<cellpack::candidate_relation> swap_edges{
        relation(0u, 1u, 10), relation(2u, 3u, 10), relation(1u, 2u, 1)};
    cellpack::packing_optimizer_config swap_config = config(2u, 2u);
    cellpack::packing_optimizer_result swap_result = optimize(swap_source, swap_support, swap_edges, swap_config);
    require(swap_result.diagnostics.swap_oracle_accepted >= 1u, "bounded cross-block swap was not accepted");
    require(swap_result.plan.feature_to_block()[0u] == swap_result.plan.feature_to_block()[2u],
        "swap refinement did not recover identical-support block");
    require(swap_result.plan.feature_to_block()[1u] == swap_result.plan.feature_to_block()[3u],
        "swap refinement did not recover second identical-support block");
}

void test_geometry_invariant_rejection_and_compatibility() {
    csr_fixture source(2u, {{0u, 1u}, {0u, 1u}});
    support_fixture support = make_support(2u, {{0u, 1u}, {0u, 1u}});
    workspace_fixture workspace(source.prepared, 2u);
    cellpack::packing_optimizer_config settings = config(2u, 2u);
    settings.objective_kind = cellpack::packing_exact_objective_kind::total_bytes;
    cellpack::packing_optimizer_result output;
    const cellpack::validation_result status = cellpack::optimize_packing_plan(
        source.prepared, support.view(), {nullptr, 0u}, settings, workspace.view(), &output);
    require(!static_cast<bool>(status), "geometry-invariant total-byte objective was accepted");

    const cellpack::packing_optimizer_result valid = optimize(source, support, {relation(0u, 1u, 2)}, config(2u, 2u));
    cellpack::packing_plan_compatibility expected;
    expected.row_count = 2u;
    expected.feature_count = 2u;
    expected.feature_axis_fingerprint = 0x12345678u;
    expected.feature_axis_fingerprint_version = 1u;
    expected.row_domain_kind = cellpack::packing_row_domain_kind::sampled_rows_identity;
    expected.row_domain_identity = 0xabcdu;
    require(static_cast<bool>(valid.plan.validate_compatibility(expected)), "valid plan compatibility rejected");
    expected.feature_axis_fingerprint ^= 1u;
    require(!static_cast<bool>(valid.plan.validate_compatibility(expected)), "feature mismatch was accepted");
    expected.feature_axis_fingerprint ^= 1u;
    expected.row_domain_kind = cellpack::packing_row_domain_kind::full_dataset_identity;
    require(!static_cast<bool>(valid.plan.validate_compatibility(expected)), "sample/full row-domain mismatch was accepted");
}

void test_determinism_and_randomized_invariants() {
    csr_fixture source(6u, {{0u, 1u}, {0u, 2u}, {1u, 3u}, {2u, 4u}, {3u, 5u}, {4u, 5u}});
    support_fixture support = make_support(6u, {{0u, 1u}, {0u, 2u}, {1u, 3u}, {2u, 4u}, {3u, 5u}, {4u, 5u}});
    std::vector<cellpack::candidate_relation> edges{
        relation(0u, 1u, 1), relation(1u, 3u, 1), relation(2u, 4u, 1), relation(4u, 5u, 1)};
    const cellpack::packing_optimizer_result first = optimize(source, support, edges, config(2u, 3u));
    std::reverse(edges.begin(), edges.end());
    const cellpack::packing_optimizer_result second = optimize(source, support, edges, config(2u, 3u));
    require(first.plan.feature_block_count() == second.plan.feature_block_count(), "candidate permutation changed block count");
    require(std::equal(first.plan.feature_permutation(), first.plan.feature_permutation() + first.plan.feature_count(),
        second.plan.feature_permutation()), "candidate permutation changed frozen feature order");
    require(std::equal(first.plan.feature_block_offsets(), first.plan.feature_block_offsets() + first.plan.feature_block_count() + 1u,
        second.plan.feature_block_offsets()), "candidate permutation changed frozen boundaries");

    std::mt19937 rng(20260814u);
    for (cellpack::u32 trial = 0u; trial < 64u; ++trial) {
        const cellpack::u32 features = 2u + rng() % 15u;
        std::vector<std::vector<cellpack::u32>> feature_rows(features);
        for (cellpack::u32 feature = 0u; feature < features; ++feature) {
            for (cellpack::u32 row = 0u; row < 17u; ++row) if ((rng() % 5u) == 0u) feature_rows[feature].push_back(row);
        }
        support_fixture random_support = make_support(17u, feature_rows);
        cellpack::detail::optimizer_state state;
        require(static_cast<bool>(state.initialize(17u, random_support.view(), 4u, 5u)), "random state initialize failed");
        for (cellpack::u32 step = 0u; step < features * 3u; ++step) {
            const cellpack::u32 a = rng() % features, b = rng() % features;
            if (a == b) continue;
            const cellpack::u32 slot_a = state.block_slot_for_feature(a), slot_b = state.block_slot_for_feature(b);
            if (slot_a == slot_b) continue;
            if ((rng() & 1u) == 0u && state.block_width(slot_a) + state.block_width(slot_b) <= 4u) {
                require(static_cast<bool>(state.merge_blocks(slot_a, slot_b)), "random legal merge failed");
            } else if (state.block_width(slot_b) < 4u) {
                require(static_cast<bool>(state.move_feature(a, slot_b)), "random legal move failed");
            } else {
                require(static_cast<bool>(state.swap_features(a, b)), "random legal swap failed");
            }
            require(static_cast<bool>(state.validate()), "random mutation broke state invariant");
        }
        require(static_cast<bool>(state.materialize_execution_geometry()), "random materialization failed");
        cellpack::packing_plan_view plan;
        require(static_cast<bool>(state.view(&plan)), "random clean view failed");
        for (cellpack::u32 feature = 0u; feature < features; ++feature) {
            require(plan.feature_permutation[plan.inverse_feature_permutation[feature]] == feature,
                "random canonical round trip failed");
        }
    }
}

} // namespace

int main() {
    test_candidate_normalization();
    test_zero_copy_gene_support_adapter();
    test_mutable_plan_and_proxy_formulas();
    test_tail_mask_and_pathological_supports();
    test_no_edge_identity_and_uneven_rows();
    test_obvious_merge_and_width_constraint();
    test_rollback_batch_shrink_and_blacklist();
    test_move_and_swap_refinement();
    test_geometry_invariant_rejection_and_compatibility();
    test_determinism_and_randomized_invariants();
    return 0;
}
