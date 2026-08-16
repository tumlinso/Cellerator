#include <CellPack/evaluator.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(double actual, double expected, const char *message) {
    if (std::fabs(actual - expected) > 1.0e-12) throw std::runtime_error(message);
}

struct plan_storage {
    std::vector<cellpack::u32> row_permutation;
    std::vector<cellpack::u32> inverse_row_permutation;
    std::vector<cellpack::u32> feature_permutation;
    std::vector<cellpack::u32> inverse_feature_permutation;
    std::vector<cellpack::u32> row_group_offsets;
    std::vector<cellpack::u32> feature_block_offsets;

    cellpack::packing_plan_view view(cellpack::u32 rows, cellpack::u32 features, bool implicit_identity = false) const {
        cellpack::packing_plan_view result;
        result.row_count = rows;
        result.feature_count = features;
        if (!implicit_identity) {
            result.row_permutation = row_permutation.empty() ? nullptr : row_permutation.data();
            result.inverse_row_permutation = inverse_row_permutation.empty() ? nullptr : inverse_row_permutation.data();
            result.feature_permutation = feature_permutation.empty() ? nullptr : feature_permutation.data();
            result.inverse_feature_permutation = inverse_feature_permutation.empty() ? nullptr : inverse_feature_permutation.data();
        }
        result.row_group_count = static_cast<cellpack::u32>(row_group_offsets.size() - 1u);
        result.row_group_offsets = row_group_offsets.data();
        result.feature_block_count = static_cast<cellpack::u32>(feature_block_offsets.size() - 1u);
        result.feature_block_offsets = feature_block_offsets.data();
        return result;
    }
};

struct evaluation_storage {
    std::vector<cellpack::packing_evaluation_entry> workspace;
    std::vector<cellpack::occupied_tile_occupancy> tiles;
    std::vector<cellpack::u32> row_active_blocks;
    std::vector<cellpack::row_group_occupancy> row_groups;
    cellpack::packing_evaluation_requirements requirements{};
    cellpack::packing_occupancy_result result{};
};

evaluation_storage evaluate(
    const cellpack::csr_support_view &source,
    const cellpack::packing_plan_view &plan) {
    evaluation_storage storage;
    cellpack::validation_result status = cellpack::query_packing_evaluation_requirements(source, plan, &storage.requirements);
    require(static_cast<bool>(status), status.message);
    storage.workspace.resize(storage.requirements.workspace_entry_capacity);
    storage.tiles.resize(storage.requirements.occupied_tile_capacity);
    storage.row_active_blocks.resize(storage.requirements.execution_row_capacity);
    storage.row_groups.resize(storage.requirements.row_group_capacity);
    cellpack::packing_evaluation_workspace_view workspace{
        storage.workspace.empty() ? nullptr : storage.workspace.data(),
        static_cast<cellpack::u32>(storage.workspace.size())
    };
    cellpack::packing_occupancy_buffers buffers{
        storage.tiles.empty() ? nullptr : storage.tiles.data(),
        static_cast<cellpack::u32>(storage.tiles.size()),
        storage.row_active_blocks.empty() ? nullptr : storage.row_active_blocks.data(),
        static_cast<cellpack::u32>(storage.row_active_blocks.size()),
        storage.row_groups.empty() ? nullptr : storage.row_groups.data(),
        static_cast<cellpack::u32>(storage.row_groups.size())
    };
    status = cellpack::evaluate_packing_plan(source, plan, workspace, buffers, &storage.result);
    require(static_cast<bool>(status), status.message);
    return storage;
}

const cellpack::occupied_tile_occupancy &find_tile(
    const evaluation_storage &evaluation,
    cellpack::u32 row_group,
    cellpack::u32 feature_block) {
    for (cellpack::u32 i = 0u; i < evaluation.result.occupied_tile_count; ++i) {
        const cellpack::occupied_tile_occupancy &tile = evaluation.tiles[i];
        if (tile.row_group == row_group && tile.feature_block == feature_block) return tile;
    }
    throw std::runtime_error("expected occupied tile was not found");
}

plan_storage identity_plan(
    cellpack::u32 rows,
    cellpack::u32 features,
    std::vector<cellpack::u32> row_groups,
    std::vector<cellpack::u32> feature_blocks) {
    plan_storage plan;
    plan.row_permutation.resize(rows);
    plan.feature_permutation.resize(features);
    std::iota(plan.row_permutation.begin(), plan.row_permutation.end(), 0u);
    std::iota(plan.feature_permutation.begin(), plan.feature_permutation.end(), 0u);
    plan.inverse_row_permutation = plan.row_permutation;
    plan.inverse_feature_permutation = plan.feature_permutation;
    plan.row_group_offsets = std::move(row_groups);
    plan.feature_block_offsets = std::move(feature_blocks);
    return plan;
}

void test_identity_plan_and_cost_models() {
    const cellpack::u32 offsets[] = { 0u, 2u, 3u, 5u };
    const cellpack::u32 features[] = { 0u, 3u, 1u, 2u, 3u };
    const cellpack::csr_support_view source{3u, 4u, 5u, offsets, features};
    plan_storage plan = identity_plan(3u, 4u, {0u, 2u, 3u}, {0u, 2u, 4u});
    evaluation_storage evaluation = evaluate(source, plan.view(3u, 4u));

    require(evaluation.result.totals.total_nnz == 5u, "identity total nnz mismatch");
    require(evaluation.result.totals.logical_tile_count == 4u, "identity logical tile count mismatch");
    require(evaluation.result.totals.occupied_tile_count == 3u, "identity occupied tile count mismatch");
    require(evaluation.result.totals.empty_tile_count == 1u, "identity empty tile count mismatch");
    require(find_tile(evaluation, 0u, 0u).nnz == 2u, "identity tile 0,0 nnz mismatch");
    require(find_tile(evaluation, 0u, 1u).nnz == 1u, "identity tile 0,1 nnz mismatch");
    require(find_tile(evaluation, 1u, 1u).nnz == 2u, "identity tile 1,1 nnz mismatch");
    require(evaluation.result.nnz_per_occupied_tile.total == 5u, "identity tile distribution lost nnz");

    cellpack::packing_cost_model compact;
    compact.value_bytes = 2u;
    compact.per_nnz_index_bytes = 4u;
    compact.occupied_tile_metadata_bytes = 8u;
    compact.row_active_block_metadata_bytes = 4u;
    cellpack::packing_cost_estimate compact_cost;
    cellpack::validation_result status = cellpack::estimate_packing_cost(evaluation.result, compact, &compact_cost);
    require(static_cast<bool>(status), status.message);

    cellpack::packing_cost_model dense = compact;
    dense.dense_values_within_occupied_tiles = true;
    dense.occupied_tile_weight = 3.0;
    cellpack::packing_cost_estimate dense_cost;
    status = cellpack::estimate_packing_cost(evaluation.result, dense, &dense_cost);
    require(static_cast<bool>(status), status.message);
    require(dense_cost.value_slots == evaluation.result.totals.occupied_dense_slots, "dense model ignored occupancy slots");
    require(compact_cost.value_slots == source.nnz_count, "compact model did not use exact nnz");
    require(dense_cost.total_bytes > compact_cost.total_bytes, "different codec assumptions produced the same estimate");
    require(dense_cost.score > static_cast<double>(dense_cost.total_bytes), "weighted score did not include tile cost");
}

void test_nontrivial_two_sided_plan() {
    const cellpack::u32 offsets[] = { 0u, 3u, 5u, 9u, 11u };
    const cellpack::u32 features[] = {
        0u, 2u, 5u,
        1u, 4u,
        0u, 1u, 3u, 5u,
        2u, 4u
    };
    const cellpack::csr_support_view source{4u, 6u, 11u, offsets, features};
    plan_storage plan;
    plan.row_permutation = {2u, 0u, 3u, 1u};
    plan.inverse_row_permutation = {1u, 3u, 0u, 2u};
    plan.feature_permutation = {5u, 0u, 3u, 2u, 4u, 1u};
    plan.inverse_feature_permutation = {1u, 5u, 3u, 2u, 4u, 0u};
    plan.row_group_offsets = {0u, 2u, 4u};
    plan.feature_block_offsets = {0u, 3u, 5u, 6u};
    const cellpack::packing_plan_view view = plan.view(4u, 6u);
    evaluation_storage evaluation = evaluate(source, view);

    require(evaluation.result.totals.total_nnz == 11u, "nontrivial total nnz mismatch");
    require(evaluation.result.totals.logical_tile_count == 6u, "nontrivial logical tiles mismatch");
    require(evaluation.result.totals.occupied_tile_count == 5u, "nontrivial occupied tiles mismatch");
    require(evaluation.result.totals.empty_tile_count == 1u, "nontrivial empty tiles mismatch");
    require(evaluation.result.totals.occupied_dense_slots == 18u, "nontrivial dense slots mismatch");
    require(evaluation.result.totals.dense_padding == 7u, "nontrivial dense padding mismatch");
    require(evaluation.result.totals.row_active_block_references == 7u, "nontrivial row/block references mismatch");

    const cellpack::occupied_tile_occupancy &g0b0 = find_tile(evaluation, 0u, 0u);
    require(g0b0.nnz == 5u && g0b0.participating_rows == 2u, "nontrivial g0b0 mismatch");
    require_close(g0b0.density, 5.0 / 6.0, "nontrivial g0b0 density mismatch");
    require(find_tile(evaluation, 0u, 1u).nnz == 1u, "nontrivial g0b1 mismatch");
    require(find_tile(evaluation, 0u, 2u).nnz == 1u, "nontrivial g0b2 mismatch");
    require(find_tile(evaluation, 1u, 1u).nnz == 3u, "nontrivial g1b1 mismatch");
    require(find_tile(evaluation, 1u, 2u).nnz == 1u, "nontrivial g1b2 mismatch");
    require(evaluation.row_active_blocks == std::vector<cellpack::u32>({2u, 2u, 1u, 2u}), "per-execution-row active blocks mismatch");
    require(evaluation.row_groups[0].active_feature_blocks == 3u, "group zero active blocks mismatch");
    require(evaluation.row_groups[1].active_feature_blocks == 2u, "group one active blocks mismatch");

    for (cellpack::u32 canonical = 0u; canonical < view.feature_count; ++canonical) {
        const cellpack::u32 execution = view.inverse_feature_permutation[canonical];
        require(view.feature_permutation[execution] == canonical, "canonical feature identity did not round trip");
    }
    for (cellpack::u32 canonical = 0u; canonical < view.row_count; ++canonical) {
        const cellpack::u32 execution = view.inverse_row_permutation[canonical];
        require(view.row_permutation[execution] == canonical, "canonical row identity did not round trip");
    }
}

void require_equivalent(const evaluation_storage &lhs, const evaluation_storage &rhs) {
    require(lhs.result.totals.total_nnz == rhs.result.totals.total_nnz, "equivalent plans changed nnz");
    require(lhs.result.totals.occupied_tile_count == rhs.result.totals.occupied_tile_count, "equivalent plans changed occupied tiles");
    require(lhs.result.totals.dense_padding == rhs.result.totals.dense_padding, "equivalent plans changed padding");
    require(lhs.row_active_blocks == rhs.row_active_blocks, "equivalent plans changed row activity");
    for (cellpack::u32 i = 0u; i < lhs.result.occupied_tile_count; ++i) {
        const cellpack::occupied_tile_occupancy &a = lhs.tiles[i], &b = rhs.tiles[i];
        require(a.row_group == b.row_group && a.feature_block == b.feature_block, "equivalent tile identity mismatch");
        require(a.nnz == b.nnz && a.participating_rows == b.participating_rows, "equivalent tile occupancy mismatch");
        require_close(a.density, b.density, "equivalent tile density mismatch");
    }
}

void test_equivalent_identity_representations() {
    const cellpack::u32 offsets[] = {0u, 1u, 3u};
    const cellpack::u32 features[] = {1u, 0u, 3u};
    const cellpack::csr_support_view source{2u, 4u, 3u, offsets, features};
    plan_storage plan = identity_plan(2u, 4u, {0u, 1u, 2u}, {0u, 2u, 4u});
    evaluation_storage explicit_identity = evaluate(source, plan.view(2u, 4u, false));
    evaluation_storage implicit_identity = evaluate(source, plan.view(2u, 4u, true));
    require_equivalent(explicit_identity, implicit_identity);
}

void test_random_plan_conservation() {
    std::mt19937 rng(20260814u);
    for (cellpack::u32 trial = 0u; trial < 64u; ++trial) {
        const cellpack::u32 rows = 1u + (rng() % 17u), features = 1u + (rng() % 19u);
        std::vector<cellpack::u32> offsets(1u, 0u), feature_ids;
        for (cellpack::u32 row = 0u; row < rows; ++row) {
            for (cellpack::u32 feature = 0u; feature < features; ++feature) {
                if ((rng() % 7u) == 0u) feature_ids.push_back(feature);
            }
            offsets.push_back(static_cast<cellpack::u32>(feature_ids.size()));
        }
        const cellpack::csr_support_view source{
            rows,
            features,
            static_cast<cellpack::u32>(feature_ids.size()),
            offsets.data(),
            feature_ids.empty() ? nullptr : feature_ids.data()
        };
        plan_storage plan;
        plan.row_permutation.resize(rows);
        plan.feature_permutation.resize(features);
        std::iota(plan.row_permutation.begin(), plan.row_permutation.end(), 0u);
        std::iota(plan.feature_permutation.begin(), plan.feature_permutation.end(), 0u);
        std::shuffle(plan.row_permutation.begin(), plan.row_permutation.end(), rng);
        std::shuffle(plan.feature_permutation.begin(), plan.feature_permutation.end(), rng);
        plan.inverse_row_permutation.resize(rows);
        plan.inverse_feature_permutation.resize(features);
        require(cellpack::build_inverse_permutation(plan.row_permutation.data(), rows, plan.inverse_row_permutation.data()), "random row inverse failed");
        require(cellpack::build_inverse_permutation(plan.feature_permutation.data(), features, plan.inverse_feature_permutation.data()), "random feature inverse failed");
        const cellpack::u32 row_width = 1u + (rng() % rows), feature_width = 1u + (rng() % features);
        plan.row_group_offsets.push_back(0u);
        for (cellpack::u32 position = row_width; position < rows; position += row_width) plan.row_group_offsets.push_back(position);
        plan.row_group_offsets.push_back(rows);
        plan.feature_block_offsets.push_back(0u);
        for (cellpack::u32 position = feature_width; position < features; position += feature_width) plan.feature_block_offsets.push_back(position);
        plan.feature_block_offsets.push_back(features);

        evaluation_storage evaluation = evaluate(source, plan.view(rows, features));
        cellpack::u64 tile_sum = 0u, group_sum = 0u, row_reference_sum = 0u;
        for (cellpack::u32 i = 0u; i < evaluation.result.occupied_tile_count; ++i) tile_sum += evaluation.tiles[i].nnz;
        for (const cellpack::row_group_occupancy &group : evaluation.row_groups) group_sum += group.nnz;
        for (cellpack::u32 active : evaluation.row_active_blocks) row_reference_sum += active;
        require(tile_sum == source.nnz_count, "random plan tile nnz conservation failed");
        require(group_sum == source.nnz_count, "random plan group nnz conservation failed");
        require(row_reference_sum == evaluation.result.totals.row_active_block_references, "random plan row-reference conservation failed");
        require(evaluation.result.totals.occupied_tile_count + evaluation.result.totals.empty_tile_count
                    == evaluation.result.totals.logical_tile_count,
                "random plan tile conservation failed");
    }
}

void test_pathological_supports() {
    {
        const cellpack::u32 offsets[] = {0u, 0u, 6u, 8u, 10u, 12u, 14u};
        const cellpack::u32 features[] = {
            0u, 1u, 2u, 3u, 4u, 5u,
            0u, 1u, 0u, 2u, 0u, 3u, 0u, 4u
        };
        const cellpack::csr_support_view source{6u, 7u, 14u, offsets, features};
        plan_storage plan = identity_plan(6u, 7u, {0u, 1u, 3u, 6u}, {0u, 2u, 6u, 7u});
        evaluation_storage evaluation = evaluate(source, plan.view(6u, 7u));
        require(evaluation.row_active_blocks[0] == 0u, "empty row became active");
        require(evaluation.result.totals.total_nnz == 14u, "mixed pathological nnz mismatch");
        require(evaluation.result.active_feature_blocks_per_row.minimum == 0u, "empty-row distribution missing zero");
        require(evaluation.result.totals.empty_tile_count != 0u, "empty feature block did not create empty tiles");
    }
    {
        const cellpack::u32 offsets[] = {0u, 2u, 4u, 6u};
        const cellpack::u32 features[] = {0u, 1u, 0u, 1u, 0u, 1u};
        const cellpack::csr_support_view source{3u, 5u, 6u, offsets, features};
        plan_storage plan = identity_plan(3u, 5u, {0u, 3u}, {0u, 2u, 5u});
        evaluation_storage evaluation = evaluate(source, plan.view(3u, 5u));
        require(evaluation.result.totals.occupied_tile_count == 1u, "concentrated support escaped one tile");
        require(evaluation.result.totals.empty_tile_count == 1u, "concentrated support empty tile mismatch");
    }
    {
        const cellpack::u32 offsets[] = {0u, 1u, 2u, 3u, 4u};
        const cellpack::u32 features[] = {0u, 2u, 1u, 3u};
        const cellpack::csr_support_view source{4u, 4u, 4u, offsets, features};
        plan_storage plan = identity_plan(4u, 4u, {0u, 2u, 4u}, {0u, 2u, 4u});
        evaluation_storage evaluation = evaluate(source, plan.view(4u, 4u));
        require(evaluation.result.totals.occupied_tile_count == 4u, "one-nnz-per-tile occupied count mismatch");
        require(evaluation.result.nnz_per_occupied_tile.minimum == 1u
                    && evaluation.result.nnz_per_occupied_tile.maximum == 1u,
                "one-nnz-per-tile distribution mismatch");
    }

    const cellpack::u32 maximum = std::numeric_limits<cellpack::u32>::max();
    const cellpack::u32 maximum_boundary[] = {0u, maximum};
    cellpack::packing_plan_view maximum_row_group;
    maximum_row_group.row_count = maximum;
    maximum_row_group.row_group_count = 1u;
    maximum_row_group.row_group_offsets = maximum_boundary;
    require(static_cast<bool>(cellpack::validate_packing_plan_view(maximum_row_group)), "maximum legal row-group width was rejected");
    cellpack::packing_plan_view maximum_feature_block;
    maximum_feature_block.feature_count = maximum;
    maximum_feature_block.feature_block_count = 1u;
    maximum_feature_block.feature_block_offsets = maximum_boundary;
    require(static_cast<bool>(cellpack::validate_packing_plan_view(maximum_feature_block)), "maximum legal feature-block width was rejected");
}

void test_static_plan_adapter() {
    const cellpack::u32 assignments[] = {7u, 7u, 3u, 3u};
    const cellpack::u32 signature_offsets[] = {0u, 1u, 2u};
    const cellpack::u32 signatures[] = {7u, 3u};
    cellpack::feature_module_assignment_view features{assignments, 4u, cellpack::invalid_id};
    cellpack::row_signature_view rows{2u, signature_offsets, signatures, 2u};
    cellpack::static_plan plan;
    cellpack::validation_result status = cellpack::build_static_plan(features, rows, cellpack::planner_config{}, &plan);
    require(static_cast<bool>(status), status.message);
    const cellpack::packing_plan_view view = cellpack::make_packing_plan_view(plan);
    require(view.row_group_count == 2u && view.feature_block_count == 2u, "static plan adapter geometry mismatch");
    require(view.row_group_offsets[0] == 0u && view.row_group_offsets[2] == 2u, "static plan row boundaries mismatch");
    require(view.feature_block_offsets[0] == 0u && view.feature_block_offsets[2] == 4u, "static plan feature boundaries mismatch");
    require(static_cast<bool>(cellpack::validate_packing_plan_view(view)), "static plan adapter produced invalid geometry");
}

void test_validation_failures() {
    const cellpack::u32 offsets[] = {0u, 1u};
    const cellpack::u32 features[] = {0u};
    const cellpack::csr_support_view source{1u, 1u, 1u, offsets, features};
    const cellpack::u32 bad_boundaries[] = {0u, 0u};
    cellpack::packing_plan_view bad_plan;
    bad_plan.row_count = 1u;
    bad_plan.feature_count = 1u;
    bad_plan.row_group_count = 1u;
    bad_plan.row_group_offsets = bad_boundaries;
    bad_plan.feature_block_count = 1u;
    bad_plan.feature_block_offsets = bad_boundaries;
    require(cellpack::validate_packing_plan_view(bad_plan).code == cellpack::validation_code::invalid_plan_geometry,
            "invalid boundaries were accepted");

    plan_storage plan = identity_plan(1u, 1u, {0u, 1u}, {0u, 1u});
    cellpack::packing_evaluation_workspace_view no_workspace{};
    cellpack::packing_occupancy_buffers no_buffers{};
    cellpack::packing_occupancy_result result;
    require(cellpack::evaluate_packing_plan(source, plan.view(1u, 1u), no_workspace, no_buffers, &result).code
                == cellpack::validation_code::insufficient_capacity,
            "insufficient evaluator buffers were accepted");
}

} // namespace

int main() {
    test_identity_plan_and_cost_models();
    test_nontrivial_two_sided_plan();
    test_equivalent_identity_representations();
    test_random_plan_conservation();
    test_pathological_supports();
    test_static_plan_adapter();
    test_validation_failures();
    return 0;
}
