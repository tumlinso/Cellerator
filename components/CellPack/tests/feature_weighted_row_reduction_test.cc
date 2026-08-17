#include "CellPack/feature_weighted_row_reduction.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>

namespace {

using cellpack::u32;
using cellpack::u64;
using storage_t = cellerator::real::storage_t;
using compute_t = cellerator::real::compute_t;
using accum_t = cellerator::real::accum_t;

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cellPackFeatureWeightedRowReductionTest: " << message << '\n';
        std::exit(1);
    }
}

void require_status(cellpack::validation_result status, const char *message) {
    if (!status) {
        std::cerr << "cellPackFeatureWeightedRowReductionTest: " << message << ": "
                  << status.message << " (index=" << status.index << ")\n";
        std::exit(1);
    }
}

storage_t store(float value) {
    return static_cast<storage_t>(value);
}

cellpack::frozen_packing_plan make_plan() {
    const std::vector<u32> permutation{4u, 0u, 5u, 2u, 1u, 3u};
    std::vector<u32> inverse(permutation.size()), feature_to_block(permutation.size()),
        feature_to_local(permutation.size());
    for (u32 execution = 0u; execution < permutation.size(); ++execution) {
        inverse[permutation[execution]] = execution;
    }
    const std::vector<u32> block_offsets{0u, 2u, 5u, 6u};
    for (u32 block = 0u; block + 1u < block_offsets.size(); ++block) {
        for (u32 execution = block_offsets[block]; execution < block_offsets[block + 1u];
             ++execution) {
            const u32 canonical = permutation[execution];
            feature_to_block[canonical] = block;
            feature_to_local[canonical] = execution - block_offsets[block];
        }
    }
    const u32 row_group_offsets[] = {0u, 8u};
    cellpack::frozen_packing_plan_build_view build;
    build.row_count = 8u;
    build.feature_count = static_cast<u32>(permutation.size());
    build.feature_permutation = permutation.data();
    build.inverse_feature_permutation = inverse.data();
    build.feature_block_count = static_cast<u32>(block_offsets.size() - 1u);
    build.feature_block_offsets = block_offsets.data();
    build.feature_to_block = feature_to_block.data();
    build.feature_to_local = feature_to_local.data();
    build.row_group_count = 1u;
    build.row_group_offsets = row_group_offsets;
    build.maximum_feature_block_width = 3u;
    build.row_group_width = 8u;
    build.identity.feature_axis_fingerprint = 0x1020304050607080ull;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind = cellpack::packing_row_domain_kind::full_dataset_identity;
    build.identity.row_domain_identity = 0x8877665544332211ull;
    build.identity.evaluation_source_identity = 0x1234u;
    build.cost_policy_identity = 0x5678u;
    cellpack::frozen_packing_plan result;
    require_status(cellpack::freeze_packing_plan(build, &result), "freeze test plan");
    return result;
}

struct canonical_fixture {
    u64 global_row_begin = 2u;
    u32 row_count = 5u;
    std::vector<u32> row_offsets{0u, 3u, 3u, 5u, 8u, 9u};
    std::vector<u32> feature_ids{0u, 2u, 5u, 1u, 3u, 0u, 4u, 5u, 2u};
    std::vector<storage_t> values{
        store(1.0f), store(2.0f), store(-1.0f),
        store(0.5f), store(4.0f),
        store(-2.0f), store(3.0f), store(1.5f),
        store(1.0f)
    };

    cellpack::plan_application_context context() const {
        cellpack::plan_application_context result;
        result.full_row_count = 8u;
        result.feature_count = 6u;
        result.feature_axis_fingerprint = 0x1020304050607080ull;
        result.feature_axis_fingerprint_version = 1u;
        result.row_domain_identity = 0x8877665544332211ull;
        return result;
    }

    cellpack::plan_application_source_view view() const {
        cellpack::plan_application_source_view result;
        result.global_row_begin = global_row_begin;
        result.row_count = row_count;
        result.feature_count = 6u;
        result.nnz_count = static_cast<u32>(feature_ids.size());
        result.value_size_bytes = sizeof(storage_t);
        result.row_offsets = row_offsets.data();
        result.canonical_feature_ids = feature_ids.data();
        result.values = values.data();
        return result;
    }
};

struct ordered_owner {
    std::vector<u32> rows, blocks, locals, canonical;
    std::vector<storage_t> values;
    cellpack::ordered_plan_partition_view metadata{};

    cellpack::ordered_plan_partition_view view() const {
        auto result = metadata;
        result.row_offsets = rows.data();
        result.block_ids = blocks.data();
        result.local_feature_ids = locals.data();
        result.canonical_feature_ids = canonical.data();
        result.values = values.data();
        return result;
    }
};

ordered_owner apply_plan(
    const cellpack::frozen_packing_plan &plan,
    const canonical_fixture &source) {
    ordered_owner result;
    result.rows.resize(source.row_offsets.size());
    result.blocks.resize(source.feature_ids.size());
    result.locals.resize(source.feature_ids.size());
    result.canonical.resize(source.feature_ids.size());
    result.values.resize(source.values.size());
    std::vector<u64> keys(source.feature_ids.size());
    std::vector<u32> source_order(source.feature_ids.size());
    cellpack::plan_application_host_workspace_view workspace;
    workspace.entry_capacity = static_cast<u32>(keys.size());
    workspace.keys = keys.data();
    workspace.source_order = source_order.data();
    cellpack::plan_application_buffers buffers;
    buffers.row_offset_capacity = result.rows.size();
    buffers.entry_capacity = result.blocks.size();
    buffers.value_capacity_bytes = result.values.size() * sizeof(storage_t);
    buffers.row_offsets = result.rows.data();
    buffers.block_ids = result.blocks.data();
    buffers.local_feature_ids = result.locals.data();
    buffers.canonical_feature_ids = result.canonical.data();
    buffers.values = result.values.data();
    require_status(cellpack::apply_frozen_plan_host(plan, source.context(), source.view(),
        workspace, buffers, &result.metadata), "apply frozen plan");
    return result;
}

struct record_owner {
    std::vector<u32> rows, blocks, masks, value_offsets;
    std::vector<unsigned char> values;
    cellpack::cell_block_record_view metadata{};

    cellpack::cell_block_record_view view() const {
        auto result = metadata;
        result.row_record_offsets = rows.data();
        result.record_block_ids = blocks.data();
        result.record_gene_masks = masks.data();
        result.record_value_offsets = value_offsets.data();
        result.values = values.data();
        return result;
    }
};

record_owner make_records(
    const cellpack::frozen_packing_plan &plan,
    const cellpack::ordered_plan_partition_view &source) {
    cellpack::cell_block_record_requirements required;
    require_status(cellpack::query_cell_block_record_requirements_host(plan, source, &required),
        "query record requirements");
    record_owner result;
    result.rows.resize(required.row_record_offset_count);
    result.blocks.resize(required.record_count);
    result.masks.resize(required.record_count);
    result.value_offsets.resize(required.record_value_offset_count);
    result.values.resize(required.value_bytes);
    cellpack::cell_block_record_buffers buffers;
    buffers.row_record_offset_capacity = result.rows.size();
    buffers.record_capacity = result.blocks.size();
    buffers.record_value_offset_capacity = result.value_offsets.size();
    buffers.value_capacity_bytes = result.values.size();
    buffers.row_record_offsets = result.rows.data();
    buffers.record_block_ids = result.blocks.data();
    buffers.record_gene_masks = result.masks.data();
    buffers.record_value_offsets = result.value_offsets.data();
    buffers.values = result.values.data();
    require_status(cellpack::build_cell_block_records_host(
        plan, source, buffers, &result.metadata), "build records");
    return result;
}

struct order_owner {
    std::vector<u64> primary;
    std::vector<u32> secondary, active, nnz, permutation, inverse;
    cellpack::local_cell_order_view metadata{};

    cellpack::local_cell_order_view view() const {
        auto result = metadata;
        result.primary_keys = primary.data();
        result.secondary_keys = secondary.data();
        result.active_block_counts = active.data();
        result.row_nnz_counts = nnz.data();
        result.row_permutation = permutation.data();
        result.inverse_row_permutation = inverse.data();
        return result;
    }
};

order_owner make_order(const cellpack::cell_block_record_view &records) {
    order_owner result;
    result.primary.resize(records.row_count);
    result.secondary.resize(records.row_count);
    result.active.resize(records.row_count);
    result.nnz.resize(records.row_count);
    result.permutation.resize(records.row_count);
    result.inverse.resize(records.row_count);
    cellpack::local_cell_order_buffers buffers;
    buffers.row_capacity = records.row_count;
    buffers.primary_keys = result.primary.data();
    buffers.secondary_keys = result.secondary.data();
    buffers.active_block_counts = result.active.data();
    buffers.row_nnz_counts = result.nnz.data();
    buffers.row_permutation = result.permutation.data();
    buffers.inverse_row_permutation = result.inverse.data();
    cellpack::local_cell_order_config config;
    config.kind = cellpack::local_cell_order_kind::row_nnz_descending;
    config.window_size = 8u;
    config.group_width = 4u;
    require_status(cellpack::build_local_cell_order_host(
        records, config, buffers, &result.metadata), "build local order");
    return result;
}

struct tile_owner {
    std::vector<u32> tile_offsets, blocks, cell_masks, entry_offsets, gene_masks,
        value_offsets;
    std::vector<unsigned char> values;
    cellpack::warp_tile_view metadata{};

    cellpack::warp_tile_view view() const {
        auto result = metadata;
        result.tile_block_offsets = tile_offsets.data();
        result.tile_block_ids = blocks.data();
        result.tile_block_cell_masks = cell_masks.data();
        result.block_row_entry_offsets = entry_offsets.data();
        result.row_block_gene_masks = gene_masks.data();
        result.row_block_value_offsets = value_offsets.data();
        result.values = values.data();
        return result;
    }
};

tile_owner make_tiles(
    const cellpack::frozen_packing_plan &plan,
    const cellpack::cell_block_record_view &records,
    const cellpack::local_cell_order_view &order) {
    cellpack::warp_tile_requirements required;
    require_status(cellpack::query_warp_tile_requirements_host(
        plan, records, order, &required), "query tile requirements");
    tile_owner result;
    result.tile_offsets.resize(required.tile_block_offset_count);
    result.blocks.resize(required.tile_block_count);
    result.cell_masks.resize(required.tile_block_count);
    result.entry_offsets.resize(required.block_row_entry_offset_count);
    result.gene_masks.resize(required.row_block_entry_count);
    result.value_offsets.resize(required.row_block_value_offset_count);
    result.values.resize(required.value_bytes);
    cellpack::warp_tile_buffers buffers;
    buffers.tile_block_offset_capacity = result.tile_offsets.size();
    buffers.tile_block_capacity = result.blocks.size();
    buffers.block_row_entry_offset_capacity = result.entry_offsets.size();
    buffers.row_block_entry_capacity = result.gene_masks.size();
    buffers.row_block_value_offset_capacity = result.value_offsets.size();
    buffers.value_capacity_bytes = result.values.size();
    buffers.tile_block_offsets = result.tile_offsets.data();
    buffers.tile_block_ids = result.blocks.data();
    buffers.tile_block_cell_masks = result.cell_masks.data();
    buffers.block_row_entry_offsets = result.entry_offsets.data();
    buffers.row_block_gene_masks = result.gene_masks.data();
    buffers.row_block_value_offsets = result.value_offsets.data();
    buffers.values = result.values.data();
    require_status(cellpack::build_warp_tiles_host(
        plan, records, order, buffers, &result.metadata), "build tiles");
    return result;
}

void require_equal(const std::vector<accum_t> &lhs, const std::vector<accum_t> &rhs,
    const char *message) {
    require(lhs.size() == rhs.size(), message);
    for (std::size_t i = 0u; i < lhs.size(); ++i) {
        require(cellpack::feature_weighted_row_reduction_within_tolerance(lhs[i], rhs[i]),
            message);
    }
}

void test_canonical_record_and_direct_tile_agreement() {
    const auto plan = make_plan();
    const canonical_fixture source;
    const auto ordered = apply_plan(plan, source);
    const auto records = make_records(plan, ordered.view());
    const auto order = make_order(records.view());
    const auto tiles = make_tiles(plan, records.view(), order.view());
    require(order.permutation != std::vector<u32>({0u, 1u, 2u, 3u, 4u}),
        "test ordering remained identity");
    require(tiles.view().tile_count == 2u, "tail tile fixture was not created");

    std::vector<compute_t> weights{
        static_cast<compute_t>(1.25), static_cast<compute_t>(-2.0),
        static_cast<compute_t>(0.5), static_cast<compute_t>(3.0),
        static_cast<compute_t>(-1.0), static_cast<compute_t>(2.0)
    };
    const auto input = cellpack::make_feature_weighted_row_reduction_view(
        plan, tiles.view(), 0xabcdef0123456789ull, weights.size(), weights.data());
    require_status(cellpack::validate_feature_weighted_row_reduction_view_host(
        plan, records.view(), order.view(), input), "validate reduction view");

    std::vector<accum_t> canonical(source.row_count), record(source.row_count),
        tile(source.row_count), repeated(source.row_count);
    cellpack::feature_weighted_row_reduction_result_view canonical_result, record_result,
        tile_result, repeated_result;
    cellpack::feature_weighted_row_reduction_buffers buffers;
    buffers.row_capacity = canonical.size();
    buffers.row_values = canonical.data();
    require_status(cellpack::evaluate_feature_weighted_row_reduction_canonical_host(
        plan, source.context(), source.view(), input, buffers, &canonical_result),
        "evaluate canonical reduction");
    buffers.row_values = record.data();
    require_status(cellpack::evaluate_feature_weighted_row_reduction_records_host(
        plan, records.view(), input, buffers, &record_result), "evaluate record reduction");
    buffers.row_values = tile.data();
    require_status(cellpack::evaluate_feature_weighted_row_reduction_tiles_host(
        plan, records.view(), order.view(), input, buffers, &tile_result),
        "evaluate direct tile reduction");
    buffers.row_values = repeated.data();
    require_status(cellpack::evaluate_feature_weighted_row_reduction_tiles_host(
        plan, records.view(), order.view(), input, buffers, &repeated_result),
        "repeat direct tile reduction");

    require_equal(canonical, record, "canonical and record reductions differ");
    require_equal(canonical, tile, "canonical and tile reductions differ");
    require_equal(tile, repeated, "direct tile reduction is not deterministic");
    const std::vector<accum_t> expected{
        static_cast<accum_t>(0.25), static_cast<accum_t>(0.0),
        static_cast<accum_t>(11.0), static_cast<accum_t>(-2.5),
        static_cast<accum_t>(0.5)
    };
    require_equal(tile, expected, "weighted reduction values are wrong");
    require(tile_result.reduction_identity == input.reduction_identity
            && tile_result.feature_weight_identity == input.feature_weight_identity
            && tile_result.global_row_begin == source.global_row_begin
            && tile_result.row_domain_identity == source.context().row_domain_identity
            && tile_result.row_values == tile.data(),
        "weighted reduction result identity is wrong");
}

void test_zero_nnz_rows() {
    const auto plan = make_plan();
    canonical_fixture source;
    source.global_row_begin = 0u;
    source.row_count = 3u;
    source.row_offsets.assign(4u, 0u);
    source.feature_ids.clear();
    source.values.clear();
    const auto ordered = apply_plan(plan, source);
    const auto records = make_records(plan, ordered.view());
    const auto order = make_order(records.view());
    const auto tiles = make_tiles(plan, records.view(), order.view());
    std::vector<compute_t> weights(6u, static_cast<compute_t>(1.0));
    const auto input = cellpack::make_feature_weighted_row_reduction_view(
        plan, tiles.view(), 0x1111222233334444ull, weights.size(), weights.data());
    std::vector<accum_t> output(3u, static_cast<accum_t>(99.0));
    cellpack::feature_weighted_row_reduction_buffers buffers{output.size(), output.data()};
    cellpack::feature_weighted_row_reduction_result_view result;
    require_status(cellpack::evaluate_feature_weighted_row_reduction_tiles_host(
        plan, records.view(), order.view(), input, buffers, &result),
        "evaluate zero-nnz tile reduction");
    require(output == std::vector<accum_t>(3u, accum_t{}),
        "zero-nnz rows did not produce exact zero outputs");
}

void test_contract_and_capacity_failures() {
    const auto plan = make_plan();
    const canonical_fixture source;
    const auto ordered = apply_plan(plan, source);
    const auto records = make_records(plan, ordered.view());
    const auto order = make_order(records.view());
    const auto tiles = make_tiles(plan, records.view(), order.view());
    std::vector<compute_t> weights(6u, static_cast<compute_t>(1.0));
    const auto valid = cellpack::make_feature_weighted_row_reduction_view(
        plan, tiles.view(), 0x9999aaaabbbbccccull, weights.size(), weights.data());
    std::vector<accum_t> output(source.row_count);
    cellpack::feature_weighted_row_reduction_buffers buffers{output.size(), output.data()};
    cellpack::feature_weighted_row_reduction_result_view result;

    auto bad = valid;
    bad.feature_weight_capacity = weights.size() - 1u;
    require(!cellpack::evaluate_feature_weighted_row_reduction_tiles_host(
        plan, records.view(), order.view(), bad, buffers, &result),
        "insufficient feature-weight capacity was accepted");
    bad = valid;
    bad.feature_weights = nullptr;
    require(!cellpack::evaluate_feature_weighted_row_reduction_tiles_host(
        plan, records.view(), order.view(), bad, buffers, &result),
        "null feature weights were accepted");
    bad = valid;
    bad.feature_weight_identity ^= 1u;
    require(!cellpack::evaluate_feature_weighted_row_reduction_tiles_host(
        plan, records.view(), order.view(), bad, buffers, &result),
        "tampered weight identity was accepted");
    bad = valid;
    bad.reduction_identity ^= 1u;
    require(!cellpack::evaluate_feature_weighted_row_reduction_tiles_host(
        plan, records.view(), order.view(), bad, buffers, &result),
        "tampered reduction identity was accepted");
    bad = valid;
    bad.plan.feature_permutation = nullptr;
    require(!cellpack::evaluate_feature_weighted_row_reduction_tiles_host(
        plan, records.view(), order.view(), bad, buffers, &result),
        "tampered plan view was accepted");
    bad = valid;
    ++bad.tiles.value_size_bytes;
    require(!cellpack::evaluate_feature_weighted_row_reduction_tiles_host(
        plan, records.view(), order.view(), bad, buffers, &result),
        "unsupported tile value width was accepted");

    auto short_buffers = buffers;
    short_buffers.row_capacity = output.size() - 1u;
    require(!cellpack::evaluate_feature_weighted_row_reduction_tiles_host(
        plan, records.view(), order.view(), valid, short_buffers, &result),
        "insufficient row-output capacity was accepted");
    auto alias_buffers = buffers;
    alias_buffers.row_values = reinterpret_cast<accum_t *>(weights.data());
    require(!cellpack::evaluate_feature_weighted_row_reduction_tiles_host(
        plan, records.view(), order.view(), valid, alias_buffers, &result),
        "weight/output alias was accepted");
    require(!cellpack::evaluate_feature_weighted_row_reduction_tiles_host(
        plan, records.view(), order.view(), valid, buffers, nullptr),
        "null result view was accepted");

    auto bad_source = source.view();
    ++bad_source.global_row_begin;
    require(!cellpack::evaluate_feature_weighted_row_reduction_canonical_host(
        plan, source.context(), bad_source, valid, buffers, &result),
        "mismatched canonical partition identity was accepted");

    const auto changed = cellpack::make_feature_weighted_row_reduction_view(
        plan, tiles.view(), 0x9999aaaabbbbcccdull, weights.size(), weights.data());
    require(changed.reduction_identity != valid.reduction_identity,
        "weight generation identity did not affect reduction identity");
}

} // namespace

int main() {
    static_assert(std::is_same<storage_t, cellerator::real::storage_t>::value,
        "test must use configured Cellerator storage values");
    test_canonical_record_and_direct_tile_agreement();
    test_zero_nnz_rows();
    test_contract_and_capacity_failures();
    return 0;
}
