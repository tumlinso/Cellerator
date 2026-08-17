#include "CellPack/warp_tiles.hh"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

using cellpack::u32;
using cellpack::u64;

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cellPackWarpTilesTest: " << message << '\n';
        std::exit(1);
    }
}

void require_status(cellpack::validation_result status, const char *message) {
    if (!status) {
        std::cerr << "cellPackWarpTilesTest: " << message << ": "
                  << status.message << " (index=" << status.index << ")\n";
        std::exit(1);
    }
}

cellpack::frozen_packing_plan make_plan() {
    std::vector<u32> permutation(36u), inverse(36u), feature_to_block(36u),
        feature_to_local(36u);
    for (u32 feature = 0u; feature < permutation.size(); ++feature) {
        permutation[feature] = feature;
        inverse[feature] = feature;
    }
    const std::vector<u32> block_offsets{0u, 32u, 34u, 36u};
    for (u32 block = 0u; block + 1u < block_offsets.size(); ++block) {
        for (u32 feature = block_offsets[block]; feature < block_offsets[block + 1u]; ++feature) {
            feature_to_block[feature] = block;
            feature_to_local[feature] = feature - block_offsets[block];
        }
    }
    const u32 row_offsets[] = {0u, 64u};
    cellpack::frozen_packing_plan_build_view build;
    build.row_count = 64u;
    build.feature_count = 36u;
    build.feature_permutation = permutation.data();
    build.inverse_feature_permutation = inverse.data();
    build.feature_block_count = 3u;
    build.feature_block_offsets = block_offsets.data();
    build.feature_to_block = feature_to_block.data();
    build.feature_to_local = feature_to_local.data();
    build.row_group_count = 1u;
    build.row_group_offsets = row_offsets;
    build.maximum_feature_block_width = 32u;
    build.row_group_width = 64u;
    build.identity.feature_axis_fingerprint = 0x123456789abcdef0ull;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind = cellpack::packing_row_domain_kind::full_dataset_identity;
    build.identity.row_domain_identity = 0x0fedcba987654321ull;
    build.identity.evaluation_source_identity = 0x11223344u;
    build.cost_policy_identity = 0x55667788u;
    cellpack::frozen_packing_plan result;
    require_status(cellpack::freeze_packing_plan(build, &result), "freeze test plan");
    return result;
}

struct ordered_storage {
    u64 global_row_begin = 0u;
    u32 row_count = 0u;
    u32 value_size_bytes = 0u;
    std::vector<u32> row_offsets, block_ids, local_ids, canonical_ids;
    std::vector<unsigned char> values;

    cellpack::ordered_plan_partition_view view() const {
        cellpack::ordered_plan_partition_view result;
        result.semantic_plan_schema_version = cellpack::packing_plan_semantic_schema_version;
        result.global_row_begin = global_row_begin;
        result.full_row_count = 64u;
        result.row_count = row_count;
        result.feature_count = 36u;
        result.nnz_count = static_cast<u32>(canonical_ids.size());
        result.value_size_bytes = value_size_bytes;
        result.feature_axis_fingerprint = 0x123456789abcdef0ull;
        result.feature_axis_fingerprint_version = 1u;
        result.row_domain_identity = 0x0fedcba987654321ull;
        result.row_offsets = row_offsets.data();
        result.block_ids = block_ids.data();
        result.local_feature_ids = local_ids.data();
        result.canonical_feature_ids = canonical_ids.data();
        result.values = values.data();
        return result;
    }
};

void append_entry(ordered_storage *fixture, u32 block, u32 local, u32 tag) {
    static constexpr u32 block_begin[] = {0u, 32u, 34u};
    fixture->block_ids.push_back(block);
    fixture->local_ids.push_back(local);
    fixture->canonical_ids.push_back(block_begin[block] + local);
    for (u32 byte = 0u; byte < fixture->value_size_bytes; ++byte) {
        fixture->values.push_back(static_cast<unsigned char>((tag * 37u + byte * 19u) & 0xffu));
    }
}

ordered_storage make_main_fixture() {
    ordered_storage result;
    result.global_row_begin = 7u;
    result.row_count = 34u;
    result.value_size_bytes = 3u;
    result.row_offsets.push_back(0u);
    for (u32 row = 0u; row < 34u; ++row) {
        if (row < 32u) {
            append_entry(&result, 0u, row, row * 10u);
            if (row == 0u) {
                append_entry(&result, 1u, 0u, 1u);
                append_entry(&result, 1u, 1u, 2u);
            }
            if (row == 1u) {
                append_entry(&result, 1u, 1u, 11u);
                append_entry(&result, 2u, 0u, 12u);
            }
        } else if (row == 32u) {
            append_entry(&result, 2u, 1u, 321u);
        }
        result.row_offsets.push_back(static_cast<u32>(result.canonical_ids.size()));
    }
    return result;
}

ordered_storage make_width_fixture(u32 value_size_bytes) {
    ordered_storage result;
    result.global_row_begin = 3u;
    result.row_count = 2u;
    result.value_size_bytes = value_size_bytes;
    result.row_offsets.push_back(0u);
    append_entry(&result, 0u, 31u, 91u);
    result.row_offsets.push_back(1u);
    append_entry(&result, 1u, 1u, 92u);
    result.row_offsets.push_back(2u);
    return result;
}

ordered_storage make_empty_fixture(u32 row_count, u32 value_size_bytes) {
    ordered_storage result;
    result.global_row_begin = row_count == 0u ? 64u : 0u;
    result.row_count = row_count;
    result.value_size_bytes = value_size_bytes;
    result.row_offsets.assign(static_cast<std::size_t>(row_count) + 1u, 0u);
    return result;
}

struct record_storage {
    std::vector<u32> row_offsets, block_ids, gene_masks, value_offsets;
    std::vector<unsigned char> values;
    cellpack::cell_block_record_view metadata{};

    cellpack::cell_block_record_view view() const {
        auto result = metadata;
        result.row_record_offsets = row_offsets.data();
        result.record_block_ids = block_ids.data();
        result.record_gene_masks = gene_masks.data();
        result.record_value_offsets = value_offsets.data();
        result.values = values.data();
        return result;
    }
};

record_storage make_records(
    const cellpack::frozen_packing_plan &plan,
    const ordered_storage &source) {
    cellpack::cell_block_record_requirements required;
    require_status(cellpack::query_cell_block_record_requirements_host(
        plan, source.view(), &required), "query record requirements");
    record_storage result;
    result.row_offsets.resize(required.row_record_offset_count);
    result.block_ids.resize(required.record_count);
    result.gene_masks.resize(required.record_count);
    result.value_offsets.resize(required.record_value_offset_count);
    result.values.resize(required.value_bytes);
    cellpack::cell_block_record_buffers buffers;
    buffers.row_record_offset_capacity = result.row_offsets.size();
    buffers.record_capacity = result.block_ids.size();
    buffers.record_value_offset_capacity = result.value_offsets.size();
    buffers.value_capacity_bytes = result.values.size();
    buffers.row_record_offsets = result.row_offsets.data();
    buffers.record_block_ids = result.block_ids.data();
    buffers.record_gene_masks = result.gene_masks.data();
    buffers.record_value_offsets = result.value_offsets.data();
    buffers.values = result.values.data();
    require_status(cellpack::build_cell_block_records_host(
        plan, source.view(), buffers, &result.metadata), "build records");
    return result;
}

struct order_storage {
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

order_storage make_order(
    const cellpack::cell_block_record_view &records,
    u32 group_width,
    u32 window_size,
    cellpack::local_cell_order_kind kind = cellpack::local_cell_order_kind::original) {
    order_storage result;
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
    config.kind = kind;
    config.group_width = group_width;
    config.window_size = window_size;
    require_status(cellpack::build_local_cell_order_host(
        records, config, buffers, &result.metadata), "build local order");
    return result;
}

struct tile_storage {
    std::vector<u32> tile_offsets, block_ids, cell_masks, entry_offsets,
        gene_masks, value_offsets;
    std::vector<unsigned char> values;
    cellpack::warp_tile_view metadata{};

    cellpack::warp_tile_view view() const {
        auto result = metadata;
        result.tile_block_offsets = tile_offsets.data();
        result.tile_block_ids = block_ids.data();
        result.tile_block_cell_masks = cell_masks.data();
        result.block_row_entry_offsets = entry_offsets.data();
        result.row_block_gene_masks = gene_masks.data();
        result.row_block_value_offsets = value_offsets.data();
        result.values = values.data();
        return result;
    }
};

tile_storage make_tiles(
    const cellpack::frozen_packing_plan &plan,
    const cellpack::cell_block_record_view &records,
    const cellpack::local_cell_order_view &order) {
    cellpack::warp_tile_requirements required;
    require_status(cellpack::query_warp_tile_requirements_host(
        plan, records, order, &required), "query tile requirements");
    tile_storage result;
    result.tile_offsets.resize(required.tile_block_offset_count);
    result.block_ids.resize(required.tile_block_count);
    result.cell_masks.resize(required.tile_block_count);
    result.entry_offsets.resize(required.block_row_entry_offset_count);
    result.gene_masks.resize(required.row_block_entry_count);
    result.value_offsets.resize(required.row_block_value_offset_count);
    result.values.resize(required.value_bytes);
    cellpack::warp_tile_buffers buffers;
    buffers.tile_block_offset_capacity = result.tile_offsets.size();
    buffers.tile_block_capacity = result.block_ids.size();
    buffers.block_row_entry_offset_capacity = result.entry_offsets.size();
    buffers.row_block_entry_capacity = result.gene_masks.size();
    buffers.row_block_value_offset_capacity = result.value_offsets.size();
    buffers.value_capacity_bytes = result.values.size();
    buffers.tile_block_offsets = result.tile_offsets.data();
    buffers.tile_block_ids = result.block_ids.data();
    buffers.tile_block_cell_masks = result.cell_masks.data();
    buffers.block_row_entry_offsets = result.entry_offsets.data();
    buffers.row_block_gene_masks = result.gene_masks.data();
    buffers.row_block_value_offsets = result.value_offsets.data();
    buffers.values = result.values.data();
    require_status(cellpack::build_warp_tiles_host(
        plan, records, order, buffers, &result.metadata), "build warp tiles");
    return result;
}

void require_exact_decode(
    const cellpack::frozen_packing_plan &plan,
    const ordered_storage &source,
    const record_storage &record_owner,
    const order_storage &order_owner,
    const tile_storage &tile_owner) {
    const auto records = record_owner.view();
    const auto order = order_owner.view();
    const auto tiles = tile_owner.view();
    std::vector<u32> cursors(source.row_count), rows(source.row_offsets.size()),
        features(source.canonical_ids.size());
    std::vector<unsigned char> values(source.values.size());
    cellpack::warp_tile_decode_workspace workspace;
    workspace.row_capacity = cursors.size();
    workspace.row_cursors = cursors.data();
    cellpack::warp_tile_decode_buffers buffers;
    buffers.row_offset_capacity = rows.size();
    buffers.entry_capacity = features.size();
    buffers.value_capacity_bytes = values.size();
    buffers.row_offsets = rows.data();
    buffers.canonical_feature_ids = features.data();
    buffers.values = values.data();
    cellpack::decoded_warp_tile_partition_view decoded;
    require_status(cellpack::decode_warp_tiles_host(
        plan, records, order, tiles, workspace, buffers, &decoded), "decode warp tiles");
    require(rows == source.row_offsets, "decoded row offsets changed");
    require(features == source.canonical_ids, "decoded canonical feature ids changed");
    require(values == source.values, "decoded value bytes changed");
    require(decoded.global_row_begin == source.global_row_begin
            && decoded.row_domain_identity == source.view().row_domain_identity,
        "decoded row identity changed");
}

void test_exact_host_contract_and_metrics() {
    const auto plan = make_plan();
    const auto source = make_main_fixture();
    const auto records = make_records(plan, source);
    const auto order = make_order(records.view(), 32u, 32u);
    const auto tiles = make_tiles(plan, records.view(), order.view());
    const auto view = tiles.view();

    require(view.tile_identity == cellpack::warp_tile_identity(records.view(), order.view())
            && view.feature_block_geometry_identity == plan.feature_block_geometry_identity()
            && view.ordering_identity == order.view().ordering_identity,
        "warp-tile semantic identity was not preserved");
    require(tiles.tile_offsets == std::vector<u32>({0u, 3u, 4u}),
        "tile-to-block offsets are wrong");
    require(tiles.block_ids == std::vector<u32>({0u, 1u, 2u, 2u}),
        "tile dictionaries are wrong");
    require(tiles.cell_masks == std::vector<u32>({0xffffffffu, 3u, 2u, 1u}),
        "tile cell masks are wrong, including bit 31 or the tail");
    require(tiles.entry_offsets == std::vector<u32>({0u, 32u, 34u, 35u, 36u}),
        "block-to-row-entry offsets are wrong");
    require(tiles.gene_masks[31] == 0x80000000u
            && tiles.gene_masks[32] == 3u
            && tiles.gene_masks[33] == 2u
            && tiles.gene_masks[34] == 1u
            && tiles.gene_masks[35] == 2u,
        "row-block gene masks are wrong");
    require(tiles.value_offsets.front() == 0u
            && tiles.value_offsets.back() == source.canonical_ids.size(),
        "compact value terminal offsets are wrong");
    require_exact_decode(plan, source, records, order, tiles);

    const auto shuffled_order = make_order(records.view(), 32u, 32u,
        cellpack::local_cell_order_kind::deterministic_random);
    const auto shuffled_tiles = make_tiles(plan, records.view(), shuffled_order.view());
    require(shuffled_order.permutation != order.permutation,
        "non-identity CP-BP-07 fixture did not permute rows");
    require(shuffled_tiles.view().ordering_identity == shuffled_order.view().ordering_identity
            && shuffled_tiles.view().tile_identity != view.tile_identity,
        "warp-tile identity did not bind the non-identity row order");
    require_exact_decode(plan, source, records, shuffled_order, shuffled_tiles);

    const auto repeated = make_tiles(plan, records.view(), order.view());
    require(tiles.tile_offsets == repeated.tile_offsets
            && tiles.block_ids == repeated.block_ids
            && tiles.cell_masks == repeated.cell_masks
            && tiles.entry_offsets == repeated.entry_offsets
            && tiles.gene_masks == repeated.gene_masks
            && tiles.value_offsets == repeated.value_offsets
            && tiles.values == repeated.values,
        "warp-tile construction is not deterministic");

    cellpack::warp_tile_metrics metrics;
    require_status(cellpack::evaluate_warp_tile_metrics_host(
        plan, records.view(), order.view(), view, &metrics), "evaluate tile metrics");
    require(metrics.tile_count == 2u && metrics.tile_block_count == 4u
            && metrics.row_block_entry_count == 36u
            && metrics.maximum_tile_block_union == 3u,
        "warp-tile count metrics are wrong");
    const u64 expected_metadata = (2u + 1u) * sizeof(u32)
        + 4u * 2u * sizeof(u32) + (4u + 1u) * sizeof(u32)
        + 36u * sizeof(u32) + (36u + 1u) * sizeof(u32);
    require(metrics.metadata_bytes == expected_metadata
            && metrics.value_bytes == source.values.size()
            && metrics.total_bytes == expected_metadata + source.values.size(),
        "warp-tile byte metrics are wrong");
}

void test_empty_inputs_and_value_widths() {
    const auto plan = make_plan();
    for (u32 value_size : {1u, 3u, 8u}) {
        const auto source = make_width_fixture(value_size);
        const auto records = make_records(plan, source);
        const auto order = make_order(records.view(), 2u, 2u);
        const auto tiles = make_tiles(plan, records.view(), order.view());
        require_exact_decode(plan, source, records, order, tiles);
    }

    for (u32 row_count : {0u, 4u}) {
        const auto source = make_empty_fixture(row_count, 5u);
        const auto records = make_records(plan, source);
        const u32 width = row_count == 0u ? 1u : 4u;
        const auto order = make_order(records.view(), width, width);
        const auto tiles = make_tiles(plan, records.view(), order.view());
        require(tiles.tile_offsets == (row_count == 0u
                ? std::vector<u32>({0u}) : std::vector<u32>({0u, 0u})),
            "empty partition/tile offsets are wrong");
        require(tiles.block_ids.empty() && tiles.gene_masks.empty()
                && tiles.value_offsets == std::vector<u32>({0u}),
            "empty partition/tile stored payload metadata");
        require_exact_decode(plan, source, records, order, tiles);
    }
}

void test_capacity_and_validation_failures() {
    const auto plan = make_plan();
    const auto source = make_main_fixture();
    const auto records = make_records(plan, source);
    const auto order = make_order(records.view(), 32u, 32u);
    const auto tiles = make_tiles(plan, records.view(), order.view());
    const auto valid = tiles.view();

    cellpack::warp_tile_requirements required;
    require_status(cellpack::query_warp_tile_requirements_host(
        plan, records.view(), order.view(), &required), "query valid tile requirements");
    std::vector<u32> tile_offsets(required.tile_block_offset_count),
        block_ids(required.tile_block_count), cell_masks(required.tile_block_count),
        entry_offsets(required.block_row_entry_offset_count),
        gene_masks(required.row_block_entry_count),
        value_offsets(required.row_block_value_offset_count);
    std::vector<unsigned char> values(required.value_bytes);
    cellpack::warp_tile_buffers short_buffers;
    short_buffers.tile_block_offset_capacity = tile_offsets.size();
    short_buffers.tile_block_capacity = block_ids.size() - 1u;
    short_buffers.block_row_entry_offset_capacity = entry_offsets.size();
    short_buffers.row_block_entry_capacity = gene_masks.size();
    short_buffers.row_block_value_offset_capacity = value_offsets.size();
    short_buffers.value_capacity_bytes = values.size();
    short_buffers.tile_block_offsets = tile_offsets.data();
    short_buffers.tile_block_ids = block_ids.data();
    short_buffers.tile_block_cell_masks = cell_masks.data();
    short_buffers.block_row_entry_offsets = entry_offsets.data();
    short_buffers.row_block_gene_masks = gene_masks.data();
    short_buffers.row_block_value_offsets = value_offsets.data();
    short_buffers.values = values.data();
    cellpack::warp_tile_view ignored;
    require(!cellpack::build_warp_tiles_host(
        plan, records.view(), order.view(), short_buffers, &ignored),
        "insufficient tile-block capacity was accepted");

    auto bad = valid;
    bad.tile_schema_version = 99u;
    require(!cellpack::validate_warp_tile_view_host(plan, records.view(), order.view(), bad),
        "unsupported tile schema was accepted");
    bad = valid;
    bad.ordering_identity ^= 1u;
    require(!cellpack::validate_warp_tile_view_host(plan, records.view(), order.view(), bad),
        "wrong ordering identity was accepted");
    bad = valid;
    bad.row_domain_identity ^= 1u;
    require(!cellpack::validate_warp_tile_view_host(plan, records.view(), order.view(), bad),
        "wrong row-domain identity was accepted");

    auto bad_tile_offsets = tiles.tile_offsets;
    bad_tile_offsets[1] = bad_tile_offsets[0];
    bad = valid;
    bad.tile_block_offsets = bad_tile_offsets.data();
    require(!cellpack::validate_warp_tile_view_host(plan, records.view(), order.view(), bad),
        "omitted tile dictionary range was accepted");
    auto bad_blocks = tiles.block_ids;
    bad_blocks[1] = bad_blocks[0];
    bad = valid;
    bad.tile_block_ids = bad_blocks.data();
    require(!cellpack::validate_warp_tile_view_host(plan, records.view(), order.view(), bad),
        "duplicate/out-of-order tile block was accepted");
    auto bad_cell_masks = tiles.cell_masks;
    bad_cell_masks.back() |= 4u;
    bad = valid;
    bad.tile_block_cell_masks = bad_cell_masks.data();
    require(!cellpack::validate_warp_tile_view_host(plan, records.view(), order.view(), bad),
        "tail cell-mask overflow was accepted");
    bad_cell_masks = tiles.cell_masks;
    bad_cell_masks[0] = 0u;
    bad = valid;
    bad.tile_block_cell_masks = bad_cell_masks.data();
    require(!cellpack::validate_warp_tile_view_host(plan, records.view(), order.view(), bad),
        "empty cell mask was accepted");
    auto bad_entry_offsets = tiles.entry_offsets;
    --bad_entry_offsets.back();
    bad = valid;
    bad.block_row_entry_offsets = bad_entry_offsets.data();
    require(!cellpack::validate_warp_tile_view_host(plan, records.view(), order.view(), bad),
        "wrong terminal row-entry offset was accepted");
    auto bad_gene_masks = tiles.gene_masks;
    bad_gene_masks[31] = 1u;
    bad = valid;
    bad.row_block_gene_masks = bad_gene_masks.data();
    require(!cellpack::validate_warp_tile_view_host(plan, records.view(), order.view(), bad),
        "tampered gene mask was accepted");
    auto bad_value_offsets = tiles.value_offsets;
    ++bad_value_offsets[1];
    bad = valid;
    bad.row_block_value_offsets = bad_value_offsets.data();
    require(!cellpack::validate_warp_tile_view_host(plan, records.view(), order.view(), bad),
        "tampered value offset was accepted");
    auto bad_values = tiles.values;
    bad_values[0] ^= 0xffu;
    bad = valid;
    bad.values = bad_values.data();
    require(!cellpack::validate_warp_tile_view_host(plan, records.view(), order.view(), bad),
        "tampered compact value bytes were accepted");

    std::vector<u32> cursors(source.row_count), rows(source.row_offsets.size()),
        features(source.canonical_ids.size() - 1u);
    std::vector<unsigned char> decoded_values(source.values.size());
    cellpack::warp_tile_decode_workspace workspace{cursors.size(), cursors.data()};
    cellpack::warp_tile_decode_buffers decode_buffers;
    decode_buffers.row_offset_capacity = rows.size();
    decode_buffers.entry_capacity = features.size();
    decode_buffers.value_capacity_bytes = decoded_values.size();
    decode_buffers.row_offsets = rows.data();
    decode_buffers.canonical_feature_ids = features.data();
    decode_buffers.values = decoded_values.data();
    cellpack::decoded_warp_tile_partition_view decoded;
    require(!cellpack::decode_warp_tiles_host(plan, records.view(), order.view(), valid,
        workspace, decode_buffers, &decoded), "insufficient decode capacity was accepted");

    const auto wide_order = make_order(records.view(), 64u, 64u);
    require(!cellpack::query_warp_tile_requirements_host(
        plan, records.view(), wide_order.view(), &required),
        "tile width greater than one warp was accepted");
}

} // namespace

int main() {
    test_exact_host_contract_and_metrics();
    test_empty_inputs_and_value_widths();
    test_capacity_and_validation_failures();
    std::cout << "cellPackWarpTilesTest: PASS\n";
    return 0;
}
