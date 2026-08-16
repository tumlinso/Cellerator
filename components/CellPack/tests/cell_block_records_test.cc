#include "CellPack/cell_block_records.hh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

namespace {

using cellpack::u32;
using cellpack::u64;

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cellPackCellBlockRecordsTest: " << message << '\n';
        std::exit(1);
    }
}

void require_status(cellpack::validation_result status, const char *message) {
    if (!status) {
        std::cerr << "cellPackCellBlockRecordsTest: " << message << ": "
                  << status.message << " (index=" << status.index << ")\n";
        std::exit(1);
    }
}

cellpack::frozen_packing_plan make_plan(
    const std::vector<u32> &feature_permutation,
    const std::vector<u32> &block_offsets,
    u32 maximum_width,
    u32 row_count = 8u) {
    std::vector<u32> inverse(feature_permutation.size());
    std::vector<u32> feature_to_block(feature_permutation.size());
    std::vector<u32> feature_to_local(feature_permutation.size());
    for (u32 execution = 0u; execution < feature_permutation.size(); ++execution) {
        inverse[feature_permutation[execution]] = execution;
    }
    for (u32 block = 0u; block + 1u < block_offsets.size(); ++block) {
        for (u32 execution = block_offsets[block]; execution < block_offsets[block + 1u]; ++execution) {
            const u32 canonical = feature_permutation[execution];
            feature_to_block[canonical] = block;
            feature_to_local[canonical] = execution - block_offsets[block];
        }
    }
    const std::vector<u32> row_offsets{0u, row_count};
    cellpack::frozen_packing_plan_build_view build;
    build.row_count = row_count;
    build.feature_count = static_cast<u32>(feature_permutation.size());
    build.feature_permutation = feature_permutation.data();
    build.inverse_feature_permutation = inverse.data();
    build.feature_block_count = static_cast<u32>(block_offsets.size() - 1u);
    build.feature_block_offsets = block_offsets.data();
    build.feature_to_block = feature_to_block.data();
    build.feature_to_local = feature_to_local.data();
    build.row_group_count = 1u;
    build.row_group_offsets = row_offsets.data();
    build.maximum_feature_block_width = maximum_width;
    build.row_group_width = row_count;
    build.identity.feature_axis_fingerprint = 0x12345678u;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind = cellpack::packing_row_domain_kind::full_dataset_identity;
    build.identity.row_domain_identity = 0xabcdu;
    build.identity.evaluation_source_identity = 0x777u;
    build.cost_policy_identity = 0x999u;
    cellpack::frozen_packing_plan result;
    require_status(cellpack::freeze_packing_plan(build, &result), "freeze test plan");
    return result;
}

cellpack::frozen_packing_plan make_fixture_plan() {
    return make_plan({3u, 1u, 5u, 0u, 4u, 2u}, {0u, 2u, 5u, 6u}, 3u);
}

struct ordered_fixture {
    std::vector<u32> row_offsets{0u, 3u, 3u, 6u, 12u};
    std::vector<u32> blocks{0u, 1u, 1u, 0u, 1u, 2u, 0u, 0u, 1u, 1u, 1u, 2u};
    std::vector<u32> locals{0u, 0u, 1u, 1u, 2u, 0u, 0u, 1u, 0u, 1u, 2u, 0u};
    std::vector<u32> features{3u, 5u, 0u, 1u, 4u, 2u, 3u, 1u, 5u, 0u, 4u, 2u};
    std::vector<u64> values{13u, 15u, 10u, 21u, 24u, 22u, 33u, 31u, 35u, 30u, 34u, 32u};

    cellpack::ordered_plan_partition_view view() const {
        cellpack::ordered_plan_partition_view result;
        result.semantic_plan_schema_version = cellpack::packing_plan_semantic_schema_version;
        result.global_row_begin = 2u;
        result.full_row_count = 8u;
        result.row_count = 4u;
        result.feature_count = 6u;
        result.nnz_count = static_cast<u32>(features.size());
        result.value_size_bytes = sizeof(u64);
        result.feature_axis_fingerprint = 0x12345678u;
        result.feature_axis_fingerprint_version = 1u;
        result.row_domain_identity = 0xabcdu;
        result.row_offsets = row_offsets.data();
        result.block_ids = blocks.data();
        result.local_feature_ids = locals.data();
        result.canonical_feature_ids = features.data();
        result.values = values.data();
        return result;
    }
};

struct record_storage {
    std::vector<u32> row_offsets, blocks, masks, value_offsets;
    std::vector<unsigned char> values;
    cellpack::cell_block_record_view view{};
};

record_storage build_records(
    const cellpack::frozen_packing_plan &plan,
    const cellpack::ordered_plan_partition_view &source) {
    cellpack::cell_block_record_requirements required;
    require_status(cellpack::query_cell_block_record_requirements_host(
        plan, source, &required), "query record requirements");
    record_storage result;
    result.row_offsets.resize(required.row_record_offset_count);
    result.blocks.resize(required.record_count);
    result.masks.resize(required.record_count);
    result.value_offsets.resize(required.record_value_offset_count);
    result.values.resize(required.value_bytes);
    cellpack::cell_block_record_buffers buffers;
    buffers.row_record_offset_capacity = result.row_offsets.size();
    buffers.record_capacity = result.blocks.size();
    buffers.record_value_offset_capacity = result.value_offsets.size();
    buffers.value_capacity_bytes = result.values.size();
    buffers.row_record_offsets = result.row_offsets.data();
    buffers.record_block_ids = result.blocks.data();
    buffers.record_gene_masks = result.masks.data();
    buffers.record_value_offsets = result.value_offsets.data();
    buffers.values = result.values.data();
    require_status(cellpack::build_cell_block_records_host(
        plan, source, buffers, &result.view), "build records");
    return result;
}

void test_exact_host_contract() {
    const cellpack::frozen_packing_plan plan = make_fixture_plan();
    const ordered_fixture fixture;
    const record_storage records = build_records(plan, fixture.view());
    require(plan.feature_block_geometry_identity() != 0u,
        "feature-block geometry identity is zero");
    require(records.view.feature_block_geometry_identity
            == plan.feature_block_geometry_identity(),
        "record geometry identity was not preserved");
    require(records.row_offsets == std::vector<u32>({0u, 2u, 2u, 5u, 8u}),
        "row-to-record offsets are wrong");
    require(records.blocks == std::vector<u32>({0u, 1u, 0u, 1u, 2u, 0u, 1u, 2u}),
        "record block ids are wrong");
    require(records.masks == std::vector<u32>({1u, 3u, 2u, 4u, 1u, 3u, 7u, 1u}),
        "record gene masks are wrong");
    require(records.value_offsets == std::vector<u32>({0u, 1u, 3u, 4u, 5u, 6u, 8u, 11u, 12u}),
        "record-to-value offsets are wrong");
    require(records.values.size() == fixture.values.size() * sizeof(u64)
            && std::equal(records.values.begin(), records.values.end(),
                reinterpret_cast<const unsigned char *>(fixture.values.data())),
        "compact value bytes changed");

    std::vector<u32> decoded_rows(fixture.row_offsets.size());
    std::vector<u32> decoded_features(fixture.features.size());
    std::vector<u64> decoded_values(fixture.values.size());
    cellpack::cell_block_decode_buffers buffers;
    buffers.row_offset_capacity = decoded_rows.size();
    buffers.entry_capacity = decoded_features.size();
    buffers.value_capacity_bytes = decoded_values.size() * sizeof(u64);
    buffers.row_offsets = decoded_rows.data();
    buffers.canonical_feature_ids = decoded_features.data();
    buffers.values = decoded_values.data();
    cellpack::decoded_cell_block_partition_view decoded;
    require_status(cellpack::decode_cell_block_records_host(
        plan, records.view, buffers, &decoded), "decode records");
    require(decoded_rows == fixture.row_offsets, "decoded row offsets changed");
    require(decoded_features == fixture.features, "decoded canonical features changed");
    require(decoded_values == fixture.values, "decoded values changed");
    require(decoded.global_row_begin == fixture.view().global_row_begin
            && decoded.row_domain_identity == fixture.view().row_domain_identity,
        "decoded partition identity changed");
}

void test_empty_and_maximum_width() {
    const cellpack::frozen_packing_plan plan = make_fixture_plan();
    const u32 empty_offsets[] = {0u, 0u, 0u};
    cellpack::ordered_plan_partition_view empty;
    empty.semantic_plan_schema_version = cellpack::packing_plan_semantic_schema_version;
    empty.global_row_begin = 3u;
    empty.full_row_count = 8u;
    empty.row_count = 2u;
    empty.feature_count = 6u;
    empty.value_size_bytes = 3u;
    empty.feature_axis_fingerprint = 0x12345678u;
    empty.feature_axis_fingerprint_version = 1u;
    empty.row_domain_identity = 0xabcdu;
    empty.row_offsets = empty_offsets;
    const record_storage empty_records = build_records(plan, empty);
    require(empty_records.row_offsets == std::vector<u32>({0u, 0u, 0u})
            && empty_records.value_offsets == std::vector<u32>({0u}),
        "all-empty record offsets are wrong");
    u32 decoded_rows[] = {99u, 99u, 99u};
    cellpack::cell_block_decode_buffers decode_buffers;
    decode_buffers.row_offset_capacity = 3u;
    decode_buffers.row_offsets = decoded_rows;
    cellpack::decoded_cell_block_partition_view decoded;
    require_status(cellpack::decode_cell_block_records_host(
        plan, empty_records.view, decode_buffers, &decoded), "decode empty records");
    require(decoded_rows[0] == 0u && decoded_rows[1] == 0u && decoded_rows[2] == 0u,
        "decoded all-empty row offsets are wrong");

    std::vector<u32> permutation(32u);
    for (u32 feature = 0u; feature < permutation.size(); ++feature) permutation[feature] = feature;
    const cellpack::frozen_packing_plan width32 = make_plan(permutation, {0u, 32u}, 32u, 1u);
    const u32 row_offsets[] = {0u, 1u}, block = 0u, local = 31u, canonical = 31u;
    const unsigned char value[] = {0xa5u, 0x5au, 0x11u};
    cellpack::ordered_plan_partition_view source;
    source.semantic_plan_schema_version = cellpack::packing_plan_semantic_schema_version;
    source.full_row_count = 1u;
    source.row_count = 1u;
    source.feature_count = 32u;
    source.nnz_count = 1u;
    source.value_size_bytes = 3u;
    source.feature_axis_fingerprint = 0x12345678u;
    source.feature_axis_fingerprint_version = 1u;
    source.row_domain_identity = 0xabcdu;
    source.row_offsets = row_offsets;
    source.block_ids = &block;
    source.local_feature_ids = &local;
    source.canonical_feature_ids = &canonical;
    source.values = value;
    const record_storage maximum = build_records(width32, source);
    require(maximum.masks == std::vector<u32>({0x80000000u}),
        "maximum-width high mask bit was lost");

    std::vector<u32> permutation33(33u);
    for (u32 feature = 0u; feature < permutation33.size(); ++feature) permutation33[feature] = feature;
    const cellpack::frozen_packing_plan width33 = make_plan(permutation33, {0u, 33u}, 33u, 1u);
    source.feature_count = 33u;
    cellpack::cell_block_record_requirements ignored;
    require(!cellpack::query_cell_block_record_requirements_host(width33, source, &ignored),
        "record v1 accepted a feature block wider than 32");
}

void test_validation_failures() {
    const cellpack::frozen_packing_plan plan = make_fixture_plan();
    const ordered_fixture fixture;
    cellpack::ordered_plan_partition_view invalid_source = fixture.view();
    std::vector<u32> bad_locals = fixture.locals;
    bad_locals[1] = bad_locals[2];
    invalid_source.local_feature_ids = bad_locals.data();
    cellpack::cell_block_record_requirements requirements;
    require(!cellpack::query_cell_block_record_requirements_host(
        plan, invalid_source, &requirements), "duplicate block/local coordinate was accepted");
    std::vector<u32> bad_features = fixture.features;
    bad_features[0] = 1u;
    invalid_source = fixture.view();
    invalid_source.canonical_feature_ids = bad_features.data();
    require(!cellpack::query_cell_block_record_requirements_host(
        plan, invalid_source, &requirements), "canonical/block geometry mismatch was accepted");

    require_status(cellpack::query_cell_block_record_requirements_host(
        plan, fixture.view(), &requirements), "query valid requirements");
    std::vector<u32> row_offsets(requirements.row_record_offset_count);
    std::vector<u32> blocks(requirements.record_count), masks(requirements.record_count);
    std::vector<u32> value_offsets(requirements.record_value_offset_count);
    std::vector<unsigned char> values(requirements.value_bytes);
    cellpack::cell_block_record_buffers short_buffers;
    short_buffers.row_record_offset_capacity = row_offsets.size();
    short_buffers.record_capacity = blocks.size() - 1u;
    short_buffers.record_value_offset_capacity = value_offsets.size();
    short_buffers.value_capacity_bytes = values.size();
    short_buffers.row_record_offsets = row_offsets.data();
    short_buffers.record_block_ids = blocks.data();
    short_buffers.record_gene_masks = masks.data();
    short_buffers.record_value_offsets = value_offsets.data();
    short_buffers.values = values.data();
    cellpack::cell_block_record_view ignored_view;
    require(!cellpack::build_cell_block_records_host(
        plan, fixture.view(), short_buffers, &ignored_view),
        "insufficient record capacity was accepted");

    const record_storage valid = build_records(plan, fixture.view());
    cellpack::cell_block_record_view bad_view = valid.view;
    bad_view.feature_block_geometry_identity ^= 1u;
    require(!cellpack::validate_cell_block_record_view_host(plan, bad_view),
        "wrong feature-block geometry identity was accepted");
    std::vector<u32> bad_masks = valid.masks;
    bad_masks[0] = 4u;
    bad_view = valid.view;
    bad_view.record_gene_masks = bad_masks.data();
    require(!cellpack::validate_cell_block_record_view_host(plan, bad_view),
        "gene mask outside block width was accepted");
    std::vector<u32> bad_value_offsets = valid.value_offsets;
    ++bad_value_offsets[1];
    bad_view = valid.view;
    bad_view.record_value_offsets = bad_value_offsets.data();
    require(!cellpack::validate_cell_block_record_view_host(plan, bad_view),
        "mask/value rank mismatch was accepted");

    std::vector<u32> bad_row_offsets = valid.row_offsets;
    bad_row_offsets[2] = bad_row_offsets[1] - 1u;
    bad_view = valid.view;
    bad_view.row_record_offsets = bad_row_offsets.data();
    require(!cellpack::validate_cell_block_record_view_host(plan, bad_view),
        "nonmonotonic row-to-record offsets were accepted");

    bad_value_offsets = valid.value_offsets;
    --bad_value_offsets.back();
    bad_view = valid.view;
    bad_view.record_value_offsets = bad_value_offsets.data();
    require(!cellpack::validate_cell_block_record_view_host(plan, bad_view),
        "wrong terminal value offset was accepted");

    std::vector<u32> bad_blocks = valid.blocks;
    bad_blocks[1] = bad_blocks[0];
    bad_view = valid.view;
    bad_view.record_block_ids = bad_blocks.data();
    require(!cellpack::validate_cell_block_record_view_host(plan, bad_view),
        "duplicate row block id was accepted");

    const cellpack::frozen_packing_plan other = make_plan(
        {1u, 3u, 5u, 0u, 4u, 2u}, {0u, 2u, 5u, 6u}, 3u);
    require(other.feature_block_geometry_identity() != plan.feature_block_geometry_identity(),
        "different feature-block geometry produced the same test identity");
    require(!cellpack::validate_cell_block_record_view_host(other, valid.view),
        "records were accepted against a different frozen geometry");

    std::vector<u32> decoded_rows(fixture.row_offsets.size());
    std::vector<u32> decoded_features(fixture.features.size() - 1u);
    std::vector<u64> decoded_values(fixture.values.size());
    cellpack::cell_block_decode_buffers short_decode;
    short_decode.row_offset_capacity = decoded_rows.size();
    short_decode.entry_capacity = decoded_features.size();
    short_decode.value_capacity_bytes = decoded_values.size() * sizeof(u64);
    short_decode.row_offsets = decoded_rows.data();
    short_decode.canonical_feature_ids = decoded_features.data();
    short_decode.values = decoded_values.data();
    cellpack::decoded_cell_block_partition_view decoded;
    require(!cellpack::decode_cell_block_records_host(
        plan, valid.view, short_decode, &decoded),
        "insufficient decode capacity was accepted");
}

} // namespace

int main() {
    test_exact_host_contract();
    test_empty_and_maximum_width();
    test_validation_failures();
    return 0;
}
