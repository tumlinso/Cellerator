#include <Cellerator/geometry/tile_statistical_validation.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

// One integrated adversarial fixture keeps plan, source, record, order, tile,
// split, null, and bootstrap identities visible in every rejection test. The
// file is intentionally larger than the usual helper threshold for that audit.
namespace {

using cellpack::u32;
using cellpack::u64;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void require_status(cellpack::validation_result status, const char *message) {
    if (!status) throw std::runtime_error(message);
}

void require_rejected(cellpack::validation_result status, const char *message) {
    if (status) throw std::runtime_error(message);
}

void require_close(double actual, double expected, const char *message) {
    if (std::fabs(actual - expected) > 1.0e-12) throw std::runtime_error(message);
}

cellpack::frozen_packing_plan make_plan(u64 row_domain = 0x726f772d646f6d31ull) {
    std::vector<u32> permutation(36u), inverse(36u), feature_to_block(36u),
        feature_to_local(36u);
    for (u32 feature = 0u; feature < 36u; ++feature) {
        permutation[feature] = feature;
        inverse[feature] = feature;
    }
    const u32 block_offsets[] = {0u, 32u, 34u, 36u};
    for (u32 block = 0u; block < 3u; ++block) {
        for (u32 feature = block_offsets[block];
             feature < block_offsets[block + 1u]; ++feature) {
            feature_to_block[feature] = block;
            feature_to_local[feature] = feature - block_offsets[block];
        }
    }
    const u32 row_offsets[] = {0u, 8u};
    cellpack::frozen_packing_plan_build_view build;
    build.row_count = 8u;
    build.feature_count = 36u;
    build.feature_permutation = permutation.data();
    build.inverse_feature_permutation = inverse.data();
    build.feature_block_count = 3u;
    build.feature_block_offsets = block_offsets;
    build.feature_to_block = feature_to_block.data();
    build.feature_to_local = feature_to_local.data();
    build.row_group_count = 1u;
    build.row_group_offsets = row_offsets;
    build.maximum_feature_block_width = 32u;
    build.row_group_width = 8u;
    build.identity.feature_axis_fingerprint = 0x6665617475726531ull;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind =
        cellpack::packing_row_domain_kind::full_dataset_identity;
    build.identity.row_domain_identity = row_domain;
    build.identity.evaluation_source_identity = 0x6576616c75617465ull;
    build.cost_policy_identity = 0x636f73742d763031ull;
    cellpack::frozen_packing_plan result;
    require_status(cellpack::freeze_packing_plan(build, &result), "freeze plan");
    return result;
}

struct canonical_matrix {
    u64 dataset_identity = 0u;
    std::vector<u32> row_offsets;
    std::vector<u32> features;
    std::vector<unsigned char> values;

    cellpack::csr_support_view support() const {
        return {8u, 36u, static_cast<u32>(features.size()), row_offsets.data(),
            features.empty() ? nullptr : features.data()};
    }

    cellpack::record_validation_source_view validation_view() const {
        cellpack::record_validation_source_view result;
        result.dataset_identity = dataset_identity;
        result.full_row_count = 8u;
        result.support = support();
        result.value_size_bytes = 3u;
        result.values = values.empty() ? nullptr : values.data();
        return result;
    }
};

canonical_matrix make_matrix(bool empty = false) {
    canonical_matrix result;
    result.dataset_identity = 0x646174612d726561ull;
    const std::vector<std::vector<u32>> rows = empty
        ? std::vector<std::vector<u32>>(8u)
        : std::vector<std::vector<u32>>{{0u, 31u, 32u}, {0u, 1u, 32u},
            {0u, 1u, 33u}, {2u, 3u, 34u}, {2u, 3u, 34u},
            {4u, 5u, 35u}, {}, {31u, 33u, 35u}};
    result.row_offsets.push_back(0u);
    for (u32 row = 0u; row < rows.size(); ++row) {
        for (u32 feature : rows[row]) {
            result.features.push_back(feature);
            for (u32 byte = 0u; byte < 3u; ++byte) {
                result.values.push_back(static_cast<unsigned char>(
                    (row * 61u + feature * 17u + byte * 29u) & 0xffu));
            }
        }
        result.row_offsets.push_back(static_cast<u32>(result.features.size()));
    }
    return result;
}

struct ordered_storage {
    std::vector<u32> blocks, locals;
    cellpack::ordered_plan_partition_view metadata{};

    cellpack::ordered_plan_partition_view view(const canonical_matrix &matrix) const {
        auto result = metadata;
        result.row_offsets = matrix.row_offsets.data();
        result.block_ids = blocks.empty() ? nullptr : blocks.data();
        result.local_feature_ids = locals.empty() ? nullptr : locals.data();
        result.canonical_feature_ids = matrix.features.empty()
            ? nullptr : matrix.features.data();
        result.values = matrix.values.empty() ? nullptr : matrix.values.data();
        return result;
    }
};

ordered_storage make_ordered(
    const cellpack::frozen_packing_plan &plan,
    const canonical_matrix &matrix) {
    ordered_storage result;
    for (u32 feature : matrix.features) {
        const u32 block = plan.feature_to_block()[feature];
        result.blocks.push_back(block);
        result.locals.push_back(plan.feature_to_local()[feature]);
    }
    result.metadata.semantic_plan_schema_version =
        cellpack::packing_plan_semantic_schema_version;
    result.metadata.full_row_count = 8u;
    result.metadata.row_count = 8u;
    result.metadata.feature_count = 36u;
    result.metadata.nnz_count = static_cast<u32>(matrix.features.size());
    result.metadata.value_size_bytes = 3u;
    result.metadata.feature_axis_fingerprint =
        plan.identity().feature_axis_fingerprint;
    result.metadata.feature_axis_fingerprint_version =
        plan.identity().feature_axis_fingerprint_version;
    result.metadata.row_domain_identity = plan.identity().row_domain_identity;
    return result;
}

struct record_storage {
    std::vector<u32> rows, blocks, masks, values_offsets;
    std::vector<unsigned char> values;
    cellpack::cell_block_record_view metadata{};

    cellpack::cell_block_record_view view() const {
        auto result = metadata;
        result.row_record_offsets = rows.data();
        result.record_block_ids = blocks.empty() ? nullptr : blocks.data();
        result.record_gene_masks = masks.empty() ? nullptr : masks.data();
        result.record_value_offsets = values_offsets.data();
        result.values = values.empty() ? nullptr : values.data();
        return result;
    }
};

record_storage make_records(
    const cellpack::frozen_packing_plan &plan,
    const canonical_matrix &matrix) {
    const auto ordered = make_ordered(plan, matrix);
    const auto source = ordered.view(matrix);
    cellpack::cell_block_record_requirements required;
    require_status(cellpack::query_cell_block_record_requirements_host(
        plan, source, &required), "query records");
    record_storage result;
    result.rows.resize(required.row_record_offset_count);
    result.blocks.resize(required.record_count);
    result.masks.resize(required.record_count);
    result.values_offsets.resize(required.record_value_offset_count);
    result.values.resize(required.value_bytes);
    cellpack::cell_block_record_buffers buffers;
    buffers.row_record_offset_capacity = result.rows.size();
    buffers.record_capacity = result.blocks.size();
    buffers.record_value_offset_capacity = result.values_offsets.size();
    buffers.value_capacity_bytes = result.values.size();
    buffers.row_record_offsets = result.rows.data();
    buffers.record_block_ids = result.blocks.data();
    buffers.record_gene_masks = result.masks.data();
    buffers.record_value_offsets = result.values_offsets.data();
    buffers.values = result.values.data();
    require_status(cellpack::build_cell_block_records_host(
        plan, source, buffers, &result.metadata), "build records");
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

order_storage make_local_order(const cellpack::cell_block_record_view &records) {
    order_storage result;
    result.primary.resize(8u);
    result.secondary.resize(8u);
    result.active.resize(8u);
    result.nnz.resize(8u);
    result.permutation.resize(8u);
    result.inverse.resize(8u);
    cellpack::local_cell_order_buffers buffers;
    buffers.row_capacity = 8u;
    buffers.primary_keys = result.primary.data();
    buffers.secondary_keys = result.secondary.data();
    buffers.active_block_counts = result.active.data();
    buffers.row_nnz_counts = result.nnz.data();
    buffers.row_permutation = result.permutation.data();
    buffers.inverse_row_permutation = result.inverse.data();
    cellpack::local_cell_order_config config;
    config.kind = cellpack::local_cell_order_kind::deterministic_random;
    config.window_size = 8u;
    config.group_width = 4u;
    config.seed = 0x6f726465722d7631ull;
    require_status(cellpack::build_local_cell_order_host(
        records, config, buffers, &result.metadata), "build local order");
    return result;
}

struct tile_storage {
    std::vector<u32> tile_offsets, blocks, cell_masks, entry_offsets, gene_masks,
        value_offsets;
    std::vector<unsigned char> values;
    cellpack::warp_tile_view metadata{};

    cellpack::warp_tile_view view() const {
        auto result = metadata;
        result.tile_block_offsets = tile_offsets.data();
        result.tile_block_ids = blocks.empty() ? nullptr : blocks.data();
        result.tile_block_cell_masks = cell_masks.empty() ? nullptr : cell_masks.data();
        result.block_row_entry_offsets = entry_offsets.data();
        result.row_block_gene_masks = gene_masks.empty() ? nullptr : gene_masks.data();
        result.row_block_value_offsets = value_offsets.data();
        result.values = values.empty() ? nullptr : values.data();
        return result;
    }
};

tile_storage make_tiles(
    const cellpack::frozen_packing_plan &plan,
    const cellpack::cell_block_record_view &records,
    const cellpack::local_cell_order_view &order) {
    cellpack::warp_tile_requirements required;
    require_status(cellpack::query_warp_tile_requirements_host(
        plan, records, order, &required), "query tiles");
    tile_storage result;
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

struct packed_fixture {
    canonical_matrix matrix;
    record_storage records;
    order_storage order;
    tile_storage tiles;

    explicit packed_fixture(const cellpack::frozen_packing_plan &plan, bool empty = false)
        : matrix(make_matrix(empty)), records(make_records(plan, matrix)),
          order(make_local_order(records.view())),
          tiles(make_tiles(plan, records.view(), order.view())) {}
};

struct split_fixture {
    std::vector<u64> rows{101u, 102u, 103u, 104u, 105u, 106u, 107u, 108u};
    std::vector<u64> groups{11u, 11u, 12u, 12u, 13u, 13u, 14u, 14u};
    std::vector<cellpack::validation_partition> partitions{8u};
    cellpack::validation_split_provenance provenance{};
    bool grouped = true;

    explicit split_fixture(bool use_groups = true) : grouped(use_groups) {
        require_status(cellpack::build_validation_split(identities(), {0x53504c4954u, 2u},
            {partitions.size(), partitions.data()}, &provenance), "build split");
    }

    cellpack::validation_identity_view identities() const {
        return {8u, rows.data(), grouped ? groups.data() : nullptr};
    }

    cellpack::record_validation_context context(
        const cellpack::frozen_packing_plan &plan) const {
        cellpack::record_validation_context result;
        result.feature_axis_identity = plan.identity().feature_axis_fingerprint;
        result.feature_axis_identity_version =
            plan.identity().feature_axis_fingerprint_version;
        result.row_domain_identity = plan.identity().row_domain_identity;
        result.plan_training_split_identity = provenance.assignment_identity;
        result.identities = identities();
        result.row_partitions = partitions.data();
        result.split_provenance = provenance;
        return result;
    }
};

canonical_matrix make_null(
    const canonical_matrix &real,
    cellpack::degree_preserving_null_provenance *provenance) {
    canonical_matrix result;
    result.row_offsets.resize(9u);
    result.features.resize(real.features.size());
    cellpack::csr_support_view output;
    require_status(cellpack::build_degree_preserving_null_reference(real.support(),
        {0x4e554c4c53454544ull, real.dataset_identity, 8u, 8192u},
        {result.row_offsets.size(), result.features.size(), result.row_offsets.data(),
            result.features.data()}, &output, provenance), "build null");
    require(provenance->target_reached, "null did not reach swap target");
    result.dataset_identity = provenance->output_identity;
    for (u32 row = 0u; row < 8u; ++row) {
        for (u32 entry = result.row_offsets[row]; entry < result.row_offsets[row + 1u];
             ++entry) {
            for (u32 byte = 0u; byte < 3u; ++byte) {
                result.values.push_back(static_cast<unsigned char>(
                    (row * 43u + result.features[entry] * 23u + byte * 31u) & 0xffu));
            }
        }
    }
    return result;
}

void test_held_out_group_scope_exactness_and_null() {
    const auto plan = make_plan();
    const packed_fixture real(plan);
    require(real.order.permutation != std::vector<u32>({0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u}),
        "test did not exercise nonidentity local order");
    require((real.records.masks[0] & (1u << 31u)) != 0u,
        "test did not exercise feature-block bit 31");
    const split_fixture split(true);
    const auto context = split.context(plan);
    cellpack::held_out_tile_validation held_out;
    require_status(cellpack::evaluate_held_out_warp_tiles(plan, context,
        real.matrix.validation_view(), real.records.view(), real.order.view(),
        real.tiles.view(), &held_out), "evaluate held-out tiles");
    require(held_out.claims_group_generalization
            && held_out.unit_kind == cellpack::validation_unit_kind::caller_group_identity,
        "group-aware held-out result lost scope");
    require(held_out.metrics.row_count == split.provenance.held_out_row_count
            && held_out.metrics.correctness_mismatches == 0u
            && held_out.metrics.tile_count != 0u
            && held_out.metrics.tile_block_union_references != 0u
            && held_out.metrics.active_block_references != 0u,
        "held-out tile raw metrics are incomplete");
    require(held_out.metrics.runtime_repeat_count == 0u
            && (held_out.metrics.available & cellpack::packing_validation_metric_runtime) == 0u,
        "host tile validation fabricated runtime evidence");

    cellpack::degree_preserving_null_provenance null_provenance;
    const canonical_matrix null_matrix = make_null(real.matrix, &null_provenance);
    const record_storage null_records = make_records(plan, null_matrix);
    const order_storage null_order = make_local_order(null_records.view());
    const tile_storage null_tiles = make_tiles(plan, null_records.view(), null_order.view());
    cellpack::held_out_tile_null_comparison comparison;
    require_status(cellpack::compare_held_out_warp_tiles_to_degree_null(plan, context,
        real.matrix.validation_view(), real.records.view(), real.order.view(),
        real.tiles.view(), null_matrix.validation_view(), null_records.view(),
        null_order.view(), null_tiles.view(), null_provenance, &comparison),
        "compare real and null tiles");
    require(comparison.exact_degree_conservation
            && comparison.real.metrics.nnz_count
                == comparison.degree_preserving_null.metrics.nnz_count
            && comparison.real.held_out_row_identity
                == comparison.degree_preserving_null.held_out_row_identity,
        "real/null tile comparison lost exact degrees or held-out rows");
}

struct bootstrap_fixture {
    std::vector<u32> multiplicities;
    std::vector<u32> materialized_rows;
    cellpack::validation_bootstrap_provenance provenance{};
    cellpack::bootstrap_tile_realization_view realization{};
};

bootstrap_fixture make_bootstrap(
    const split_fixture &split,
    const order_storage &order,
    u64 seed,
    u32 draws) {
    bootstrap_fixture result;
    result.multiplicities.resize(8u);
    require_status(cellpack::build_validation_bootstrap(split.identities(),
        {seed, draws}, {result.multiplicities.size(), result.multiplicities.data()},
        &result.provenance), "build bootstrap");
    u32 maximum = 0u;
    for (u32 count : result.multiplicities) maximum = std::max(maximum, count);
    for (u32 layer = 0u; layer < maximum; ++layer) {
        for (u32 execution = 0u; execution < 8u; ++execution) {
            const u32 row = order.permutation[execution];
            if (result.multiplicities[row] > layer) result.materialized_rows.push_back(row);
        }
    }
    require(result.materialized_rows.size() == result.provenance.materialized_row_count,
        "bootstrap materialization count mismatch");
    result.realization.bootstrap_identity = result.provenance.bootstrap_identity;
    result.realization.materialized_row_count = result.materialized_rows.size();
    result.realization.global_row_indices = result.materialized_rows.data();
    result.realization.realization_identity =
        cellpack::bootstrap_tile_realization_identity(result.provenance,
            result.materialized_rows.data(), result.materialized_rows.size());
    return result;
}

cellpack::bootstrap_tile_replicate_input make_input(
    const bootstrap_fixture &bootstrap,
    const packed_fixture &packed,
    const cellpack::record_validation_source_view *source,
    const cellpack::cell_block_record_view *records,
    const cellpack::local_cell_order_view *order,
    const cellpack::warp_tile_view *tiles) {
    cellpack::bootstrap_tile_replicate_input result;
    result.bootstrap_provenance = &bootstrap.provenance;
    result.row_multiplicities = bootstrap.multiplicities.data();
    result.source = source;
    result.records = records;
    result.order = order;
    result.tiles = tiles;
    result.realization = bootstrap.realization;
    (void)packed;
    return result;
}

void test_bootstrap_repeats_and_deterministic_summary() {
    const auto plan = make_plan();
    const packed_fixture packed(plan);
    const split_fixture split(true);
    const auto context = split.context(plan);
    std::vector<bootstrap_fixture> bootstraps;
    bootstraps.push_back(make_bootstrap(split, packed.order, 1001u, 7u));
    bootstraps.push_back(make_bootstrap(split, packed.order, 1002u, 9u));
    bootstraps.push_back(make_bootstrap(split, packed.order, 1003u, 11u));
    bool repeated = false;
    for (const auto &bootstrap : bootstraps) {
        repeated = repeated || std::any_of(bootstrap.multiplicities.begin(),
            bootstrap.multiplicities.end(), [](u32 count) { return count > 1u; });
    }
    require(repeated, "bootstrap tests did not repeat any source row");
    const auto source = packed.matrix.validation_view();
    const auto records = packed.records.view();
    const auto order = packed.order.view();
    const auto tiles = packed.tiles.view();
    std::vector<cellpack::bootstrap_tile_replicate_input> inputs;
    for (const auto &bootstrap : bootstraps) {
        inputs.push_back(make_input(bootstrap, packed, &source, &records, &order, &tiles));
    }
    std::vector<cellpack::bootstrap_tile_replicate_validation> raw_a(3u), raw_b(3u);
    cellpack::bootstrap_tile_stability_summary summary_a, summary_b;
    require_status(cellpack::evaluate_bootstrap_warp_tile_stability(plan, context,
        inputs.data(), inputs.size(), {raw_a.size(), raw_a.data()}, &summary_a),
        "evaluate bootstrap tile stability");
    require_status(cellpack::evaluate_bootstrap_warp_tile_stability(plan, context,
        inputs.data(), inputs.size(), {raw_b.size(), raw_b.data()}, &summary_b),
        "repeat bootstrap tile stability");
    require(summary_a.repeat_count == 3u
            && summary_a.encoded_bytes.observation_count == 3u
            && summary_a.encoded_bytes_per_nnz.observation_count == 3u
            && summary_a.encoded_bytes.minimum <= summary_a.encoded_bytes.mean
            && summary_a.encoded_bytes.mean <= summary_a.encoded_bytes.maximum
            && summary_a.encoded_bytes.sample_standard_deviation >= 0.0,
        "bootstrap summary omitted deterministic repeat statistics");
    double expected_mean = 0.0;
    for (const auto &replicate : raw_a) {
        expected_mean += static_cast<double>(replicate.metrics.encoded_bytes);
    }
    expected_mean /= static_cast<double>(raw_a.size());
    double squared_delta = 0.0;
    for (const auto &replicate : raw_a) {
        const double delta = static_cast<double>(replicate.metrics.encoded_bytes)
            - expected_mean;
        squared_delta += delta * delta;
    }
    const double expected_sample_sd = std::sqrt(
        squared_delta / static_cast<double>(raw_a.size() - 1u));
    require_close(summary_a.encoded_bytes.mean, expected_mean,
        "bootstrap mean arithmetic is wrong");
    require_close(summary_a.encoded_bytes.sample_standard_deviation,
        expected_sample_sd, "bootstrap sample standard deviation is wrong");
    require_close(summary_a.encoded_bytes.mean, summary_b.encoded_bytes.mean,
        "bootstrap summary is not deterministic");
    require_close(summary_a.padding_slots.sample_standard_deviation,
        summary_b.padding_slots.sample_standard_deviation,
        "bootstrap sample standard deviation is not deterministic");
    for (u32 index = 0u; index < raw_a.size(); ++index) {
        require(raw_a[index].bootstrap_identity == bootstraps[index].provenance.bootstrap_identity
                && raw_a[index].metrics.row_count
                    == bootstraps[index].provenance.materialized_row_count
                && raw_a[index].metrics.correctness_mismatches == 0u,
            "bootstrap raw replicate was not preserved");
    }
}

void test_cell_level_zero_nnz_and_tail_tiles() {
    const auto plan = make_plan();
    const packed_fixture empty(plan, true);
    const split_fixture split(false);
    const auto context = split.context(plan);
    cellpack::held_out_tile_validation held_out;
    require_status(cellpack::evaluate_held_out_warp_tiles(plan, context,
        empty.matrix.validation_view(), empty.records.view(), empty.order.view(),
        empty.tiles.view(), &held_out), "evaluate empty held-out tiles");
    require(!held_out.claims_group_generalization
            && held_out.metrics.nnz_count == 0u
            && held_out.metrics.tile_block_union_references == 0u
            && held_out.metrics.padding_slots == 0u
            && held_out.metrics.encoded_bytes == held_out.metrics.metadata_bytes
            && (held_out.metrics.available & cellpack::packing_validation_metric_storage) == 0u,
        "zero-NNZ tile metrics fabricated a denominator or group claim");

    std::vector<bootstrap_fixture> bootstraps;
    bootstraps.push_back(make_bootstrap(split, empty.order, 2001u, 5u));
    bootstraps.push_back(make_bootstrap(split, empty.order, 2002u, 6u));
    const auto source = empty.matrix.validation_view();
    const auto records = empty.records.view();
    const auto order = empty.order.view();
    const auto tiles = empty.tiles.view();
    std::vector<cellpack::bootstrap_tile_replicate_input> inputs;
    for (const auto &bootstrap : bootstraps) {
        inputs.push_back(make_input(bootstrap, empty, &source, &records, &order, &tiles));
    }
    std::vector<cellpack::bootstrap_tile_replicate_validation> raw(2u);
    cellpack::bootstrap_tile_stability_summary summary;
    require_status(cellpack::evaluate_bootstrap_warp_tile_stability(plan, context,
        inputs.data(), inputs.size(), {raw.size(), raw.data()}, &summary),
        "evaluate empty bootstrap tiles");
    require(summary.encoded_bytes_per_nnz.observation_count == 0u
            && summary.metadata_bytes_per_nnz.observation_count == 0u
            && summary.padding_slots_per_nnz.observation_count == 0u
            && summary.tile_count.observation_count == 2u
            && raw[0].metrics.tile_count == 2u,
        "zero-denominator bootstrap summary or tail tile is wrong");
}

void test_identity_and_payload_tamper_rejection() {
    const auto plan = make_plan();
    packed_fixture packed(plan);
    split_fixture split(true);
    auto context = split.context(plan);
    cellpack::held_out_tile_validation ignored;

    auto bad_context = context;
    ++bad_context.feature_axis_identity;
    require_rejected(cellpack::evaluate_held_out_warp_tiles(plan, bad_context,
        packed.matrix.validation_view(), packed.records.view(), packed.order.view(),
        packed.tiles.view(), &ignored), "tampered feature-axis identity was accepted");
    bad_context = context;
    ++bad_context.row_domain_identity;
    require_rejected(cellpack::evaluate_held_out_warp_tiles(plan, bad_context,
        packed.matrix.validation_view(), packed.records.view(), packed.order.view(),
        packed.tiles.view(), &ignored), "tampered row-domain identity was accepted");

    std::vector<cellpack::validation_partition> leaked = split.partitions;
    leaked[1u] = leaked[0u] == cellpack::validation_partition::training
        ? cellpack::validation_partition::held_out
        : cellpack::validation_partition::training;
    bad_context = context;
    bad_context.row_partitions = leaked.data();
    require_rejected(cellpack::evaluate_held_out_warp_tiles(plan, bad_context,
        packed.matrix.validation_view(), packed.records.view(), packed.order.view(),
        packed.tiles.view(), &ignored), "group-overlapping split was accepted");

    auto bad_tiles = packed.tiles.view();
    ++bad_tiles.tile_identity;
    require_rejected(cellpack::evaluate_held_out_warp_tiles(plan, context,
        packed.matrix.validation_view(), packed.records.view(), packed.order.view(),
        bad_tiles, &ignored), "tampered tile identity was accepted");
    auto bad_order = packed.order.view();
    ++bad_order.ordering_identity;
    require_rejected(cellpack::evaluate_held_out_warp_tiles(plan, context,
        packed.matrix.validation_view(), packed.records.view(), bad_order,
        packed.tiles.view(), &ignored), "tampered order identity was accepted");
    auto bad_source = packed.matrix.validation_view();
    bad_source.dataset_identity = 0u;
    require_rejected(cellpack::evaluate_held_out_warp_tiles(plan, context, bad_source,
        packed.records.view(), packed.order.view(), packed.tiles.view(), &ignored),
        "missing source identity was accepted");
    canonical_matrix changed_source = packed.matrix;
    for (u32 row = 0u; row < 8u; ++row) {
        if (split.partitions[row] == cellpack::validation_partition::held_out
            && changed_source.row_offsets[row + 1u] != changed_source.row_offsets[row]) {
            changed_source.values[static_cast<std::size_t>(changed_source.row_offsets[row])
                * 3u] ^= 0xffu;
            break;
        }
    }
    require_rejected(cellpack::evaluate_held_out_warp_tiles(plan, context,
        changed_source.validation_view(), packed.records.view(), packed.order.view(),
        packed.tiles.view(), &ignored), "tampered source value bytes were accepted");
    require_rejected(cellpack::evaluate_held_out_warp_tiles(
        make_plan(0x77726f6e672d726full), context, packed.matrix.validation_view(),
        packed.records.view(), packed.order.view(), packed.tiles.view(), &ignored),
        "wrong frozen plan was accepted");

    const bootstrap_fixture bootstrap = make_bootstrap(split, packed.order, 3001u, 9u);
    const auto source = packed.matrix.validation_view();
    const auto records = packed.records.view();
    const auto order = packed.order.view();
    const auto tiles = packed.tiles.view();
    auto input = make_input(bootstrap, packed, &source, &records, &order, &tiles);
    cellpack::bootstrap_tile_replicate_validation raw;
    cellpack::bootstrap_tile_stability_summary summary;
    auto bad_realization = bootstrap.realization;
    ++bad_realization.realization_identity;
    input.realization = bad_realization;
    require_rejected(cellpack::evaluate_bootstrap_warp_tile_stability(plan, context,
        &input, 1u, {1u, &raw}, &summary),
        "tampered bootstrap realization identity was accepted");
    std::vector<u32> bad_multiplicities = bootstrap.multiplicities;
    ++bad_multiplicities[0];
    input = make_input(bootstrap, packed, &source, &records, &order, &tiles);
    input.row_multiplicities = bad_multiplicities.data();
    require_rejected(cellpack::evaluate_bootstrap_warp_tile_stability(plan, context,
        &input, 1u, {1u, &raw}, &summary),
        "tampered bootstrap multiplicity was accepted");

    cellpack::degree_preserving_null_provenance provenance;
    const canonical_matrix null_matrix = make_null(packed.matrix, &provenance);
    const record_storage null_records = make_records(plan, null_matrix);
    const order_storage null_order = make_local_order(null_records.view());
    const tile_storage null_tiles = make_tiles(plan, null_records.view(), null_order.view());
    ++provenance.output_identity;
    cellpack::held_out_tile_null_comparison comparison;
    require_rejected(cellpack::compare_held_out_warp_tiles_to_degree_null(plan, context,
        packed.matrix.validation_view(), packed.records.view(), packed.order.view(),
        packed.tiles.view(), null_matrix.validation_view(), null_records.view(),
        null_order.view(), null_tiles.view(), provenance, &comparison),
        "tampered null provenance was accepted");
}

} // namespace

int main() {
    test_held_out_group_scope_exactness_and_null();
    test_bootstrap_repeats_and_deterministic_summary();
    test_cell_level_zero_nnz_and_tail_tiles();
    test_identity_and_payload_tamper_rejection();
    return 0;
}
