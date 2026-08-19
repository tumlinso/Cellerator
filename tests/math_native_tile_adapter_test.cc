#include <Cellerator/compute/math/native_tile_view.hh>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace math = cellerator::compute::math;

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "math_native_tile_adapter_test: " << message << '\n';
    std::abort();
}

void require(math::physical_view_status status, const char *message) {
    require(static_cast<bool>(status), message);
}

struct fixture {
    std::vector<std::uint32_t> feature_blocks{0u, 3u, 5u};
    std::vector<std::uint32_t> feature_permutation{2u, 0u, 4u, 1u, 3u};
    std::vector<std::uint32_t> row_permutation{2u, 0u, 1u};
    std::vector<std::uint32_t> inverse_rows{1u, 2u, 0u};
    std::vector<std::uint32_t> tile_blocks{0u, 2u};
    std::vector<std::uint32_t> block_ids{0u, 1u};
    std::vector<std::uint32_t> cell_masks{0x5u, 0x6u};
    std::vector<std::uint32_t> block_entries{0u, 2u, 4u};
    std::vector<std::uint32_t> gene_masks{0x5u, 0x2u, 0x3u, 0x1u};
    std::vector<std::uint32_t> value_offsets{0u, 2u, 3u, 5u, 6u};
    std::vector<float> values{10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
    cellpack::feature_weighted_row_reduction_plan_view plan{};
    cellpack::local_cell_order_view order{};
    cellpack::warp_tile_view tiles{};

    fixture() {
        plan.semantic_plan_schema_version = cellpack::packing_plan_semantic_schema_version;
        plan.geometry_identity_version = cellpack::feature_block_geometry_identity_version;
        plan.feature_count = 5u;
        plan.feature_block_count = 2u;
        plan.feature_block_geometry_identity = 0x1234u;
        plan.feature_block_offsets = feature_blocks.data();
        plan.feature_permutation = feature_permutation.data();

        order.order_schema_version = cellpack::local_cell_order_schema_version;
        order.signature_algorithm_version = cellpack::local_cell_signature_algorithm_version;
        order.kind = cellpack::local_cell_order_kind::inferred_minhash;
        order.window_size = 3u;
        order.group_width = 3u;
        order.ordering_identity = 0x5678u;
        order.global_row_begin = 100u;
        order.full_row_count = 3u;
        order.row_count = 3u;
        order.feature_block_count = 2u;
        order.feature_block_geometry_identity = plan.feature_block_geometry_identity;
        order.row_domain_identity = 0x9abcu;
        order.row_permutation = row_permutation.data();
        order.inverse_row_permutation = inverse_rows.data();

        tiles.tile_schema_version = cellpack::warp_tile_schema_version;
        tiles.record_schema_version = cellpack::cell_block_record_schema_version;
        tiles.semantic_plan_schema_version = plan.semantic_plan_schema_version;
        tiles.geometry_identity_version = plan.geometry_identity_version;
        tiles.order_schema_version = order.order_schema_version;
        tiles.tile_identity = 0xdef0u;
        tiles.feature_block_geometry_identity = plan.feature_block_geometry_identity;
        tiles.ordering_identity = order.ordering_identity;
        tiles.global_row_begin = order.global_row_begin;
        tiles.full_row_count = order.full_row_count;
        tiles.row_count = order.row_count;
        tiles.feature_count = plan.feature_count;
        tiles.feature_block_count = plan.feature_block_count;
        tiles.tile_row_width = order.group_width;
        tiles.tile_count = 1u;
        tiles.nnz_count = 6u;
        tiles.tile_block_count = 2u;
        tiles.row_block_entry_count = 4u;
        tiles.value_size_bytes = sizeof(float);
        tiles.row_domain_identity = order.row_domain_identity;
        tiles.tile_block_offsets = tile_blocks.data();
        tiles.tile_block_ids = block_ids.data();
        tiles.tile_block_cell_masks = cell_masks.data();
        tiles.block_row_entry_offsets = block_entries.data();
        tiles.row_block_gene_masks = gene_masks.data();
        tiles.row_block_value_offsets = value_offsets.data();
        tiles.values = values.data();
    }

    cellpack::persistent_packing_payload_view payload() const {
        cellpack::persistent_packing_payload_view result;
        result.payload_schema_version = cellpack::persistent_packing_payload_schema_version;
        result.payload_kind = cellpack::persistent_packing_payload_kind;
        result.payload_identity = 0x4242u;
        result.plan = plan;
        result.order = order;
        result.tiles = tiles;
        return result;
    }
};

struct sidecars {
    std::vector<std::uint32_t> unions;
    std::vector<std::uint32_t> offsets;
    std::vector<math::native_tile_block_metrics> metrics;

    explicit sidecars(const math::native_tile_requirements &required)
        : unions(required.union_mask_count), offsets(required.packed_offset_count),
          metrics(required.block_metric_count) {}

    math::native_tile_buffers buffers() {
        return {unions.size(), unions.data(), offsets.size(), offsets.data(),
            metrics.size(), metrics.data()};
    }
};

void test_sidecars_and_aliasing() {
    fixture data;
    math::native_tile_requirements required;
    require(math::query_native_tile_requirements_host(
        data.plan, data.order, data.tiles, &required), "requirements query failed");
    require(required.union_mask_count == 2u && required.packed_offset_count == 3u
        && required.block_metric_count == 2u, "requirements counts changed");
    sidecars storage(required);
    math::native_tile_view view;
    require(math::build_native_tile_view_host(
        data.plan, data.order, data.tiles, storage.buffers(), &view), "adapter build failed");

    require(view.tiles.values == data.values.data(), "adapter copied compact values");
    require(view.tiles.tile_block_offsets == data.tile_blocks.data(),
        "adapter copied tile metadata");
    require(storage.unions == std::vector<std::uint32_t>({0x7u, 0x3u}),
        "union masks are incorrect");
    require(storage.offsets == std::vector<std::uint32_t>({0u, 3u, 6u}),
        "packed offsets are incorrect");
    require(view.dense_workload == 10u, "aggregate workload is incorrect");
    require(storage.metrics[0].active_rows == 2u
        && storage.metrics[0].active_features == 3u
        && storage.metrics[0].nnz == 3u
        && storage.metrics[0].dense_workload == 6u
        && std::abs(storage.metrics[0].density - 0.5) < 1.0e-12
        && std::abs(storage.metrics[0].feature_reuse - 1.0) < 1.0e-12,
        "first tile-block metrics are incorrect");
    require(std::abs(storage.metrics[1].density - 0.75) < 1.0e-12
        && std::abs(storage.metrics[1].feature_reuse - 1.5) < 1.0e-12,
        "second tile-block metrics are incorrect");
}

void test_exact_decode() {
    fixture data;
    math::native_tile_requirements required;
    require(math::query_native_tile_requirements_from_cpk1_host(
        data.payload(), &required), "CPK1 requirements query failed");
    sidecars storage(required);
    math::native_tile_view view;
    require(math::build_native_tile_view_from_cpk1_host(
        data.payload(), storage.buffers(), &view), "CPK1 adapter build failed");

    const std::uint32_t expected[6][4] = {
        {0u, 2u, 0u, 2u}, {0u, 2u, 2u, 4u}, {2u, 1u, 1u, 0u},
        {1u, 0u, 3u, 1u}, {1u, 0u, 4u, 3u}, {2u, 1u, 3u, 1u}};
    for (std::uint32_t index = 0u; index < data.values.size(); ++index) {
        math::native_tile_decoded_value decoded;
        require(math::decode_native_tile_value_host(view, index, &decoded),
            "packed value decode failed");
        require(decoded.execution_row == expected[index][0]
            && decoded.canonical_row == expected[index][1]
            && decoded.execution_feature == expected[index][2]
            && decoded.canonical_feature == expected[index][3],
            "packed value decoded to the wrong logical coordinate");
        require(decoded.global_row == 100u + decoded.canonical_row,
            "global row identity is incorrect");
        require(decoded.value == &data.values[index], "decoder did not alias source value");
    }
    math::native_tile_decoded_value decoded;
    require(!math::decode_native_tile_value_host(view, data.tiles.nnz_count, &decoded),
        "out-of-range packed value was accepted");
    storage.offsets[1] = 4u;
    require(!math::decode_native_tile_value_host(view, 1u, &decoded),
        "corrupt derived packed offsets were accepted");
}

void test_rejections() {
    fixture data;
    math::native_tile_requirements required;
    require(math::query_native_tile_requirements_host(
        data.plan, data.order, data.tiles, &required), "requirements query failed");
    sidecars storage(required);
    math::native_tile_view view;
    auto too_small = storage.buffers();
    too_small.packed_offset_capacity -= 1u;
    require(!math::build_native_tile_view_host(
        data.plan, data.order, data.tiles, too_small, &view),
        "undersized sidecar buffer was accepted");

    data.inverse_rows[2] = 2u;
    require(!math::query_native_tile_requirements_host(
        data.plan, data.order, data.tiles, &required),
        "non-invertible row order was accepted");
    data.inverse_rows[2] = 0u;
    data.gene_masks[0] = 0x8u;
    require(!math::query_native_tile_requirements_host(
        data.plan, data.order, data.tiles, &required),
        "out-of-block gene mask was accepted");
}

} // namespace

int main() {
    test_sidecars_and_aliasing();
    test_exact_decode();
    test_rejections();
    std::cout << "math_native_tile_adapter_test: ok\n";
    return 0;
}
