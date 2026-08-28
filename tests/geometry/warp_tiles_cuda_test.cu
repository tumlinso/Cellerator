#include "Cellerator/geometry/warp_tiles_cuda.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>

namespace cp = ::cellpack;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "cellPackWarpTilesCudaTest: %s\n", message);
        std::exit(1);
    }
}

void require_cuda(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "cellPackWarpTilesCudaTest: %s: %s\n",
            message, cudaGetErrorString(error));
        std::exit(1);
    }
}

void require_status(cp::validation_result status, const char *message) {
    if (!status) {
        std::fprintf(stderr, "cellPackWarpTilesCudaTest: %s: %s (index=%u)\n",
            message, status.message, status.index);
        std::exit(1);
    }
}

template <typename T>
struct device_array {
    T *data = nullptr;
    std::size_t size = 0u;

    device_array() = default;
    explicit device_array(std::size_t count) : size(count) {
        if (count != 0u) require_cuda(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc");
    }
    explicit device_array(const std::vector<T> &host) : device_array(host.size()) {
        if (!host.empty()) require_cuda(cudaMemcpy(data, host.data(), host.size() * sizeof(T),
            cudaMemcpyHostToDevice), "upload device array");
    }
    device_array(const device_array &) = delete;
    device_array &operator=(const device_array &) = delete;
    ~device_array() { if (data != nullptr) cudaFree(data); }

    std::vector<T> download() const {
        std::vector<T> host(size);
        if (size != 0u) require_cuda(cudaMemcpy(host.data(), data, size * sizeof(T),
            cudaMemcpyDeviceToHost), "download device array");
        return host;
    }
};

cp::frozen_packing_plan make_plan() {
    constexpr cp::u32 features = 64u, rows = 64u;
    std::vector<cp::u32> permutation(features), inverse(features), to_block(features),
        to_local(features);
    std::iota(permutation.begin(), permutation.end(), 0u);
    std::iota(inverse.begin(), inverse.end(), 0u);
    for (cp::u32 feature = 0u; feature < features; ++feature) {
        to_block[feature] = feature / 32u;
        to_local[feature] = feature % 32u;
    }
    const cp::u32 block_offsets[] = {0u, 32u, 64u};
    const cp::u32 row_offsets[] = {0u, rows};
    cp::frozen_packing_plan_build_view build;
    build.row_count = rows;
    build.feature_count = features;
    build.feature_permutation = permutation.data();
    build.inverse_feature_permutation = inverse.data();
    build.feature_block_count = 2u;
    build.feature_block_offsets = block_offsets;
    build.feature_to_block = to_block.data();
    build.feature_to_local = to_local.data();
    build.row_group_count = 1u;
    build.row_group_offsets = row_offsets;
    build.maximum_feature_block_width = 32u;
    build.row_group_width = rows;
    build.identity.feature_axis_fingerprint = 0x4350303856414c31ull;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind = cp::packing_row_domain_kind::full_dataset_identity;
    build.identity.row_domain_identity = 0x43503038524f5731ull;
    build.identity.evaluation_source_identity = 0x20260817u;
    build.cost_policy_identity = 0x43503033u;
    cp::frozen_packing_plan plan;
    require_status(cp::freeze_packing_plan(build, &plan), "freeze plan");
    return plan;
}

struct ordered_fixture {
    cp::u64 global_row_begin = 0u;
    cp::u32 rows = 0u, value_size = 0u;
    std::vector<cp::u32> row_offsets, blocks, locals, features;
    std::vector<unsigned char> values;

    cp::ordered_plan_partition_view view() const {
        cp::ordered_plan_partition_view result;
        result.semantic_plan_schema_version = cp::packing_plan_semantic_schema_version;
        result.global_row_begin = global_row_begin;
        result.full_row_count = 64u;
        result.row_count = rows;
        result.feature_count = 64u;
        result.nnz_count = static_cast<cp::u32>(features.size());
        result.value_size_bytes = value_size;
        result.feature_axis_fingerprint = 0x4350303856414c31ull;
        result.feature_axis_fingerprint_version = 1u;
        result.row_domain_identity = 0x43503038524f5731ull;
        result.row_offsets = row_offsets.data();
        result.block_ids = blocks.data();
        result.local_feature_ids = locals.data();
        result.canonical_feature_ids = features.data();
        result.values = values.data();
        return result;
    }
};

void append(ordered_fixture *fixture, cp::u32 block, cp::u32 local, cp::u32 tag) {
    fixture->blocks.push_back(block);
    fixture->locals.push_back(local);
    fixture->features.push_back(block * 32u + local);
    for (cp::u32 byte = 0u; byte < fixture->value_size; ++byte) {
        fixture->values.push_back(static_cast<unsigned char>((tag * 53u + byte * 17u) & 0xffu));
    }
}

ordered_fixture main_fixture(cp::u32 value_size) {
    ordered_fixture fixture;
    fixture.global_row_begin = 7u;
    fixture.rows = 34u;
    fixture.value_size = value_size;
    fixture.row_offsets.push_back(0u);
    for (cp::u32 row = 0u; row < fixture.rows; ++row) {
        if (row < 32u) {
            append(&fixture, 0u, row, row + 1u);
            if (row == 0u) {
                append(&fixture, 1u, 0u, 100u);
                append(&fixture, 1u, 1u, 101u);
            } else if (row == 1u) {
                append(&fixture, 1u, 31u, 102u);
            }
        } else if (row == 32u) {
            append(&fixture, 1u, 1u, 103u);
        }
        fixture.row_offsets.push_back(static_cast<cp::u32>(fixture.features.size()));
    }
    return fixture;
}

ordered_fixture empty_fixture(cp::u32 rows, cp::u32 value_size) {
    ordered_fixture fixture;
    fixture.global_row_begin = rows == 0u ? 64u : 0u;
    fixture.rows = rows;
    fixture.value_size = value_size;
    fixture.row_offsets.assign(static_cast<std::size_t>(rows) + 1u, 0u);
    return fixture;
}

struct host_records {
    std::vector<cp::u32> rows, blocks, masks, values_offsets;
    std::vector<unsigned char> values;
    cp::cell_block_record_view metadata{};
    cp::cell_block_record_view view() const {
        auto result = metadata;
        result.row_record_offsets = rows.data();
        result.record_block_ids = blocks.data();
        result.record_gene_masks = masks.data();
        result.record_value_offsets = values_offsets.data();
        result.values = values.data();
        return result;
    }
};

host_records build_records(const cp::frozen_packing_plan &plan, const ordered_fixture &source) {
    cp::cell_block_record_requirements required;
    require_status(cp::query_cell_block_record_requirements_host(plan, source.view(), &required),
        "query records");
    host_records result;
    result.rows.resize(required.row_record_offset_count);
    result.blocks.resize(required.record_count);
    result.masks.resize(required.record_count);
    result.values_offsets.resize(required.record_value_offset_count);
    result.values.resize(required.value_bytes);
    cp::cell_block_record_buffers buffers{result.rows.size(), result.blocks.size(),
        result.values_offsets.size(), result.values.size(), result.rows.data(),
        result.blocks.data(), result.masks.data(), result.values_offsets.data(),
        result.values.data()};
    require_status(cp::build_cell_block_records_host(
        plan, source.view(), buffers, &result.metadata), "build records");
    return result;
}

struct host_order {
    std::vector<cp::u64> primary;
    std::vector<cp::u32> secondary, active, nnz, permutation, inverse;
    cp::local_cell_order_view metadata{};
    cp::local_cell_order_view view() const {
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

host_order build_order(const cp::cell_block_record_view &records, cp::u32 width,
    cp::local_cell_order_kind kind) {
    host_order result;
    result.primary.resize(records.row_count);
    result.secondary.resize(records.row_count);
    result.active.resize(records.row_count);
    result.nnz.resize(records.row_count);
    result.permutation.resize(records.row_count);
    result.inverse.resize(records.row_count);
    cp::local_cell_order_buffers buffers{records.row_count, result.primary.data(),
        result.secondary.data(), result.active.data(), result.nnz.data(),
        result.permutation.data(), result.inverse.data()};
    cp::local_cell_order_config config;
    config.kind = kind;
    config.window_size = width == 32u ? 64u : width;
    config.group_width = width;
    config.seed = 0x123456789abcdef0ull;
    require_status(cp::build_local_cell_order_host(records, config, buffers, &result.metadata),
        "build order");
    return result;
}

struct host_tiles {
    std::vector<cp::u32> tile_offsets, blocks, cell_masks, entry_offsets, masks,
        value_offsets;
    std::vector<unsigned char> values;
    cp::warp_tile_view metadata{};
    cp::warp_tile_requirements required{};
    cp::warp_tile_view view() const {
        auto result = metadata;
        result.tile_block_offsets = tile_offsets.data();
        result.tile_block_ids = blocks.data();
        result.tile_block_cell_masks = cell_masks.data();
        result.block_row_entry_offsets = entry_offsets.data();
        result.row_block_gene_masks = masks.data();
        result.row_block_value_offsets = value_offsets.data();
        result.values = values.data();
        return result;
    }
};

host_tiles build_host_tiles(const cp::frozen_packing_plan &plan,
    const cp::cell_block_record_view &records, const cp::local_cell_order_view &order) {
    host_tiles result;
    require_status(cp::query_warp_tile_requirements_host(
        plan, records, order, &result.required), "query host tiles");
    result.tile_offsets.resize(result.required.tile_block_offset_count);
    result.blocks.resize(result.required.tile_block_count);
    result.cell_masks.resize(result.required.tile_block_count);
    result.entry_offsets.resize(result.required.block_row_entry_offset_count);
    result.masks.resize(result.required.row_block_entry_count);
    result.value_offsets.resize(result.required.row_block_value_offset_count);
    result.values.resize(result.required.value_bytes);
    cp::warp_tile_buffers buffers{result.tile_offsets.size(), result.blocks.size(),
        result.entry_offsets.size(), result.masks.size(), result.value_offsets.size(),
        result.values.size(), result.tile_offsets.data(), result.blocks.data(),
        result.cell_masks.data(), result.entry_offsets.data(), result.masks.data(),
        result.value_offsets.data(), result.values.data()};
    require_status(cp::build_warp_tiles_host(
        plan, records, order, buffers, &result.metadata), "build host tiles");
    return result;
}

void run_exact_case(const cp::frozen_packing_plan &plan, const ordered_fixture &source,
    cp::u32 width, cp::local_cell_order_kind kind) {
    const host_records records = build_records(plan, source);
    const host_order order = build_order(records.view(), width, kind);
    const host_tiles oracle = build_host_tiles(plan, records.view(), order.view());
    device_array<cp::u32> d_record_rows(records.rows), d_record_blocks(records.blocks),
        d_record_masks(records.masks), d_record_values_offsets(records.values_offsets),
        d_permutation(order.permutation);
    device_array<unsigned char> d_record_values(records.values);
    cp::cell_block_record_view device_records = records.view();
    device_records.row_record_offsets = d_record_rows.data;
    device_records.record_block_ids = d_record_blocks.data;
    device_records.record_gene_masks = d_record_masks.data;
    device_records.record_value_offsets = d_record_values_offsets.data;
    device_records.values = d_record_values.data;
    cp::local_cell_order_view device_order = order.view();
    device_order.row_permutation = d_permutation.data;

    cp::warp_tile_cuda_requirements scratch;
    require_status(cp::query_warp_tile_cuda_requirements(oracle.metadata.tile_count,
        oracle.required.tile_block_count, oracle.required.row_block_entry_count, &scratch),
        "query CUDA tiles");
    device_array<cp::u32> d_tile_offsets(oracle.required.tile_block_offset_count),
        d_blocks(oracle.required.tile_block_count), d_cell_masks(oracle.required.tile_block_count),
        d_entry_offsets(oracle.required.block_row_entry_offset_count),
        d_masks(oracle.required.row_block_entry_count),
        d_value_offsets(oracle.required.row_block_value_offset_count),
        d_tile_counts(scratch.tile_count_capacity),
        d_descriptor_counts(scratch.tile_block_capacity),
        d_descriptor_tiles(scratch.tile_block_capacity),
        d_source_records(scratch.row_block_entry_capacity),
        d_row_value_counts(scratch.row_block_entry_capacity);
    device_array<unsigned char> d_values(oracle.required.value_bytes),
        d_cub(scratch.cub_temporary_bytes);
    cp::warp_tile_cuda_workspace workspace{scratch.tile_count_capacity,
        scratch.tile_block_capacity, scratch.row_block_entry_capacity, d_tile_counts.data,
        d_descriptor_counts.data, d_descriptor_tiles.data, d_source_records.data,
        d_row_value_counts.data, d_cub.data, scratch.cub_temporary_bytes};
    cp::warp_tile_buffers buffers{oracle.required.tile_block_offset_count,
        oracle.required.tile_block_count, oracle.required.block_row_entry_offset_count,
        oracle.required.row_block_entry_count, oracle.required.row_block_value_offset_count,
        oracle.required.value_bytes, d_tile_offsets.data, d_blocks.data, d_cell_masks.data,
        d_entry_offsets.data, d_masks.data, d_value_offsets.data, d_values.data};
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    cp::warp_tile_view device_tiles;
    require_status(cp::build_warp_tiles_cuda(plan, device_records, device_order,
        oracle.required, workspace, buffers, stream, &device_tiles), "build CUDA tiles");
    require_cuda(cudaStreamSynchronize(stream), "finish CUDA tiles");

    host_tiles actual;
    actual.required = oracle.required;
    actual.metadata = device_tiles;
    actual.tile_offsets = d_tile_offsets.download();
    actual.blocks = d_blocks.download();
    actual.cell_masks = d_cell_masks.download();
    actual.entry_offsets = d_entry_offsets.download();
    actual.masks = d_masks.download();
    actual.value_offsets = d_value_offsets.download();
    actual.values = d_values.download();
    require(actual.tile_offsets == oracle.tile_offsets && actual.blocks == oracle.blocks
            && actual.cell_masks == oracle.cell_masks
            && actual.entry_offsets == oracle.entry_offsets && actual.masks == oracle.masks
            && actual.value_offsets == oracle.value_offsets && actual.values == oracle.values,
        "CPU and CUDA tile bytes differ");
    require(device_tiles.tile_identity == oracle.metadata.tile_identity
            && device_tiles.ordering_identity == oracle.metadata.ordering_identity
            && device_tiles.tile_count == oracle.metadata.tile_count
            && device_tiles.tile_block_count == oracle.metadata.tile_block_count
            && device_tiles.row_block_entry_count == oracle.metadata.row_block_entry_count,
        "CUDA tile metadata differs from CPU oracle");
    require_status(cp::validate_warp_tile_view_host(
        plan, records.view(), order.view(), actual.view()), "validate downloaded CUDA tiles");
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
}

void test_exact_and_empty() {
    const cp::frozen_packing_plan plan = make_plan();
    run_exact_case(plan, main_fixture(3u), 32u,
        cp::local_cell_order_kind::deterministic_random);
    run_exact_case(plan, main_fixture(1u), 32u, cp::local_cell_order_kind::original);
    run_exact_case(plan, main_fixture(8u), 32u, cp::local_cell_order_kind::original);
    run_exact_case(plan, empty_fixture(0u, 5u), 1u, cp::local_cell_order_kind::original);
    run_exact_case(plan, empty_fixture(4u, 5u), 4u, cp::local_cell_order_kind::original);
}

void test_query_and_metadata_failures() {
    cp::warp_tile_cuda_requirements ignored;
    require(cp::query_warp_tile_cuda_requirements(0u, 0u, 0u, nullptr).code
            == cp::validation_code::null_pointer,
        "null CUDA requirements output was accepted");
    require(cp::query_warp_tile_cuda_requirements(
            std::numeric_limits<cp::u32>::max(), 0u, 0u, &ignored).code
            == cp::validation_code::integer_overflow,
        "signed CUB tile-count overflow was accepted");

    const cp::frozen_packing_plan plan = make_plan();
    const ordered_fixture source = main_fixture(3u);
    const host_records records = build_records(plan, source);
    const host_order order = build_order(records.view(), 32u, cp::local_cell_order_kind::original);
    const host_tiles oracle = build_host_tiles(plan, records.view(), order.view());
    cp::warp_tile_cuda_workspace workspace;
    cp::warp_tile_buffers buffers;
    cp::warp_tile_view output;
    cp::warp_tile_requirements bad = oracle.required;
    --bad.tile_block_count;
    require(cp::build_warp_tiles_cuda(plan, records.view(), order.view(), bad,
            workspace, buffers, nullptr, &output).code == cp::validation_code::invalid_matrix_view,
        "inconsistent expected descriptor count was accepted");
    require(cp::build_warp_tiles_cuda(plan, records.view(), order.view(), oracle.required,
            workspace, buffers, nullptr, nullptr).code == cp::validation_code::null_pointer,
        "null CUDA tile view output was accepted");
    require(cp::build_warp_tiles_cuda(plan, records.view(), order.view(), oracle.required,
            workspace, buffers, nullptr, &output).code == cp::validation_code::insufficient_capacity,
        "insufficient CUDA tile workspace/output capacity was accepted");

    cp::warp_tile_cuda_requirements scratch;
    require_status(cp::query_warp_tile_cuda_requirements(oracle.metadata.tile_count,
        oracle.required.tile_block_count, oracle.required.row_block_entry_count, &scratch),
        "query alias-validation scratch");
    workspace.tile_count_capacity = scratch.tile_count_capacity;
    workspace.tile_block_capacity = scratch.tile_block_capacity;
    workspace.row_block_entry_capacity = scratch.row_block_entry_capacity;
    workspace.cub_temporary_bytes = scratch.cub_temporary_bytes;
    workspace.tile_block_counts = reinterpret_cast<cp::u32 *>(0x1000u);
    workspace.descriptor_row_counts = reinterpret_cast<cp::u32 *>(0x2000u);
    workspace.descriptor_tile_ids = reinterpret_cast<cp::u32 *>(0x3000u);
    workspace.source_record_indices = reinterpret_cast<cp::u32 *>(0x4000u);
    workspace.row_value_counts = reinterpret_cast<cp::u32 *>(0x5000u);
    workspace.cub_temporary_storage = reinterpret_cast<void *>(0x6000u);
    buffers.tile_block_offset_capacity = oracle.required.tile_block_offset_count;
    buffers.tile_block_capacity = oracle.required.tile_block_count;
    buffers.block_row_entry_offset_capacity = oracle.required.block_row_entry_offset_count;
    buffers.row_block_entry_capacity = oracle.required.row_block_entry_count;
    buffers.row_block_value_offset_capacity = oracle.required.row_block_value_offset_count;
    buffers.value_capacity_bytes = oracle.required.value_bytes;
    buffers.tile_block_offsets = workspace.tile_block_counts;
    buffers.tile_block_ids = reinterpret_cast<cp::u32 *>(0x7000u);
    buffers.tile_block_cell_masks = reinterpret_cast<cp::u32 *>(0x8000u);
    buffers.block_row_entry_offsets = reinterpret_cast<cp::u32 *>(0x9000u);
    buffers.row_block_gene_masks = reinterpret_cast<cp::u32 *>(0xa000u);
    buffers.row_block_value_offsets = reinterpret_cast<cp::u32 *>(0xb000u);
    buffers.values = reinterpret_cast<void *>(0xc000u);
    require(cp::build_warp_tiles_cuda(plan, records.view(), order.view(), oracle.required,
            workspace, buffers, nullptr, &output).code == cp::validation_code::invalid_matrix_view,
        "aliased CUDA tile output/workspace was accepted");

    auto bad_order = order.view();
    bad_order.ordering_identity = 0u;
    require(cp::build_warp_tiles_cuda(plan, records.view(), bad_order, oracle.required,
            workspace, buffers, nullptr, &output).code
            == cp::validation_code::invalid_plan_geometry,
        "invalid CUDA order identity was accepted");
}

} // namespace

int main() {
    int devices = 0;
    require_cuda(cudaGetDeviceCount(&devices), "query CUDA devices");
    require(devices > 0, "no CUDA device available");
    test_exact_and_empty();
    test_query_and_metadata_failures();
    std::fprintf(stdout, "cellPackWarpTilesCudaTest: PASS on device 0\n");
    return 0;
}
