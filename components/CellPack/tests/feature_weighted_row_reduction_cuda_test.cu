#include "CellPack/feature_weighted_row_reduction_cuda.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

namespace cp = cellpack;
using storage_t = cellerator::real::storage_t;
using compute_t = cellerator::real::compute_t;
using accum_t = cellerator::real::accum_t;

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cellPackFeatureWeightedRowReductionCudaTest: " << message << '\n';
        std::exit(1);
    }
}

void require_status(cp::validation_result status, const char *message) {
    if (!status) {
        std::cerr << "cellPackFeatureWeightedRowReductionCudaTest: " << message
                  << ": " << status.message << " (index=" << status.index << ")\n";
        std::exit(1);
    }
}

void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::cerr << "cellPackFeatureWeightedRowReductionCudaTest: " << message
                  << ": " << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
}

template<typename T>
struct device_array {
    T *data = nullptr;
    std::size_t size = 0u;

    device_array() = default;
    explicit device_array(std::size_t count) : size(count) {
        if (count != 0u) {
            require_cuda(cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)),
                "cudaMalloc");
        }
    }
    ~device_array() { if (data != nullptr) cudaFree(data); }
    device_array(const device_array &) = delete;
    device_array &operator=(const device_array &) = delete;
};

template<typename T>
void upload(device_array<T> &device, const std::vector<T> &host) {
    require(device.size >= host.size(), "device upload capacity");
    if (!host.empty()) {
        require_cuda(cudaMemcpy(device.data, host.data(), host.size() * sizeof(T),
            cudaMemcpyHostToDevice), "cudaMemcpy host-to-device");
    }
}

storage_t stored(float value) { return static_cast<storage_t>(value); }

cp::frozen_packing_plan make_plan() {
    constexpr cp::u32 feature_count = 36u;
    std::vector<cp::u32> permutation(feature_count), inverse(feature_count),
        feature_to_block(feature_count), feature_to_local(feature_count);
    for (cp::u32 i = 0u; i < 32u; ++i) permutation[i] = 31u - i;
    permutation[32] = 35u;
    permutation[33] = 32u;
    permutation[34] = 34u;
    permutation[35] = 33u;
    const std::vector<cp::u32> block_offsets{0u, 32u, 36u};
    for (cp::u32 block = 0u; block + 1u < block_offsets.size(); ++block) {
        for (cp::u32 execution = block_offsets[block];
             execution < block_offsets[block + 1u]; ++execution) {
            const cp::u32 canonical = permutation[execution];
            inverse[canonical] = execution;
            feature_to_block[canonical] = block;
            feature_to_local[canonical] = execution - block_offsets[block];
        }
    }
    const cp::u32 row_group_offsets[] = {0u, 8u};
    cp::frozen_packing_plan_build_view build;
    build.row_count = 8u;
    build.feature_count = feature_count;
    build.feature_permutation = permutation.data();
    build.inverse_feature_permutation = inverse.data();
    build.feature_block_count = 2u;
    build.feature_block_offsets = block_offsets.data();
    build.feature_to_block = feature_to_block.data();
    build.feature_to_local = feature_to_local.data();
    build.row_group_count = 1u;
    build.row_group_offsets = row_group_offsets;
    build.maximum_feature_block_width = 32u;
    build.row_group_width = 8u;
    build.identity.feature_axis_fingerprint = 0x1020304050607080ull;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind = cp::packing_row_domain_kind::full_dataset_identity;
    build.identity.row_domain_identity = 0x8877665544332211ull;
    build.identity.evaluation_source_identity = 0x1234u;
    build.cost_policy_identity = 0x5678u;
    cp::frozen_packing_plan plan;
    require_status(cp::freeze_packing_plan(build, &plan), "freeze plan");
    return plan;
}

struct fixture {
    cp::u64 global_row_begin = 2u;
    cp::u32 row_count = 5u;
    std::vector<cp::u32> row_offsets;
    std::vector<cp::u32> features;
    std::vector<storage_t> values;
};

fixture make_fixture(bool all_empty) {
    fixture result;
    result.row_offsets.push_back(0u);
    std::vector<std::vector<cp::u32>> rows(5u);
    if (!all_empty) {
        rows[0].resize(32u);
        std::iota(rows[0].begin(), rows[0].end(), 0u); // full width, including bit 31
        rows[0].push_back(35u);
        rows[2] = {0u, 5u, 31u, 32u, 35u};
        rows[3] = {3u, 7u};
        rows[4] = {31u};
    }
    for (cp::u32 row = 0u; row < rows.size(); ++row) {
        for (cp::u32 feature : rows[row]) {
            result.features.push_back(feature);
            const float sign = ((row + feature) & 1u) == 0u ? 1.0f : -1.0f;
            result.values.push_back(stored(sign * (0.25f + 0.125f * feature)));
        }
        result.row_offsets.push_back(static_cast<cp::u32>(result.features.size()));
    }
    return result;
}

cp::plan_application_context context() {
    cp::plan_application_context result;
    result.full_row_count = 8u;
    result.feature_count = 36u;
    result.feature_axis_fingerprint = 0x1020304050607080ull;
    result.feature_axis_fingerprint_version = 1u;
    result.row_domain_identity = 0x8877665544332211ull;
    return result;
}

cp::plan_application_source_view source_view(const fixture &source) {
    cp::plan_application_source_view result;
    result.global_row_begin = source.global_row_begin;
    result.row_count = source.row_count;
    result.feature_count = 36u;
    result.nnz_count = static_cast<cp::u32>(source.features.size());
    result.value_size_bytes = sizeof(storage_t);
    result.row_offsets = source.row_offsets.data();
    result.canonical_feature_ids = source.features.empty() ? nullptr : source.features.data();
    result.values = source.values.empty() ? nullptr : source.values.data();
    return result;
}

struct packed_fixture {
    std::vector<cp::u32> ordered_rows, ordered_blocks, ordered_locals,
        ordered_canonical;
    std::vector<storage_t> ordered_values;
    cp::ordered_plan_partition_view ordered{};
    std::vector<cp::u32> record_rows, record_blocks, record_masks,
        record_value_offsets;
    std::vector<unsigned char> record_values;
    cp::cell_block_record_view records{};
    std::vector<cp::u64> order_primary;
    std::vector<cp::u32> order_secondary, order_active, order_nnz,
        order_permutation, order_inverse;
    cp::local_cell_order_view order{};
    std::vector<cp::u32> tile_offsets, tile_blocks, tile_cell_masks,
        tile_entry_offsets, tile_gene_masks, tile_value_offsets;
    std::vector<unsigned char> tile_values;
    cp::warp_tile_view tiles{};
};

packed_fixture pack(
    const cp::frozen_packing_plan &plan,
    const fixture &source) {
    packed_fixture result;
    const cp::u32 nnz = static_cast<cp::u32>(source.features.size());
    result.ordered_rows.resize(source.row_offsets.size());
    result.ordered_blocks.resize(nnz);
    result.ordered_locals.resize(nnz);
    result.ordered_canonical.resize(nnz);
    result.ordered_values.resize(nnz);
    std::vector<cp::u64> keys(nnz);
    std::vector<cp::u32> source_order(nnz);
    cp::plan_application_host_workspace_view workspace{nnz, keys.data(), source_order.data()};
    cp::plan_application_buffers ordered_buffers{result.ordered_rows.size(), nnz,
        nnz * sizeof(storage_t), result.ordered_rows.data(), result.ordered_blocks.data(),
        result.ordered_locals.data(), result.ordered_canonical.data(),
        result.ordered_values.data()};
    require_status(cp::apply_frozen_plan_host(plan, context(), source_view(source),
        workspace, ordered_buffers, &result.ordered), "apply plan");

    cp::cell_block_record_requirements record_required;
    require_status(cp::query_cell_block_record_requirements_host(
        plan, result.ordered, &record_required), "query records");
    result.record_rows.resize(record_required.row_record_offset_count);
    result.record_blocks.resize(record_required.record_count);
    result.record_masks.resize(record_required.record_count);
    result.record_value_offsets.resize(record_required.record_value_offset_count);
    result.record_values.resize(record_required.value_bytes);
    cp::cell_block_record_buffers record_buffers{result.record_rows.size(),
        result.record_blocks.size(), result.record_value_offsets.size(),
        result.record_values.size(), result.record_rows.data(), result.record_blocks.data(),
        result.record_masks.data(), result.record_value_offsets.data(),
        result.record_values.data()};
    require_status(cp::build_cell_block_records_host(
        plan, result.ordered, record_buffers, &result.records), "build records");

    result.order_primary.resize(source.row_count);
    result.order_secondary.resize(source.row_count);
    result.order_active.resize(source.row_count);
    result.order_nnz.resize(source.row_count);
    result.order_permutation.resize(source.row_count);
    result.order_inverse.resize(source.row_count);
    cp::local_cell_order_buffers order_buffers{source.row_count,
        result.order_primary.data(), result.order_secondary.data(),
        result.order_active.data(), result.order_nnz.data(),
        result.order_permutation.data(), result.order_inverse.data()};
    cp::local_cell_order_config order_config;
    order_config.kind = cp::local_cell_order_kind::row_nnz_descending;
    order_config.window_size = 8u;
    order_config.group_width = 4u;
    require_status(cp::build_local_cell_order_host(
        result.records, order_config, order_buffers, &result.order), "build order");

    cp::warp_tile_requirements tile_required;
    require_status(cp::query_warp_tile_requirements_host(
        plan, result.records, result.order, &tile_required), "query tiles");
    result.tile_offsets.resize(tile_required.tile_block_offset_count);
    result.tile_blocks.resize(tile_required.tile_block_count);
    result.tile_cell_masks.resize(tile_required.tile_block_count);
    result.tile_entry_offsets.resize(tile_required.block_row_entry_offset_count);
    result.tile_gene_masks.resize(tile_required.row_block_entry_count);
    result.tile_value_offsets.resize(tile_required.row_block_value_offset_count);
    result.tile_values.resize(tile_required.value_bytes);
    cp::warp_tile_buffers tile_buffers{result.tile_offsets.size(), result.tile_blocks.size(),
        result.tile_entry_offsets.size(), result.tile_gene_masks.size(),
        result.tile_value_offsets.size(), result.tile_values.size(),
        result.tile_offsets.data(), result.tile_blocks.data(), result.tile_cell_masks.data(),
        result.tile_entry_offsets.data(), result.tile_gene_masks.data(),
        result.tile_value_offsets.data(), result.tile_values.data()};
    require_status(cp::build_warp_tiles_host(
        plan, result.records, result.order, tile_buffers, &result.tiles), "build tiles");
    return result;
}

struct device_fixture {
    device_array<cp::u32> plan_offsets, plan_permutation, order_permutation,
        tile_offsets, tile_blocks, tile_cell_masks, tile_entry_offsets,
        tile_gene_masks, tile_value_offsets;
    device_array<unsigned char> tile_values;
    device_array<compute_t> weights;
    device_array<accum_t> output;

    device_fixture(const cp::frozen_packing_plan &plan, const packed_fixture &packed,
        const std::vector<compute_t> &host_weights)
        : plan_offsets(plan.feature_block_count() + 1u),
          plan_permutation(plan.feature_count()),
          order_permutation(packed.order_permutation.size()),
          tile_offsets(packed.tile_offsets.size()), tile_blocks(packed.tile_blocks.size()),
          tile_cell_masks(packed.tile_cell_masks.size()),
          tile_entry_offsets(packed.tile_entry_offsets.size()),
          tile_gene_masks(packed.tile_gene_masks.size()),
          tile_value_offsets(packed.tile_value_offsets.size()),
          tile_values(packed.tile_values.size()), weights(host_weights.size()),
          output(packed.tiles.row_count) {
        std::vector<cp::u32> offsets(plan.feature_block_offsets(),
            plan.feature_block_offsets() + plan.feature_block_count() + 1u);
        std::vector<cp::u32> permutation(plan.feature_permutation(),
            plan.feature_permutation() + plan.feature_count());
        upload(plan_offsets, offsets);
        upload(plan_permutation, permutation);
        upload(order_permutation, packed.order_permutation);
        upload(tile_offsets, packed.tile_offsets);
        upload(tile_blocks, packed.tile_blocks);
        upload(tile_cell_masks, packed.tile_cell_masks);
        upload(tile_entry_offsets, packed.tile_entry_offsets);
        upload(tile_gene_masks, packed.tile_gene_masks);
        upload(tile_value_offsets, packed.tile_value_offsets);
        upload(tile_values, packed.tile_values);
        upload(weights, host_weights);
    }
};

cp::feature_weighted_row_reduction_view device_input(
    const cp::feature_weighted_row_reduction_view &host,
    const device_fixture &device) {
    auto result = host;
    result.plan.feature_block_offsets = device.plan_offsets.data;
    result.plan.feature_permutation = device.plan_permutation.data;
    result.tiles.tile_block_offsets = device.tile_offsets.data;
    result.tiles.tile_block_ids = device.tile_blocks.data;
    result.tiles.tile_block_cell_masks = device.tile_cell_masks.data;
    result.tiles.block_row_entry_offsets = device.tile_entry_offsets.data;
    result.tiles.row_block_gene_masks = device.tile_gene_masks.data;
    result.tiles.row_block_value_offsets = device.tile_value_offsets.data;
    result.tiles.values = device.tile_values.data;
    result.feature_weights = device.weights.data;
    return result;
}

void run_case(bool all_empty) {
    const auto plan = make_plan();
    const auto source = make_fixture(all_empty);
    const auto packed = pack(plan, source);
    require(packed.tiles.tile_count == 2u, "tail tile was not built");
    if (!all_empty) {
        require(packed.order_permutation
                != std::vector<cp::u32>({0u, 1u, 2u, 3u, 4u}),
            "local order unexpectedly remained identity");
        require(std::find(packed.tile_gene_masks.begin(), packed.tile_gene_masks.end(),
                    0xffffffffu) != packed.tile_gene_masks.end(),
            "full gene mask including bit 31 was not represented");
    }

    std::vector<compute_t> weights(plan.feature_count());
    for (cp::u32 feature = 0u; feature < weights.size(); ++feature) {
        weights[feature] = static_cast<compute_t>(0.5 + 0.03125 * feature);
    }
    const auto host_input = cp::make_feature_weighted_row_reduction_view(
        plan, packed.tiles, 0xabcdef0123456789ull, weights.size(), weights.data());
    std::vector<accum_t> canonical(source.row_count), tile(source.row_count);
    cp::feature_weighted_row_reduction_result_view canonical_result, tile_result;
    cp::feature_weighted_row_reduction_buffers canonical_buffers{
        canonical.size(), canonical.data()};
    cp::feature_weighted_row_reduction_buffers tile_buffers{tile.size(), tile.data()};
    require_status(cp::evaluate_feature_weighted_row_reduction_canonical_host(
        plan, context(), source_view(source), host_input, canonical_buffers,
        &canonical_result), "canonical host reference");
    require_status(cp::evaluate_feature_weighted_row_reduction_tiles_host(
        plan, packed.records, packed.order, host_input, tile_buffers, &tile_result),
        "direct tile host reference");

    device_fixture device(plan, packed, weights);
    auto input = device_input(host_input, device);
    auto order = packed.order;
    order.row_permutation = device.order_permutation.data;
    cp::feature_weighted_row_reduction_buffers buffers{source.row_count, device.output.data};
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    cp::feature_weighted_row_reduction_result_view result;
    require_status(cp::evaluate_feature_weighted_row_reduction_tiles_cuda(
        input, order, buffers, stream, &result), "direct CUDA reduction");
    require(result.row_values == device.output.data
            && result.reduction_identity == input.reduction_identity
            && result.feature_weight_identity == input.feature_weight_identity
            && result.global_row_begin == source.global_row_begin
            && result.row_count == source.row_count,
        "CUDA result metadata");
    require_cuda(cudaStreamSynchronize(stream), "synchronize first reduction");
    std::vector<accum_t> first(source.row_count), repeated(source.row_count);
    require_cuda(cudaMemcpy(first.data(), device.output.data,
        first.size() * sizeof(accum_t), cudaMemcpyDeviceToHost), "download first output");
    require_status(cp::evaluate_feature_weighted_row_reduction_tiles_cuda(
        input, order, buffers, stream, &result), "repeat direct CUDA reduction");
    require_cuda(cudaStreamSynchronize(stream), "synchronize repeated reduction");
    require_cuda(cudaMemcpy(repeated.data(), device.output.data,
        repeated.size() * sizeof(accum_t), cudaMemcpyDeviceToHost), "download repeat output");
    for (std::size_t row = 0u; row < first.size(); ++row) {
        require(cp::feature_weighted_row_reduction_within_tolerance(
            canonical[row], first[row]), "canonical/CUDA numerical mismatch");
        require(cp::feature_weighted_row_reduction_within_tolerance(
            tile[row], first[row]), "tile-host/CUDA numerical mismatch");
        require(first[row] == repeated[row], "repeat CUDA output changed");
    }

    if (!all_empty) {
        auto tampered = input;
        tampered.reduction_identity ^= 1u;
        require(!cp::evaluate_feature_weighted_row_reduction_tiles_cuda(
            tampered, order, buffers, stream, &result), "identity tamper accepted");
        tampered = input;
        tampered.feature_weight_capacity = input.tiles.feature_count - 1u;
        require(!cp::evaluate_feature_weighted_row_reduction_tiles_cuda(
            tampered, order, buffers, stream, &result), "weight capacity tamper accepted");
        auto tampered_order = order;
        tampered_order.ordering_identity ^= 1u;
        require(!cp::evaluate_feature_weighted_row_reduction_tiles_cuda(
            input, tampered_order, buffers, stream, &result), "order identity tamper accepted");
        auto small = buffers;
        --small.row_capacity;
        require(!cp::evaluate_feature_weighted_row_reduction_tiles_cuda(
            input, order, small, stream, &result), "output capacity tamper accepted");
        auto alias = buffers;
        alias.row_values = reinterpret_cast<accum_t *>(device.weights.data);
        require(!cp::evaluate_feature_weighted_row_reduction_tiles_cuda(
            input, order, alias, stream, &result), "output alias accepted");
        require(!cp::evaluate_feature_weighted_row_reduction_tiles_cuda(
            input, order, buffers, stream, nullptr), "null result accepted");

    }
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
}

} // namespace

int main() {
    run_case(false);
    run_case(true);
    std::cout << "cellPackFeatureWeightedRowReductionCudaTest: passed\n";
    return 0;
}
