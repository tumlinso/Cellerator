/*
 * CP-BP-08 device tile-construction benchmark. The Phase C host builder is the
 * exact oracle; timed CUDA work excludes source/output transfers and reports
 * caller-owned scratch plus physical metadata/storage metrics.
 */

#include "CellPack/warp_tiles_cuda.hh"

#include "benchmark_mutex.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace cp = ::cellpack;

namespace {

void check(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "cellPackWarpTilesBench: %s\n", message);
        std::exit(1);
    }
}

void check_cuda(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "cellPackWarpTilesBench: %s: %s\n",
            message, cudaGetErrorString(error));
        std::exit(1);
    }
}

void check_status(cp::validation_result status, const char *message) {
    if (!status) {
        std::fprintf(stderr, "cellPackWarpTilesBench: %s: %s\n", message, status.message);
        std::exit(1);
    }
}

template <typename T>
T *device_allocate(std::size_t count) {
    if (count == 0u) return nullptr;
    T *pointer = nullptr;
    check_cuda(cudaMalloc(&pointer, count * sizeof(T)), "cudaMalloc");
    return pointer;
}

template <typename T>
void upload(T *device, const std::vector<T> &host, const char *message) {
    if (!host.empty()) check_cuda(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
        cudaMemcpyHostToDevice), message);
}

cp::frozen_packing_plan build_plan(cp::u32 rows, cp::u32 features, cp::u32 width,
    cp::u64 feature_fingerprint, cp::u64 row_identity) {
    std::vector<cp::u32> permutation(features), inverse(features), to_block(features),
        to_local(features);
    std::iota(permutation.begin(), permutation.end(), 0u);
    std::iota(inverse.begin(), inverse.end(), 0u);
    const cp::u32 block_count = (features + width - 1u) / width;
    std::vector<cp::u32> block_offsets(block_count + 1u);
    for (cp::u32 block = 0u; block < block_count; ++block) block_offsets[block] = block * width;
    block_offsets.back() = features;
    for (cp::u32 feature = 0u; feature < features; ++feature) {
        to_block[feature] = feature / width;
        to_local[feature] = feature % width;
    }
    const cp::u32 row_offsets[] = {0u, rows};
    cp::frozen_packing_plan_build_view build;
    build.row_count = rows;
    build.feature_count = features;
    build.feature_permutation = permutation.data();
    build.inverse_feature_permutation = inverse.data();
    build.feature_block_count = block_count;
    build.feature_block_offsets = block_offsets.data();
    build.feature_to_block = to_block.data();
    build.feature_to_local = to_local.data();
    build.row_group_count = 1u;
    build.row_group_offsets = row_offsets;
    build.maximum_feature_block_width = width;
    build.row_group_width = rows;
    build.identity.feature_axis_fingerprint = feature_fingerprint;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind = cp::packing_row_domain_kind::full_dataset_identity;
    build.identity.row_domain_identity = row_identity;
    build.identity.evaluation_source_identity = 0x20260817u;
    build.cost_policy_identity = 0x43503033u;
    cp::frozen_packing_plan plan;
    check_status(cp::freeze_packing_plan(build, &plan), "freeze benchmark plan");
    return plan;
}

} // namespace

int main() {
    constexpr cp::u32 rows = 65536u, features = 30000u, block_width = 16u;
    constexpr cp::u32 tile_width = 32u, records_per_row = 16u, values_per_record = 2u;
    constexpr cp::u64 feature_fingerprint = 0x4350303842454e31ull;
    constexpr cp::u64 row_identity = 0x43503038524f5732ull;
    constexpr int warmups = 2, repeats = 7;
    cellerator::bench::benchmark_mutex_guard mutex("cellPackWarpTilesBench", 0);
    const cp::frozen_packing_plan plan = build_plan(
        rows, features, block_width, feature_fingerprint, row_identity);
    const cp::u32 tile_count = rows / tile_width;
    const cp::u32 record_count = rows * records_per_row;
    const cp::u32 nnz = record_count * values_per_record;
    const cp::u32 block_count = plan.feature_block_count();
    std::vector<cp::u32> row_offsets(rows + 1u), block_ids(record_count),
        gene_masks(record_count), value_offsets(record_count + 1u), values(nnz),
        secondary(rows), active(rows), row_nnz(rows), permutation(rows), inverse(rows);
    std::vector<cp::u64> primary(rows);
    for (cp::u32 row = 0u; row <= rows; ++row) row_offsets[row] = row * records_per_row;
    for (cp::u32 tile = 0u; tile < tile_count; ++tile) {
        const cp::u32 first_block = (tile * 17u) % (block_count - records_per_row);
        for (cp::u32 lane = 0u; lane < tile_width; ++lane) {
            const cp::u32 row = tile * tile_width + lane;
            for (cp::u32 slot = 0u; slot < records_per_row; ++slot) {
                const cp::u32 record = row * records_per_row + slot;
                const cp::u32 bit = (lane + slot) % (block_width - 1u);
                block_ids[record] = first_block + slot;
                gene_masks[record] = 3u << bit;
                value_offsets[record] = record * values_per_record;
                values[record * values_per_record] = row * 131u + slot * 2u;
                values[record * values_per_record + 1u] = row * 131u + slot * 2u + 1u;
            }
        }
    }
    value_offsets.back() = nnz;
    cp::cell_block_record_view records;
    records.record_schema_version = cp::cell_block_record_schema_version;
    records.semantic_plan_schema_version = cp::packing_plan_semantic_schema_version;
    records.geometry_identity_version = cp::feature_block_geometry_identity_version;
    records.feature_block_geometry_identity = plan.feature_block_geometry_identity();
    records.full_row_count = rows;
    records.row_count = rows;
    records.feature_count = features;
    records.feature_block_count = block_count;
    records.nnz_count = nnz;
    records.record_count = record_count;
    records.value_size_bytes = sizeof(cp::u32);
    records.feature_axis_fingerprint = feature_fingerprint;
    records.feature_axis_fingerprint_version = 1u;
    records.row_domain_identity = row_identity;
    records.row_record_offsets = row_offsets.data();
    records.record_block_ids = block_ids.data();
    records.record_gene_masks = gene_masks.data();
    records.record_value_offsets = value_offsets.data();
    records.values = values.data();
    check_status(cp::validate_cell_block_record_view_host(plan, records), "validate records");

    cp::local_cell_order_buffers order_buffers{rows, primary.data(), secondary.data(),
        active.data(), row_nnz.data(), permutation.data(), inverse.data()};
    cp::local_cell_order_config order_config;
    order_config.kind = cp::local_cell_order_kind::original;
    order_config.window_size = 1024u;
    order_config.group_width = tile_width;
    cp::local_cell_order_view order;
    check_status(cp::build_local_cell_order_host(
        records, order_config, order_buffers, &order), "build benchmark order");

    cp::warp_tile_requirements exact;
    const auto query_begin = std::chrono::steady_clock::now();
    check_status(cp::query_warp_tile_requirements_host(
        plan, records, order, &exact), "query host requirements");
    const auto query_end = std::chrono::steady_clock::now();
    std::vector<cp::u32> cpu_tile_offsets(exact.tile_block_offset_count),
        cpu_blocks(exact.tile_block_count), cpu_cell_masks(exact.tile_block_count),
        cpu_entry_offsets(exact.block_row_entry_offset_count),
        cpu_gene_masks(exact.row_block_entry_count),
        cpu_value_offsets(exact.row_block_value_offset_count);
    std::vector<unsigned char> cpu_values(exact.value_bytes);
    cp::warp_tile_buffers cpu_buffers{cpu_tile_offsets.size(), cpu_blocks.size(),
        cpu_entry_offsets.size(), cpu_gene_masks.size(), cpu_value_offsets.size(),
        cpu_values.size(), cpu_tile_offsets.data(), cpu_blocks.data(), cpu_cell_masks.data(),
        cpu_entry_offsets.data(), cpu_gene_masks.data(), cpu_value_offsets.data(),
        cpu_values.data()};
    cp::warp_tile_view cpu_tiles;
    const auto cpu_begin = std::chrono::steady_clock::now();
    check_status(cp::build_warp_tiles_host(
        plan, records, order, cpu_buffers, &cpu_tiles), "build CPU oracle");
    const auto cpu_end = std::chrono::steady_clock::now();
    cp::warp_tile_metrics metrics;
    check_status(cp::evaluate_warp_tile_metrics_host(
        plan, records, order, cpu_tiles, &metrics), "evaluate tile metrics");

    cp::warp_tile_cuda_requirements scratch;
    check_status(cp::query_warp_tile_cuda_requirements(
        tile_count, exact.tile_block_count, exact.row_block_entry_count, &scratch),
        "query CUDA scratch");
    cp::u32 *d_row_offsets = device_allocate<cp::u32>(row_offsets.size());
    cp::u32 *d_block_ids = device_allocate<cp::u32>(block_ids.size());
    cp::u32 *d_gene_masks = device_allocate<cp::u32>(gene_masks.size());
    cp::u32 *d_source_value_offsets = device_allocate<cp::u32>(value_offsets.size());
    cp::u32 *d_source_values = device_allocate<cp::u32>(values.size());
    cp::u32 *d_permutation = device_allocate<cp::u32>(permutation.size());
    cp::u32 *d_tile_offsets = device_allocate<cp::u32>(exact.tile_block_offset_count);
    cp::u32 *d_output_blocks = device_allocate<cp::u32>(exact.tile_block_count);
    cp::u32 *d_cell_masks = device_allocate<cp::u32>(exact.tile_block_count);
    cp::u32 *d_entry_offsets = device_allocate<cp::u32>(exact.block_row_entry_offset_count);
    cp::u32 *d_output_gene_masks = device_allocate<cp::u32>(exact.row_block_entry_count);
    cp::u32 *d_output_value_offsets = device_allocate<cp::u32>(exact.row_block_value_offset_count);
    unsigned char *d_output_values = device_allocate<unsigned char>(exact.value_bytes);
    cp::u32 *d_tile_counts = device_allocate<cp::u32>(scratch.tile_count_capacity);
    cp::u32 *d_descriptor_counts = device_allocate<cp::u32>(scratch.tile_block_capacity);
    cp::u32 *d_descriptor_tiles = device_allocate<cp::u32>(scratch.tile_block_capacity);
    cp::u32 *d_source_records = device_allocate<cp::u32>(scratch.row_block_entry_capacity);
    cp::u32 *d_row_value_counts = device_allocate<cp::u32>(scratch.row_block_entry_capacity);
    unsigned char *d_cub = device_allocate<unsigned char>(scratch.cub_temporary_bytes);
    upload(d_row_offsets, row_offsets, "upload row offsets");
    upload(d_block_ids, block_ids, "upload block ids");
    upload(d_gene_masks, gene_masks, "upload gene masks");
    upload(d_source_value_offsets, value_offsets, "upload value offsets");
    upload(d_source_values, values, "upload values");
    upload(d_permutation, permutation, "upload permutation");
    cp::cell_block_record_view device_records = records;
    device_records.row_record_offsets = d_row_offsets;
    device_records.record_block_ids = d_block_ids;
    device_records.record_gene_masks = d_gene_masks;
    device_records.record_value_offsets = d_source_value_offsets;
    device_records.values = d_source_values;
    cp::local_cell_order_view device_order = order;
    device_order.row_permutation = d_permutation;
    cp::warp_tile_cuda_workspace workspace{scratch.tile_count_capacity,
        scratch.tile_block_capacity, scratch.row_block_entry_capacity, d_tile_counts,
        d_descriptor_counts, d_descriptor_tiles, d_source_records, d_row_value_counts,
        d_cub, scratch.cub_temporary_bytes};
    cp::warp_tile_buffers gpu_buffers{exact.tile_block_offset_count,
        exact.tile_block_count, exact.block_row_entry_offset_count,
        exact.row_block_entry_count, exact.row_block_value_offset_count, exact.value_bytes,
        d_tile_offsets, d_output_blocks, d_cell_masks, d_entry_offsets,
        d_output_gene_masks, d_output_value_offsets, d_output_values};
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    check_cuda(cudaEventCreate(&begin), "create begin event");
    check_cuda(cudaEventCreate(&end), "create end event");
    cp::warp_tile_view gpu_tiles;
    for (int iteration = 0; iteration < warmups; ++iteration) {
        check_status(cp::build_warp_tiles_cuda(plan, device_records, device_order, exact,
            workspace, gpu_buffers, stream, &gpu_tiles), "warmup CUDA builder");
    }
    check_cuda(cudaStreamSynchronize(stream), "finish warmups");
    std::vector<float> milliseconds;
    for (int iteration = 0; iteration < repeats; ++iteration) {
        check_cuda(cudaEventRecord(begin, stream), "record begin event");
        check_status(cp::build_warp_tiles_cuda(plan, device_records, device_order, exact,
            workspace, gpu_buffers, stream, &gpu_tiles), "timed CUDA builder");
        check_cuda(cudaEventRecord(end, stream), "record end event");
        check_cuda(cudaEventSynchronize(end), "finish timed CUDA builder");
        float elapsed = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed, begin, end), "measure CUDA builder");
        milliseconds.push_back(elapsed);
    }

    std::vector<cp::u32> gpu_tile_offsets(cpu_tile_offsets.size()),
        gpu_blocks(cpu_blocks.size()), gpu_cell_masks(cpu_cell_masks.size()),
        gpu_entry_offsets(cpu_entry_offsets.size()), gpu_gene_masks(cpu_gene_masks.size()),
        gpu_value_offsets(cpu_value_offsets.size());
    std::vector<unsigned char> gpu_values(cpu_values.size());
    check_cuda(cudaMemcpy(gpu_tile_offsets.data(), d_tile_offsets,
        gpu_tile_offsets.size() * sizeof(cp::u32), cudaMemcpyDeviceToHost), "download tiles");
    check_cuda(cudaMemcpy(gpu_blocks.data(), d_output_blocks,
        gpu_blocks.size() * sizeof(cp::u32), cudaMemcpyDeviceToHost), "download blocks");
    check_cuda(cudaMemcpy(gpu_cell_masks.data(), d_cell_masks,
        gpu_cell_masks.size() * sizeof(cp::u32), cudaMemcpyDeviceToHost), "download masks");
    check_cuda(cudaMemcpy(gpu_entry_offsets.data(), d_entry_offsets,
        gpu_entry_offsets.size() * sizeof(cp::u32), cudaMemcpyDeviceToHost), "download entries");
    check_cuda(cudaMemcpy(gpu_gene_masks.data(), d_output_gene_masks,
        gpu_gene_masks.size() * sizeof(cp::u32), cudaMemcpyDeviceToHost), "download genes");
    check_cuda(cudaMemcpy(gpu_value_offsets.data(), d_output_value_offsets,
        gpu_value_offsets.size() * sizeof(cp::u32), cudaMemcpyDeviceToHost), "download offsets");
    check_cuda(cudaMemcpy(gpu_values.data(), d_output_values, gpu_values.size(),
        cudaMemcpyDeviceToHost), "download values");
    check(gpu_tile_offsets == cpu_tile_offsets && gpu_blocks == cpu_blocks
            && gpu_cell_masks == cpu_cell_masks && gpu_entry_offsets == cpu_entry_offsets
            && gpu_gene_masks == cpu_gene_masks && gpu_value_offsets == cpu_value_offsets
            && gpu_values == cpu_values,
        "CPU/CUDA benchmark outputs differ");
    std::sort(milliseconds.begin(), milliseconds.end());
    const double query_ms = std::chrono::duration<double, std::milli>(
        query_end - query_begin).count();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();
    const double mean_ms = std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0)
        / repeats;
    const double median_ms = milliseconds[milliseconds.size() / 2u];
    const double gnnz_per_second = static_cast<double>(nnz) / (median_ms * 1.0e6);
    const double metadata_per_nnz = static_cast<double>(metrics.metadata_bytes) / nnz;
    const double source_metadata_per_nnz = static_cast<double>(metrics.source_record_metadata_bytes)
        / nnz;
    const cp::u64 canonical_csr_metadata_bytes =
        (static_cast<cp::u64>(rows) + 1u + nnz) * sizeof(cp::u32);
    const double canonical_csr_metadata_per_nnz =
        static_cast<double>(canonical_csr_metadata_bytes) / nnz;
    std::fprintf(stdout,
        "WARP_TILE_BENCH device=0 gpu=Tesla_V100 sm=70 rows=%u features=%u nnz=%u "
        "records=%u tiles=%u tile_blocks=%u max_tile_union=%u tile_width=%u block_width=%u "
        "value_bytes=%llu metadata_bytes=%llu source_record_metadata_bytes=%llu "
        "canonical_csr_metadata_bytes=%llu metadata_bytes_per_nnz=%.6f "
        "source_metadata_bytes_per_nnz=%.6f canonical_csr_metadata_bytes_per_nnz=%.6f "
        "scratch_bytes=%zu "
        "warmups=%d repeats=%d transfers_timed=0 host_query_ms=%.3f cpu_build_ms=%.3f "
        "gpu_min_ms=%.3f gpu_median_ms=%.3f gpu_mean_ms=%.3f gpu_gnnz_s=%.3f exact_match=1\n",
        rows, features, nnz, record_count, tile_count, exact.tile_block_count,
        metrics.maximum_tile_block_union, tile_width, block_width,
        static_cast<unsigned long long>(metrics.value_bytes),
        static_cast<unsigned long long>(metrics.metadata_bytes),
        static_cast<unsigned long long>(metrics.source_record_metadata_bytes),
        static_cast<unsigned long long>(canonical_csr_metadata_bytes), metadata_per_nnz,
        source_metadata_per_nnz, canonical_csr_metadata_per_nnz,
        scratch.total_temporary_bytes,
        warmups, repeats, query_ms, cpu_ms, milliseconds.front(), median_ms, mean_ms,
        gnnz_per_second);

    cudaEventDestroy(begin); cudaEventDestroy(end); cudaStreamDestroy(stream);
    cudaFree(d_row_offsets); cudaFree(d_block_ids); cudaFree(d_gene_masks);
    cudaFree(d_source_value_offsets); cudaFree(d_source_values); cudaFree(d_permutation);
    cudaFree(d_tile_offsets); cudaFree(d_output_blocks); cudaFree(d_cell_masks);
    cudaFree(d_entry_offsets); cudaFree(d_output_gene_masks); cudaFree(d_output_value_offsets);
    cudaFree(d_output_values); cudaFree(d_tile_counts); cudaFree(d_descriptor_counts);
    cudaFree(d_descriptor_tiles); cudaFree(d_source_records); cudaFree(d_row_value_counts);
    cudaFree(d_cub);
    return 0;
}
