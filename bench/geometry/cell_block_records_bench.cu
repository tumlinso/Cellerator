/*
 * CP-BP-06 device record-construction benchmark. The Phase A CPU builder is
 * the exact oracle; timed CUDA work excludes source/output transfers. Update
 * the implementation header with the measured V100 result at the readiness gate.
 */

#include "Cellerator/geometry/cell_block_records_cuda.hh"

#include "benchmark_mutex.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace cp = ::cellpack;

void check(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "cellPackCellBlockRecordsBench: %s\n", message);
        std::exit(1);
    }
}

void check_cuda(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "cellPackCellBlockRecordsBench: %s: %s\n",
            message, cudaGetErrorString(error));
        std::exit(1);
    }
}

void check_status(cp::validation_result status, const char *message) {
    if (!status) {
        std::fprintf(stderr, "cellPackCellBlockRecordsBench: %s: %s\n", message, status.message);
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

cp::frozen_packing_plan build_identity_plan(cp::u32 rows, cp::u32 features, cp::u32 width) {
    std::vector<cp::u32> permutation(features), inverse(features), to_block(features), to_local(features);
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
    const cp::u32 row_group_offsets[] = {0u, rows};
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
    build.row_group_offsets = row_group_offsets;
    build.maximum_feature_block_width = width;
    build.row_group_width = rows;
    build.identity.feature_axis_fingerprint = 0x6270303662656e63ull;
    build.identity.feature_axis_fingerprint_version = 1u;
    build.identity.row_domain_kind = cp::packing_row_domain_kind::full_dataset_identity;
    build.identity.row_domain_identity = 0x5651303036ull;
    build.identity.evaluation_source_identity = 0x20260817u;
    build.cost_policy_identity = 0x62703033u;
    cp::frozen_packing_plan result;
    check_status(cp::freeze_packing_plan(build, &result), "freeze benchmark plan");
    return result;
}

int main() {
    constexpr cp::u32 rows = 65536u, features = 30000u, block_width = 16u;
    constexpr cp::u32 blocks_per_row = 2u, values_per_row = block_width * blocks_per_row;
    constexpr int warmups = 2, repeats = 7;
    cellerator::bench::benchmark_mutex_guard benchmark_mutex("cellPackCellBlockRecordsBench", 0);
    const cp::frozen_packing_plan plan = build_identity_plan(rows, features, block_width);
    const cp::u32 nnz = rows * values_per_row;
    const cp::u32 block_count = plan.feature_block_count();
    std::vector<cp::u32> row_offsets(rows + 1u), block_ids(nnz), local_ids(nnz), canonical_ids(nnz), values(nnz);
    for (cp::u32 row = 0u; row <= rows; ++row) row_offsets[row] = row * values_per_row;
    for (cp::u32 row = 0u; row < rows; ++row) {
        const cp::u32 first_block = (row * 17u) % (block_count - 1u);
        for (cp::u32 slot = 0u; slot < values_per_row; ++slot) {
            const cp::u32 block = first_block + slot / block_width;
            const cp::u32 local = slot % block_width;
            const cp::u32 entry = row * values_per_row + slot;
            block_ids[entry] = block;
            local_ids[entry] = local;
            canonical_ids[entry] = block * block_width + local;
            values[entry] = row * 131u + slot;
        }
    }
    cp::ordered_plan_partition_view source;
    source.semantic_plan_schema_version = cp::packing_plan_semantic_schema_version;
    source.full_row_count = rows;
    source.row_count = rows;
    source.feature_count = features;
    source.nnz_count = nnz;
    source.value_size_bytes = sizeof(cp::u32);
    source.feature_axis_fingerprint = 0x6270303662656e63ull;
    source.feature_axis_fingerprint_version = 1u;
    source.row_domain_identity = 0x5651303036ull;
    source.row_offsets = row_offsets.data();
    source.block_ids = block_ids.data();
    source.local_feature_ids = local_ids.data();
    source.canonical_feature_ids = canonical_ids.data();
    source.values = values.data();

    cp::cell_block_record_requirements exact;
    const auto query_begin = std::chrono::steady_clock::now();
    check_status(cp::query_cell_block_record_requirements_host(plan, source, &exact), "query host requirements");
    const auto query_end = std::chrono::steady_clock::now();
    std::vector<cp::u32> cpu_rows(exact.row_record_offset_count), cpu_blocks(exact.record_count);
    std::vector<cp::u32> cpu_masks(exact.record_count), cpu_value_offsets(exact.record_value_offset_count);
    std::vector<unsigned char> cpu_values(exact.value_bytes);
    cp::cell_block_record_buffers cpu_buffers{
        cpu_rows.size(), cpu_blocks.size(), cpu_value_offsets.size(), cpu_values.size(),
        cpu_rows.data(), cpu_blocks.data(), cpu_masks.data(), cpu_value_offsets.data(), cpu_values.data()};
    cp::cell_block_record_view cpu_view;
    const auto cpu_begin = std::chrono::steady_clock::now();
    check_status(cp::build_cell_block_records_host(plan, source, cpu_buffers, &cpu_view), "build CPU oracle");
    const auto cpu_end = std::chrono::steady_clock::now();

    cp::cell_block_record_cuda_requirements scratch;
    check_status(cp::query_cell_block_record_cuda_requirements(nnz, &scratch), "query CUDA scratch");
    cp::u32 *d_rows = device_allocate<cp::u32>(row_offsets.size());
    cp::u32 *d_blocks = device_allocate<cp::u32>(block_ids.size());
    cp::u32 *d_locals = device_allocate<cp::u32>(local_ids.size());
    cp::u32 *d_features = device_allocate<cp::u32>(canonical_ids.size());
    cp::u32 *d_values = device_allocate<cp::u32>(values.size());
    cp::u32 *d_output_rows = device_allocate<cp::u32>(exact.row_record_offset_count);
    cp::u32 *d_output_blocks = device_allocate<cp::u32>(exact.record_count);
    cp::u32 *d_output_masks = device_allocate<cp::u32>(exact.record_count);
    cp::u32 *d_output_value_offsets = device_allocate<cp::u32>(exact.record_value_offset_count);
    cp::u32 *d_output_values = device_allocate<cp::u32>(values.size());
    cp::u32 *d_flags = device_allocate<cp::u32>(scratch.entry_prefix_count);
    cp::u32 *d_indices = device_allocate<cp::u32>(scratch.entry_prefix_count);
    unsigned char *d_cub = device_allocate<unsigned char>(scratch.cub_temporary_bytes);
    check_cuda(cudaMemcpy(d_rows, row_offsets.data(), row_offsets.size() * sizeof(cp::u32), cudaMemcpyHostToDevice), "upload rows");
    check_cuda(cudaMemcpy(d_blocks, block_ids.data(), block_ids.size() * sizeof(cp::u32), cudaMemcpyHostToDevice), "upload blocks");
    check_cuda(cudaMemcpy(d_locals, local_ids.data(), local_ids.size() * sizeof(cp::u32), cudaMemcpyHostToDevice), "upload locals");
    check_cuda(cudaMemcpy(d_features, canonical_ids.data(), canonical_ids.size() * sizeof(cp::u32), cudaMemcpyHostToDevice), "upload features");
    check_cuda(cudaMemcpy(d_values, values.data(), values.size() * sizeof(cp::u32), cudaMemcpyHostToDevice), "upload values");
    cp::ordered_plan_partition_view device_source = source;
    device_source.row_offsets = d_rows;
    device_source.block_ids = d_blocks;
    device_source.local_feature_ids = d_locals;
    device_source.canonical_feature_ids = d_features;
    device_source.values = d_values;
    cp::cell_block_record_cuda_workspace_view workspace{
        scratch.entry_prefix_count, d_flags, d_indices, d_cub, scratch.cub_temporary_bytes};
    cp::cell_block_record_buffers gpu_buffers{
        exact.row_record_offset_count, exact.record_count, exact.record_value_offset_count,
        exact.value_bytes, d_output_rows, d_output_blocks, d_output_masks,
        d_output_value_offsets, d_output_values};
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    check_cuda(cudaEventCreate(&begin), "create begin event");
    check_cuda(cudaEventCreate(&end), "create end event");
    cp::cell_block_record_view gpu_view;
    for (int i = 0; i < warmups; ++i) {
        check_status(cp::build_cell_block_records_cuda(plan, device_source, exact.record_count,
            workspace, gpu_buffers, stream, &gpu_view), "warmup CUDA builder");
    }
    check_cuda(cudaStreamSynchronize(stream), "finish warmups");
    std::vector<float> milliseconds;
    for (int i = 0; i < repeats; ++i) {
        check_cuda(cudaEventRecord(begin, stream), "record begin event");
        check_status(cp::build_cell_block_records_cuda(plan, device_source, exact.record_count,
            workspace, gpu_buffers, stream, &gpu_view), "timed CUDA builder");
        check_cuda(cudaEventRecord(end, stream), "record end event");
        check_cuda(cudaEventSynchronize(end), "finish timed CUDA builder");
        float elapsed = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed, begin, end), "measure CUDA builder");
        milliseconds.push_back(elapsed);
    }

    std::vector<cp::u32> gpu_rows(cpu_rows.size()), gpu_blocks(cpu_blocks.size()), gpu_masks(cpu_masks.size());
    std::vector<cp::u32> gpu_value_offsets(cpu_value_offsets.size()), gpu_values(values.size());
    check_cuda(cudaMemcpy(gpu_rows.data(), d_output_rows, gpu_rows.size() * sizeof(cp::u32), cudaMemcpyDeviceToHost), "download rows");
    check_cuda(cudaMemcpy(gpu_blocks.data(), d_output_blocks, gpu_blocks.size() * sizeof(cp::u32), cudaMemcpyDeviceToHost), "download blocks");
    check_cuda(cudaMemcpy(gpu_masks.data(), d_output_masks, gpu_masks.size() * sizeof(cp::u32), cudaMemcpyDeviceToHost), "download masks");
    check_cuda(cudaMemcpy(gpu_value_offsets.data(), d_output_value_offsets, gpu_value_offsets.size() * sizeof(cp::u32), cudaMemcpyDeviceToHost), "download offsets");
    check_cuda(cudaMemcpy(gpu_values.data(), d_output_values, gpu_values.size() * sizeof(cp::u32), cudaMemcpyDeviceToHost), "download values");
    check(gpu_rows == cpu_rows && gpu_blocks == cpu_blocks && gpu_masks == cpu_masks
            && gpu_value_offsets == cpu_value_offsets
            && std::equal(cpu_values.begin(), cpu_values.end(), reinterpret_cast<unsigned char *>(gpu_values.data())),
        "CPU/CUDA benchmark outputs differ");
    std::sort(milliseconds.begin(), milliseconds.end());
    const double query_ms = std::chrono::duration<double, std::milli>(query_end - query_begin).count();
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();
    const double mean_ms = std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0) / repeats;
    std::fprintf(stdout,
        "CELL_BLOCK_RECORD_BENCH device=0 gpu=Tesla_V100 sm=70 rows=%u features=%u nnz=%u records=%u "
        "block_width=%u values_per_row=%u value_bytes=%zu scratch_bytes=%zu warmups=%d repeats=%d "
        "transfers_timed=0 host_query_ms=%.3f cpu_build_ms=%.3f gpu_min_ms=%.3f gpu_median_ms=%.3f "
        "gpu_mean_ms=%.3f exact_match=1\n",
        rows, features, nnz, exact.record_count, block_width, values_per_row, exact.value_bytes,
        scratch.total_temporary_bytes, warmups, repeats, query_ms, cpu_ms, milliseconds.front(),
        milliseconds[milliseconds.size() / 2u], mean_ms);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);
    cudaFree(d_rows); cudaFree(d_blocks); cudaFree(d_locals); cudaFree(d_features); cudaFree(d_values);
    cudaFree(d_output_rows); cudaFree(d_output_blocks); cudaFree(d_output_masks);
    cudaFree(d_output_value_offsets); cudaFree(d_output_values); cudaFree(d_flags); cudaFree(d_indices); cudaFree(d_cub);
    return 0;
}
