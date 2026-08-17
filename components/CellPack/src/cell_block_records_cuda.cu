/*
CP-BP-06 custom CUDA benchmark, 2026-08-17, Barrier-A base 1e25e11. Reference:
exact Phase A CPU builder. Target: Tesla V100-SXM2-16GB, sm_70. Shape: 65,536
rows, 30,000 features, 2,097,152 ordered NNZ, 131,072 width-16 records, u32
values. Command: ./Cellerator/build-cp-bp06/cellPackCellBlockRecordsBench.
Transfers excluded; two warmups/seven repeats. CPU build 13.055 ms; CUB plus
regular-kernel CUDA min/median/mean 0.391/0.393/0.394 ms; caller scratch
16,786,695 bytes. Every offset, block id, mask, and value byte matched exactly.
The custom kernels remain justified because record-boundary detection, compact
mask emission, and row-offset construction are project-specific sparse grammar
operations not provided as a complete maintained-library primitive; CUB owns
the general exclusive scan.
*/
#include "CellPack/cell_block_records_cuda.hh"

#include <cub/cub.cuh>

#include <climits>
#include <limits>

namespace cellpack {
namespace {

__global__ void detect_block_transitions_kernel(
    u32 nnz_count,
    const u32 *block_ids,
    u32 *record_start_flags) {
    const u32 entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= nnz_count) return;
    record_start_flags[entry] = entry == 0u || block_ids[entry] != block_ids[entry - 1u]
        ? 1u
        : 0u;
}

__global__ void mark_nonempty_row_starts_kernel(
    u32 row_count,
    const u32 *row_offsets,
    u32 *record_start_flags) {
    const u32 row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    const u32 begin = row_offsets[row];
    if (begin != row_offsets[row + 1u]) record_start_flags[begin] = 1u;
}

__global__ void emit_cell_block_records_kernel(
    u32 nnz_count,
    const u32 *source_block_ids,
    const u32 *source_local_feature_ids,
    const u32 *record_start_flags,
    const u32 *record_indices,
    u32 *record_block_ids,
    u32 *record_gene_masks,
    u32 *record_value_offsets) {
    const u32 entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= nnz_count || record_start_flags[entry] == 0u) return;
    u32 mask = 0u, cursor = entry;
    do {
        mask |= 1u << source_local_feature_ids[cursor];
        ++cursor;
    } while (cursor < nnz_count && record_start_flags[cursor] == 0u);
    const u32 record = record_indices[entry];
    record_block_ids[record] = source_block_ids[entry];
    record_gene_masks[record] = mask;
    record_value_offsets[record] = entry;
}

__global__ void emit_cell_block_row_offsets_kernel(
    u32 row_count,
    u32 nnz_count,
    u32 expected_record_count,
    const u32 *source_row_offsets,
    const u32 *record_indices,
    u32 *row_record_offsets,
    u32 *record_value_offsets) {
    const u32 row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row <= row_count) {
        row_record_offsets[row] = record_indices[source_row_offsets[row]];
    }
    if (row == 0u) record_value_offsets[expected_record_count] = nnz_count;
}

validation_result cuda_failure(const char *message) {
    return validation_error(validation_code::invalid_matrix_view, invalid_id, message);
}

bool add_size(std::size_t value, std::size_t *total) noexcept {
    if (total == nullptr || value > std::numeric_limits<std::size_t>::max() - *total) {
        return false;
    }
    *total += value;
    return true;
}

u32 launch_blocks(u32 count) noexcept {
    return count == 0u ? 0u : ((count - 1u) / 256u) + 1u;
}

validation_result validate_cuda_metadata(
    const frozen_packing_plan &plan,
    const ordered_plan_partition_view &source,
    u32 expected_record_count) {
    validation_result status = plan.validate();
    if (!status) return status;
    if (plan.identity().row_domain_kind != packing_row_domain_kind::full_dataset_identity
        || plan.maximum_feature_block_width() > cell_block_gene_mask_bits) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "CUDA cell-block records require a full-domain width-32 frozen plan");
    }
    if (source.semantic_plan_schema_version != packing_plan_semantic_schema_version) {
        return validation_error(validation_code::unsupported_version,
            source.semantic_plan_schema_version,
            "CUDA ordered partition semantic plan version is unsupported");
    }
    if (source.full_row_count != plan.row_count()
        || source.feature_count != plan.feature_count()
        || source.feature_axis_fingerprint != plan.identity().feature_axis_fingerprint
        || source.feature_axis_fingerprint_version
            != plan.identity().feature_axis_fingerprint_version
        || source.row_domain_identity != plan.identity().row_domain_identity) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "CUDA ordered partition is incompatible with the frozen plan");
    }
    const u64 row_end = source.global_row_begin + static_cast<u64>(source.row_count);
    if (row_end < source.global_row_begin || row_end > source.full_row_count
        || source.row_count == std::numeric_limits<u32>::max()) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "CUDA ordered partition is outside the full row domain");
    }
    if (source.value_size_bytes == 0u || source.row_offsets == nullptr) {
        return validation_error(source.row_offsets == nullptr
                ? validation_code::null_pointer : validation_code::invalid_matrix_view,
            invalid_id, "CUDA ordered partition row offsets or value size is invalid");
    }
    if (source.nnz_count != 0u
        && (source.block_ids == nullptr || source.local_feature_ids == nullptr
            || source.canonical_feature_ids == nullptr || source.values == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA ordered partition entry arrays are null");
    }
    if (expected_record_count > source.nnz_count
        || (source.nnz_count == 0u) != (expected_record_count == 0u)) {
        return validation_error(validation_code::invalid_matrix_view, expected_record_count,
            "CUDA expected record count is outside its exact NNZ bound");
    }
    return validation_ok();
}

validation_result validate_cuda_buffers(
    const ordered_plan_partition_view &source,
    u32 expected_record_count,
    const cell_block_record_cuda_requirements &required,
    const cell_block_record_cuda_workspace_view &workspace,
    const cell_block_record_buffers &buffers) {
    if (source.nnz_count != 0u
        && source.value_size_bytes > std::numeric_limits<std::size_t>::max() / source.nnz_count) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "CUDA cell-block value byte count overflows size_t");
    }
    const std::size_t row_offset_count = static_cast<std::size_t>(source.row_count) + 1u;
    const std::size_t value_bytes = static_cast<std::size_t>(source.nnz_count)
        * source.value_size_bytes;
    if (buffers.row_record_offset_capacity < row_offset_count
        || buffers.record_capacity < expected_record_count
        || buffers.record_value_offset_capacity
            < static_cast<std::size_t>(expected_record_count) + 1u
        || buffers.value_capacity_bytes < value_bytes) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "CUDA cell-block output capacity is insufficient");
    }
    if (buffers.row_record_offsets == nullptr || buffers.record_value_offsets == nullptr
        || (expected_record_count != 0u
            && (buffers.record_block_ids == nullptr || buffers.record_gene_masks == nullptr))
        || (source.nnz_count != 0u && buffers.values == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA cell-block output arrays are null");
    }
    if (source.nnz_count != 0u
        && (source.values == buffers.values
            || source.block_ids == buffers.record_block_ids
            || source.local_feature_ids == buffers.record_gene_masks)) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "CUDA cell-block construction is out-of-place");
    }
    if (buffers.row_record_offsets == buffers.record_value_offsets
        || (expected_record_count != 0u
            && (buffers.row_record_offsets == buffers.record_block_ids
                || buffers.row_record_offsets == buffers.record_gene_masks
                || buffers.record_value_offsets == buffers.record_block_ids
                || buffers.record_value_offsets == buffers.record_gene_masks
                || buffers.record_block_ids == buffers.record_gene_masks))) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "CUDA cell-block output arrays must be distinct");
    }
    if (source.nnz_count != 0u
        && (workspace.entry_prefix_capacity < required.entry_prefix_count
            || workspace.record_start_flags == nullptr || workspace.record_indices == nullptr
            || workspace.cub_temporary_bytes < required.cub_temporary_bytes
            || (required.cub_temporary_bytes != 0u
                && workspace.cub_temporary_storage == nullptr))) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "CUDA cell-block workspace is insufficient");
    }
    if (source.nnz_count != 0u
        && (workspace.record_start_flags == workspace.record_indices
            || workspace.record_start_flags == buffers.row_record_offsets
            || workspace.record_start_flags == buffers.record_block_ids
            || workspace.record_start_flags == buffers.record_gene_masks
            || workspace.record_start_flags == buffers.record_value_offsets
            || workspace.record_indices == buffers.row_record_offsets
            || workspace.record_indices == buffers.record_block_ids
            || workspace.record_indices == buffers.record_gene_masks
            || workspace.record_indices == buffers.record_value_offsets)) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "CUDA cell-block workspace and output arrays must be distinct");
    }
    return validation_ok();
}

void set_cuda_record_view(
    const frozen_packing_plan &plan,
    const ordered_plan_partition_view &source,
    u32 record_count,
    const cell_block_record_buffers &buffers,
    cell_block_record_view *out) {
    cell_block_record_view result;
    result.record_schema_version = cell_block_record_schema_version;
    result.semantic_plan_schema_version = packing_plan_semantic_schema_version;
    result.geometry_identity_version = feature_block_geometry_identity_version;
    result.feature_block_geometry_identity = plan.feature_block_geometry_identity();
    result.global_row_begin = source.global_row_begin;
    result.full_row_count = source.full_row_count;
    result.row_count = source.row_count;
    result.feature_count = source.feature_count;
    result.feature_block_count = plan.feature_block_count();
    result.nnz_count = source.nnz_count;
    result.record_count = record_count;
    result.value_size_bytes = source.value_size_bytes;
    result.feature_axis_fingerprint = source.feature_axis_fingerprint;
    result.feature_axis_fingerprint_version = source.feature_axis_fingerprint_version;
    result.row_domain_identity = source.row_domain_identity;
    result.row_record_offsets = buffers.row_record_offsets;
    result.record_block_ids = buffers.record_block_ids;
    result.record_gene_masks = buffers.record_gene_masks;
    result.record_value_offsets = buffers.record_value_offsets;
    result.values = buffers.values;
    *out = result;
}

} // namespace

validation_result query_cell_block_record_cuda_requirements(
    u32 nnz_count,
    cell_block_record_cuda_requirements *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA cell-block requirements output is null");
    }
    if (nnz_count >= static_cast<u32>(INT_MAX)) {
        return validation_error(validation_code::integer_overflow, nnz_count,
            "CUB cell-block scan uses a signed count including the NNZ sentinel");
    }
    cell_block_record_cuda_requirements result;
    if (nnz_count != 0u) {
        result.entry_prefix_count = static_cast<std::size_t>(nnz_count) + 1u;
        result.record_start_flag_bytes = result.entry_prefix_count * sizeof(u32);
        result.record_index_bytes = result.entry_prefix_count * sizeof(u32);
        cudaError_t error = cub::DeviceScan::ExclusiveSum(
            nullptr, result.cub_temporary_bytes,
            static_cast<const u32 *>(nullptr), static_cast<u32 *>(nullptr),
            static_cast<int>(result.entry_prefix_count));
        if (error != cudaSuccess) return cuda_failure("CUB cell-block scan size query failed");
        if (!add_size(result.record_start_flag_bytes, &result.total_temporary_bytes)
            || !add_size(result.record_index_bytes, &result.total_temporary_bytes)
            || !add_size(result.cub_temporary_bytes, &result.total_temporary_bytes)) {
            return validation_error(validation_code::integer_overflow, invalid_id,
                "CUDA cell-block temporary byte count overflows size_t");
        }
    }
    *out = result;
    return validation_ok();
}

validation_result build_cell_block_records_cuda(
    const frozen_packing_plan &plan,
    const ordered_plan_partition_view &device_source,
    u32 expected_record_count,
    const cell_block_record_cuda_workspace_view &workspace,
    const cell_block_record_buffers &device_buffers,
    cudaStream_t stream,
    cell_block_record_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA cell-block record view output is null");
    }
    validation_result status = validate_cuda_metadata(plan, device_source, expected_record_count);
    if (!status) return status;
    cell_block_record_cuda_requirements required;
    status = query_cell_block_record_cuda_requirements(device_source.nnz_count, &required);
    if (!status) return status;
    status = validate_cuda_buffers(
        device_source, expected_record_count, required, workspace, device_buffers);
    if (!status) return status;

    cudaError_t error = cudaSuccess;
    if (device_source.nnz_count == 0u) {
        error = cudaMemsetAsync(device_buffers.row_record_offsets, 0,
            (static_cast<std::size_t>(device_source.row_count) + 1u) * sizeof(u32), stream);
        if (error != cudaSuccess) return cuda_failure("CUDA empty row-record offset clear failed");
        error = cudaMemsetAsync(device_buffers.record_value_offsets, 0, sizeof(u32), stream);
        if (error != cudaSuccess) return cuda_failure("CUDA empty value-offset clear failed");
        set_cuda_record_view(plan, device_source, 0u, device_buffers, out);
        return validation_ok();
    }

    error = cudaMemsetAsync(workspace.record_start_flags, 0,
        required.record_start_flag_bytes, stream);
    if (error != cudaSuccess) return cuda_failure("CUDA record-start flag clear failed");
    detect_block_transitions_kernel<<<launch_blocks(device_source.nnz_count), 256u, 0u, stream>>>(
        device_source.nnz_count, device_source.block_ids, workspace.record_start_flags);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return cuda_failure("CUDA block-transition launch failed");
    if (device_source.row_count != 0u) {
        mark_nonempty_row_starts_kernel<<<launch_blocks(device_source.row_count), 256u, 0u, stream>>>(
            device_source.row_count, device_source.row_offsets, workspace.record_start_flags);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return cuda_failure("CUDA row-start launch failed");
    }

    std::size_t cub_bytes = workspace.cub_temporary_bytes;
    error = cub::DeviceScan::ExclusiveSum(
        workspace.cub_temporary_storage, cub_bytes,
        workspace.record_start_flags, workspace.record_indices,
        static_cast<int>(required.entry_prefix_count), stream);
    if (error != cudaSuccess) return cuda_failure("CUB cell-block exclusive scan failed");

    emit_cell_block_records_kernel<<<launch_blocks(device_source.nnz_count), 256u, 0u, stream>>>(
        device_source.nnz_count,
        device_source.block_ids,
        device_source.local_feature_ids,
        workspace.record_start_flags,
        workspace.record_indices,
        device_buffers.record_block_ids,
        device_buffers.record_gene_masks,
        device_buffers.record_value_offsets);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return cuda_failure("CUDA compact record emission launch failed");

    emit_cell_block_row_offsets_kernel<<<launch_blocks(device_source.row_count + 1u),
        256u, 0u, stream>>>(
        device_source.row_count,
        device_source.nnz_count,
        expected_record_count,
        device_source.row_offsets,
        workspace.record_indices,
        device_buffers.row_record_offsets,
        device_buffers.record_value_offsets);
    error = cudaPeekAtLastError();
    if (error != cudaSuccess) return cuda_failure("CUDA record-offset emission launch failed");

    const std::size_t value_bytes = static_cast<std::size_t>(device_source.nnz_count)
        * device_source.value_size_bytes;
    error = cudaMemcpyAsync(device_buffers.values, device_source.values, value_bytes,
        cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) return cuda_failure("CUDA compact value-byte copy failed");

    set_cuda_record_view(plan, device_source, expected_record_count, device_buffers, out);
    return validation_ok();
}

} // namespace cellpack
