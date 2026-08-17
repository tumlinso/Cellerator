/*
CP-BP-08 benchmark, 2026-08-17, Tesla V100-SXM2-16GB (`sm_70`), CUDA
12.9.86. Command: `./build-cp-bp08/cellPackWarpTilesBench`. Shape: 65,536
rows, 2,097,152 NNZ, 1,048,576 records, 2,048 tiles, 32,768 tile blocks,
width-32 rows/width-16 feature blocks, u32 values. Transfers excluded; two
warmups/seven repeats. Exact Phase C CPU build: 31.954 ms. CUDA
min/median/mean: 0.756/0.756/0.756 ms, 2.775 GNNZ/s, 8,664,075 caller-scratch
bytes, exact byte agreement. Metadata fell from 6.125 CP-BP-06 record bytes/NNZ
to 4.191 tile bytes/NNZ. The project-specific warp union/mask/value emission is
custom CUDA because no maintained library owns this compact grammar; CUB owns
all global exclusive scans. Integer/mask construction has no numerical
tolerance and is not Tensor Core eligible.
*/
#include "CellPack/warp_tiles_cuda.hh"

#include <cub/cub.cuh>

#include <climits>
#include <limits>

namespace cellpack {
namespace {

constexpr u32 warp_width = 32u;
constexpr u32 warps_per_block = 8u;
constexpr u32 threads_per_block = warp_width * warps_per_block;

__device__ u32 warp_min(u32 value) noexcept {
    for (u32 delta = warp_width / 2u; delta != 0u; delta /= 2u) {
        const u32 other = __shfl_down_sync(0xffffffffu, value, delta);
        value = other < value ? other : value;
    }
    return __shfl_sync(0xffffffffu, value, 0u);
}

__global__ void count_tile_blocks_kernel(
    u32 tile_count,
    u32 row_count,
    u32 tile_width,
    const u32 *row_permutation,
    const u32 *row_record_offsets,
    const u32 *record_block_ids,
    u32 *tile_block_counts) {
    const u32 tile = blockIdx.x;
    const u32 lane = threadIdx.x;
    if (tile >= tile_count || lane >= warp_width) return;
    const u32 execution_row = tile * tile_width + lane;
    u32 cursor = 0u, end = 0u, block = invalid_id;
    if (lane < tile_width && execution_row < row_count) {
        const u32 row = row_permutation[execution_row];
        cursor = row_record_offsets[row];
        end = row_record_offsets[row + 1u];
        block = cursor < end ? record_block_ids[cursor] : invalid_id;
    }
    u32 count = 0u;
    for (u32 next = warp_min(block); next != invalid_id; next = warp_min(block)) {
        if (block == next) {
            ++cursor;
            block = cursor < end ? record_block_ids[cursor] : invalid_id;
        }
        if (lane == 0u) ++count;
    }
    if (lane == 0u) tile_block_counts[tile] = count;
}

__global__ void emit_tile_descriptors_kernel(
    u32 tile_count,
    u32 row_count,
    u32 tile_width,
    const u32 *row_permutation,
    const u32 *row_record_offsets,
    const u32 *record_block_ids,
    const u32 *tile_block_offsets,
    u32 *tile_block_ids,
    u32 *tile_block_cell_masks,
    u32 *descriptor_row_counts,
    u32 *descriptor_tile_ids) {
    const u32 tile = blockIdx.x;
    const u32 lane = threadIdx.x;
    if (tile >= tile_count || lane >= warp_width) return;
    const u32 execution_row = tile * tile_width + lane;
    u32 cursor = 0u, end = 0u, block = invalid_id;
    if (lane < tile_width && execution_row < row_count) {
        const u32 row = row_permutation[execution_row];
        cursor = row_record_offsets[row];
        end = row_record_offsets[row + 1u];
        block = cursor < end ? record_block_ids[cursor] : invalid_id;
    }
    u32 local_descriptor = 0u;
    for (u32 next = warp_min(block); next != invalid_id; next = warp_min(block)) {
        const u32 cell_mask = __ballot_sync(0xffffffffu, block == next);
        if (lane == 0u) {
            const u32 descriptor = tile_block_offsets[tile] + local_descriptor;
            tile_block_ids[descriptor] = next;
            tile_block_cell_masks[descriptor] = cell_mask;
            descriptor_row_counts[descriptor] = __popc(cell_mask);
            descriptor_tile_ids[descriptor] = tile;
            ++local_descriptor;
        }
        if (block == next) {
            ++cursor;
            block = cursor < end ? record_block_ids[cursor] : invalid_id;
        }
    }
}

__device__ u32 find_record(
    const u32 *record_block_ids,
    u32 begin,
    u32 end,
    u32 block_id) noexcept {
    while (begin < end) {
        const u32 middle = begin + (end - begin) / 2u;
        if (record_block_ids[middle] < block_id) begin = middle + 1u;
        else end = middle;
    }
    return begin;
}

__global__ void emit_row_block_entries_kernel(
    u32 tile_block_count,
    u32 row_count,
    u32 tile_width,
    const u32 *descriptor_tile_ids,
    const u32 *tile_block_ids,
    const u32 *tile_block_cell_masks,
    const u32 *block_row_entry_offsets,
    const u32 *row_permutation,
    const u32 *row_record_offsets,
    const u32 *record_block_ids,
    const u32 *record_gene_masks,
    u32 *source_record_indices,
    u32 *row_value_counts,
    u32 *row_block_gene_masks) {
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / warp_width;
    const u32 lane = threadIdx.x & (warp_width - 1u);
    if (warp >= tile_block_count) return;
    const u32 mask = tile_block_cell_masks[warp];
    if ((mask & (1u << lane)) == 0u) return;
    const u32 tile = descriptor_tile_ids[warp];
    const u32 execution_row = tile * tile_width + lane;
    if (execution_row >= row_count) return;
    const u32 row = row_permutation[execution_row];
    const u32 begin = row_record_offsets[row], end = row_record_offsets[row + 1u];
    const u32 source_record = find_record(record_block_ids, begin, end, tile_block_ids[warp]);
    const u32 lower_mask = lane == 0u ? 0u : ((1u << lane) - 1u);
    const u32 entry = block_row_entry_offsets[warp] + __popc(mask & lower_mask);
    const u32 gene_mask = record_gene_masks[source_record];
    source_record_indices[entry] = source_record;
    row_value_counts[entry] = __popc(gene_mask);
    row_block_gene_masks[entry] = gene_mask;
}

__global__ void copy_compact_values_kernel(
    u32 row_block_entry_count,
    u32 value_size_bytes,
    const u32 *source_record_indices,
    const u32 *source_value_offsets,
    const unsigned char *source_values,
    const u32 *row_block_value_offsets,
    unsigned char *output_values) {
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / warp_width;
    const u32 lane = threadIdx.x & (warp_width - 1u);
    if (warp >= row_block_entry_count) return;
    const u32 source_record = source_record_indices[warp];
    const u32 source_begin = source_value_offsets[source_record];
    const u32 value_count = source_value_offsets[source_record + 1u] - source_begin;
    if (lane >= value_count) return;
    const u32 output_value = row_block_value_offsets[warp] + lane;
    const std::size_t source_byte = static_cast<std::size_t>(source_begin + lane)
        * value_size_bytes;
    const std::size_t output_byte = static_cast<std::size_t>(output_value)
        * value_size_bytes;
    for (u32 byte = 0u; byte < value_size_bytes; ++byte) {
        output_values[output_byte + byte] = source_values[source_byte + byte];
    }
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

u32 launch_blocks_for_warps(u32 warp_count) noexcept {
    return warp_count == 0u ? 0u : (warp_count + warps_per_block - 1u) / warps_per_block;
}

validation_result validate_cuda_metadata(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_requirements &expected,
    u32 *tile_count_out) {
    validation_result status = plan.validate();
    if (!status) return status;
    if (records.record_schema_version != cell_block_record_schema_version
        || records.semantic_plan_schema_version != packing_plan_semantic_schema_version
        || records.geometry_identity_version != feature_block_geometry_identity_version
        || order.order_schema_version != local_cell_order_schema_version
        || order.signature_algorithm_version != local_cell_signature_algorithm_version) {
        return validation_error(validation_code::unsupported_version, invalid_id,
            "CUDA warp-tile input version is unsupported");
    }
    if (plan.identity().row_domain_kind != packing_row_domain_kind::full_dataset_identity
        || plan.maximum_feature_block_width() > cell_block_gene_mask_bits
        || records.feature_block_geometry_identity != plan.feature_block_geometry_identity()
        || records.full_row_count != plan.row_count()
        || records.feature_count != plan.feature_count()
        || records.feature_block_count != plan.feature_block_count()
        || records.feature_axis_fingerprint != plan.identity().feature_axis_fingerprint
        || records.feature_axis_fingerprint_version
            != plan.identity().feature_axis_fingerprint_version
        || records.row_domain_identity != plan.identity().row_domain_identity
        || order.feature_block_geometry_identity != records.feature_block_geometry_identity
        || order.row_domain_identity != records.row_domain_identity
        || order.global_row_begin != records.global_row_begin
        || order.full_row_count != records.full_row_count
        || order.row_count != records.row_count
        || order.feature_block_count != records.feature_block_count
        || order.ordering_identity == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "CUDA warp-tile plan, record, or order identity is incompatible");
    }
    const u64 row_end = records.global_row_begin + static_cast<u64>(records.row_count);
    if (row_end < records.global_row_begin || row_end > records.full_row_count
        || records.value_size_bytes == 0u || order.group_width == 0u
        || order.group_width > warp_tile_cell_mask_bits) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "CUDA warp-tile row domain, value width, or tile width is invalid");
    }
    if (records.row_record_offsets == nullptr || records.record_value_offsets == nullptr
        || (records.row_count != 0u && order.row_permutation == nullptr)
        || (records.record_count != 0u
            && (records.record_block_ids == nullptr || records.record_gene_masks == nullptr))
        || (records.nnz_count != 0u && records.values == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA warp-tile device input array is null");
    }
    const u32 tile_count = records.row_count / order.group_width
        + (records.row_count % order.group_width != 0u ? 1u : 0u);
    if (tile_count >= static_cast<u32>(INT_MAX)
        || expected.tile_block_count >= static_cast<u32>(INT_MAX)
        || expected.row_block_entry_count >= static_cast<u32>(INT_MAX)
        || expected.tile_block_offset_count != static_cast<std::size_t>(tile_count) + 1u
        || expected.block_row_entry_offset_count
            != static_cast<std::size_t>(expected.tile_block_count) + 1u
        || expected.row_block_entry_count != records.record_count
        || expected.row_block_value_offset_count
            != static_cast<std::size_t>(records.record_count) + 1u
        || expected.tile_block_count > records.record_count
        || (records.record_count == 0u) != (expected.tile_block_count == 0u)) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "CUDA warp-tile expected counts are inconsistent");
    }
    if (records.nnz_count != 0u
        && records.value_size_bytes > std::numeric_limits<std::size_t>::max()
            / records.nnz_count) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "CUDA warp-tile value byte count overflows size_t");
    }
    const std::size_t value_bytes = static_cast<std::size_t>(records.nnz_count)
        * records.value_size_bytes;
    if (expected.value_bytes != value_bytes) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "CUDA warp-tile expected value bytes are inconsistent");
    }
    *tile_count_out = tile_count;
    return validation_ok();
}

validation_result validate_cuda_buffers(
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    u32 tile_count,
    const warp_tile_requirements &expected,
    const warp_tile_cuda_requirements &required,
    const warp_tile_cuda_workspace &workspace,
    const warp_tile_buffers &buffers) {
    if (buffers.tile_block_offset_capacity < expected.tile_block_offset_count
        || buffers.tile_block_capacity < expected.tile_block_count
        || buffers.block_row_entry_offset_capacity < expected.block_row_entry_offset_count
        || buffers.row_block_entry_capacity < expected.row_block_entry_count
        || buffers.row_block_value_offset_capacity < expected.row_block_value_offset_count
        || buffers.value_capacity_bytes < expected.value_bytes
        || workspace.tile_count_capacity < required.tile_count_capacity
        || workspace.tile_block_capacity < required.tile_block_capacity
        || workspace.row_block_entry_capacity < required.row_block_entry_capacity
        || workspace.cub_temporary_bytes < required.cub_temporary_bytes) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "CUDA warp-tile output or workspace capacity is insufficient");
    }
    if (buffers.tile_block_offsets == nullptr || buffers.block_row_entry_offsets == nullptr
        || buffers.row_block_value_offsets == nullptr || workspace.tile_block_counts == nullptr
        || workspace.descriptor_row_counts == nullptr || workspace.row_value_counts == nullptr
        || (expected.tile_block_count != 0u
            && (buffers.tile_block_ids == nullptr || buffers.tile_block_cell_masks == nullptr
                || workspace.descriptor_tile_ids == nullptr))
        || (expected.row_block_entry_count != 0u
            && (buffers.row_block_gene_masks == nullptr
                || workspace.source_record_indices == nullptr))
        || (expected.value_bytes != 0u && buffers.values == nullptr)
        || (required.cub_temporary_bytes != 0u
            && workspace.cub_temporary_storage == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA warp-tile output or workspace pointer is null");
    }
    const void *outputs[] = {
        buffers.tile_block_offsets, buffers.tile_block_ids,
        buffers.tile_block_cell_masks, buffers.block_row_entry_offsets,
        buffers.row_block_gene_masks, buffers.row_block_value_offsets, buffers.values,
        workspace.tile_block_counts, workspace.descriptor_row_counts,
        workspace.descriptor_tile_ids, workspace.source_record_indices,
        workspace.row_value_counts, workspace.cub_temporary_storage
    };
    for (std::size_t lhs = 0u; lhs < sizeof(outputs) / sizeof(outputs[0]); ++lhs) {
        if (outputs[lhs] == nullptr) continue;
        for (std::size_t rhs = lhs + 1u; rhs < sizeof(outputs) / sizeof(outputs[0]); ++rhs) {
            if (outputs[lhs] == outputs[rhs]) {
                return validation_error(validation_code::invalid_matrix_view, invalid_id,
                    "CUDA warp-tile output and workspace arrays must be distinct");
            }
        }
    }
    const void *inputs[] = {
        records.row_record_offsets, records.record_block_ids, records.record_gene_masks,
        records.record_value_offsets, records.values, order.row_permutation
    };
    for (const void *output : outputs) {
        if (output == nullptr) continue;
        for (const void *input : inputs) {
            if (input != nullptr && output == input) {
                return validation_error(validation_code::invalid_matrix_view, invalid_id,
                    "CUDA warp-tile construction is out-of-place");
            }
        }
    }
    (void) tile_count;
    return validation_ok();
}

void set_cuda_view(
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    u32 tile_count,
    const warp_tile_requirements &expected,
    const warp_tile_buffers &buffers,
    warp_tile_view *out) {
    warp_tile_view result;
    result.tile_schema_version = warp_tile_schema_version;
    result.record_schema_version = records.record_schema_version;
    result.semantic_plan_schema_version = records.semantic_plan_schema_version;
    result.geometry_identity_version = records.geometry_identity_version;
    result.order_schema_version = order.order_schema_version;
    result.tile_identity = warp_tile_identity(records, order);
    result.feature_block_geometry_identity = records.feature_block_geometry_identity;
    result.ordering_identity = order.ordering_identity;
    result.global_row_begin = records.global_row_begin;
    result.full_row_count = records.full_row_count;
    result.row_count = records.row_count;
    result.feature_count = records.feature_count;
    result.feature_block_count = records.feature_block_count;
    result.tile_row_width = order.group_width;
    result.tile_count = tile_count;
    result.nnz_count = records.nnz_count;
    result.tile_block_count = expected.tile_block_count;
    result.row_block_entry_count = expected.row_block_entry_count;
    result.value_size_bytes = records.value_size_bytes;
    result.feature_axis_fingerprint = records.feature_axis_fingerprint;
    result.feature_axis_fingerprint_version = records.feature_axis_fingerprint_version;
    result.row_domain_identity = records.row_domain_identity;
    result.tile_block_offsets = buffers.tile_block_offsets;
    result.tile_block_ids = buffers.tile_block_ids;
    result.tile_block_cell_masks = buffers.tile_block_cell_masks;
    result.block_row_entry_offsets = buffers.block_row_entry_offsets;
    result.row_block_gene_masks = buffers.row_block_gene_masks;
    result.row_block_value_offsets = buffers.row_block_value_offsets;
    result.values = buffers.values;
    *out = result;
}

} // namespace

validation_result query_warp_tile_cuda_requirements(
    u32 tile_count,
    u32 tile_block_count,
    u32 row_block_entry_count,
    warp_tile_cuda_requirements *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA warp-tile requirements output is null");
    }
    if (tile_count >= static_cast<u32>(INT_MAX)
        || tile_block_count >= static_cast<u32>(INT_MAX)
        || row_block_entry_count >= static_cast<u32>(INT_MAX)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "CUB warp-tile scans use signed counts including terminal offsets");
    }
    warp_tile_cuda_requirements result;
    result.tile_count_capacity = static_cast<std::size_t>(tile_count) + 1u;
    result.tile_block_capacity = static_cast<std::size_t>(tile_block_count) + 1u;
    result.row_block_entry_capacity = static_cast<std::size_t>(row_block_entry_count) + 1u;
    std::size_t tile_bytes = 0u, descriptor_bytes = 0u, row_bytes = 0u;
    cudaError_t error = cub::DeviceScan::ExclusiveSum(
        nullptr, tile_bytes, static_cast<const u32 *>(nullptr), static_cast<u32 *>(nullptr),
        static_cast<int>(result.tile_count_capacity));
    if (error != cudaSuccess) return cuda_failure("CUB tile-offset scan size query failed");
    error = cub::DeviceScan::ExclusiveSum(
        nullptr, descriptor_bytes,
        static_cast<const u32 *>(nullptr), static_cast<u32 *>(nullptr),
        static_cast<int>(result.tile_block_capacity));
    if (error != cudaSuccess) return cuda_failure("CUB row-entry scan size query failed");
    error = cub::DeviceScan::ExclusiveSum(
        nullptr, row_bytes, static_cast<const u32 *>(nullptr), static_cast<u32 *>(nullptr),
        static_cast<int>(result.row_block_entry_capacity));
    if (error != cudaSuccess) return cuda_failure("CUB value-offset scan size query failed");
    result.cub_temporary_bytes = tile_bytes > descriptor_bytes ? tile_bytes : descriptor_bytes;
    if (row_bytes > result.cub_temporary_bytes) result.cub_temporary_bytes = row_bytes;
    std::size_t total = 0u;
    if (!add_size(result.tile_count_capacity * sizeof(u32), &total)
        || !add_size(result.tile_block_capacity * sizeof(u32), &total)
        || !add_size(static_cast<std::size_t>(tile_block_count) * sizeof(u32), &total)
        || !add_size(static_cast<std::size_t>(row_block_entry_count) * sizeof(u32), &total)
        || !add_size(result.row_block_entry_capacity * sizeof(u32), &total)
        || !add_size(result.cub_temporary_bytes, &total)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "CUDA warp-tile temporary byte count overflows size_t");
    }
    result.total_temporary_bytes = total;
    *out = result;
    return validation_ok();
}

validation_result build_warp_tiles_cuda(
    const frozen_packing_plan &plan,
    const cell_block_record_view &device_records,
    const local_cell_order_view &device_order,
    const warp_tile_requirements &expected,
    const warp_tile_cuda_workspace &workspace,
    const warp_tile_buffers &device_buffers,
    cudaStream_t stream,
    warp_tile_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA warp-tile view output is null");
    }
    u32 tile_count = 0u;
    validation_result status = validate_cuda_metadata(
        plan, device_records, device_order, expected, &tile_count);
    if (!status) return status;
    warp_tile_cuda_requirements required;
    status = query_warp_tile_cuda_requirements(
        tile_count, expected.tile_block_count, expected.row_block_entry_count, &required);
    if (!status) return status;
    status = validate_cuda_buffers(device_records, device_order, tile_count,
        expected, required, workspace, device_buffers);
    if (!status) return status;

    cudaError_t error = cudaMemsetAsync(workspace.tile_block_counts + tile_count,
        0, sizeof(u32), stream);
    if (error != cudaSuccess) return cuda_failure("CUDA tile-count sentinel clear failed");
    if (tile_count != 0u) {
        count_tile_blocks_kernel<<<tile_count, warp_width, 0u, stream>>>(
            tile_count, device_records.row_count, device_order.group_width,
            device_order.row_permutation, device_records.row_record_offsets,
            device_records.record_block_ids, workspace.tile_block_counts);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return cuda_failure("CUDA tile-union count launch failed");
    }
    std::size_t cub_bytes = workspace.cub_temporary_bytes;
    error = cub::DeviceScan::ExclusiveSum(workspace.cub_temporary_storage, cub_bytes,
        workspace.tile_block_counts, device_buffers.tile_block_offsets,
        static_cast<int>(required.tile_count_capacity), stream);
    if (error != cudaSuccess) return cuda_failure("CUB tile-offset scan failed");

    error = cudaMemsetAsync(
        workspace.descriptor_row_counts + expected.tile_block_count, 0, sizeof(u32), stream);
    if (error != cudaSuccess) return cuda_failure("CUDA descriptor-count sentinel clear failed");
    if (tile_count != 0u) {
        emit_tile_descriptors_kernel<<<tile_count, warp_width, 0u, stream>>>(
            tile_count, device_records.row_count, device_order.group_width,
            device_order.row_permutation, device_records.row_record_offsets,
            device_records.record_block_ids, device_buffers.tile_block_offsets,
            device_buffers.tile_block_ids, device_buffers.tile_block_cell_masks,
            workspace.descriptor_row_counts, workspace.descriptor_tile_ids);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return cuda_failure("CUDA tile descriptor launch failed");
    }
    cub_bytes = workspace.cub_temporary_bytes;
    error = cub::DeviceScan::ExclusiveSum(workspace.cub_temporary_storage, cub_bytes,
        workspace.descriptor_row_counts, device_buffers.block_row_entry_offsets,
        static_cast<int>(required.tile_block_capacity), stream);
    if (error != cudaSuccess) return cuda_failure("CUB row-entry offset scan failed");

    error = cudaMemsetAsync(
        workspace.row_value_counts + expected.row_block_entry_count, 0, sizeof(u32), stream);
    if (error != cudaSuccess) return cuda_failure("CUDA row-value sentinel clear failed");
    if (expected.tile_block_count != 0u) {
        emit_row_block_entries_kernel<<<launch_blocks_for_warps(expected.tile_block_count),
            threads_per_block, 0u, stream>>>(
            expected.tile_block_count, device_records.row_count, device_order.group_width,
            workspace.descriptor_tile_ids, device_buffers.tile_block_ids,
            device_buffers.tile_block_cell_masks, device_buffers.block_row_entry_offsets,
            device_order.row_permutation, device_records.row_record_offsets,
            device_records.record_block_ids, device_records.record_gene_masks,
            workspace.source_record_indices, workspace.row_value_counts,
            device_buffers.row_block_gene_masks);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return cuda_failure("CUDA row-block entry launch failed");
    }
    cub_bytes = workspace.cub_temporary_bytes;
    error = cub::DeviceScan::ExclusiveSum(workspace.cub_temporary_storage, cub_bytes,
        workspace.row_value_counts, device_buffers.row_block_value_offsets,
        static_cast<int>(required.row_block_entry_capacity), stream);
    if (error != cudaSuccess) return cuda_failure("CUB value-offset scan failed");

    if (expected.row_block_entry_count != 0u) {
        copy_compact_values_kernel<<<launch_blocks_for_warps(expected.row_block_entry_count),
            threads_per_block, 0u, stream>>>(
            expected.row_block_entry_count, device_records.value_size_bytes,
            workspace.source_record_indices, device_records.record_value_offsets,
            static_cast<const unsigned char *>(device_records.values),
            device_buffers.row_block_value_offsets,
            static_cast<unsigned char *>(device_buffers.values));
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return cuda_failure("CUDA compact value-copy launch failed");
    }
    set_cuda_view(device_records, device_order, tile_count, expected, device_buffers, out);
    return validation_ok();
}

} // namespace cellpack
