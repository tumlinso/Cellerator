/*
CP-BP-07 benchmark, 2026-08-17, Tesla V100-SXM2-16GB (`sm_70`), CUDA
12.9.86: `cellPackLocalCellOrderingBench --warmup 1 --repeats 5` ordered
65,536 rows with 16 active blocks/row in 1,024-row windows and 32-row groups.
The CUB path measured 0.233472 ms median versus 22.7307 ms for the CPU
reference, excluding host/device transfers and including all enqueued signature,
two-sort, and inverse-map work; exact integer maps agreed. Inferred ordering
reduced group-union block-id metadata from 4,194,304 original/row-NNZ bytes and
2,701,568 deterministic-random bytes to 131,072 bytes. CUB owns both stable
segmented radix sorts. Custom kernels are restricted to sparse signature/count
construction, window offsets, primary-key gathering, and the inverse map. This
integer/hash/sort workload is not Tensor Core eligible.
*/
#include "Cellerator/geometry/local_cell_ordering_cuda.hh"

#include <cub/cub.cuh>

#include <climits>
#include <limits>

namespace cellpack {
namespace {

__host__ __device__ u64 splitmix64(u64 value) noexcept {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

__global__ void build_keys_kernel(
    u32 row_count,
    u64 global_row_begin,
    local_cell_order_kind kind,
    u64 seed,
    const u32 *row_record_offsets,
    const u32 *record_block_ids,
    const u32 *record_value_offsets,
    u64 *primary_keys,
    u32 *secondary_keys,
    u32 *active_block_counts,
    u32 *row_nnz_counts,
    u32 *rows) {
    const u32 row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    const u32 begin = row_record_offsets[row];
    const u32 end = row_record_offsets[row + 1u];
    const u32 active = end - begin;
    const u32 nnz = record_value_offsets[end] - record_value_offsets[begin];
    active_block_counts[row] = active;
    row_nnz_counts[row] = nnz;
    secondary_keys[row] = active;
    rows[row] = row;
    if (kind == local_cell_order_kind::inferred_minhash) {
        if (begin == end) {
            primary_keys[row] = ~static_cast<u64>(0u);
            return;
        }
        u64 packed = 0u;
        for (u32 lane = 0u; lane < local_cell_signature_lane_count; ++lane) {
            u64 minimum = ~static_cast<u64>(0u);
            const u64 lane_seed = splitmix64(seed ^ (static_cast<u64>(lane) << 32u));
            for (u32 record = begin; record < end; ++record) {
                const u64 hash = splitmix64(lane_seed ^ static_cast<u64>(record_block_ids[record]));
                minimum = hash < minimum ? hash : minimum;
            }
            packed = (packed << 16u) | ((minimum >> 48u) & 0xffffu);
        }
        primary_keys[row] = packed;
    } else if (kind == local_cell_order_kind::original) {
        primary_keys[row] = row;
        secondary_keys[row] = 0u;
    } else if (kind == local_cell_order_kind::deterministic_random) {
        primary_keys[row] = splitmix64(seed ^ (global_row_begin + row));
        secondary_keys[row] = 0u;
    } else {
        primary_keys[row] = ~static_cast<u64>(0u) - nnz;
    }
}

__global__ void build_window_offsets_kernel(
    u32 window_count,
    u32 window_size,
    u32 row_count,
    u32 *offsets) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > window_count) return;
    const u64 offset = static_cast<u64>(index) * window_size;
    offsets[index] = offset < row_count ? static_cast<u32>(offset) : row_count;
}

__global__ void gather_primary_kernel(
    u32 row_count,
    const u64 *canonical_primary,
    const u32 *secondary_ordered_rows,
    u64 *gathered_primary) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < row_count) gathered_primary[index] = canonical_primary[secondary_ordered_rows[index]];
}

__global__ void build_inverse_kernel(
    u32 row_count,
    const u32 *permutation,
    u32 *inverse) {
    const u32 execution = blockIdx.x * blockDim.x + threadIdx.x;
    if (execution < row_count) inverse[permutation[execution]] = execution;
}

bool valid_kind(local_cell_order_kind kind) noexcept {
    return kind == local_cell_order_kind::inferred_minhash
        || kind == local_cell_order_kind::original
        || kind == local_cell_order_kind::deterministic_random
        || kind == local_cell_order_kind::row_nnz_descending;
}

validation_result cuda_failure(const char *message) {
    return validation_error(validation_code::invalid_matrix_view, invalid_id, message);
}

bool add_size(std::size_t value, std::size_t *total) noexcept {
    if (value > std::numeric_limits<std::size_t>::max() - *total) return false;
    *total += value;
    return true;
}

validation_result validate_device_metadata(
    const cell_block_record_view &records,
    const local_cell_order_config &config) {
    if (!valid_kind(config.kind) || config.window_size == 0u || config.group_width == 0u
        || config.group_width > config.window_size
        || config.window_size % config.group_width != 0u) {
        return validation_error(validation_code::invalid_signature, invalid_id,
            "CUDA local-cell order configuration is invalid");
    }
    if (records.record_schema_version != cell_block_record_schema_version
        || records.semantic_plan_schema_version != packing_plan_semantic_schema_version
        || records.geometry_identity_version != feature_block_geometry_identity_version) {
        return validation_error(validation_code::unsupported_version,
            records.record_schema_version, "CUDA cell-block record version is unsupported");
    }
    const u64 row_end = records.global_row_begin + static_cast<u64>(records.row_count);
    if (records.feature_block_geometry_identity == 0u || records.row_domain_identity == 0u
        || row_end < records.global_row_begin || row_end > records.full_row_count) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "CUDA cell-block record identity is invalid");
    }
    if (records.row_record_offsets == nullptr || records.record_value_offsets == nullptr
        || (records.record_count != 0u && records.record_block_ids == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA cell-block record array is null");
    }
    return validation_ok();
}

void set_view(
    const cell_block_record_view &records,
    const local_cell_order_config &config,
    const local_cell_order_buffers &buffers,
    local_cell_order_view *out) {
    local_cell_order_view result;
    result.order_schema_version = local_cell_order_schema_version;
    result.signature_algorithm_version = local_cell_signature_algorithm_version;
    result.kind = config.kind;
    result.window_size = config.window_size;
    result.group_width = config.group_width;
    result.seed = config.seed;
    result.ordering_identity = local_cell_order_identity(records, config);
    result.global_row_begin = records.global_row_begin;
    result.full_row_count = records.full_row_count;
    result.row_count = records.row_count;
    result.feature_block_count = records.feature_block_count;
    result.feature_block_geometry_identity = records.feature_block_geometry_identity;
    result.row_domain_identity = records.row_domain_identity;
    result.primary_keys = buffers.primary_keys;
    result.secondary_keys = buffers.secondary_keys;
    result.active_block_counts = buffers.active_block_counts;
    result.row_nnz_counts = buffers.row_nnz_counts;
    result.row_permutation = buffers.row_permutation;
    result.inverse_row_permutation = buffers.inverse_row_permutation;
    *out = result;
}

} // namespace

validation_result query_local_cell_order_cuda_requirements(
    u32 row_count,
    const local_cell_order_config &config,
    local_cell_order_cuda_requirements *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA local-cell order requirements output is null");
    }
    if (!valid_kind(config.kind) || config.window_size == 0u || config.group_width == 0u
        || config.group_width > config.window_size
        || config.window_size % config.group_width != 0u) {
        return validation_error(validation_code::invalid_signature, invalid_id,
            "CUDA local-cell order configuration is invalid");
    }
    const u64 window_count64 = (static_cast<u64>(row_count) + config.window_size - 1u)
        / config.window_size;
    if (row_count > static_cast<u32>(INT_MAX) || window_count64 > INT_MAX) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "CUB local-cell sort uses signed 32-bit item and segment counts");
    }
    local_cell_order_cuda_requirements result;
    result.row_capacity = row_count;
    result.window_offset_capacity = static_cast<std::size_t>(window_count64) + 1u;
    if (row_count != 0u) {
        std::size_t secondary_bytes = 0u, primary_bytes = 0u;
        cudaError_t error = cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, secondary_bytes,
            static_cast<const u32 *>(nullptr), static_cast<u32 *>(nullptr),
            static_cast<const u32 *>(nullptr), static_cast<u32 *>(nullptr),
            static_cast<int>(row_count), static_cast<int>(window_count64),
            static_cast<const u32 *>(nullptr), static_cast<const u32 *>(nullptr),
            0, 32);
        if (error != cudaSuccess) return cuda_failure("CUB secondary sort size query failed");
        error = cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, primary_bytes,
            static_cast<const u64 *>(nullptr), static_cast<u64 *>(nullptr),
            static_cast<const u32 *>(nullptr), static_cast<u32 *>(nullptr),
            static_cast<int>(row_count), static_cast<int>(window_count64),
            static_cast<const u32 *>(nullptr), static_cast<const u32 *>(nullptr),
            0, 64);
        if (error != cudaSuccess) return cuda_failure("CUB primary sort size query failed");
        result.cub_temporary_bytes = secondary_bytes > primary_bytes
            ? secondary_bytes : primary_bytes;
    }
    std::size_t total = 0u;
    if (!add_size(static_cast<std::size_t>(row_count) * sizeof(u64), &total)
        || !add_size(static_cast<std::size_t>(row_count) * sizeof(u64), &total)
        || !add_size(static_cast<std::size_t>(row_count) * sizeof(u32), &total)
        || !add_size(static_cast<std::size_t>(row_count) * sizeof(u32), &total)
        || !add_size(result.window_offset_capacity * sizeof(u32), &total)
        || !add_size(result.cub_temporary_bytes, &total)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "CUDA local-cell temporary byte count overflows size_t");
    }
    result.total_temporary_bytes = total;
    *out = result;
    return validation_ok();
}

validation_result build_local_cell_order_cuda(
    const cell_block_record_view &device_records,
    const local_cell_order_config &config,
    const local_cell_order_buffers &buffers,
    const local_cell_order_cuda_workspace &workspace,
    cudaStream_t stream,
    local_cell_order_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA local-cell order view output is null");
    }
    validation_result status = validate_device_metadata(device_records, config);
    if (!status) return status;
    local_cell_order_cuda_requirements required;
    status = query_local_cell_order_cuda_requirements(device_records.row_count, config, &required);
    if (!status) return status;
    const u32 rows = device_records.row_count;
    if (buffers.row_capacity < rows || workspace.row_capacity < rows
        || workspace.window_offset_capacity < required.window_offset_capacity
        || workspace.cub_temporary_bytes < required.cub_temporary_bytes) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "CUDA local-cell output or workspace capacity is insufficient");
    }
    if (rows != 0u
        && (buffers.primary_keys == nullptr || buffers.secondary_keys == nullptr
            || buffers.active_block_counts == nullptr || buffers.row_nnz_counts == nullptr
            || buffers.row_permutation == nullptr || buffers.inverse_row_permutation == nullptr
            || workspace.primary_gathered == nullptr || workspace.primary_sorted == nullptr
            || workspace.secondary_sorted == nullptr || workspace.row_scratch == nullptr
            || workspace.window_offsets == nullptr
            || (required.cub_temporary_bytes != 0u
                && workspace.cub_temporary_storage == nullptr))) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA local-cell output or workspace array is null");
    }
    if (rows != 0u) {
        const u32 blocks = (rows + 255u) / 256u;
        build_keys_kernel<<<blocks, 256u, 0u, stream>>>(rows,
            device_records.global_row_begin, config.kind, config.seed,
            device_records.row_record_offsets, device_records.record_block_ids,
            device_records.record_value_offsets, buffers.primary_keys,
            buffers.secondary_keys, buffers.active_block_counts,
            buffers.row_nnz_counts, buffers.row_permutation);
        cudaError_t error = cudaPeekAtLastError();
        if (error != cudaSuccess) return cuda_failure("CUDA local-cell key launch failed");
        const u32 window_count = static_cast<u32>(required.window_offset_capacity - 1u);
        build_window_offsets_kernel<<<(window_count + 256u) / 256u, 256u, 0u, stream>>>(
            window_count, config.window_size, rows, workspace.window_offsets);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return cuda_failure("CUDA local-cell window launch failed");
        std::size_t cub_bytes = workspace.cub_temporary_bytes;
        error = cub::DeviceSegmentedRadixSort::SortPairs(
            workspace.cub_temporary_storage, cub_bytes,
            buffers.secondary_keys, workspace.secondary_sorted,
            buffers.row_permutation, workspace.row_scratch,
            static_cast<int>(rows), static_cast<int>(window_count),
            workspace.window_offsets, workspace.window_offsets + 1u,
            0, 32, stream);
        if (error != cudaSuccess) return cuda_failure("CUB local-cell secondary sort failed");
        gather_primary_kernel<<<blocks, 256u, 0u, stream>>>(rows,
            buffers.primary_keys, workspace.row_scratch, workspace.primary_gathered);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return cuda_failure("CUDA local-cell gather launch failed");
        cub_bytes = workspace.cub_temporary_bytes;
        error = cub::DeviceSegmentedRadixSort::SortPairs(
            workspace.cub_temporary_storage, cub_bytes,
            workspace.primary_gathered, workspace.primary_sorted,
            workspace.row_scratch, buffers.row_permutation,
            static_cast<int>(rows), static_cast<int>(window_count),
            workspace.window_offsets, workspace.window_offsets + 1u,
            0, 64, stream);
        if (error != cudaSuccess) return cuda_failure("CUB local-cell primary sort failed");
        build_inverse_kernel<<<blocks, 256u, 0u, stream>>>(rows,
            buffers.row_permutation, buffers.inverse_row_permutation);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return cuda_failure("CUDA local-cell inverse launch failed");
    }
    set_view(device_records, config, buffers, out);
    return validation_ok();
}

} // namespace cellpack
