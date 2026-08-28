/*
CP-BP-05 CUDA application benchmark, 2026-08-16, Cellerator worktree:
`./build-cp-bp05/cellPackApplyPlanBench --warmup 1 --repeats 5` on Tesla
V100-SXM2-16GB (`sm_70`), 65,536 rows, 30,000 features, 2,097,152 NNZ,
32 entries/row, 16-feature blocks. The CUB segmented-radix path measured
34.4607 ms minimum / 35.5920 ms mean versus a 51.4423 ms CPU reference, with
25,166,079 CUB scratch bytes and no transfers or synchronization in the timed
API. CPU/CUDA outputs matched exactly, including value bytes. CUB owns sorting;
the two custom kernels only form plan keys and gather application-specific
block/local/canonical/value outputs. A specialized short-row sort remains
deferred until it beats this same end-to-end boundary.
*/
#include "Cellerator/geometry/apply_plan.hh"

#include <cub/cub.cuh>

#include <climits>
#include <limits>

namespace cellpack {
namespace {

__global__ void map_plan_entries_kernel(
    u32 nnz_count,
    const u32 *canonical_feature_ids,
    const u32 *feature_to_block,
    const u32 *feature_to_local,
    u64 *keys,
    u32 *source_order) {
    const u32 entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= nnz_count) return;
    const u32 canonical = canonical_feature_ids[entry];
    keys[entry] = (static_cast<u64>(feature_to_block[canonical]) << 32u)
        | feature_to_local[canonical];
    source_order[entry] = entry;
}

__global__ void gather_plan_entries_kernel(
    u32 nnz_count,
    u32 value_size_bytes,
    const u64 *keys,
    const u32 *source_order,
    const u32 *source_canonical_features,
    const unsigned char *source_values,
    u32 *block_ids,
    u32 *local_feature_ids,
    u32 *canonical_feature_ids,
    unsigned char *values) {
    const u32 output_entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_entry >= nnz_count) return;
    const u32 source_entry = source_order[output_entry];
    const u64 key = keys[output_entry];
    block_ids[output_entry] = static_cast<u32>(key >> 32u);
    local_feature_ids[output_entry] = static_cast<u32>(key);
    canonical_feature_ids[output_entry] = source_canonical_features[source_entry];
    const std::size_t output_offset = static_cast<std::size_t>(output_entry) * value_size_bytes;
    const std::size_t source_offset = static_cast<std::size_t>(source_entry) * value_size_bytes;
    for (u32 byte = 0u; byte < value_size_bytes; ++byte) {
        values[output_offset + byte] = source_values[source_offset + byte];
    }
}

validation_result cuda_failure(const char *message) {
    return validation_error(validation_code::invalid_matrix_view, invalid_id, message);
}

bool add_size(std::size_t value, std::size_t *total) {
    if (total == nullptr || value > std::numeric_limits<std::size_t>::max() - *total) return false;
    *total += value;
    return true;
}

validation_result validate_device_buffers(
    const plan_application_source_view &source,
    const plan_application_buffers &buffers) {
    const std::size_t row_offsets = static_cast<std::size_t>(source.row_count) + 1u;
    if (source.value_size_bytes != 0u
        && source.nnz_count > std::numeric_limits<std::size_t>::max() / source.value_size_bytes) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "CUDA plan application value byte count overflows size_t");
    }
    const std::size_t value_bytes = static_cast<std::size_t>(source.nnz_count)
        * source.value_size_bytes;
    if (buffers.row_offset_capacity < row_offsets
        || buffers.entry_capacity < source.nnz_count
        || buffers.value_capacity_bytes < value_bytes) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "CUDA plan application output capacity is insufficient");
    }
    if (buffers.row_offsets == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA plan application output row offsets are null");
    }
    if (source.nnz_count != 0u
        && (buffers.block_ids == nullptr || buffers.local_feature_ids == nullptr
            || buffers.canonical_feature_ids == nullptr || buffers.values == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA plan application output arrays are null");
    }
    if (source.nnz_count != 0u
        && (source.canonical_feature_ids == buffers.canonical_feature_ids
            || source.values == buffers.values
            || buffers.block_ids == buffers.local_feature_ids
            || buffers.block_ids == buffers.canonical_feature_ids
            || buffers.local_feature_ids == buffers.canonical_feature_ids)) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "CUDA plan application is out-of-place and requires distinct entry buffers");
    }
    return validation_ok();
}

void set_result_view(
    const plan_application_context &context,
    const plan_application_source_view &source,
    const plan_application_buffers &buffers,
    ordered_plan_partition_view *out) {
    ordered_plan_partition_view result;
    result.semantic_plan_schema_version = packing_plan_semantic_schema_version;
    result.global_row_begin = source.global_row_begin;
    result.full_row_count = context.full_row_count;
    result.row_count = source.row_count;
    result.feature_count = source.feature_count;
    result.nnz_count = source.nnz_count;
    result.value_size_bytes = source.value_size_bytes;
    result.feature_axis_fingerprint = context.feature_axis_fingerprint;
    result.feature_axis_fingerprint_version = context.feature_axis_fingerprint_version;
    result.row_domain_identity = context.row_domain_identity;
    result.row_offsets = buffers.row_offsets;
    result.block_ids = buffers.block_ids;
    result.local_feature_ids = buffers.local_feature_ids;
    result.canonical_feature_ids = buffers.canonical_feature_ids;
    result.values = buffers.values;
    *out = result;
}

} // namespace

validation_result query_plan_application_cuda_requirements(
    u32 row_count,
    u32 nnz_count,
    plan_application_cuda_requirements *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA plan application requirements output is null");
    }
    if (row_count > static_cast<u32>(INT_MAX) || nnz_count > static_cast<u32>(INT_MAX)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "CUB segmented radix sort uses signed 32-bit item and segment counts");
    }
    plan_application_cuda_requirements result;
    result.key_bytes_each = static_cast<std::size_t>(nnz_count) * sizeof(u64);
    result.order_bytes_each = static_cast<std::size_t>(nnz_count) * sizeof(u32);
    if (row_count != 0u && nnz_count != 0u) {
        cudaError_t error = cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, result.cub_temporary_bytes,
            static_cast<const u64 *>(nullptr), static_cast<u64 *>(nullptr),
            static_cast<const u32 *>(nullptr), static_cast<u32 *>(nullptr),
            static_cast<int>(nnz_count), static_cast<int>(row_count),
            static_cast<const u32 *>(nullptr), static_cast<const u32 *>(nullptr),
            0, static_cast<int>(sizeof(u64) * 8u));
        if (error != cudaSuccess) return cuda_failure("CUB segmented radix sort size query failed");
    }
    std::size_t total = 0u;
    if (!add_size(result.key_bytes_each, &total)
        || !add_size(result.key_bytes_each, &total)
        || !add_size(result.order_bytes_each, &total)
        || !add_size(result.order_bytes_each, &total)
        || !add_size(result.cub_temporary_bytes, &total)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "CUDA plan application temporary byte count overflows size_t");
    }
    result.total_temporary_bytes = total;
    *out = result;
    return validation_ok();
}

validation_result apply_frozen_plan_cuda(
    const frozen_packing_plan &plan,
    const plan_application_context &context,
    const plan_application_source_view &device_source,
    const plan_application_device_feature_view &device_plan,
    const plan_application_cuda_workspace_view &workspace,
    const plan_application_buffers &device_buffers,
    cudaStream_t stream,
    ordered_plan_partition_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA ordered plan partition output is null");
    }
    validation_result status = validate_plan_application_metadata(plan, context, device_source);
    if (!status) return status;
    status = validate_device_buffers(device_source, device_buffers);
    if (!status) return status;
    if (device_plan.feature_count != plan.feature_count()
        || device_plan.feature_block_count != plan.feature_block_count()) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "CUDA feature lookup dimensions do not match the frozen plan");
    }
    if (device_plan.feature_to_block == nullptr || device_plan.feature_to_local == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA feature lookup arrays are null");
    }
    plan_application_cuda_requirements required;
    status = query_plan_application_cuda_requirements(
        device_source.row_count, device_source.nnz_count, &required);
    if (!status) return status;
    if (workspace.entry_capacity < device_source.nnz_count
        || (device_source.nnz_count != 0u
            && (workspace.keys_in == nullptr || workspace.keys_out == nullptr
                || workspace.source_order_in == nullptr || workspace.source_order_out == nullptr))
        || workspace.cub_temporary_bytes < required.cub_temporary_bytes
        || (required.cub_temporary_bytes != 0u && workspace.cub_temporary_storage == nullptr)) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "CUDA plan application workspace is insufficient");
    }

    cudaError_t error = cudaMemcpyAsync(device_buffers.row_offsets, device_source.row_offsets,
        (static_cast<std::size_t>(device_source.row_count) + 1u) * sizeof(u32),
        cudaMemcpyDeviceToDevice, stream);
    if (error != cudaSuccess) return cuda_failure("CUDA plan application row-offset copy failed");
    if (device_source.nnz_count != 0u) {
        const u32 blocks = (device_source.nnz_count + 255u) / 256u;
        map_plan_entries_kernel<<<blocks, 256u, 0u, stream>>>(
            device_source.nnz_count,
            device_source.canonical_feature_ids,
            device_plan.feature_to_block,
            device_plan.feature_to_local,
            workspace.keys_in,
            workspace.source_order_in);
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return cuda_failure("CUDA plan application map launch failed");

        std::size_t cub_bytes = workspace.cub_temporary_bytes;
        error = cub::DeviceSegmentedRadixSort::SortPairs(
            workspace.cub_temporary_storage, cub_bytes,
            workspace.keys_in, workspace.keys_out,
            workspace.source_order_in, workspace.source_order_out,
            static_cast<int>(device_source.nnz_count), static_cast<int>(device_source.row_count),
            device_source.row_offsets, device_source.row_offsets + 1u,
            0, static_cast<int>(sizeof(u64) * 8u), stream);
        if (error != cudaSuccess) return cuda_failure("CUB segmented radix sort failed");

        gather_plan_entries_kernel<<<blocks, 256u, 0u, stream>>>(
            device_source.nnz_count,
            device_source.value_size_bytes,
            workspace.keys_out,
            workspace.source_order_out,
            device_source.canonical_feature_ids,
            static_cast<const unsigned char *>(device_source.values),
            device_buffers.block_ids,
            device_buffers.local_feature_ids,
            device_buffers.canonical_feature_ids,
            static_cast<unsigned char *>(device_buffers.values));
        error = cudaPeekAtLastError();
        if (error != cudaSuccess) return cuda_failure("CUDA plan application gather launch failed");
    }
    set_result_view(context, device_source, device_buffers, out);
    return validation_ok();
}

} // namespace cellpack
