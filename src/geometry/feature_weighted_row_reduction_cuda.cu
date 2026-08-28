/*
CP-BP-09 benchmark, 2026-08-17, Tesla V100-SXM2-16GB (`sm_70`), CUDA 12.9.86.
Command: `./build-cp-bp09/cellPackFeatureWeightedRowReductionBench`. Shape:
65,536 rows, 32,768 features, 2,097,152 f16 NNZ, width-32 row tiles, width-16
feature blocks; three warmups/eleven repeats; resident inputs/outputs with
setup, transfers, allocation, and synchronization excluded. Regular direct
tile min/median/mean was 0.016/0.017/0.017 ms for high sharing, 0.040/0.041/
0.041 ms for medium sharing, and 0.116/0.117/0.117 ms for low sharing. The
existing Cellerator f16/f32 CSR kernel measured 0.075/0.075/0.075, 0.077/0.079/
0.078, and 0.094/0.095/0.096 ms respectively. Exact configured-tolerance
agreement held. cuSPARSE was not applicable because the configured values are
f16 while the existing Cellerator cuSPARSE SpMV wrapper requires f32. Scratch
is zero and each path launches once. Effective direct bandwidth was 350.119,
157.200, and 65.263 GB/s. No low-occupancy packed specialization was retained:
the declared low-sharing regime loses to the maintained CSR fallback, and no
candidate demonstrated the required 5% repeated-median gain. The compact tile
grammar has no maintained library-native descriptor, so the regular direct
kernel remains justified for its measured high/medium-sharing regimes. This
irregular single-RHS reduction is not Tensor Core eligible.
*/
#include "Cellerator/geometry/feature_weighted_row_reduction_cuda.hh"

#include <climits>
#include <cstdint>

namespace cellpack {
namespace {

using storage_t = cellerator::real::storage_t;
using compute_t = cellerator::real::compute_t;
using accum_t = cellerator::real::accum_t;

constexpr u32 warp_width = 32u;
constexpr u32 warps_per_block = 4u;
constexpr u32 threads_per_block = warp_width * warps_per_block;

u64 splitmix64(u64 value) noexcept {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

u64 reduction_identity(const feature_weighted_row_reduction_view &input) noexcept {
    u64 identity = splitmix64(input.tiles.tile_identity);
    identity = splitmix64(identity ^ input.plan.feature_block_geometry_identity);
    identity = splitmix64(identity ^ input.feature_weight_identity);
    identity = splitmix64(identity
        ^ (static_cast<u64>(feature_weighted_row_reduction_schema_version) << 32u)
        ^ static_cast<u32>(cellerator::real::code_of<storage_t>::code));
    identity = splitmix64(identity
        ^ (static_cast<u64>(cellerator::real::code_of<compute_t>::code) << 32u)
        ^ static_cast<u32>(cellerator::real::code_of<accum_t>::code));
    return identity == 0u ? 1u : identity;
}

validation_result validate_metadata(
    const feature_weighted_row_reduction_view &input,
    const local_cell_order_view &order) {
    const warp_tile_view &tiles = input.tiles;
    if (input.schema_version != feature_weighted_row_reduction_schema_version
        || tiles.tile_schema_version != warp_tile_schema_version
        || input.plan.semantic_plan_schema_version != packing_plan_semantic_schema_version
        || tiles.semantic_plan_schema_version != packing_plan_semantic_schema_version
        || input.plan.geometry_identity_version != feature_block_geometry_identity_version
        || tiles.geometry_identity_version != feature_block_geometry_identity_version
        || order.order_schema_version != local_cell_order_schema_version
        || order.signature_algorithm_version != local_cell_signature_algorithm_version) {
        return validation_error(validation_code::unsupported_version, invalid_id,
            "CUDA weighted-row-reduction version is unsupported");
    }
    if (input.storage_type_code
            != static_cast<u32>(cellerator::real::code_of<storage_t>::code)
        || input.weight_type_code
            != static_cast<u32>(cellerator::real::code_of<compute_t>::code)
        || input.accumulation_type_code
            != static_cast<u32>(cellerator::real::code_of<accum_t>::code)
        || tiles.value_size_bytes != sizeof(storage_t)) {
        return validation_error(validation_code::invalid_matrix_view,
            tiles.value_size_bytes,
            "CUDA weighted-row-reduction numeric contract is incompatible");
    }
    if (input.feature_weight_identity == 0u || input.reduction_identity == 0u
        || tiles.tile_identity == 0u || tiles.ordering_identity == 0u
        || tiles.feature_axis_fingerprint == 0u
        || tiles.feature_axis_fingerprint_version == 0u
        || tiles.row_domain_identity == 0u
        || input.reduction_identity != reduction_identity(input)) {
        return validation_error(validation_code::invalid_signature, invalid_id,
            "CUDA weighted-row-reduction identity is invalid");
    }
    if (input.plan.feature_count != tiles.feature_count
        || input.plan.feature_block_count != tiles.feature_block_count
        || input.plan.feature_block_geometry_identity
            != tiles.feature_block_geometry_identity
        || order.feature_block_geometry_identity != tiles.feature_block_geometry_identity
        || order.ordering_identity != tiles.ordering_identity
        || order.global_row_begin != tiles.global_row_begin
        || order.full_row_count != tiles.full_row_count
        || order.row_count != tiles.row_count
        || order.feature_block_count != tiles.feature_block_count
        || order.row_domain_identity != tiles.row_domain_identity
        || order.group_width != tiles.tile_row_width) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "CUDA weighted-row-reduction plan, order, and tile domains disagree");
    }
    if (tiles.tile_row_width == 0u || tiles.tile_row_width > warp_width
        || tiles.row_count == UINT_MAX
        || tiles.tile_count != (tiles.row_count / tiles.tile_row_width
            + (tiles.row_count % tiles.tile_row_width != 0u ? 1u : 0u))
        || tiles.tile_count >= static_cast<u32>(INT_MAX)
        || tiles.tile_block_count > tiles.row_block_entry_count
        || tiles.row_block_entry_count > tiles.nnz_count
        || input.feature_weight_capacity < tiles.feature_count) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "CUDA weighted-row-reduction dimensions or capacity are invalid");
    }
    if (input.plan.feature_block_offsets == nullptr
        || (tiles.feature_count != 0u
            && (input.plan.feature_permutation == nullptr
                || input.feature_weights == nullptr))
        || tiles.tile_block_offsets == nullptr
        || tiles.block_row_entry_offsets == nullptr
        || (tiles.row_count != 0u && order.row_permutation == nullptr)
        || (tiles.tile_block_count != 0u
            && (tiles.tile_block_ids == nullptr
                || tiles.tile_block_cell_masks == nullptr))
        || (tiles.row_block_entry_count != 0u
            && (tiles.row_block_gene_masks == nullptr
                || tiles.row_block_value_offsets == nullptr))
        || (tiles.nnz_count != 0u && tiles.values == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA weighted-row-reduction device input pointer is null");
    }
    return validation_ok();
}

validation_result validate_output(
    const feature_weighted_row_reduction_view &input,
    const local_cell_order_view &order,
    const feature_weighted_row_reduction_buffers &buffers,
    feature_weighted_row_reduction_result_view *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA weighted-row-reduction result output is null");
    }
    if (buffers.row_capacity < input.tiles.row_count) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
            "CUDA weighted-row-reduction output capacity is insufficient");
    }
    if (input.tiles.row_count != 0u && buffers.row_values == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "CUDA weighted-row-reduction output values are null");
    }
    const void *output = buffers.row_values;
    const void *inputs[] = {
        input.feature_weights,
        input.plan.feature_block_offsets,
        input.plan.feature_permutation,
        input.tiles.tile_block_offsets,
        input.tiles.tile_block_ids,
        input.tiles.tile_block_cell_masks,
        input.tiles.block_row_entry_offsets,
        input.tiles.row_block_gene_masks,
        input.tiles.row_block_value_offsets,
        input.tiles.values,
        order.row_permutation
    };
    if (output != nullptr) {
        for (const void *source : inputs) {
            if (source != nullptr && output == source) {
                return validation_error(validation_code::invalid_matrix_view, invalid_id,
                    "CUDA weighted-row-reduction output aliases an input");
            }
        }
    }
    return validation_ok();
}

__global__ void feature_weighted_row_reduction_kernel(
    feature_weighted_row_reduction_view input,
    const u32 *row_permutation,
    accum_t *row_values) {
    const u32 warp = (blockIdx.x * blockDim.x + threadIdx.x) / warp_width;
    const u32 lane = threadIdx.x & (warp_width - 1u);
    if (warp >= input.tiles.tile_count) return;

    const u32 execution_row = warp * input.tiles.tile_row_width + lane;
    const bool row_active = lane < input.tiles.tile_row_width
        && execution_row < input.tiles.row_count;
    accum_t sum{};
    const storage_t *values = static_cast<const storage_t *>(input.tiles.values);
    const u32 descriptor_begin = input.tiles.tile_block_offsets[warp];
    const u32 descriptor_end = input.tiles.tile_block_offsets[warp + 1u];
    for (u32 descriptor = descriptor_begin; descriptor < descriptor_end; ++descriptor) {
        const u32 cell_mask = input.tiles.tile_block_cell_masks[descriptor];
        const u32 lane_bit = 1u << lane;
        if (!row_active || (cell_mask & lane_bit) == 0u) continue;
        const u32 lower_mask = lane == 0u ? 0u : lane_bit - 1u;
        const u32 entry = input.tiles.block_row_entry_offsets[descriptor]
            + __popc(cell_mask & lower_mask);
        const u32 block = input.tiles.tile_block_ids[descriptor];
        const u32 block_begin = input.plan.feature_block_offsets[block];
        const u32 block_end = input.plan.feature_block_offsets[block + 1u];
        const u32 gene_mask = input.tiles.row_block_gene_masks[entry];
        u32 value = input.tiles.row_block_value_offsets[entry];
        for (u32 local = 0u; local < block_end - block_begin; ++local) {
            if ((gene_mask & (1u << local)) == 0u) continue;
            const u32 canonical_feature =
                input.plan.feature_permutation[block_begin + local];
            const compute_t product = static_cast<compute_t>(values[value])
                * input.feature_weights[canonical_feature];
            sum += static_cast<accum_t>(product);
            ++value;
        }
    }
    if (row_active) row_values[row_permutation[execution_row]] = sum;
}

void set_result(
    const feature_weighted_row_reduction_view &input,
    const feature_weighted_row_reduction_buffers &buffers,
    feature_weighted_row_reduction_result_view *out) {
    feature_weighted_row_reduction_result_view result;
    result.schema_version = input.schema_version;
    result.reduction_identity = input.reduction_identity;
    result.feature_weight_identity = input.feature_weight_identity;
    result.global_row_begin = input.tiles.global_row_begin;
    result.full_row_count = input.tiles.full_row_count;
    result.row_count = input.tiles.row_count;
    result.row_domain_identity = input.tiles.row_domain_identity;
    result.row_values = buffers.row_values;
    *out = result;
}

} // namespace

validation_result evaluate_feature_weighted_row_reduction_tiles_cuda(
    const feature_weighted_row_reduction_view &input,
    const local_cell_order_view &device_order,
    const feature_weighted_row_reduction_buffers &buffers,
    cudaStream_t caller_stream,
    feature_weighted_row_reduction_result_view *out) {
    validation_result status = validate_metadata(input, device_order);
    if (!status) return status;
    status = validate_output(input, device_order, buffers, out);
    if (!status) return status;
    set_result(input, buffers, out);
    if (input.tiles.row_count == 0u) return validation_ok();

    const u32 blocks = (input.tiles.tile_count + warps_per_block - 1u)
        / warps_per_block;
    feature_weighted_row_reduction_kernel<<<blocks, threads_per_block, 0,
        caller_stream>>>(input, device_order.row_permutation, buffers.row_values);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        *out = feature_weighted_row_reduction_result_view{};
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "CUDA weighted-row-reduction kernel launch failed");
    }
    return validation_ok();
}

} // namespace cellpack
