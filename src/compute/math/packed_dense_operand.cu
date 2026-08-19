/*
CP-MATH-03 custom CUDA justification, 2026-08-19 precommit:
No single cuBLAS/cuSPARSE call performs this device-resident feature-row
permutation. On Tesla V100 sm_70, 32,768 x 512 f32 (64 MiB output), the command
`cuda_controller.py run --spec /tmp/cp_math_03_foreground.json --json` measured
a 15-sample median 0.191168 ms per fused pack versus 89.9482 ms for the exact
host-issued cudaMemcpyAsync-per-row reference (470.5x); outputs agreed bitwise.
*/

#include <Cellerator/compute/math/packed_dense_operand.hh>

#include <algorithm>
#include <cstring>
#include <limits>

namespace cellerator::compute::math {
namespace {

constexpr u64 fnv_offset = 1469598103934665603ull;
constexpr u64 fnv_prime = 1099511628211ull;

physical_view_status fail(physical_view_status_code code, const char *message) noexcept {
    return {code, message};
}

void hash_u64(u64 *hash, u64 value) noexcept {
    for (u32 byte = 0u; byte < 8u; ++byte) {
        *hash ^= (value >> (byte * 8u)) & 0xffu;
        *hash *= fnv_prime;
    }
}

bool supported_value_size(u32 bytes) noexcept {
    return bytes == 2u || bytes == 4u || bytes == 8u;
}

bool valid_plan_metadata(
    const cellpack::feature_weighted_row_reduction_plan_view &plan) noexcept {
    return plan.semantic_plan_schema_version
            == cellpack::packing_plan_semantic_schema_version
        && plan.geometry_identity_version
            == cellpack::feature_block_geometry_identity_version
        && plan.feature_block_geometry_identity != 0u
        && plan.feature_block_count != 0u
        && plan.feature_block_offsets != nullptr
        && (plan.feature_count == 0u || plan.feature_permutation != nullptr);
}

physical_view_status validate_source(
    const cellpack::feature_weighted_row_reduction_plan_view &plan,
    const canonical_dense_operand_view &source,
    std::size_t *value_bytes) noexcept {
    if (value_bytes == nullptr || !valid_plan_metadata(plan)
        || source.feature_count != plan.feature_count
        || source.feature_order.schema_version != feature_order_identity_schema_version
        || source.feature_order.kind != feature_order_kind::canonical
        || source.feature_order.feature_count != source.feature_count
        || source.feature_order.feature_axis_identity == 0u
        || source.feature_order.feature_axis_identity_version == 0u
        || source.feature_order.packing_geometry_identity != 0u
        || source.operand_identity == 0u || !supported_value_size(source.value_size_bytes)
        || (source.layout != dense_layout_kind::row_major
            && source.layout != dense_layout_kind::column_major)) {
        return fail(physical_view_status_code::invalid_argument,
            "packed dense source metadata is invalid");
    }
    const u64 minimum = source.layout == dense_layout_kind::row_major
        ? source.column_count : source.feature_count;
    if ((source.feature_count != 0u && source.column_count != 0u)
        && (source.values == nullptr || source.leading_dimension < minimum)) {
        return fail(physical_view_status_code::invalid_argument,
            "packed dense source storage is invalid");
    }
    if (source.column_count != 0u
        && source.feature_count > std::numeric_limits<u64>::max()
            / source.column_count) {
        return fail(physical_view_status_code::overflow,
            "packed dense element count overflows");
    }
    const u64 elements = static_cast<u64>(source.feature_count) * source.column_count;
    if (elements > std::numeric_limits<std::size_t>::max() / source.value_size_bytes) {
        return fail(physical_view_status_code::overflow,
            "packed dense byte count overflows");
    }
    *value_bytes = static_cast<std::size_t>(elements) * source.value_size_bytes;
    return {};
}

packed_dense_operand_requirements make_requirements(
    const cellpack::feature_weighted_row_reduction_plan_view &plan,
    const canonical_dense_operand_view &source,
    std::size_t bytes) noexcept {
    packed_dense_operand_requirements result;
    result.value_bytes = bytes;
    result.leading_dimension = source.column_count;
    result.feature_order = source.feature_order;
    result.feature_order.kind = feature_order_kind::packed;
    result.feature_order.packing_geometry_identity =
        plan.feature_block_geometry_identity;
    u64 identity = fnv_offset;
    hash_u64(&identity, packed_dense_operand_schema_version);
    hash_u64(&identity, source.operand_identity);
    hash_u64(&identity, plan.feature_block_geometry_identity);
    hash_u64(&identity, source.feature_order.feature_axis_identity);
    hash_u64(&identity, source.feature_order.feature_axis_identity_version);
    hash_u64(&identity, source.feature_count);
    hash_u64(&identity, source.column_count);
    hash_u64(&identity, source.value_size_bytes);
    result.operand_identity = identity == 0u ? 1u : identity;
    return result;
}

void set_result(
    const canonical_dense_operand_view &source,
    const packed_dense_operand_requirements &required,
    const packed_dense_operand_buffers &buffers,
    packed_dense_operand_view *out) noexcept {
    packed_dense_operand_view result;
    result.values = buffers.values;
    result.feature_count = source.feature_count;
    result.column_count = source.column_count;
    result.leading_dimension = required.leading_dimension;
    result.value_size_bytes = source.value_size_bytes;
    result.feature_order = required.feature_order;
    result.operand_identity = required.operand_identity;
    result.storage_bytes = required.value_bytes;
    *out = result;
}

template <typename T>
__global__ void pack_dense_rows_kernel(
    const u32 *feature_permutation,
    const T *source,
    u32 features,
    u64 columns,
    u64 source_leading_dimension,
    dense_layout_kind source_layout,
    T *packed) {
    const u64 count = static_cast<u64>(features) * columns;
    for (u64 logical = static_cast<u64>(blockIdx.x) * blockDim.x + threadIdx.x;
         logical < count; logical += static_cast<u64>(blockDim.x) * gridDim.x) {
        const u32 execution_feature = static_cast<u32>(logical / columns);
        const u64 column = logical - static_cast<u64>(execution_feature) * columns;
        const u32 canonical_feature = feature_permutation[execution_feature];
        const u64 source_index = source_layout == dense_layout_kind::row_major
            ? static_cast<u64>(canonical_feature) * source_leading_dimension + column
            : column * source_leading_dimension + canonical_feature;
        packed[logical] = source[source_index];
    }
}

template <typename T>
cudaError_t launch_pack(
    const cellpack::feature_weighted_row_reduction_plan_view &plan,
    const canonical_dense_operand_view &source,
    void *target,
    cudaStream_t stream) noexcept {
    const u64 count = static_cast<u64>(source.feature_count) * source.column_count;
    if (count == 0u) return cudaSuccess;
    constexpr u32 threads = 256u;
    const u64 needed = (count + threads - 1u) / threads;
    const u32 blocks = static_cast<u32>(std::min<u64>(needed, 65535u));
    pack_dense_rows_kernel<<<blocks, threads, 0u, stream>>>(
        plan.feature_permutation, static_cast<const T *>(source.values),
        source.feature_count, source.column_count, source.leading_dimension,
        source.layout, static_cast<T *>(target));
    return cudaPeekAtLastError();
}

} // namespace

physical_view_status query_packed_dense_operand_requirements(
    const cellpack::feature_weighted_row_reduction_plan_view &plan,
    const canonical_dense_operand_view &canonical,
    packed_dense_operand_requirements *out) noexcept {
    if (out == nullptr) return fail(
        physical_view_status_code::invalid_argument,
        "packed dense requirements output is null");
    std::size_t bytes = 0u;
    const physical_view_status status = validate_source(plan, canonical, &bytes);
    if (!status) return status;
    *out = make_requirements(plan, canonical, bytes);
    return {};
}

physical_view_status pack_dense_operand_host(
    const cellpack::feature_weighted_row_reduction_plan_view &plan,
    const canonical_dense_operand_view &canonical,
    const packed_dense_operand_buffers &buffers,
    packed_dense_operand_view *out) noexcept {
    packed_dense_operand_requirements required;
    physical_view_status status = query_packed_dense_operand_requirements(
        plan, canonical, &required);
    if (!status) return status;
    if (out == nullptr || (required.value_bytes != 0u
            && buffers.values == canonical.values)) {
        return fail(physical_view_status_code::invalid_argument,
            "packed dense host output is null or aliases its source");
    }
    if (buffers.value_capacity_bytes < required.value_bytes
        || (required.value_bytes != 0u && buffers.values == nullptr)) {
        return fail(physical_view_status_code::insufficient_capacity,
            "packed dense host buffer is too small");
    }
    const auto *source = static_cast<const unsigned char *>(canonical.values);
    auto *target = static_cast<unsigned char *>(buffers.values);
    for (u32 execution = 0u; execution < canonical.feature_count; ++execution) {
        const u32 feature = plan.feature_permutation[execution];
        if (feature >= canonical.feature_count) return fail(
            physical_view_status_code::invalid_geometry,
            "feature permutation is out of range");
        for (u64 column = 0u; column < canonical.column_count; ++column) {
            const u64 source_index = canonical.layout == dense_layout_kind::row_major
                ? static_cast<u64>(feature) * canonical.leading_dimension + column
                : column * canonical.leading_dimension + feature;
            const u64 target_index = static_cast<u64>(execution)
                * canonical.column_count + column;
            std::memcpy(target + target_index * canonical.value_size_bytes,
                source + source_index * canonical.value_size_bytes,
                canonical.value_size_bytes);
        }
    }
    set_result(canonical, required, buffers, out);
    return {};
}

physical_view_status pack_dense_operand_cuda(
    const cellpack::feature_weighted_row_reduction_plan_view &plan,
    const canonical_dense_operand_view &canonical,
    const packed_dense_operand_buffers &buffers,
    cudaStream_t stream,
    packed_dense_operand_view *out) noexcept {
    packed_dense_operand_requirements required;
    physical_view_status status = query_packed_dense_operand_requirements(
        plan, canonical, &required);
    if (!status) return status;
    if (out == nullptr || (required.value_bytes != 0u
            && buffers.values == canonical.values)) {
        return fail(physical_view_status_code::invalid_argument,
            "packed dense device output is null or aliases its source");
    }
    if (buffers.value_capacity_bytes < required.value_bytes
        || (required.value_bytes != 0u && buffers.values == nullptr)) {
        return fail(physical_view_status_code::insufficient_capacity,
            "packed dense device buffer is too small");
    }
    cudaError_t error = cudaSuccess;
    if (canonical.value_size_bytes == 2u) {
        error = launch_pack<std::uint16_t>(plan, canonical, buffers.values, stream);
    } else if (canonical.value_size_bytes == 4u) {
        error = launch_pack<std::uint32_t>(plan, canonical, buffers.values, stream);
    } else {
        error = launch_pack<std::uint64_t>(plan, canonical, buffers.values, stream);
    }
    if (error != cudaSuccess) return fail(
        physical_view_status_code::cuda_failure,
        "packed dense CUDA launch failed");
    set_result(canonical, required, buffers, out);
    return {};
}

} // namespace cellerator::compute::math
