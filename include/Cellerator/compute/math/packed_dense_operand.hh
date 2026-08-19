#pragma once

#include "physical_csr.hh"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <type_traits>

namespace cellerator::compute::math {

inline constexpr u32 packed_dense_operand_schema_version = 1u;

// Canonical feature-major mathematical RHS. Either physical layout may be
// supplied; the reusable packed result is always contiguous row-major KxN.
struct canonical_dense_operand_view {
    const void *values = nullptr;
    u32 feature_count = 0u;
    u64 column_count = 0u;
    u64 leading_dimension = 0u;
    dense_layout_kind layout = dense_layout_kind::row_major;
    u32 value_size_bytes = 0u;
    feature_order_identity feature_order{};
    u64 operand_identity = 0u;
};

struct packed_dense_operand_view {
    u32 schema_version = packed_dense_operand_schema_version;
    const void *values = nullptr;
    u32 feature_count = 0u;
    u64 column_count = 0u;
    u64 leading_dimension = 0u;
    dense_layout_kind layout = dense_layout_kind::row_major;
    u32 value_size_bytes = 0u;
    feature_order_identity feature_order{};
    u64 operand_identity = 0u;
    std::size_t storage_bytes = 0u;
};

struct packed_dense_operand_requirements {
    std::size_t value_bytes = 0u;
    u64 leading_dimension = 0u;
    feature_order_identity feature_order{};
    u64 operand_identity = 0u;
};

struct packed_dense_operand_buffers {
    std::size_t value_capacity_bytes = 0u;
    void *values = nullptr;
};

physical_view_status query_packed_dense_operand_requirements(
    const cellpack::feature_weighted_row_reduction_plan_view &plan,
    const canonical_dense_operand_view &canonical,
    packed_dense_operand_requirements *out) noexcept;

physical_view_status pack_dense_operand_host(
    const cellpack::feature_weighted_row_reduction_plan_view &host_plan,
    const canonical_dense_operand_view &canonical,
    const packed_dense_operand_buffers &buffers,
    packed_dense_operand_view *out) noexcept;

// The plan permutation and both value pointers are device-resident. The call
// only enqueues the one-time pack; callers synchronize through their stream.
physical_view_status pack_dense_operand_cuda(
    const cellpack::feature_weighted_row_reduction_plan_view &device_plan,
    const canonical_dense_operand_view &device_canonical,
    const packed_dense_operand_buffers &device_buffers,
    cudaStream_t stream,
    packed_dense_operand_view *out) noexcept;

static_assert(std::is_trivially_copyable<packed_dense_operand_view>::value,
    "packed dense operand view must remain pointer-copyable");

} // namespace cellerator::compute::math
