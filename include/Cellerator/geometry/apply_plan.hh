#pragma once

#include "Cellerator/geometry/packing_plan.hh"

#include <cuda_runtime_api.h>

#include <cstddef>

namespace cellpack {

// Identity of the full row/feature universe receiving a frozen semantic plan.
// CP-BP-05 deliberately rejects sample-scoped plans: a sampled optimizer result
// must be re-frozen/re-evaluated against this full domain before application.
struct plan_application_context {
    u32 full_row_count = 0u;
    u32 feature_count = 0u;
    u64 feature_axis_fingerprint = 0u;
    u32 feature_axis_fingerprint_version = 0u;
    u64 row_domain_identity = 0u;
};

// CSR-like partition in canonical feature coordinates. Pointers are host
// resident for the host path and device resident for the CUDA path. The CUDA
// path requires the same source to have passed host validation before upload.
struct plan_application_source_view {
    u64 global_row_begin = 0u;
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 nnz_count = 0u;
    u32 value_size_bytes = 0u;
    const u32 *row_offsets = nullptr;
    const u32 *canonical_feature_ids = nullptr;
    const void *values = nullptr;
};

// Caller-owned output. Application is intentionally out-of-place so canonical
// source tuples remain available for exact reconstruction and comparison.
struct plan_application_buffers {
    std::size_t row_offset_capacity = 0u;
    std::size_t entry_capacity = 0u;
    std::size_t value_capacity_bytes = 0u;
    u32 *row_offsets = nullptr;
    u32 *block_ids = nullptr;
    u32 *local_feature_ids = nullptr;
    u32 *canonical_feature_ids = nullptr;
    void *values = nullptr;
};

struct ordered_plan_partition_view {
    u32 semantic_plan_schema_version = 0u;
    u64 global_row_begin = 0u;
    u32 full_row_count = 0u;
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 nnz_count = 0u;
    u32 value_size_bytes = 0u;
    u64 feature_axis_fingerprint = 0u;
    u32 feature_axis_fingerprint_version = 0u;
    u64 row_domain_identity = 0u;
    const u32 *row_offsets = nullptr;
    const u32 *block_ids = nullptr;
    const u32 *local_feature_ids = nullptr;
    const u32 *canonical_feature_ids = nullptr;
    const void *values = nullptr;
};

struct plan_application_host_workspace_view {
    u32 entry_capacity = 0u;
    u64 *keys = nullptr;
    u32 *source_order = nullptr;
};

struct plan_application_device_feature_view {
    u32 feature_count = 0u;
    u32 feature_block_count = 0u;
    const u32 *feature_to_block = nullptr;
    const u32 *feature_to_local = nullptr;
};

struct plan_application_cuda_workspace_view {
    u32 entry_capacity = 0u;
    u64 *keys_in = nullptr;
    u64 *keys_out = nullptr;
    u32 *source_order_in = nullptr;
    u32 *source_order_out = nullptr;
    void *cub_temporary_storage = nullptr;
    std::size_t cub_temporary_bytes = 0u;
};

struct plan_application_cuda_requirements {
    std::size_t key_bytes_each = 0u;
    std::size_t order_bytes_each = 0u;
    std::size_t cub_temporary_bytes = 0u;
    std::size_t total_temporary_bytes = 0u;
};

validation_result validate_plan_application_metadata(
    const frozen_packing_plan &plan,
    const plan_application_context &context,
    const plan_application_source_view &source);

validation_result validate_plan_application_source_host(
    const frozen_packing_plan &plan,
    const plan_application_context &context,
    const plan_application_source_view &source);

validation_result apply_frozen_plan_host(
    const frozen_packing_plan &plan,
    const plan_application_context &context,
    const plan_application_source_view &source,
    const plan_application_host_workspace_view &workspace,
    const plan_application_buffers &buffers,
    ordered_plan_partition_view *out);

validation_result query_plan_application_cuda_requirements(
    u32 row_count,
    u32 nnz_count,
    plan_application_cuda_requirements *out);

// Enqueues map, CUB segmented radix sort, and gather work without synchronizing.
// Device feature maps must be exact uploads of the frozen plan's lookup arrays.
validation_result apply_frozen_plan_cuda(
    const frozen_packing_plan &plan,
    const plan_application_context &context,
    const plan_application_source_view &device_source,
    const plan_application_device_feature_view &device_plan,
    const plan_application_cuda_workspace_view &workspace,
    const plan_application_buffers &device_buffers,
    cudaStream_t stream,
    ordered_plan_partition_view *out);

} // namespace cellpack
