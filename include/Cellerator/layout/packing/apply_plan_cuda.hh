#pragma once

#include <Cellerator/layout/packing/apply_plan.hh>
#include <cuda_runtime_api.h>

namespace cellpack {

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

}
