#pragma once

#include <Cellerator/layout/projection/packed_dense_operand.hh>
#include <cuda_runtime_api.h>

namespace cellerator::layout::projection {

// The plan permutation and both value pointers are device-resident. The call
// only enqueues the one-time pack; callers synchronize through their stream.
physical_view_status pack_dense_operand_cuda(
    const cellpack::feature_weighted_row_reduction_plan_view &device_plan,
    const canonical_dense_operand_view &device_canonical,
    const packed_dense_operand_buffers &device_buffers,
    cudaStream_t stream,
    packed_dense_operand_view *out) noexcept;

}
