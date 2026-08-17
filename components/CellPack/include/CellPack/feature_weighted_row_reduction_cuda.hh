#pragma once

#include "CellPack/feature_weighted_row_reduction.hh"

#include <cuda_runtime_api.h>

namespace cellpack {

// Enqueues the v1 configured-precision feature-weighted row reduction directly
// over caller-owned device-resident warp tiles. All pointers in input.plan,
// input.tiles, device_order.row_permutation, input.feature_weights, and buffers
// must be device-accessible. The output is canonical partition-local row order.
//
// This call allocates no memory, performs no transfer or synchronization, uses
// no scratch, and launches at most one kernel on caller_stream. The returned
// result view is host metadata whose row_values pointer remains caller-owned.
validation_result evaluate_feature_weighted_row_reduction_tiles_cuda(
    const feature_weighted_row_reduction_view &input,
    const local_cell_order_view &device_order,
    const feature_weighted_row_reduction_buffers &buffers,
    cudaStream_t caller_stream,
    feature_weighted_row_reduction_result_view *out);

} // namespace cellpack
