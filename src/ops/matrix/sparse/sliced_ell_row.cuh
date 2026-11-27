#pragma once

#include <Cellerator/tensor/base/device_views.cuh>

namespace cellerator::ops::matrix::sparse::detail {

struct sliced_ell_row {
    unsigned long slot_offset;
    unsigned int width;
};

// An empty slice lookup yields zero slots. A nonzero slice_rows supplies the
// starting slice for uniform batches.
__device__ __forceinline__ sliced_ell_row locate_sliced_ell_row(
    const tensor::base::device::SlicedEllView &matrix, unsigned int row) noexcept {
    unsigned int slice = matrix.slice_rows != 0u ? row / matrix.slice_rows : 0u;
    while (slice + 1u < matrix.slice_count && row >= matrix.slice_row_offsets[slice + 1u]) {
        ++slice;
    }
    if (slice >= matrix.slice_count) return {0u, 0u};

    const unsigned int width = matrix.slice_widths[slice];
    const unsigned long offset = static_cast<unsigned long>(matrix.slice_slot_offsets[slice])
        + static_cast<unsigned long>(row - matrix.slice_row_offsets[slice]) * width;
    return {offset, width};
}

} // namespace cellerator::ops::matrix::sparse::detail
