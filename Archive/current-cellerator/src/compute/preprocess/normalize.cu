/*
Preprocess normalization kernel note, 2026-05-06, Tesla V100-SXM2-16GB sm_70.
These layout-specific kernels are the existing separate primitive path, kept as
the reference and partial-workflow API beside the fused high-level traversal.
Validated after the split with celleratorPreprocessRuntimeTest,
QCMetricsEquivalenceTest, and QCMaskGroupsTest; no numerical divergence beyond
existing test tolerances.
*/
#include "preprocess_internal.cuh"

#include "kernels/preprocess_math.cuh"

#include <cuda_fp16.h>

namespace cellerator::compute::preprocess {

namespace {

__global__ void normalize_log1p_blocked_ell_kernel(
    cs_device::blocked_ell_view src,
    const float *__restrict__ total_counts,
    const unsigned char *__restrict__ keep_cells,
    float target_sum
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    unsigned int row = warp_global;

    while (row < src.rows) {
        const float denom = total_counts[row];
        const float scale = denom > 0.0f ? target_sum / denom : 0.0f;
        const int keep = keep_cells == nullptr ? 1 : keep_cells[row] != 0u;
        unsigned int ell_col = lane;

        if (keep) {
            while (ell_col < src.ell_cols) {
                const unsigned long offset = (unsigned long) row * src.ell_cols + ell_col;
                const float value = __half2float(src.val[offset]);
                src.val[offset] = value != 0.0f ? __float2half(log1pf(value * scale)) : __float2half(0.0f);
                ell_col += 32u;
            }
        }
        row += warp_stride;
    }
}

__global__ void normalize_log1p_sliced_ell_kernel(
    cs_device::sliced_ell_view src,
    const float *__restrict__ total_counts,
    const unsigned char *__restrict__ keep_cells,
    float target_sum
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    unsigned int row = warp_global;

    while (row < src.rows) {
        unsigned int slice = 0u, row_begin = 0u, width = 0u;
        unsigned long slot_base = 0ul;
        if (src.slice_count != 0u) {
            if (src.slice_rows == 32u) {
                slice = row >> 5;
                if (slice >= src.slice_count) slice = src.slice_count - 1u;
            } else if (src.slice_rows != 0u) {
                slice = row / src.slice_rows;
                if (slice >= src.slice_count) slice = src.slice_count - 1u;
            } else {
                while (slice + 1u < src.slice_count && row >= src.slice_row_offsets[slice + 1u]) ++slice;
            }
            row_begin = src.slice_row_offsets[slice];
            width = src.slice_widths[slice];
            slot_base = (unsigned long) src.slice_slot_offsets[slice]
                + (unsigned long) (row - row_begin) * (unsigned long) width;
        }

        const float denom = total_counts[row];
        const float scale = denom > 0.0f ? target_sum / denom : 0.0f;
        const int keep = keep_cells == nullptr ? 1 : keep_cells[row] != 0u;
        if (keep) {
            for (unsigned int slot = lane; slot < width; slot += 32u) {
                const unsigned int col = src.col_idx[slot_base + slot];
                const float value = __half2float(src.val[slot_base + slot]);
                if (col < src.cols && value != 0.0f) {
                    src.val[slot_base + slot] = __float2half(log1pf(value * scale));
                }
            }
        }
        row += warp_stride;
    }
}

__global__ void normalize_log1p_compressed_kernel(
    cs_device::compressed_view src,
    const float *__restrict__ total_counts,
    const unsigned char *__restrict__ keep_cells,
    float target_sum
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    unsigned int row = warp_global;

    while (row < src.rows) {
        const float denom = total_counts[row];
        const float scale = denom > 0.0f ? target_sum / denom : 0.0f;
        const int keep = keep_cells == nullptr ? 1 : keep_cells[row] != 0u;
        if (keep) {
            const unsigned int end = src.majorPtr[row + 1u];
            for (unsigned int idx = src.majorPtr[row] + lane; idx < end; idx += 32u) {
                const float value = __half2float(src.val[idx]);
                src.val[idx] = value != 0.0f ? __float2half(log1pf(value * scale)) : __float2half(0.0f);
            }
        }
        row += warp_stride;
    }
}

} // namespace

int normalize_log1p_inplace(cs_device::blocked_ell_view *src,
                            preprocess_workspace *workspace,
                            const float *device_total_counts,
                            const unsigned char *device_keep_cells,
                            float target_sum) {
    if (src == nullptr || workspace == nullptr || device_total_counts == nullptr) return 0;
    if (!cuda_ok(cudaSetDevice(workspace->device >= 0 ? workspace->device : 0), "cudaSetDevice preprocess normalize")) return 0;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    normalize_log1p_blocked_ell_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        device_total_counts,
        device_keep_cells,
        target_sum);
    return cuda_ok(cudaGetLastError(), "normalize_log1p_blocked_ell_kernel");
}

int normalize_log1p_inplace(cs_device::sliced_ell_view *src,
                            preprocess_workspace *workspace,
                            const float *device_total_counts,
                            const unsigned char *device_keep_cells,
                            float target_sum) {
    if (src == nullptr || workspace == nullptr || device_total_counts == nullptr) return 0;
    if (!cuda_ok(cudaSetDevice(workspace->device >= 0 ? workspace->device : 0), "cudaSetDevice preprocess normalize sliced")) return 0;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    normalize_log1p_sliced_ell_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        device_total_counts,
        device_keep_cells,
        target_sum);
    return cuda_ok(cudaGetLastError(), "normalize_log1p_sliced_ell_kernel");
}

int normalize_log1p_compressed_fallback_inplace(cs_device::compressed_view *src,
                                                preprocess_workspace *workspace,
                                                const float *device_total_counts,
                                                const unsigned char *device_keep_cells,
                                                float target_sum) {
    if (src == nullptr || workspace == nullptr || device_total_counts == nullptr) return 0;
    if (src->axis != cellshard::sparse::compressed_by_row) return 0;
    if (!cuda_ok(cudaSetDevice(workspace->device >= 0 ? workspace->device : 0), "cudaSetDevice preprocess normalize compressed")) return 0;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    normalize_log1p_compressed_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        device_total_counts,
        device_keep_cells,
        target_sum);
    return cuda_ok(cudaGetLastError(), "normalize_log1p_compressed_kernel");
}

} // namespace cellerator::compute::preprocess
