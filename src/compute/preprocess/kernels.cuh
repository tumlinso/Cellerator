#pragma once

#include "types.cuh"
#include "../primitives/warp_reduce.cuh"

#include <cuda_runtime.h>

namespace cellerator {
namespace compute {
namespace preprocess {
namespace kernels {

namespace reduce = ::cellerator::compute::primitives::reduce;

__global__ static void compute_cell_metrics_kernel(
    csv::compressed_view src,
    const unsigned char * __restrict__ gene_flags,
    cell_filter_params filter,
    float * __restrict__ total_counts,
    float * __restrict__ mito_counts,
    float * __restrict__ max_counts,
    unsigned int * __restrict__ detected_genes,
    unsigned char * __restrict__ keep_cells
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    unsigned int row = warp_global;

    while (row < src.rows) {
        const unsigned int begin = src.majorPtr[row];
        const unsigned int end = src.majorPtr[row + 1u];
        const unsigned int count = end - begin;
        float sum = 0.0f;
        float mito = 0.0f;
        float vmax = 0.0f;
        unsigned int idx = begin + lane;

        while (idx < end) {
            const float value = __half2float(src.val[idx]);
            const unsigned int gene = src.minorIdx[idx];
            sum += value;
            if (gene_flags != 0 && (gene_flags[gene] & gene_flag_mito) != 0u) mito += value;
            vmax = fmaxf(vmax, value);
            idx += 32u;
        }

        sum = reduce::warp_sum(sum);
        mito = reduce::warp_sum(mito);
        vmax = reduce::warp_max(vmax);

        if (lane == 0u) {
            total_counts[row] = sum;
            mito_counts[row] = mito;
            max_counts[row] = vmax;
            detected_genes[row] = count;
            if (keep_cells != 0) {
                const float mito_fraction = sum > 0.0f ? mito / sum : 0.0f;
                keep_cells[row] = (unsigned char) (sum >= filter.min_counts &&
                                                   count >= filter.min_genes &&
                                                   mito_fraction <= filter.max_mito_fraction);
            }
        }

        row += warp_stride;
    }
}

__global__ static void normalize_log1p_kernel(
    csv::compressed_view src,
    const float * __restrict__ total_counts,
    const unsigned char * __restrict__ keep_cells,
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
        const int keep = keep_cells == 0 ? 1 : keep_cells[row] != 0;
        unsigned int idx = src.majorPtr[row] + lane;
        const unsigned int end = src.majorPtr[row + 1u];

        if (keep) {
            while (idx < end) {
                const float value = __half2float(src.val[idx]) * scale;
                src.val[idx] = __float2half(log1pf(value));
                idx += 32u;
            }
        }

        row += warp_stride;
    }
}

__global__ static void square_values_kernel(
    unsigned int nnz,
    const __half * __restrict__ src,
    float * __restrict__ dst
) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int i = tid;

    while (i < nnz) {
        const float value = __half2float(src[i]);
        dst[i] = value * value;
        i += stride;
    }
}

__global__ static void convert_values_kernel(
    unsigned int nnz,
    const __half * __restrict__ src,
    float * __restrict__ dst
) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int i = tid;

    while (i < nnz) {
        dst[i] = __half2float(src[i]);
        i += stride;
    }
}

__global__ static void fill_ones_kernel(unsigned int n, float * __restrict__ dst) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int i = tid;

    while (i < n) {
        dst[i] = 1.0f;
        i += stride;
    }
}

__global__ static void add_scalar_kernel(float * __restrict__ dst, const float * __restrict__ src) {
    if (threadIdx.x == 0 && blockIdx.x == 0) dst[0] += src[0];
}

__global__ static void expand_keep_mask_kernel(unsigned int n,
                                              const unsigned char * __restrict__ keep,
                                              float * __restrict__ dst) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int i = tid;

    while (i < n) {
        dst[i] = keep != 0 && keep[i] != 0 ? 1.0f : 0.0f;
        i += stride;
    }
}

__global__ static void build_gene_filter_mask_kernel(
    unsigned int cols,
    float inv_cells,
    gene_filter_params filter,
    const float * __restrict__ sum,
    const float * __restrict__ sq_sum,
    const float * __restrict__ detected_cells,
    unsigned char * __restrict__ keep
) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int gene = tid;

    while (gene < cols) {
        const float mean = sum[gene] * inv_cells;
        const float var = fmaxf(sq_sum[gene] * inv_cells - mean * mean, 0.0f);
        keep[gene] = (unsigned char) (sum[gene] >= filter.min_sum &&
                                      detected_cells[gene] >= filter.min_detected_cells &&
                                      var >= filter.min_variance);
        gene += stride;
    }
}

} // namespace kernels
} // namespace preprocess
} // namespace compute
} // namespace cellerator
