#pragma once

#include <cuda_fp16.h>

// Private preprocess math helpers. Prefer these inline device helpers over
// copy-pasting per-entry normalization or gene-metric microkernels into callers.
namespace cellerator::compute::preprocess::kernels {

__device__ inline int row_is_kept(const unsigned char *__restrict__ keep_cells, unsigned int row) {
    return keep_cells == nullptr || keep_cells[row] != 0u;
}

__device__ inline float normalized_log1p_value(float value, float total_count, float target_sum) {
    const float scale = total_count > 0.0f ? target_sum / total_count : 0.0f;
    return value != 0.0f ? log1pf(value * scale) : 0.0f;
}

__device__ inline void accumulate_gene_stat(float value,
                                            unsigned int col,
                                            unsigned int cols,
                                            float *__restrict__ gene_sum,
                                            float *__restrict__ gene_detected,
                                            float *__restrict__ gene_sq_sum) {
    if (col >= cols || value == 0.0f) return;
    atomicAdd(gene_sum + col, value);
    atomicAdd(gene_detected + col, 1.0f);
    atomicAdd(gene_sq_sum + col, value * value);
}

} // namespace cellerator::compute::preprocess::kernels
