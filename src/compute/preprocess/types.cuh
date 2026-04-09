#pragma once

#include "../../../extern/CellShard/src/CellShard.hh"

#include <cstddef>

#include <cuda_runtime.h>

namespace cellerator {
namespace compute {
namespace preprocess {

namespace cs = ::cellshard;
namespace csd = ::cellshard::distributed;
namespace csv = ::cellshard::device;

enum {
    gene_flag_none = 0u,
    gene_flag_mito = 1u << 0
};

struct alignas(16) cell_filter_params {
    float min_counts;
    unsigned int min_genes;
    float max_mito_fraction;
};

struct alignas(16) gene_filter_params {
    float min_sum;
    float min_detected_cells;
    float min_variance;
};

struct alignas(16) cell_metrics_view {
    unsigned int rows;
    float *total_counts;
    float *mito_counts;
    float *max_counts;
    unsigned int *detected_genes;
    unsigned char *keep_cells;
};

struct alignas(16) gene_metrics_view {
    unsigned int cols;
    float *sum;
    float *sq_sum;
    float *detected_cells;
    unsigned char *keep_genes;
};

struct alignas(16) device_workspace {
    int device;
    cudaStream_t stream;
    int owns_stream;

    unsigned int rows_capacity;
    unsigned int cols_capacity;
    unsigned int nnz_capacity;

    void *d_cell_block;
    void *d_gene_block;
    void *d_spmv_tmp;
    std::size_t d_spmv_tmp_bytes;

    float *d_total_counts;
    float *d_mito_counts;
    float *d_max_counts;
    unsigned int *d_detected_genes;
    unsigned char *d_keep_cells;

    float *d_gene_sum;
    float *d_gene_sq_sum;
    float *d_gene_detected;
    unsigned char *d_keep_genes;
    unsigned char *d_gene_flags;
    float *d_ones_rows;
    float *d_tmp_nnz;
    float *d_kept_cells_tmp;
    float *d_active_rows;
    void *d_reduce_tmp;
    std::size_t d_reduce_tmp_bytes;
};

struct alignas(16) distributed_workspace {
    unsigned int device_count;
    device_workspace *devices;

    unsigned int cols_capacity;
    void *h_block;
    float *h_gene_sum;
    float *h_gene_sq_sum;
    float *h_gene_detected;
    float *h_active_rows;
};

struct alignas(16) part_preprocess_result {
    cell_metrics_view cell;
    gene_metrics_view gene;
    float *active_rows;
};

} // namespace preprocess
} // namespace compute
} // namespace cellerator
