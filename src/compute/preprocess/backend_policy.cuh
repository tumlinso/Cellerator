#pragma once

#include "types.cuh"

namespace cellerator {
namespace compute {
namespace preprocess {

enum {
    preprocess_operator_cell_metrics = 1u,
    preprocess_operator_normalize_log1p = 2u,
    preprocess_operator_gene_metrics = 3u,
    preprocess_operator_gene_filter_mask = 4u
};

enum {
    preprocess_backend_custom_kernel = 1u,
    preprocess_backend_cusparse_spmv = 2u
};

enum {
    preprocess_layout_native_blocked_ell = 1u,
    preprocess_layout_temporary_csr_analysis = 2u,
    preprocess_layout_native_sliced_ell = 3u
};

struct alignas(8) preprocess_backend_plan {
    unsigned int backend;
    unsigned int layout;
};

static constexpr const char *preprocess_operator_name(unsigned int op) {
    switch (op) {
        case preprocess_operator_cell_metrics: return "cell_metrics";
        case preprocess_operator_normalize_log1p: return "normalize_log1p";
        case preprocess_operator_gene_metrics: return "gene_metrics";
        case preprocess_operator_gene_filter_mask: return "gene_filter_mask";
        default: return "unknown";
    }
}

static constexpr const char *preprocess_backend_name(unsigned int backend) {
    switch (backend) {
        case preprocess_backend_custom_kernel: return "custom_kernel";
        case preprocess_backend_cusparse_spmv: return "cusparse_spmv";
        default: return "unknown";
    }
}

static constexpr const char *preprocess_layout_name(unsigned int layout) {
    switch (layout) {
        case preprocess_layout_native_blocked_ell: return "native_blocked_ell";
        case preprocess_layout_temporary_csr_analysis: return "temporary_csr_analysis";
        case preprocess_layout_native_sliced_ell: return "native_sliced_ell";
        default: return "unknown";
    }
}

static constexpr preprocess_backend_plan classify_backend(unsigned int op,
                                                          const csv::blocked_ell_view *) {
    switch (op) {
        case preprocess_operator_cell_metrics:
        case preprocess_operator_normalize_log1p:
        case preprocess_operator_gene_metrics:
        case preprocess_operator_gene_filter_mask:
            return {preprocess_backend_custom_kernel, preprocess_layout_native_blocked_ell};
        default:
            return {0u, 0u};
    }
}

static constexpr preprocess_backend_plan classify_backend(unsigned int op,
                                                          const csv::sliced_ell_view *) {
    switch (op) {
        case preprocess_operator_cell_metrics:
        case preprocess_operator_normalize_log1p:
        case preprocess_operator_gene_metrics:
        case preprocess_operator_gene_filter_mask:
            return {preprocess_backend_custom_kernel, preprocess_layout_native_sliced_ell};
        default:
            return {0u, 0u};
    }
}

static constexpr preprocess_backend_plan classify_backend(unsigned int op,
                                                          const csv::compressed_view *) {
    switch (op) {
        case preprocess_operator_gene_metrics:
            return {preprocess_backend_cusparse_spmv, preprocess_layout_temporary_csr_analysis};
        case preprocess_operator_cell_metrics:
        case preprocess_operator_normalize_log1p:
        case preprocess_operator_gene_filter_mask:
            return {preprocess_backend_custom_kernel, preprocess_layout_temporary_csr_analysis};
        default:
            return {0u, 0u};
    }
}

} // namespace preprocess
} // namespace compute
} // namespace cellerator
