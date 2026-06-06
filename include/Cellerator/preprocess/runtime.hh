#pragma once

#include "preprocess.cuh"

#include <cstddef>
#include <cstdint>

#include <Cellerator/optimize/runtime_optimizer.hh>
#include <Cellerator/preprocess/aliases.hh>

namespace cellerator::preprocess {

enum status_code : int {
    status_ok = 0,
    status_invalid_argument = 1,
    status_not_raw_counts = 2,
    status_already_preprocessed = 3,
    status_unsupported_layout = 4
};

enum native_sparse_layout : std::uint32_t {
    native_sparse_unknown = 0u,
    native_sparse_blocked_ell = 1u,
    native_sparse_sliced_ell = 2u,
    native_sparse_compressed_fallback = 3u
};

enum value_precision : std::uint32_t {
    value_precision_unknown = 0u,
    value_precision_fp16 = 1u,
    value_precision_fp32_accumulator = 2u
};

enum preprocess_reduction_mode : std::uint32_t {
    preprocess_reduction_unknown = 0u,
    preprocess_reduction_single_device = 1u,
    preprocess_reduction_nccl = 2u,
    preprocess_reduction_peer_copy = 3u
};

struct status {
    int code;
    char message[192];
};

struct preprocess_state_view {
    const char *assay;
    const char *matrix_orientation;
    const char *matrix_state;
    const char *feature_namespace;
    unsigned int preprocess_available;
    unsigned int raw_counts_available;
    unsigned int processed_matrix_available;
};

struct adapter_source_view {
    const char *path;
    const char *format;
    const char *matrix_source;
    unsigned int allow_processed;
};

struct cellshard_stage_plan {
    native_sparse_layout layout;
    value_precision value_type;
    value_precision accumulator_type;
    unsigned int adapt_to_cellshard_first;
    unsigned int direct_external_kernels;
};

struct qc_feature_annotation_view {
    const char * const *feature_ids;
    const char * const *feature_names;
    const char * const *feature_types;
    const char * const *modalities;
    std::uint32_t feature_count;
};

struct qc_group_rule_view {
    std::uint32_t group_index;
    const char *group_name;
    const char *prefix;
    const char *exact_feature_id;
    const char *exact_feature_name;
    const char *feature_type;
    const char *modality;
};

struct preprocess_cellshard_session_options {
    const char *input_path = nullptr;
    const int *device_ids = nullptr;
    std::uint32_t device_count = 0u;
    std::uint32_t enable_peer_access = 1u;
    std::uint32_t stream_flags = cudaStreamNonBlocking;
    const preprocess_ranked_nccl_config *ranked_nccl = nullptr;

    const char *assay = "scrna";
    const char *matrix_orientation = "observations_by_features";
    const char *matrix_state = "raw_counts";
    const char *feature_namespace = "gene_symbol";
    const char *mito_prefix = "MT-";

    float target_sum = 10000.0f;
    float min_counts = 1.0f;
    std::uint32_t min_features = 1u;
    float max_group_fraction = 1.0f;
    std::uint32_t fraction_group_index = qc_group_mt;
    float min_gene_sum = 0.0f;
    float min_detected_cells = 0.0f;
    float min_variance = 0.0f;

    const std::uint32_t *feature_group_masks = nullptr;
    const char * const *group_names = nullptr;
    std::uint32_t group_count = 0u;

    const ::cellerator::optimize::optimizer_options *optimizer = nullptr;
};

struct preprocess_cellshard_session_result {
    native_sparse_layout layout = native_sparse_unknown;
    preprocess_reduction_mode reduction_mode = preprocess_reduction_unknown;

    std::uint64_t rows = 0u;
    std::uint64_t cols = 0u;
    std::uint64_t nnz = 0u;
    std::uint64_t partitions_processed = 0u;
    std::uint64_t shards_visited = 0u;
    std::uint64_t kept_cells = 0u;
    std::uint32_t kept_genes = 0u;
    std::uint32_t group_count = 0u;
    std::uint32_t device_count = 0u;
    preprocess_execution_plan execution_plan = preprocess_execution_default;
    ::cellerator::optimize::optimizer_result optimizer_result{};
    double gene_sum_checksum = 0.0;

    const char *group_names[CELLERATOR_PREPROCESS_MAX_QC_GROUPS] = {};
    std::uint32_t *feature_group_masks = nullptr;

    std::uint8_t *cell_keep = nullptr;
    float *cell_total_counts = nullptr;
    float *cell_mito_counts = nullptr;
    float *cell_max_counts = nullptr;
    std::uint32_t *cell_detected_genes = nullptr;
    float *cell_group_counts = nullptr;
    float *cell_group_pct = nullptr;

    std::uint8_t *gene_keep = nullptr;
    float *gene_sum = nullptr;
    float *gene_sq_sum = nullptr;
    float *gene_detected_cells = nullptr;
};

const char *version();

void clear_status(status *out);

int validate_raw_count_state(const preprocess_state_view *state, status *out);

int reject_double_preprocess(const preprocess_state_view *state, status *out);

int mark_mito_features_by_prefix(const char * const *feature_names,
                                 std::uint32_t feature_count,
                                 const char *prefix,
                                 unsigned char *gene_flags);

int compile_qc_feature_group_masks(const qc_feature_annotation_view *features,
                                   const qc_group_rule_view *rules,
                                   std::uint32_t rule_count,
                                   const std::uint32_t *explicit_masks,
                                   std::uint32_t *feature_group_masks);

int compile_default_qc_feature_group_masks(const qc_feature_annotation_view *features,
                                           const std::uint32_t *explicit_masks,
                                           std::uint32_t *feature_group_masks);

int plan_cellshard_adapter_stage(const adapter_source_view *source,
                                 cellshard_stage_plan *plan,
                                 status *out);

void clear(preprocess_cellshard_session_result *result);

int preprocess_cellshard_session_all_gpus(const preprocess_cellshard_session_options *options,
                                          preprocess_cellshard_session_result *result,
                                          status *out);

} // namespace cellerator::preprocess
