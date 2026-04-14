#pragma once

#include "../ingest/dataset/dataset_manifest.cuh"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::apps::workbench {

enum class issue_severity {
    info,
    warning,
    error
};

struct issue {
    issue_severity severity = issue_severity::info;
    std::string scope;
    std::string message;
};

struct source_entry {
    bool included = true;
    std::string dataset_id;
    std::string matrix_path;
    std::string feature_path;
    std::string barcode_path;
    std::string metadata_path;
    std::string matrix_source = "x";
    bool allow_processed = false;
    unsigned int format = ingest::dataset::source_unknown;
    unsigned long rows = 0;
    unsigned long cols = 0;
    unsigned long nnz = 0;
    unsigned long feature_count = 0;
    unsigned long barcode_count = 0;
    unsigned long metadata_rows = 0;
    unsigned long metadata_cols = 0;
    bool matrix_sparse = false;
    bool matrix_count_like = false;
    std::string matrix_encoding;
    bool probe_ok = false;
};

struct manifest_inspection {
    std::string manifest_path;
    std::vector<source_entry> sources;
    std::vector<issue> issues;
    bool ok = false;
};

enum class builder_path_role {
    none = 0,
    matrix,
    features,
    barcodes,
    metadata
};

struct filesystem_entry {
    std::string name;
    std::string path;
    std::uint64_t size = 0;
    bool is_regular = false;
    bool is_directory = false;
    bool is_symlink = false;
    bool readable = false;
};

struct draft_dataset {
    bool included = true;
    std::string dataset_id;
    std::string matrix_path;
    std::string feature_path;
    std::string barcode_path;
    std::string metadata_path;
    std::string matrix_source = "x";
    bool allow_processed = false;
    unsigned int format = ingest::dataset::source_unknown;
};

struct ingest_policy {
    unsigned long max_part_nnz = 1ul << 26ul;
    unsigned long convert_window_bytes = 1ul << 30ul;
    unsigned long target_shard_bytes = 1ul << 30ul;
    std::size_t reader_bytes = (std::size_t) 8u << 20u;
    unsigned int blocked_ell_block_sizes[3] = {8u, 16u, 32u};
    unsigned int blocked_ell_candidate_count = 3u;
    double blocked_ell_min_fill_ratio = 0.30;
    std::string output_path;
    std::string cache_dir;
    bool verify_after_write = true;
    int device = 0;
    bool embed_metadata = true;
    bool build_browse_cache = true;
    unsigned int browse_top_features = 16u;
    unsigned int browse_sample_rows_per_partition = 8u;
};

struct planned_dataset {
    std::size_t source_index = 0;
    std::string dataset_id;
    unsigned long global_row_begin = 0;
    unsigned long global_row_end = 0;
    unsigned long rows = 0;
    unsigned long cols = 0;
    unsigned long nnz = 0;
    unsigned long partition_begin = 0;
    unsigned long partition_count = 0;
    unsigned long feature_count = 0;
    unsigned long barcode_count = 0;
};

enum class execution_format : std::uint32_t {
    unknown = 0u,
    compressed = 1u,
    blocked_ell = 2u,
    mixed = 3u,
    bucketed_blocked_ell = 4u
};

struct planned_part {
    unsigned long partition_id = 0;
    std::size_t source_index = 0;
    std::string dataset_id;
    unsigned long row_begin = 0;
    unsigned long row_end = 0;
    unsigned long rows = 0;
    unsigned long nnz = 0;
    std::size_t estimated_bytes = 0;
    std::size_t execution_bytes = 0;
    std::size_t blocked_ell_bytes = 0;
    std::size_t bucketed_blocked_ell_bytes = 0;
    double blocked_ell_fill_ratio = 0.0;
    unsigned int blocked_ell_block_size = 0u;
    unsigned int blocked_ell_ell_cols = 0u;
    unsigned int blocked_ell_bucket_count = 1u;
    execution_format preferred_format = execution_format::blocked_ell;
    unsigned long shard_id = 0;
};

struct planned_shard {
    unsigned long shard_id = 0;
    unsigned long partition_begin = 0;
    unsigned long partition_end = 0;
    unsigned long row_begin = 0;
    unsigned long row_end = 0;
    unsigned long rows = 0;
    unsigned long nnz = 0;
    std::size_t estimated_bytes = 0;
    std::size_t execution_bytes = 0;
    std::size_t blocked_ell_bytes = 0;
    std::size_t bucketed_blocked_ell_bytes = 0;
    double blocked_ell_fill_ratio = 0.0;
    unsigned int blocked_ell_block_size = 0u;
    std::uint32_t bucketed_partition_count = 0u;
    std::uint32_t bucketed_segment_count = 0u;
    std::uint32_t preferred_pair = 0u;
    execution_format preferred_format = execution_format::blocked_ell;
};

struct ingest_plan {
    ingest_policy policy;
    std::vector<source_entry> sources;
    std::vector<planned_dataset> datasets;
    std::vector<planned_part> parts;
    std::vector<planned_shard> shards;
    std::vector<issue> issues;
    unsigned long total_rows = 0;
    unsigned long total_cols = 0;
    unsigned long total_nnz = 0;
    std::size_t total_estimated_bytes = 0;
    bool ok = false;
};

struct run_event {
    std::string phase;
    std::string message;
};

struct conversion_report {
    bool ok = false;
    std::vector<run_event> events;
    std::vector<issue> issues;
};

struct source_dataset_summary {
    std::string dataset_id;
    std::string matrix_path;
    std::string feature_path;
    std::string barcode_path;
    std::string metadata_path;
    unsigned int format = ingest::dataset::source_unknown;
    std::uint64_t row_begin = 0;
    std::uint64_t row_end = 0;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t nnz = 0;
};

struct dataset_partition_summary {
    std::uint64_t partition_id = 0;
    std::uint64_t row_begin = 0;
    std::uint64_t row_end = 0;
    std::uint64_t rows = 0;
    std::uint64_t nnz = 0;
    std::uint32_t dataset_id = 0;
    std::uint32_t axis = 0;
    std::uint32_t codec_id = 0;
    std::uint32_t execution_format = 0;
    std::uint32_t blocked_ell_block_size = 0;
    std::uint32_t blocked_ell_bucket_count = 0;
    float blocked_ell_fill_ratio = 0.0f;
    std::uint64_t execution_bytes = 0;
    std::uint64_t blocked_ell_bytes = 0;
    std::uint64_t bucketed_blocked_ell_bytes = 0;
};

struct dataset_shard_summary {
    std::uint64_t shard_id = 0;
    std::uint64_t partition_begin = 0;
    std::uint64_t partition_end = 0;
    std::uint64_t row_begin = 0;
    std::uint64_t row_end = 0;
    std::uint32_t execution_format = 0;
    std::uint32_t blocked_ell_block_size = 0;
    std::uint32_t bucketed_partition_count = 0;
    std::uint32_t bucketed_segment_count = 0;
    float blocked_ell_fill_ratio = 0.0f;
    std::uint64_t execution_bytes = 0;
    std::uint64_t bucketed_blocked_ell_bytes = 0;
    std::uint32_t preferred_pair = 0;
};

struct codec_summary {
    std::uint32_t codec_id = 0;
    std::uint32_t family = 0;
    std::uint32_t value_code = 0;
    std::uint32_t scale_value_code = 0;
    std::uint32_t bits = 0;
    std::uint32_t flags = 0;
};

struct embedded_metadata_dataset_summary {
    std::uint32_t dataset_index = 0;
    std::uint64_t row_begin = 0;
    std::uint64_t row_end = 0;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::vector<std::string> column_names;
};

struct embedded_metadata_table {
    bool available = false;
    std::string error;
    std::uint32_t dataset_index = 0;
    std::uint64_t row_begin = 0;
    std::uint64_t row_end = 0;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::vector<std::string> column_names;
    std::vector<std::string> field_values;
    std::vector<std::uint32_t> row_offsets;
};

struct observation_metadata_column_summary {
    std::string name;
    std::uint32_t type = 0;
};

struct observation_metadata_summary {
    bool available = false;
    std::uint64_t rows = 0;
    std::uint32_t cols = 0;
    std::vector<observation_metadata_column_summary> columns;
};

struct observation_metadata_column {
    std::string name;
    std::uint32_t type = 0;
    std::vector<std::string> text_values;
    std::vector<float> float32_values;
    std::vector<std::uint8_t> uint8_values;
};

struct observation_metadata_table {
    bool available = false;
    std::string error;
    std::uint64_t rows = 0;
    std::uint32_t cols = 0;
    std::vector<observation_metadata_column> columns;
};

struct browse_cache_summary {
    bool available = false;
    std::uint32_t selected_feature_count = 0;
    std::uint32_t sample_rows_per_partition = 0;
    std::vector<std::uint32_t> selected_feature_indices;
    std::vector<std::string> selected_feature_names;
    std::vector<float> gene_sum;
    std::vector<float> gene_detected;
    std::vector<float> gene_sq_sum;
    std::vector<float> dataset_feature_mean;
    std::vector<float> shard_feature_mean;
    std::vector<std::uint32_t> partition_sample_row_offsets;
    std::vector<std::uint64_t> partition_sample_global_rows;
    std::vector<float> partition_sample_values;
};

struct persisted_preprocess_summary {
    bool available = false;
    std::string assay;
    std::string matrix_orientation;
    std::string matrix_state;
    std::string pipeline_scope;
    std::string raw_matrix_name;
    std::string active_matrix_name;
    std::string feature_namespace;
    std::string mito_prefix;
    bool raw_counts_available = false;
    bool processed_matrix_available = false;
    bool normalized_log1p_metrics = false;
    bool hvg_available = false;
    bool mark_mito_from_feature_names = false;
    std::uint64_t rows = 0;
    std::uint32_t cols = 0;
    std::uint64_t nnz = 0;
    std::uint32_t partitions_processed = 0;
    std::uint32_t mito_feature_count = 0;
    float target_sum = 0.0f;
    float min_counts = 0.0f;
    std::uint32_t min_genes = 0;
    float max_mito_fraction = 0.0f;
    float min_gene_sum = 0.0f;
    float min_detected_cells = 0.0f;
    float min_variance = 0.0f;
    double kept_cells = 0.0;
    std::uint32_t kept_genes = 0;
    double gene_sum_checksum = 0.0;
};

struct persisted_preprocess_table {
    bool available = false;
    std::string error;
    persisted_preprocess_summary summary;
    std::vector<float> cell_total_counts;
    std::vector<float> cell_mito_counts;
    std::vector<float> cell_max_counts;
    std::vector<std::uint32_t> cell_detected_genes;
    std::vector<std::uint8_t> cell_keep;
    std::vector<float> gene_sum;
    std::vector<float> gene_sq_sum;
    std::vector<float> gene_detected_cells;
    std::vector<std::uint8_t> gene_keep;
    std::vector<std::uint8_t> gene_flags;
};

struct runtime_service_summary {
    bool available = false;
    std::uint32_t service_mode = 0;
    std::uint32_t live_write_mode = 0;
    std::uint32_t prefer_pack_delivery = 0;
    std::uint32_t remote_pack_delivery = 0;
    std::uint32_t single_reader_coordinator = 0;
    std::uint32_t maintenance_lock_blocks_overwrite = 0;
    std::uint64_t canonical_generation = 0;
    std::uint64_t execution_plan_generation = 0;
    std::uint64_t pack_generation = 0;
    std::uint64_t service_epoch = 0;
    std::uint64_t active_read_generation = 0;
    std::uint64_t staged_write_generation = 0;
};

struct dataset_summary {
    std::string path;
    std::string matrix_format;
    std::string payload_layout;
    std::vector<issue> issues;
    std::vector<source_dataset_summary> datasets;
    std::vector<dataset_partition_summary> partitions;
    std::vector<dataset_shard_summary> shards;
    std::vector<codec_summary> codecs;
    std::vector<std::string> feature_names;
    std::vector<embedded_metadata_dataset_summary> embedded_metadata;
    observation_metadata_summary observation_metadata;
    browse_cache_summary browse;
    persisted_preprocess_summary preprocess;
    runtime_service_summary runtime_service;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t nnz = 0;
    std::uint64_t num_partitions = 0;
    std::uint64_t num_shards = 0;
    std::uint64_t num_datasets = 0;
    std::uint32_t preferred_base_format = 0;
    bool ok = false;
};

struct preprocess_config {
    float target_sum = 10000.0f;
    float min_counts = 500.0f;
    unsigned int min_genes = 200u;
    float max_mito_fraction = 0.2f;
    float min_gene_sum = 1.0f;
    float min_detected_cells = 5.0f;
    float min_variance = 0.01f;
    int device = 0;
    bool drop_host_parts = true;
    bool mark_mito_from_feature_names = true;
    std::string mito_prefix = "MT-";
    std::string cache_dir;
};

struct preprocess_summary {
    bool ok = false;
    int device = -1;
    unsigned long partitions_processed = 0;
    unsigned long rows = 0;
    unsigned long cols = 0;
    unsigned long nnz = 0;
    double kept_cells = 0.0;
    unsigned long kept_genes = 0;
    double gene_sum_checksum = 0.0;
    std::vector<issue> issues;
};

std::string format_name(unsigned int format);
std::string severity_name(issue_severity severity);
std::string execution_format_name(execution_format format);
std::string builder_path_role_name(builder_path_role role);

builder_path_role infer_builder_path_role(const std::string &path);

std::vector<filesystem_entry> list_filesystem_entries(const std::string &dir_path,
                                                      std::vector<issue> *issues = nullptr);

std::vector<draft_dataset> discover_dataset_drafts(const std::string &dir_path,
                                                   std::vector<issue> *issues = nullptr);

std::vector<source_entry> sources_from_dataset_drafts(const std::vector<draft_dataset> &drafts);

manifest_inspection inspect_source_entries(const std::vector<source_entry> &sources,
                                           const std::string &label = "<builder>",
                                           std::size_t reader_bytes = (std::size_t) 8u << 20u);

bool export_manifest_tsv(const std::string &path,
                         const std::vector<draft_dataset> &drafts,
                         std::size_t reader_bytes = (std::size_t) 8u << 20u,
                         std::vector<issue> *issues = nullptr);

manifest_inspection inspect_manifest_tsv(const std::string &manifest_path,
                                         std::size_t reader_bytes = (std::size_t) 8u << 20u);

ingest_plan plan_dataset_ingest(const std::vector<source_entry> &sources,
                               const ingest_policy &policy = ingest_policy());

conversion_report convert_plan_to_dataset_csh5(const ingest_plan &plan);

dataset_summary summarize_dataset_csh5(const std::string &path);

embedded_metadata_table load_embedded_metadata_table(const std::string &path,
                                                     std::size_t table_index);

observation_metadata_table load_observation_metadata_table(const std::string &path);

persisted_preprocess_table load_persisted_preprocess_table(const std::string &path);

preprocess_summary run_preprocess_pass(const std::string &path,
                                       const preprocess_config &config = preprocess_config());

} // namespace cellerator::apps::workbench
