#pragma once

#include "../ingest/series/series_manifest.cuh"

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
    unsigned int format = ingest::series::source_unknown;
    unsigned long rows = 0;
    unsigned long cols = 0;
    unsigned long nnz = 0;
    unsigned long feature_count = 0;
    unsigned long barcode_count = 0;
    unsigned long metadata_rows = 0;
    unsigned long metadata_cols = 0;
    bool probe_ok = false;
};

struct manifest_inspection {
    std::string manifest_path;
    std::vector<source_entry> sources;
    std::vector<issue> issues;
    bool ok = false;
};

struct ingest_policy {
    unsigned long max_part_nnz = 1ul << 26ul;
    unsigned long max_window_bytes = 1ul << 30ul;
    std::size_t reader_bytes = (std::size_t) 8u << 20u;
    std::string output_path;
    bool verify_after_write = true;
    int device = 0;
};

struct planned_dataset {
    std::size_t source_index = 0;
    std::string dataset_id;
    unsigned long global_row_begin = 0;
    unsigned long global_row_end = 0;
    unsigned long rows = 0;
    unsigned long cols = 0;
    unsigned long nnz = 0;
    unsigned long part_begin = 0;
    unsigned long part_count = 0;
    unsigned long feature_count = 0;
    unsigned long barcode_count = 0;
};

struct planned_part {
    unsigned long part_id = 0;
    std::size_t source_index = 0;
    std::string dataset_id;
    unsigned long row_begin = 0;
    unsigned long row_end = 0;
    unsigned long rows = 0;
    unsigned long nnz = 0;
    std::size_t estimated_bytes = 0;
    unsigned long shard_id = 0;
};

struct planned_shard {
    unsigned long shard_id = 0;
    unsigned long part_begin = 0;
    unsigned long part_end = 0;
    unsigned long row_begin = 0;
    unsigned long row_end = 0;
    unsigned long rows = 0;
    unsigned long nnz = 0;
    std::size_t estimated_bytes = 0;
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

struct series_dataset_summary {
    std::string dataset_id;
    std::string matrix_path;
    std::string feature_path;
    std::string barcode_path;
    std::string metadata_path;
    unsigned int format = ingest::series::source_unknown;
    std::uint64_t row_begin = 0;
    std::uint64_t row_end = 0;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t nnz = 0;
};

struct series_part_summary {
    std::uint64_t part_id = 0;
    std::uint64_t row_begin = 0;
    std::uint64_t row_end = 0;
    std::uint64_t rows = 0;
    std::uint64_t nnz = 0;
    std::uint32_t dataset_id = 0;
    std::uint32_t axis = 0;
    std::uint32_t codec_id = 0;
};

struct series_shard_summary {
    std::uint64_t shard_id = 0;
    std::uint64_t part_begin = 0;
    std::uint64_t part_end = 0;
    std::uint64_t row_begin = 0;
    std::uint64_t row_end = 0;
};

struct codec_summary {
    std::uint32_t codec_id = 0;
    std::uint32_t family = 0;
    std::uint32_t value_code = 0;
    std::uint32_t scale_value_code = 0;
    std::uint32_t bits = 0;
    std::uint32_t flags = 0;
};

struct series_summary {
    std::string path;
    std::vector<issue> issues;
    std::vector<series_dataset_summary> datasets;
    std::vector<series_part_summary> parts;
    std::vector<series_shard_summary> shards;
    std::vector<codec_summary> codecs;
    std::vector<std::string> feature_names;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t nnz = 0;
    std::uint64_t num_parts = 0;
    std::uint64_t num_shards = 0;
    std::uint64_t num_datasets = 0;
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
    unsigned long parts_processed = 0;
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

manifest_inspection inspect_manifest_tsv(const std::string &manifest_path,
                                         std::size_t reader_bytes = (std::size_t) 8u << 20u);

ingest_plan plan_series_ingest(const std::vector<source_entry> &sources,
                               const ingest_policy &policy = ingest_policy());

conversion_report convert_plan_to_series_csh5(const ingest_plan &plan);

series_summary summarize_series_csh5(const std::string &path);

preprocess_summary run_preprocess_pass(const std::string &path,
                                       const preprocess_config &config = preprocess_config());

} // namespace cellerator::apps::workbench
