#include "../src/workbench/dataset_workbench.hh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace wb = ::cellerator::apps::workbench;

namespace {

std::string join_issues(const std::vector<wb::issue> &issues) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < issues.size(); ++i) {
        if (i != 0u) oss << "; ";
        oss << wb::severity_name(issues[i].severity) << ":" << issues[i].scope << ":" << issues[i].message;
    }
    return oss.str();
}

std::vector<unsigned int> get_blocked_ell_block_sizes(const wb::ingest_policy &policy) {
    std::vector<unsigned int> out;
    out.reserve(policy.blocked_ell_candidate_count);
    for (unsigned int i = 0u; i < policy.blocked_ell_candidate_count && i < 3u; ++i) {
        out.push_back(policy.blocked_ell_block_sizes[i]);
    }
    return out;
}

void set_blocked_ell_block_sizes(wb::ingest_policy &policy, const std::vector<unsigned int> &values) {
    if (values.empty()) throw py::value_error("blocked_ell_block_sizes must not be empty");
    if (values.size() > 3u) throw py::value_error("blocked_ell_block_sizes supports at most 3 candidates");
    for (std::size_t i = 0; i < values.size(); ++i) policy.blocked_ell_block_sizes[i] = values[i];
    policy.blocked_ell_candidate_count = static_cast<unsigned int>(values.size());
}

wb::manifest_inspection inspect_directory_impl(const std::string &dir_path, std::size_t reader_bytes) {
    const std::vector<wb::draft_dataset> drafts = wb::discover_dataset_drafts(dir_path, nullptr);
    const std::vector<wb::source_entry> sources = wb::sources_from_dataset_drafts(drafts);
    return wb::inspect_source_entries(sources, dir_path, reader_bytes);
}

wb::manifest_inspection export_manifest_impl(const std::string &path,
                                             const std::vector<wb::draft_dataset> &drafts,
                                             std::size_t reader_bytes) {
    std::vector<wb::issue> issues;
    if (!wb::export_manifest_tsv(path, drafts, reader_bytes, &issues)) {
        throw std::runtime_error(join_issues(issues));
    }
    return wb::inspect_manifest_tsv(path, reader_bytes);
}

} // namespace

PYBIND11_MODULE(_cellerator, m) {
    m.doc() = "Thin Cellerator Python bindings over the dataset workbench surface.";
    m.attr("__version__") = "0.1.0";

    py::enum_<wb::issue_severity>(m, "issue_severity")
        .value("info", wb::issue_severity::info)
        .value("warning", wb::issue_severity::warning)
        .value("error", wb::issue_severity::error);

    py::enum_<wb::builder_path_role>(m, "builder_path_role")
        .value("none", wb::builder_path_role::none)
        .value("matrix", wb::builder_path_role::matrix)
        .value("features", wb::builder_path_role::features)
        .value("barcodes", wb::builder_path_role::barcodes)
        .value("metadata", wb::builder_path_role::metadata);

    py::enum_<wb::execution_format>(m, "execution_format")
        .value("unknown", wb::execution_format::unknown)
        .value("compressed", wb::execution_format::compressed)
        .value("blocked_ell", wb::execution_format::blocked_ell)
        .value("mixed", wb::execution_format::mixed)
        .value("bucketed_blocked_ell", wb::execution_format::bucketed_blocked_ell);

    py::class_<wb::issue>(m, "issue")
        .def(py::init<>())
        .def_readwrite("severity", &wb::issue::severity)
        .def_readwrite("scope", &wb::issue::scope)
        .def_readwrite("message", &wb::issue::message);

    py::class_<wb::source_entry>(m, "source_entry")
        .def(py::init<>())
        .def_readwrite("included", &wb::source_entry::included)
        .def_readwrite("dataset_id", &wb::source_entry::dataset_id)
        .def_readwrite("matrix_path", &wb::source_entry::matrix_path)
        .def_readwrite("feature_path", &wb::source_entry::feature_path)
        .def_readwrite("barcode_path", &wb::source_entry::barcode_path)
        .def_readwrite("metadata_path", &wb::source_entry::metadata_path)
        .def_readwrite("matrix_source", &wb::source_entry::matrix_source)
        .def_readwrite("allow_processed", &wb::source_entry::allow_processed)
        .def_readwrite("format", &wb::source_entry::format)
        .def_readwrite("rows", &wb::source_entry::rows)
        .def_readwrite("cols", &wb::source_entry::cols)
        .def_readwrite("nnz", &wb::source_entry::nnz)
        .def_readwrite("feature_count", &wb::source_entry::feature_count)
        .def_readwrite("barcode_count", &wb::source_entry::barcode_count)
        .def_readwrite("metadata_rows", &wb::source_entry::metadata_rows)
        .def_readwrite("metadata_cols", &wb::source_entry::metadata_cols)
        .def_readwrite("matrix_sparse", &wb::source_entry::matrix_sparse)
        .def_readwrite("matrix_count_like", &wb::source_entry::matrix_count_like)
        .def_readwrite("matrix_encoding", &wb::source_entry::matrix_encoding)
        .def_readwrite("probe_ok", &wb::source_entry::probe_ok);

    py::class_<wb::manifest_inspection>(m, "manifest_inspection")
        .def(py::init<>())
        .def_readwrite("manifest_path", &wb::manifest_inspection::manifest_path)
        .def_readwrite("sources", &wb::manifest_inspection::sources)
        .def_readwrite("issues", &wb::manifest_inspection::issues)
        .def_readwrite("ok", &wb::manifest_inspection::ok);

    py::class_<wb::filesystem_entry>(m, "filesystem_entry")
        .def(py::init<>())
        .def_readwrite("name", &wb::filesystem_entry::name)
        .def_readwrite("path", &wb::filesystem_entry::path)
        .def_readwrite("size", &wb::filesystem_entry::size)
        .def_readwrite("is_regular", &wb::filesystem_entry::is_regular)
        .def_readwrite("is_directory", &wb::filesystem_entry::is_directory)
        .def_readwrite("is_symlink", &wb::filesystem_entry::is_symlink)
        .def_readwrite("readable", &wb::filesystem_entry::readable);

    py::class_<wb::draft_dataset>(m, "draft_dataset")
        .def(py::init<>())
        .def_readwrite("included", &wb::draft_dataset::included)
        .def_readwrite("dataset_id", &wb::draft_dataset::dataset_id)
        .def_readwrite("matrix_path", &wb::draft_dataset::matrix_path)
        .def_readwrite("feature_path", &wb::draft_dataset::feature_path)
        .def_readwrite("barcode_path", &wb::draft_dataset::barcode_path)
        .def_readwrite("metadata_path", &wb::draft_dataset::metadata_path)
        .def_readwrite("matrix_source", &wb::draft_dataset::matrix_source)
        .def_readwrite("allow_processed", &wb::draft_dataset::allow_processed)
        .def_readwrite("format", &wb::draft_dataset::format);

    py::class_<wb::ingest_policy>(m, "ingest_policy")
        .def(py::init<>())
        .def_readwrite("max_part_nnz", &wb::ingest_policy::max_part_nnz)
        .def_readwrite("convert_window_bytes", &wb::ingest_policy::convert_window_bytes)
        .def_readwrite("target_shard_bytes", &wb::ingest_policy::target_shard_bytes)
        .def_readwrite("reader_bytes", &wb::ingest_policy::reader_bytes)
        .def_readwrite("blocked_ell_min_fill_ratio", &wb::ingest_policy::blocked_ell_min_fill_ratio)
        .def_readwrite("output_path", &wb::ingest_policy::output_path)
        .def_readwrite("cache_dir", &wb::ingest_policy::cache_dir)
        .def_readwrite("verify_after_write", &wb::ingest_policy::verify_after_write)
        .def_readwrite("device", &wb::ingest_policy::device)
        .def_readwrite("embed_metadata", &wb::ingest_policy::embed_metadata)
        .def_readwrite("build_browse_cache", &wb::ingest_policy::build_browse_cache)
        .def_readwrite("browse_top_features", &wb::ingest_policy::browse_top_features)
        .def_readwrite("browse_sample_rows_per_partition", &wb::ingest_policy::browse_sample_rows_per_partition)
        .def_property("blocked_ell_block_sizes", &get_blocked_ell_block_sizes, &set_blocked_ell_block_sizes);

    py::class_<wb::planned_dataset>(m, "planned_dataset")
        .def(py::init<>())
        .def_readwrite("source_index", &wb::planned_dataset::source_index)
        .def_readwrite("dataset_id", &wb::planned_dataset::dataset_id)
        .def_readwrite("global_row_begin", &wb::planned_dataset::global_row_begin)
        .def_readwrite("global_row_end", &wb::planned_dataset::global_row_end)
        .def_readwrite("rows", &wb::planned_dataset::rows)
        .def_readwrite("cols", &wb::planned_dataset::cols)
        .def_readwrite("nnz", &wb::planned_dataset::nnz)
        .def_readwrite("partition_begin", &wb::planned_dataset::partition_begin)
        .def_readwrite("partition_count", &wb::planned_dataset::partition_count)
        .def_readwrite("feature_count", &wb::planned_dataset::feature_count)
        .def_readwrite("barcode_count", &wb::planned_dataset::barcode_count);

    py::class_<wb::planned_part>(m, "planned_part")
        .def(py::init<>())
        .def_readwrite("partition_id", &wb::planned_part::partition_id)
        .def_readwrite("source_index", &wb::planned_part::source_index)
        .def_readwrite("dataset_id", &wb::planned_part::dataset_id)
        .def_readwrite("row_begin", &wb::planned_part::row_begin)
        .def_readwrite("row_end", &wb::planned_part::row_end)
        .def_readwrite("rows", &wb::planned_part::rows)
        .def_readwrite("nnz", &wb::planned_part::nnz)
        .def_readwrite("estimated_bytes", &wb::planned_part::estimated_bytes)
        .def_readwrite("execution_bytes", &wb::planned_part::execution_bytes)
        .def_readwrite("blocked_ell_bytes", &wb::planned_part::blocked_ell_bytes)
        .def_readwrite("bucketed_blocked_ell_bytes", &wb::planned_part::bucketed_blocked_ell_bytes)
        .def_readwrite("blocked_ell_fill_ratio", &wb::planned_part::blocked_ell_fill_ratio)
        .def_readwrite("blocked_ell_block_size", &wb::planned_part::blocked_ell_block_size)
        .def_readwrite("blocked_ell_ell_cols", &wb::planned_part::blocked_ell_ell_cols)
        .def_readwrite("blocked_ell_bucket_count", &wb::planned_part::blocked_ell_bucket_count)
        .def_readwrite("preferred_format", &wb::planned_part::preferred_format)
        .def_readwrite("shard_id", &wb::planned_part::shard_id);

    py::class_<wb::planned_shard>(m, "planned_shard")
        .def(py::init<>())
        .def_readwrite("shard_id", &wb::planned_shard::shard_id)
        .def_readwrite("partition_begin", &wb::planned_shard::partition_begin)
        .def_readwrite("partition_end", &wb::planned_shard::partition_end)
        .def_readwrite("row_begin", &wb::planned_shard::row_begin)
        .def_readwrite("row_end", &wb::planned_shard::row_end)
        .def_readwrite("rows", &wb::planned_shard::rows)
        .def_readwrite("nnz", &wb::planned_shard::nnz)
        .def_readwrite("estimated_bytes", &wb::planned_shard::estimated_bytes)
        .def_readwrite("execution_bytes", &wb::planned_shard::execution_bytes)
        .def_readwrite("blocked_ell_bytes", &wb::planned_shard::blocked_ell_bytes)
        .def_readwrite("bucketed_blocked_ell_bytes", &wb::planned_shard::bucketed_blocked_ell_bytes)
        .def_readwrite("blocked_ell_fill_ratio", &wb::planned_shard::blocked_ell_fill_ratio)
        .def_readwrite("blocked_ell_block_size", &wb::planned_shard::blocked_ell_block_size)
        .def_readwrite("bucketed_partition_count", &wb::planned_shard::bucketed_partition_count)
        .def_readwrite("bucketed_segment_count", &wb::planned_shard::bucketed_segment_count)
        .def_readwrite("preferred_pair", &wb::planned_shard::preferred_pair)
        .def_readwrite("preferred_format", &wb::planned_shard::preferred_format);

    py::class_<wb::ingest_plan>(m, "ingest_plan")
        .def(py::init<>())
        .def_readwrite("policy", &wb::ingest_plan::policy)
        .def_readwrite("sources", &wb::ingest_plan::sources)
        .def_readwrite("datasets", &wb::ingest_plan::datasets)
        .def_readwrite("parts", &wb::ingest_plan::parts)
        .def_readwrite("shards", &wb::ingest_plan::shards)
        .def_readwrite("issues", &wb::ingest_plan::issues)
        .def_readwrite("total_rows", &wb::ingest_plan::total_rows)
        .def_readwrite("total_cols", &wb::ingest_plan::total_cols)
        .def_readwrite("total_nnz", &wb::ingest_plan::total_nnz)
        .def_readwrite("total_estimated_bytes", &wb::ingest_plan::total_estimated_bytes)
        .def_readwrite("ok", &wb::ingest_plan::ok);

    py::class_<wb::run_event>(m, "run_event")
        .def(py::init<>())
        .def_readwrite("phase", &wb::run_event::phase)
        .def_readwrite("message", &wb::run_event::message);

    py::class_<wb::conversion_report>(m, "conversion_report")
        .def(py::init<>())
        .def_readwrite("ok", &wb::conversion_report::ok)
        .def_readwrite("events", &wb::conversion_report::events)
        .def_readwrite("issues", &wb::conversion_report::issues);

    py::class_<wb::source_dataset_summary>(m, "source_dataset_summary")
        .def(py::init<>())
        .def_readwrite("dataset_id", &wb::source_dataset_summary::dataset_id)
        .def_readwrite("matrix_path", &wb::source_dataset_summary::matrix_path)
        .def_readwrite("feature_path", &wb::source_dataset_summary::feature_path)
        .def_readwrite("barcode_path", &wb::source_dataset_summary::barcode_path)
        .def_readwrite("metadata_path", &wb::source_dataset_summary::metadata_path)
        .def_readwrite("format", &wb::source_dataset_summary::format)
        .def_readwrite("row_begin", &wb::source_dataset_summary::row_begin)
        .def_readwrite("row_end", &wb::source_dataset_summary::row_end)
        .def_readwrite("rows", &wb::source_dataset_summary::rows)
        .def_readwrite("cols", &wb::source_dataset_summary::cols)
        .def_readwrite("nnz", &wb::source_dataset_summary::nnz);

    py::class_<wb::dataset_partition_summary>(m, "dataset_partition_summary")
        .def(py::init<>())
        .def_readwrite("partition_id", &wb::dataset_partition_summary::partition_id)
        .def_readwrite("row_begin", &wb::dataset_partition_summary::row_begin)
        .def_readwrite("row_end", &wb::dataset_partition_summary::row_end)
        .def_readwrite("rows", &wb::dataset_partition_summary::rows)
        .def_readwrite("nnz", &wb::dataset_partition_summary::nnz)
        .def_readwrite("dataset_id", &wb::dataset_partition_summary::dataset_id)
        .def_readwrite("axis", &wb::dataset_partition_summary::axis)
        .def_readwrite("codec_id", &wb::dataset_partition_summary::codec_id)
        .def_readwrite("execution_format", &wb::dataset_partition_summary::execution_format)
        .def_readwrite("blocked_ell_block_size", &wb::dataset_partition_summary::blocked_ell_block_size)
        .def_readwrite("blocked_ell_bucket_count", &wb::dataset_partition_summary::blocked_ell_bucket_count)
        .def_readwrite("blocked_ell_fill_ratio", &wb::dataset_partition_summary::blocked_ell_fill_ratio)
        .def_readwrite("execution_bytes", &wb::dataset_partition_summary::execution_bytes)
        .def_readwrite("blocked_ell_bytes", &wb::dataset_partition_summary::blocked_ell_bytes)
        .def_readwrite("bucketed_blocked_ell_bytes", &wb::dataset_partition_summary::bucketed_blocked_ell_bytes);

    py::class_<wb::dataset_shard_summary>(m, "dataset_shard_summary")
        .def(py::init<>())
        .def_readwrite("shard_id", &wb::dataset_shard_summary::shard_id)
        .def_readwrite("partition_begin", &wb::dataset_shard_summary::partition_begin)
        .def_readwrite("partition_end", &wb::dataset_shard_summary::partition_end)
        .def_readwrite("row_begin", &wb::dataset_shard_summary::row_begin)
        .def_readwrite("row_end", &wb::dataset_shard_summary::row_end)
        .def_readwrite("execution_format", &wb::dataset_shard_summary::execution_format)
        .def_readwrite("blocked_ell_block_size", &wb::dataset_shard_summary::blocked_ell_block_size)
        .def_readwrite("bucketed_partition_count", &wb::dataset_shard_summary::bucketed_partition_count)
        .def_readwrite("bucketed_segment_count", &wb::dataset_shard_summary::bucketed_segment_count)
        .def_readwrite("blocked_ell_fill_ratio", &wb::dataset_shard_summary::blocked_ell_fill_ratio)
        .def_readwrite("execution_bytes", &wb::dataset_shard_summary::execution_bytes)
        .def_readwrite("bucketed_blocked_ell_bytes", &wb::dataset_shard_summary::bucketed_blocked_ell_bytes)
        .def_readwrite("preferred_pair", &wb::dataset_shard_summary::preferred_pair);

    py::class_<wb::codec_summary>(m, "codec_summary")
        .def(py::init<>())
        .def_readwrite("codec_id", &wb::codec_summary::codec_id)
        .def_readwrite("family", &wb::codec_summary::family)
        .def_readwrite("value_code", &wb::codec_summary::value_code)
        .def_readwrite("scale_value_code", &wb::codec_summary::scale_value_code)
        .def_readwrite("bits", &wb::codec_summary::bits)
        .def_readwrite("flags", &wb::codec_summary::flags);

    py::class_<wb::embedded_metadata_dataset_summary>(m, "embedded_metadata_dataset_summary")
        .def(py::init<>())
        .def_readwrite("dataset_index", &wb::embedded_metadata_dataset_summary::dataset_index)
        .def_readwrite("row_begin", &wb::embedded_metadata_dataset_summary::row_begin)
        .def_readwrite("row_end", &wb::embedded_metadata_dataset_summary::row_end)
        .def_readwrite("rows", &wb::embedded_metadata_dataset_summary::rows)
        .def_readwrite("cols", &wb::embedded_metadata_dataset_summary::cols)
        .def_readwrite("column_names", &wb::embedded_metadata_dataset_summary::column_names);

    py::class_<wb::embedded_metadata_table>(m, "embedded_metadata_table")
        .def(py::init<>())
        .def_readwrite("available", &wb::embedded_metadata_table::available)
        .def_readwrite("error", &wb::embedded_metadata_table::error)
        .def_readwrite("dataset_index", &wb::embedded_metadata_table::dataset_index)
        .def_readwrite("row_begin", &wb::embedded_metadata_table::row_begin)
        .def_readwrite("row_end", &wb::embedded_metadata_table::row_end)
        .def_readwrite("rows", &wb::embedded_metadata_table::rows)
        .def_readwrite("cols", &wb::embedded_metadata_table::cols)
        .def_readwrite("column_names", &wb::embedded_metadata_table::column_names)
        .def_readwrite("field_values", &wb::embedded_metadata_table::field_values)
        .def_readwrite("row_offsets", &wb::embedded_metadata_table::row_offsets);

    py::class_<wb::observation_metadata_column_summary>(m, "observation_metadata_column_summary")
        .def(py::init<>())
        .def_readwrite("name", &wb::observation_metadata_column_summary::name)
        .def_readwrite("type", &wb::observation_metadata_column_summary::type);

    py::class_<wb::observation_metadata_summary>(m, "observation_metadata_summary")
        .def(py::init<>())
        .def_readwrite("available", &wb::observation_metadata_summary::available)
        .def_readwrite("rows", &wb::observation_metadata_summary::rows)
        .def_readwrite("cols", &wb::observation_metadata_summary::cols)
        .def_readwrite("columns", &wb::observation_metadata_summary::columns);

    py::class_<wb::observation_metadata_column>(m, "observation_metadata_column")
        .def(py::init<>())
        .def_readwrite("name", &wb::observation_metadata_column::name)
        .def_readwrite("type", &wb::observation_metadata_column::type)
        .def_readwrite("text_values", &wb::observation_metadata_column::text_values)
        .def_readwrite("float32_values", &wb::observation_metadata_column::float32_values)
        .def_readwrite("uint8_values", &wb::observation_metadata_column::uint8_values);

    py::class_<wb::observation_metadata_table>(m, "observation_metadata_table")
        .def(py::init<>())
        .def_readwrite("available", &wb::observation_metadata_table::available)
        .def_readwrite("error", &wb::observation_metadata_table::error)
        .def_readwrite("rows", &wb::observation_metadata_table::rows)
        .def_readwrite("cols", &wb::observation_metadata_table::cols)
        .def_readwrite("columns", &wb::observation_metadata_table::columns);

    py::class_<wb::browse_cache_summary>(m, "browse_cache_summary")
        .def(py::init<>())
        .def_readwrite("available", &wb::browse_cache_summary::available)
        .def_readwrite("selected_feature_count", &wb::browse_cache_summary::selected_feature_count)
        .def_readwrite("sample_rows_per_partition", &wb::browse_cache_summary::sample_rows_per_partition)
        .def_readwrite("selected_feature_indices", &wb::browse_cache_summary::selected_feature_indices)
        .def_readwrite("selected_feature_names", &wb::browse_cache_summary::selected_feature_names)
        .def_readwrite("gene_sum", &wb::browse_cache_summary::gene_sum)
        .def_readwrite("gene_detected", &wb::browse_cache_summary::gene_detected)
        .def_readwrite("gene_sq_sum", &wb::browse_cache_summary::gene_sq_sum)
        .def_readwrite("dataset_feature_mean", &wb::browse_cache_summary::dataset_feature_mean)
        .def_readwrite("shard_feature_mean", &wb::browse_cache_summary::shard_feature_mean)
        .def_readwrite("partition_sample_row_offsets", &wb::browse_cache_summary::partition_sample_row_offsets)
        .def_readwrite("partition_sample_global_rows", &wb::browse_cache_summary::partition_sample_global_rows)
        .def_readwrite("partition_sample_values", &wb::browse_cache_summary::partition_sample_values);

    py::class_<wb::persisted_preprocess_summary>(m, "persisted_preprocess_summary")
        .def(py::init<>())
        .def_readwrite("available", &wb::persisted_preprocess_summary::available)
        .def_readwrite("assay", &wb::persisted_preprocess_summary::assay)
        .def_readwrite("matrix_orientation", &wb::persisted_preprocess_summary::matrix_orientation)
        .def_readwrite("matrix_state", &wb::persisted_preprocess_summary::matrix_state)
        .def_readwrite("pipeline_scope", &wb::persisted_preprocess_summary::pipeline_scope)
        .def_readwrite("raw_matrix_name", &wb::persisted_preprocess_summary::raw_matrix_name)
        .def_readwrite("active_matrix_name", &wb::persisted_preprocess_summary::active_matrix_name)
        .def_readwrite("feature_namespace", &wb::persisted_preprocess_summary::feature_namespace)
        .def_readwrite("mito_prefix", &wb::persisted_preprocess_summary::mito_prefix)
        .def_readwrite("raw_counts_available", &wb::persisted_preprocess_summary::raw_counts_available)
        .def_readwrite("processed_matrix_available", &wb::persisted_preprocess_summary::processed_matrix_available)
        .def_readwrite("normalized_log1p_metrics", &wb::persisted_preprocess_summary::normalized_log1p_metrics)
        .def_readwrite("hvg_available", &wb::persisted_preprocess_summary::hvg_available)
        .def_readwrite("mark_mito_from_feature_names", &wb::persisted_preprocess_summary::mark_mito_from_feature_names)
        .def_readwrite("rows", &wb::persisted_preprocess_summary::rows)
        .def_readwrite("cols", &wb::persisted_preprocess_summary::cols)
        .def_readwrite("nnz", &wb::persisted_preprocess_summary::nnz)
        .def_readwrite("partitions_processed", &wb::persisted_preprocess_summary::partitions_processed)
        .def_readwrite("mito_feature_count", &wb::persisted_preprocess_summary::mito_feature_count)
        .def_readwrite("target_sum", &wb::persisted_preprocess_summary::target_sum)
        .def_readwrite("min_counts", &wb::persisted_preprocess_summary::min_counts)
        .def_readwrite("min_genes", &wb::persisted_preprocess_summary::min_genes)
        .def_readwrite("max_mito_fraction", &wb::persisted_preprocess_summary::max_mito_fraction)
        .def_readwrite("min_gene_sum", &wb::persisted_preprocess_summary::min_gene_sum)
        .def_readwrite("min_detected_cells", &wb::persisted_preprocess_summary::min_detected_cells)
        .def_readwrite("min_variance", &wb::persisted_preprocess_summary::min_variance)
        .def_readwrite("kept_cells", &wb::persisted_preprocess_summary::kept_cells)
        .def_readwrite("kept_genes", &wb::persisted_preprocess_summary::kept_genes)
        .def_readwrite("gene_sum_checksum", &wb::persisted_preprocess_summary::gene_sum_checksum);

    py::class_<wb::persisted_preprocess_table>(m, "persisted_preprocess_table")
        .def(py::init<>())
        .def_readwrite("available", &wb::persisted_preprocess_table::available)
        .def_readwrite("error", &wb::persisted_preprocess_table::error)
        .def_readwrite("summary", &wb::persisted_preprocess_table::summary)
        .def_readwrite("cell_total_counts", &wb::persisted_preprocess_table::cell_total_counts)
        .def_readwrite("cell_mito_counts", &wb::persisted_preprocess_table::cell_mito_counts)
        .def_readwrite("cell_max_counts", &wb::persisted_preprocess_table::cell_max_counts)
        .def_readwrite("cell_detected_genes", &wb::persisted_preprocess_table::cell_detected_genes)
        .def_readwrite("cell_keep", &wb::persisted_preprocess_table::cell_keep)
        .def_readwrite("gene_sum", &wb::persisted_preprocess_table::gene_sum)
        .def_readwrite("gene_sq_sum", &wb::persisted_preprocess_table::gene_sq_sum)
        .def_readwrite("gene_detected_cells", &wb::persisted_preprocess_table::gene_detected_cells)
        .def_readwrite("gene_keep", &wb::persisted_preprocess_table::gene_keep)
        .def_readwrite("gene_flags", &wb::persisted_preprocess_table::gene_flags);

    py::class_<wb::runtime_service_summary>(m, "runtime_service_summary")
        .def(py::init<>())
        .def_readwrite("available", &wb::runtime_service_summary::available)
        .def_readwrite("service_mode", &wb::runtime_service_summary::service_mode)
        .def_readwrite("live_write_mode", &wb::runtime_service_summary::live_write_mode)
        .def_readwrite("prefer_pack_delivery", &wb::runtime_service_summary::prefer_pack_delivery)
        .def_readwrite("remote_pack_delivery", &wb::runtime_service_summary::remote_pack_delivery)
        .def_readwrite("single_reader_coordinator", &wb::runtime_service_summary::single_reader_coordinator)
        .def_readwrite("maintenance_lock_blocks_overwrite", &wb::runtime_service_summary::maintenance_lock_blocks_overwrite)
        .def_readwrite("canonical_generation", &wb::runtime_service_summary::canonical_generation)
        .def_readwrite("execution_plan_generation", &wb::runtime_service_summary::execution_plan_generation)
        .def_readwrite("pack_generation", &wb::runtime_service_summary::pack_generation)
        .def_readwrite("service_epoch", &wb::runtime_service_summary::service_epoch)
        .def_readwrite("active_read_generation", &wb::runtime_service_summary::active_read_generation)
        .def_readwrite("staged_write_generation", &wb::runtime_service_summary::staged_write_generation);

    py::class_<wb::dataset_summary>(m, "dataset_summary")
        .def(py::init<>())
        .def_readwrite("path", &wb::dataset_summary::path)
        .def_readwrite("matrix_format", &wb::dataset_summary::matrix_format)
        .def_readwrite("payload_layout", &wb::dataset_summary::payload_layout)
        .def_readwrite("issues", &wb::dataset_summary::issues)
        .def_readwrite("datasets", &wb::dataset_summary::datasets)
        .def_readwrite("partitions", &wb::dataset_summary::partitions)
        .def_readwrite("shards", &wb::dataset_summary::shards)
        .def_readwrite("codecs", &wb::dataset_summary::codecs)
        .def_readwrite("feature_names", &wb::dataset_summary::feature_names)
        .def_readwrite("embedded_metadata", &wb::dataset_summary::embedded_metadata)
        .def_readwrite("observation_metadata", &wb::dataset_summary::observation_metadata)
        .def_readwrite("browse", &wb::dataset_summary::browse)
        .def_readwrite("preprocess", &wb::dataset_summary::preprocess)
        .def_readwrite("runtime_service", &wb::dataset_summary::runtime_service)
        .def_readwrite("rows", &wb::dataset_summary::rows)
        .def_readwrite("cols", &wb::dataset_summary::cols)
        .def_readwrite("nnz", &wb::dataset_summary::nnz)
        .def_readwrite("num_partitions", &wb::dataset_summary::num_partitions)
        .def_readwrite("num_shards", &wb::dataset_summary::num_shards)
        .def_readwrite("num_datasets", &wb::dataset_summary::num_datasets)
        .def_readwrite("preferred_base_format", &wb::dataset_summary::preferred_base_format)
        .def_readwrite("ok", &wb::dataset_summary::ok);

    py::class_<wb::preprocess_config>(m, "preprocess_config")
        .def(py::init<>())
        .def_readwrite("target_sum", &wb::preprocess_config::target_sum)
        .def_readwrite("min_counts", &wb::preprocess_config::min_counts)
        .def_readwrite("min_genes", &wb::preprocess_config::min_genes)
        .def_readwrite("max_mito_fraction", &wb::preprocess_config::max_mito_fraction)
        .def_readwrite("min_gene_sum", &wb::preprocess_config::min_gene_sum)
        .def_readwrite("min_detected_cells", &wb::preprocess_config::min_detected_cells)
        .def_readwrite("min_variance", &wb::preprocess_config::min_variance)
        .def_readwrite("device", &wb::preprocess_config::device)
        .def_readwrite("drop_host_parts", &wb::preprocess_config::drop_host_parts)
        .def_readwrite("mark_mito_from_feature_names", &wb::preprocess_config::mark_mito_from_feature_names)
        .def_readwrite("mito_prefix", &wb::preprocess_config::mito_prefix)
        .def_readwrite("cache_dir", &wb::preprocess_config::cache_dir);

    py::class_<wb::preprocess_summary>(m, "preprocess_summary")
        .def(py::init<>())
        .def_readwrite("ok", &wb::preprocess_summary::ok)
        .def_readwrite("device", &wb::preprocess_summary::device)
        .def_readwrite("partitions_processed", &wb::preprocess_summary::partitions_processed)
        .def_readwrite("rows", &wb::preprocess_summary::rows)
        .def_readwrite("cols", &wb::preprocess_summary::cols)
        .def_readwrite("nnz", &wb::preprocess_summary::nnz)
        .def_readwrite("kept_cells", &wb::preprocess_summary::kept_cells)
        .def_readwrite("kept_genes", &wb::preprocess_summary::kept_genes)
        .def_readwrite("gene_sum_checksum", &wb::preprocess_summary::gene_sum_checksum)
        .def_readwrite("issues", &wb::preprocess_summary::issues);

    m.def("format_name", &wb::format_name, py::arg("format"));
    m.def("severity_name", &wb::severity_name, py::arg("severity"));
    m.def("execution_format_name", &wb::execution_format_name, py::arg("format"));
    m.def("builder_path_role_name", &wb::builder_path_role_name, py::arg("role"));
    m.def("infer_builder_path_role", &wb::infer_builder_path_role, py::arg("path"));
    m.def("list_filesystem_entries",
          [](const std::string &dir_path) { return wb::list_filesystem_entries(dir_path, nullptr); },
          py::arg("dir_path"));
    m.def("discover_drafts",
          [](const std::string &dir_path) { return wb::discover_dataset_drafts(dir_path, nullptr); },
          py::arg("dir_path"));
    m.def("sources_from_drafts", &wb::sources_from_dataset_drafts, py::arg("drafts"));
    m.def("inspect_sources",
          &wb::inspect_source_entries,
          py::arg("sources"),
          py::arg("label") = "<builder>",
          py::arg("reader_bytes") = static_cast<std::size_t>(8u) << 20u);
    m.def("inspect_manifest",
          &wb::inspect_manifest_tsv,
          py::arg("manifest_path"),
          py::arg("reader_bytes") = static_cast<std::size_t>(8u) << 20u);
    m.def("inspect_directory",
          &inspect_directory_impl,
          py::arg("dir_path"),
          py::arg("reader_bytes") = static_cast<std::size_t>(8u) << 20u);
    m.def("export_manifest",
          &export_manifest_impl,
          py::arg("path"),
          py::arg("drafts"),
          py::arg("reader_bytes") = static_cast<std::size_t>(8u) << 20u);
    m.def("plan_ingest",
          &wb::plan_dataset_ingest,
          py::arg("sources"),
          py::arg("policy") = wb::ingest_policy());
    m.def("convert", &wb::convert_plan_to_dataset_csh5, py::arg("plan"));
    m.def("summarize_dataset", &wb::summarize_dataset_csh5, py::arg("path"));
    m.def("load_embedded_metadata_table", &wb::load_embedded_metadata_table, py::arg("path"), py::arg("table_index"));
    m.def("load_observation_metadata_table", &wb::load_observation_metadata_table, py::arg("path"));
    m.def("load_persisted_preprocess_table", &wb::load_persisted_preprocess_table, py::arg("path"));
    m.def("preprocess",
          &wb::run_preprocess_pass,
          py::arg("path"),
          py::arg("config") = wb::preprocess_config());
}
