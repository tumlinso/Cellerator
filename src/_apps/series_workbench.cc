#include "series_workbench.hh"

#include <cuda_runtime.h>

#include "../ingest/common/barcode_table.cuh"
#include "../ingest/common/feature_table.cuh"
#include "../ingest/common/metadata_table.cuh"
#include "../ingest/mtx/mtx_reader.cuh"
#include "../ingest/series/series_partition.cuh"

#include <hdf5.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cellerator::apps::workbench {

namespace ccommon = ::cellerator::ingest::common;
namespace cmtx = ::cellerator::ingest::mtx;
namespace cseries = ::cellerator::ingest::series;

namespace {

inline void push_issue(std::vector<issue> *issues,
                       issue_severity severity,
                       const std::string &scope,
                       const std::string &message) {
    if (issues == nullptr) return;
    issues->push_back(issue{severity, scope, message});
}

inline bool has_errors(const std::vector<issue> &issues) {
    for (const issue &entry : issues) {
        if (entry.severity == issue_severity::error) return true;
    }
    return false;
}

inline bool file_exists(const std::string &path) {
    if (path.empty()) return false;
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    std::fclose(f);
    return true;
}

inline unsigned int infer_format(const source_entry &source) {
    if (source.format != cseries::source_unknown) return source.format;
    if (source.matrix_path.find(".mtx") != std::string::npos) return cseries::source_mtx;
    return cseries::source_unknown;
}

inline source_entry probe_source_entry(const source_entry &input,
                                       std::size_t reader_bytes,
                                       std::vector<issue> *issues,
                                       const std::string &scope_prefix) {
    source_entry output = input;
    const std::string scope = scope_prefix.empty() ? output.dataset_id : scope_prefix;
    output.probe_ok = false;
    output.rows = 0;
    output.cols = 0;
    output.nnz = 0;
    output.feature_count = 0;
    output.barcode_count = 0;
    output.metadata_rows = 0;
    output.metadata_cols = 0;
    output.format = infer_format(output);

    if (!output.included) {
        push_issue(issues, issue_severity::info, scope, "source excluded from the active ingest plan");
        return output;
    }

    if (output.matrix_path.empty()) {
        push_issue(issues, issue_severity::error, scope, "matrix path is missing");
        return output;
    }
    if (!file_exists(output.matrix_path)) {
        push_issue(issues, issue_severity::error, scope, "matrix path is not readable");
        return output;
    }
    if (output.dataset_id.empty()) {
        push_issue(issues, issue_severity::warning, scope, "dataset id is empty");
    }

    if (output.format != cseries::source_mtx && output.format != cseries::source_tenx_mtx) {
        push_issue(issues, issue_severity::error, scope, "only MTX / 10x MTX sources are currently supported by the workbench");
        return output;
    }

    {
        cmtx::header header;
        unsigned long *row_nnz = 0;
        cmtx::init(&header);
        if (!cmtx::scan_row_nnz(output.matrix_path.c_str(), &header, &row_nnz, reader_bytes)) {
            push_issue(issues, issue_severity::error, scope, "failed to probe matrix market header and row counts");
            return output;
        }
        output.rows = header.rows;
        output.cols = header.cols;
        output.nnz = header.nnz_loaded;
        std::free(row_nnz);
    }

    if (!output.feature_path.empty()) {
        ccommon::feature_table features;
        ccommon::init(&features);
        if (ccommon::load_tsv(output.feature_path.c_str(), &features, 0)) {
            output.feature_count = ccommon::count(&features);
            if (output.cols != 0 && output.feature_count != output.cols) {
                push_issue(issues,
                           issue_severity::warning,
                           scope,
                           "feature table row count does not match matrix columns");
            }
        } else {
            push_issue(issues, issue_severity::warning, scope, "failed to read feature table");
        }
        ccommon::clear(&features);
    } else {
        push_issue(issues, issue_severity::warning, scope, "feature path is missing");
    }

    if (!output.barcode_path.empty()) {
        ccommon::barcode_table barcodes;
        ccommon::init(&barcodes);
        if (ccommon::load_lines(output.barcode_path.c_str(), &barcodes)) {
            output.barcode_count = ccommon::count(&barcodes);
            if (output.rows != 0 && output.barcode_count != output.rows) {
                push_issue(issues,
                           issue_severity::warning,
                           scope,
                           "barcode table row count does not match matrix rows");
            }
        } else {
            push_issue(issues, issue_severity::warning, scope, "failed to read barcode table");
        }
        ccommon::clear(&barcodes);
    } else {
        push_issue(issues, issue_severity::warning, scope, "barcode path is missing");
    }

    if (!output.metadata_path.empty()) {
        ccommon::metadata_table metadata;
        ccommon::init(&metadata);
        if (ccommon::load_tsv(output.metadata_path.c_str(), &metadata, 1)) {
            output.metadata_rows = metadata.num_rows;
            output.metadata_cols = metadata.num_cols;
        } else {
            push_issue(issues, issue_severity::warning, scope, "failed to read metadata table");
        }
        ccommon::clear(&metadata);
    }

    output.probe_ok = true;
    return output;
}

inline std::size_t estimate_standard_csr_bytes(unsigned long rows, unsigned long nnz) {
    return (std::size_t) (rows + 1ul) * sizeof(::cellshard::types::ptr_t)
        + (std::size_t) nnz * sizeof(::cellshard::types::idx_t)
        + (std::size_t) nnz * sizeof(::real::storage_t);
}

inline hid_t open_group(hid_t parent, const char *path) {
    return H5Gopen2(parent, path, H5P_DEFAULT);
}

inline bool read_attr_u64(hid_t obj, const char *name, std::uint64_t *value) {
    hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
    const bool ok = attr >= 0 && H5Aread(attr, H5T_NATIVE_UINT64, value) >= 0;
    if (attr >= 0) H5Aclose(attr);
    return ok;
}

inline bool read_attr_u32(hid_t obj, const char *name, std::uint32_t *value) {
    hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
    const bool ok = attr >= 0 && H5Aread(attr, H5T_NATIVE_UINT32, value) >= 0;
    if (attr >= 0) H5Aclose(attr);
    return ok;
}

template<typename T>
bool read_dataset_vector(hid_t parent, const char *name, hid_t dtype, std::vector<T> *out) {
    hid_t dset = H5Dopen2(parent, name, H5P_DEFAULT);
    hid_t space = (hid_t) -1;
    hsize_t dims[1] = {0};
    int ndims = 0;
    if (dset < 0 || out == nullptr) {
        if (dset >= 0) H5Dclose(dset);
        return false;
    }
    space = H5Dget_space(dset);
    if (space < 0) {
        H5Dclose(dset);
        return false;
    }
    ndims = H5Sget_simple_extent_dims(space, dims, 0);
    if (ndims != 1) {
        H5Sclose(space);
        H5Dclose(dset);
        return false;
    }
    out->assign((std::size_t) dims[0], T{});
    const bool ok = out->empty() || H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, out->data()) >= 0;
    H5Sclose(space);
    H5Dclose(dset);
    return ok;
}

inline bool read_text_column_strings(hid_t parent, const char *name, std::vector<std::string> *out) {
    hid_t group = H5Gopen2(parent, name, H5P_DEFAULT);
    std::uint32_t count = 0;
    std::uint32_t bytes = 0;
    std::vector<std::uint32_t> offsets;
    std::vector<char> data;
    if (out == nullptr) return false;
    out->clear();
    if (group < 0) return false;
    if (!read_attr_u32(group, "count", &count) || !read_attr_u32(group, "bytes", &bytes)) {
        H5Gclose(group);
        return false;
    }
    offsets.assign((std::size_t) count + 1u, 0u);
    data.assign((std::size_t) bytes, '\0');
    if (!read_dataset_vector(group, "offsets", H5T_NATIVE_UINT32, &offsets)
        || !read_dataset_vector(group, "data", H5T_NATIVE_CHAR, &data)) {
        H5Gclose(group);
        return false;
    }
    out->reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t begin = offsets[i];
        const std::uint32_t end = offsets[i + 1u];
        if (end <= begin || end > data.size()) {
            out->push_back(std::string());
            continue;
        }
        out->emplace_back(data.data() + begin);
    }
    H5Gclose(group);
    return true;
}

inline unsigned long find_part_index_for_row(const std::vector<std::uint64_t> &part_row_offsets,
                                             std::uint64_t row_begin) {
    if (part_row_offsets.size() < 2u) return 0;
    const auto it = std::upper_bound(part_row_offsets.begin(), part_row_offsets.end(), row_begin);
    if (it == part_row_offsets.begin()) return 0;
    return (unsigned long) std::distance(part_row_offsets.begin(), it - 1);
}

inline unsigned long find_part_end_for_row(const std::vector<std::uint64_t> &part_row_offsets,
                                           std::uint64_t row_end) {
    const unsigned long begin = find_part_index_for_row(part_row_offsets, row_end);
    if (begin + 1u < part_row_offsets.size() && part_row_offsets[begin] < row_end) return begin + 1u;
    return begin;
}

} // namespace

std::string format_name(unsigned int format) {
    switch (format) {
        case cseries::source_mtx: return "mtx";
        case cseries::source_tenx_mtx: return "tenx_mtx";
        case cseries::source_tenx_h5: return "tenx_h5";
        case cseries::source_h5ad: return "h5ad";
        case cseries::source_loom: return "loom";
        case cseries::source_binary: return "binary";
        default: return "unknown";
    }
}

std::string severity_name(issue_severity severity) {
    switch (severity) {
        case issue_severity::info: return "info";
        case issue_severity::warning: return "warning";
        case issue_severity::error: return "error";
    }
    return "info";
}

manifest_inspection inspect_manifest_tsv(const std::string &manifest_path, std::size_t reader_bytes) {
    manifest_inspection inspection;
    cseries::manifest manifest;
    inspection.manifest_path = manifest_path;
    cseries::init(&manifest);

    if (manifest_path.empty()) {
        push_issue(&inspection.issues, issue_severity::error, "manifest", "manifest path is empty");
        return inspection;
    }

    if (!cseries::load_tsv(manifest_path.c_str(), &manifest, 1)) {
        push_issue(&inspection.issues, issue_severity::error, "manifest", "failed to load manifest TSV");
        cseries::clear(&manifest);
        return inspection;
    }

    inspection.sources.reserve(manifest.count);
    for (unsigned int i = 0; i < manifest.count; ++i) {
        source_entry source;
        source.dataset_id = ccommon::get(&manifest.dataset_ids, i) != nullptr ? ccommon::get(&manifest.dataset_ids, i) : "";
        source.matrix_path = ccommon::get(&manifest.matrix_paths, i) != nullptr ? ccommon::get(&manifest.matrix_paths, i) : "";
        source.feature_path = ccommon::get(&manifest.feature_paths, i) != nullptr ? ccommon::get(&manifest.feature_paths, i) : "";
        source.barcode_path = ccommon::get(&manifest.barcode_paths, i) != nullptr ? ccommon::get(&manifest.barcode_paths, i) : "";
        source.metadata_path = ccommon::get(&manifest.metadata_paths, i) != nullptr ? ccommon::get(&manifest.metadata_paths, i) : "";
        source.format = manifest.formats != nullptr ? manifest.formats[i] : cseries::source_unknown;
        source.rows = manifest.rows != nullptr ? manifest.rows[i] : 0ul;
        source.cols = manifest.cols != nullptr ? manifest.cols[i] : 0ul;
        source.nnz = manifest.nnz != nullptr ? manifest.nnz[i] : 0ul;
        inspection.sources.push_back(probe_source_entry(source,
                                                        reader_bytes,
                                                        &inspection.issues,
                                                        source.dataset_id.empty()
                                                            ? ("manifest[" + std::to_string(i) + "]")
                                                            : source.dataset_id));
    }
    inspection.ok = !has_errors(inspection.issues);
    cseries::clear(&manifest);
    return inspection;
}

ingest_plan plan_series_ingest(const std::vector<source_entry> &sources, const ingest_policy &policy) {
    ingest_plan plan;
    std::vector<unsigned long> part_rows;
    std::vector<unsigned long> part_nnz;
    std::vector<unsigned long> part_bytes;
    cseries::partition shard_plan;
    std::unordered_set<std::string> unique_feature_ids;

    plan.policy = policy;
    plan.sources.reserve(sources.size());
    cseries::init(&shard_plan);

    for (std::size_t source_index = 0; source_index < sources.size(); ++source_index) {
        source_entry source = probe_source_entry(
            sources[source_index],
            policy.reader_bytes,
            &plan.issues,
            sources[source_index].dataset_id.empty()
                ? ("source[" + std::to_string(source_index) + "]")
                : sources[source_index].dataset_id);
        plan.sources.push_back(source);

        if (!source.included || !source.probe_ok) continue;
        if (source.rows == 0 && source.nnz == 0) {
            push_issue(&plan.issues, issue_severity::warning, source.dataset_id, "matrix has zero rows and zero nnz");
        }

        cmtx::header header;
        unsigned long *row_nnz = 0;
        unsigned long *row_offsets = 0;
        unsigned long *part_nnz_raw = 0;
        unsigned long num_parts = 0;
        const unsigned long global_row_begin = plan.total_rows;
        const unsigned long global_part_begin = (unsigned long) plan.parts.size();
        cmtx::init(&header);
        if (!cmtx::scan_row_nnz(source.matrix_path.c_str(), &header, &row_nnz, policy.reader_bytes)) {
            push_issue(&plan.issues, issue_severity::error, source.dataset_id, "failed to rescan matrix rows during plan generation");
            continue;
        }
        if (!cmtx::plan_row_partitions_by_nnz(row_nnz, header.rows, policy.max_part_nnz, &row_offsets, &num_parts)
            || !cmtx::build_part_nnz_from_row_nnz(row_nnz, row_offsets, num_parts, &part_nnz_raw)) {
            push_issue(&plan.issues, issue_severity::error, source.dataset_id, "failed to build row partitions");
            std::free(row_nnz);
            std::free(row_offsets);
            std::free(part_nnz_raw);
            continue;
        }

        if (!source.feature_path.empty()) {
            ccommon::feature_table features;
            ccommon::init(&features);
            if (ccommon::load_tsv(source.feature_path.c_str(), &features, 0)) {
                for (unsigned int i = 0; i < ccommon::count(&features); ++i) {
                    const char *id = ccommon::id(&features, i);
                    if (id != nullptr && *id != 0) unique_feature_ids.insert(id);
                }
            }
            ccommon::clear(&features);
        }

        for (unsigned long local_part = 0; local_part < num_parts; ++local_part) {
            const unsigned long rows = row_offsets[local_part + 1u] - row_offsets[local_part];
            const unsigned long nnz = part_nnz_raw[local_part];
            const std::size_t bytes = estimate_standard_csr_bytes(rows, nnz);
            planned_part part;
            part.part_id = (unsigned long) plan.parts.size();
            part.source_index = source_index;
            part.dataset_id = source.dataset_id;
            part.row_begin = global_row_begin + row_offsets[local_part];
            part.row_end = global_row_begin + row_offsets[local_part + 1u];
            part.rows = rows;
            part.nnz = nnz;
            part.estimated_bytes = bytes;
            plan.parts.push_back(part);
            part_rows.push_back(rows);
            part_nnz.push_back(nnz);
            part_bytes.push_back((unsigned long) bytes);
            plan.total_estimated_bytes += bytes;
        }

        planned_dataset dataset;
        dataset.source_index = source_index;
        dataset.dataset_id = source.dataset_id;
        dataset.global_row_begin = global_row_begin;
        dataset.global_row_end = global_row_begin + source.rows;
        dataset.rows = source.rows;
        dataset.cols = source.cols;
        dataset.nnz = source.nnz;
        dataset.part_begin = global_part_begin;
        dataset.part_count = num_parts;
        dataset.feature_count = source.feature_count;
        dataset.barcode_count = source.barcode_count;
        plan.datasets.push_back(dataset);
        plan.total_rows += source.rows;
        plan.total_nnz += source.nnz;

        std::free(row_nnz);
        std::free(row_offsets);
        std::free(part_nnz_raw);
    }

    if (plan.datasets.empty()) {
        push_issue(&plan.issues, issue_severity::error, "plan", "no supported included datasets are available for ingest");
    }

    if (!unique_feature_ids.empty()) {
        plan.total_cols = (unsigned long) unique_feature_ids.size();
    } else {
        for (const planned_dataset &dataset : plan.datasets) {
            plan.total_cols = std::max(plan.total_cols, dataset.cols);
        }
    }

    if (!part_rows.empty()) {
        if (!cseries::build_by_bytes(&shard_plan,
                                     part_rows.data(),
                                     part_bytes.data(),
                                     (unsigned long) part_rows.size(),
                                     policy.max_window_bytes)) {
            push_issue(&plan.issues, issue_severity::error, "plan", "failed to build shard plan");
        } else {
            for (unsigned long shard_id = 0; shard_id < shard_plan.count; ++shard_id) {
                const cseries::shard_range &range = shard_plan.ranges[shard_id];
                planned_shard shard;
                shard.shard_id = shard_id;
                shard.part_begin = range.part_begin;
                shard.part_end = range.part_end;
                shard.row_begin = range.row_begin;
                shard.row_end = range.row_end;
                shard.rows = range.row_end - range.row_begin;
                shard.nnz = range.nnz;
                shard.estimated_bytes = (std::size_t) range.bytes;
                plan.shards.push_back(shard);
                for (unsigned long part_id = range.part_begin; part_id < range.part_end && part_id < plan.parts.size(); ++part_id) {
                    plan.parts[part_id].shard_id = shard_id;
                }
            }
        }
    }

    plan.ok = !has_errors(plan.issues);
    cseries::clear(&shard_plan);
    return plan;
}

series_summary summarize_series_csh5(const std::string &path) {
    series_summary summary;
    hid_t file = (hid_t) -1;
    hid_t datasets = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    hid_t provenance = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    hid_t embedded_metadata = (hid_t) -1;
    hid_t browse = (hid_t) -1;
    std::vector<std::uint32_t> dataset_formats;
    std::vector<std::uint64_t> dataset_row_begin;
    std::vector<std::uint64_t> dataset_row_end;
    std::vector<std::uint64_t> dataset_rows;
    std::vector<std::uint64_t> dataset_cols;
    std::vector<std::uint64_t> dataset_nnz;
    std::vector<std::string> dataset_ids;
    std::vector<std::string> matrix_paths;
    std::vector<std::string> feature_paths;
    std::vector<std::string> barcode_paths;
    std::vector<std::string> metadata_paths;
    std::vector<std::uint64_t> part_rows;
    std::vector<std::uint64_t> part_nnz;
    std::vector<std::uint32_t> part_axes;
    std::vector<std::uint64_t> part_row_offsets;
    std::vector<std::uint32_t> part_dataset_ids;
    std::vector<std::uint32_t> part_codec_ids;
    std::vector<std::uint64_t> shard_offsets;
    std::vector<std::uint32_t> codec_ids;
    std::vector<std::uint32_t> codec_families;
    std::vector<std::uint32_t> codec_value_codes;
    std::vector<std::uint32_t> codec_scale_codes;
    std::vector<std::uint32_t> codec_bits;
    std::vector<std::uint32_t> codec_flags;

    summary.path = path;
    if (path.empty()) {
        push_issue(&summary.issues, issue_severity::error, "inspect", "series.csh5 path is empty");
        return summary;
    }

    file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        push_issue(&summary.issues, issue_severity::error, "inspect", "failed to open series.csh5");
        return summary;
    }

    if (!read_attr_u64(file, "rows", &summary.rows)
        || !read_attr_u64(file, "cols", &summary.cols)
        || !read_attr_u64(file, "nnz", &summary.nnz)
        || !read_attr_u64(file, "num_parts", &summary.num_parts)
        || !read_attr_u64(file, "num_shards", &summary.num_shards)
        || !read_attr_u64(file, "num_datasets", &summary.num_datasets)) {
        push_issue(&summary.issues, issue_severity::error, "inspect", "failed to read top-level series attributes");
        goto done;
    }

    datasets = open_group(file, "/datasets");
    matrix = open_group(file, "/matrix");
    provenance = open_group(file, "/provenance");
    codecs = open_group(file, "/codecs");
    if (datasets < 0 || matrix < 0 || provenance < 0 || codecs < 0) {
        push_issue(&summary.issues, issue_severity::error, "inspect", "series.csh5 is missing one or more required groups");
        goto done;
    }

    if (!read_text_column_strings(datasets, "dataset_ids", &dataset_ids)
        || !read_text_column_strings(datasets, "matrix_paths", &matrix_paths)
        || !read_text_column_strings(datasets, "feature_paths", &feature_paths)
        || !read_text_column_strings(datasets, "barcode_paths", &barcode_paths)
        || !read_text_column_strings(datasets, "metadata_paths", &metadata_paths)
        || !read_dataset_vector(datasets, "formats", H5T_NATIVE_UINT32, &dataset_formats)
        || !read_dataset_vector(datasets, "row_begin", H5T_NATIVE_UINT64, &dataset_row_begin)
        || !read_dataset_vector(datasets, "row_end", H5T_NATIVE_UINT64, &dataset_row_end)
        || !read_dataset_vector(datasets, "rows", H5T_NATIVE_UINT64, &dataset_rows)
        || !read_dataset_vector(datasets, "cols", H5T_NATIVE_UINT64, &dataset_cols)
        || !read_dataset_vector(datasets, "nnz", H5T_NATIVE_UINT64, &dataset_nnz)) {
        push_issue(&summary.issues, issue_severity::error, "inspect", "failed to read dataset metadata");
        goto done;
    }

    summary.datasets.reserve(dataset_ids.size());
    for (std::size_t i = 0; i < dataset_ids.size(); ++i) {
        summary.datasets.push_back(series_dataset_summary{
            dataset_ids[i],
            i < matrix_paths.size() ? matrix_paths[i] : std::string(),
            i < feature_paths.size() ? feature_paths[i] : std::string(),
            i < barcode_paths.size() ? barcode_paths[i] : std::string(),
            i < metadata_paths.size() ? metadata_paths[i] : std::string(),
            i < dataset_formats.size() ? dataset_formats[i] : 0u,
            i < dataset_row_begin.size() ? dataset_row_begin[i] : 0ull,
            i < dataset_row_end.size() ? dataset_row_end[i] : 0ull,
            i < dataset_rows.size() ? dataset_rows[i] : 0ull,
            i < dataset_cols.size() ? dataset_cols[i] : 0ull,
            i < dataset_nnz.size() ? dataset_nnz[i] : 0ull
        });
    }

    if (!read_dataset_vector(matrix, "part_rows", H5T_NATIVE_UINT64, &part_rows)
        || !read_dataset_vector(matrix, "part_nnz", H5T_NATIVE_UINT64, &part_nnz)
        || !read_dataset_vector(matrix, "part_axes", H5T_NATIVE_UINT32, &part_axes)
        || !read_dataset_vector(matrix, "part_row_offsets", H5T_NATIVE_UINT64, &part_row_offsets)
        || !read_dataset_vector(matrix, "part_dataset_ids", H5T_NATIVE_UINT32, &part_dataset_ids)
        || !read_dataset_vector(matrix, "part_codec_ids", H5T_NATIVE_UINT32, &part_codec_ids)
        || !read_dataset_vector(matrix, "shard_offsets", H5T_NATIVE_UINT64, &shard_offsets)) {
        push_issue(&summary.issues, issue_severity::error, "inspect", "failed to read matrix layout metadata");
        goto done;
    }

    summary.parts.reserve(part_rows.size());
    for (std::size_t i = 0; i < part_rows.size(); ++i) {
        summary.parts.push_back(series_part_summary{
            (std::uint64_t) i,
            i < part_row_offsets.size() ? part_row_offsets[i] : 0ull,
            i + 1u < part_row_offsets.size() ? part_row_offsets[i + 1u] : 0ull,
            i < part_rows.size() ? part_rows[i] : 0ull,
            i < part_nnz.size() ? part_nnz[i] : 0ull,
            i < part_dataset_ids.size() ? part_dataset_ids[i] : 0u,
            i < part_axes.size() ? part_axes[i] : 0u,
            i < part_codec_ids.size() ? part_codec_ids[i] : 0u
        });
    }

    summary.shards.reserve(shard_offsets.size() > 0 ? shard_offsets.size() - 1u : 0u);
    for (std::size_t i = 0; i + 1u < shard_offsets.size(); ++i) {
        const std::uint64_t row_begin = shard_offsets[i];
        const std::uint64_t row_end = shard_offsets[i + 1u];
        const unsigned long part_begin = find_part_index_for_row(part_row_offsets, row_begin);
        const unsigned long part_end = find_part_end_for_row(part_row_offsets, row_end);
        summary.shards.push_back(series_shard_summary{
            (std::uint64_t) i,
            part_begin,
            row_end == row_begin ? part_begin : std::max<unsigned long>(part_begin, part_end),
            row_begin,
            row_end
        });
    }

    if (!read_dataset_vector(codecs, "codec_id", H5T_NATIVE_UINT32, &codec_ids)
        || !read_dataset_vector(codecs, "family", H5T_NATIVE_UINT32, &codec_families)
        || !read_dataset_vector(codecs, "value_code", H5T_NATIVE_UINT32, &codec_value_codes)
        || !read_dataset_vector(codecs, "scale_value_code", H5T_NATIVE_UINT32, &codec_scale_codes)
        || !read_dataset_vector(codecs, "bits", H5T_NATIVE_UINT32, &codec_bits)
        || !read_dataset_vector(codecs, "flags", H5T_NATIVE_UINT32, &codec_flags)) {
        push_issue(&summary.issues, issue_severity::warning, "inspect", "failed to read codec table");
    } else {
        for (std::size_t i = 0; i < codec_ids.size(); ++i) {
            summary.codecs.push_back(codec_summary{
                codec_ids[i],
                i < codec_families.size() ? codec_families[i] : 0u,
                i < codec_value_codes.size() ? codec_value_codes[i] : 0u,
                i < codec_scale_codes.size() ? codec_scale_codes[i] : 0u,
                i < codec_bits.size() ? codec_bits[i] : 0u,
                i < codec_flags.size() ? codec_flags[i] : 0u
            });
        }
    }

    if (!read_text_column_strings(provenance, "feature_names", &summary.feature_names)) {
        push_issue(&summary.issues, issue_severity::warning, "inspect", "failed to read feature names");
    }

    embedded_metadata = open_group(file, "/embedded_metadata");
    if (embedded_metadata >= 0) {
        std::uint32_t metadata_count = 0;
        std::vector<std::uint32_t> dataset_indices;
        std::vector<std::uint64_t> global_row_begin;
        std::vector<std::uint64_t> global_row_end;
        if (!read_attr_u32(embedded_metadata, "count", &metadata_count)
            || !read_dataset_vector(embedded_metadata, "dataset_indices", H5T_NATIVE_UINT32, &dataset_indices)
            || !read_dataset_vector(embedded_metadata, "global_row_begin", H5T_NATIVE_UINT64, &global_row_begin)
            || !read_dataset_vector(embedded_metadata, "global_row_end", H5T_NATIVE_UINT64, &global_row_end)) {
            push_issue(&summary.issues, issue_severity::warning, "inspect", "failed to read embedded metadata directory");
        } else {
            summary.embedded_metadata.reserve(metadata_count);
            for (std::uint32_t i = 0; i < metadata_count; ++i) {
                char table_name[64];
                hid_t table = (hid_t) -1;
                std::uint32_t rows = 0;
                std::uint32_t cols = 0;
                std::vector<std::string> column_names;
                if (std::snprintf(table_name, sizeof(table_name), "table_%u", i) <= 0) continue;
                table = H5Gopen2(embedded_metadata, table_name, H5P_DEFAULT);
                if (table < 0) continue;
                if (!read_attr_u32(table, "rows", &rows)
                    || !read_attr_u32(table, "cols", &cols)
                    || !read_text_column_strings(table, "column_names", &column_names)) {
                    push_issue(&summary.issues, issue_severity::warning, "inspect", "failed to read one embedded metadata table");
                    H5Gclose(table);
                    continue;
                }
                summary.embedded_metadata.push_back(embedded_metadata_dataset_summary{
                    i < dataset_indices.size() ? dataset_indices[i] : i,
                    i < global_row_begin.size() ? global_row_begin[i] : 0ull,
                    i < global_row_end.size() ? global_row_end[i] : 0ull,
                    rows,
                    cols,
                    std::move(column_names)
                });
                H5Gclose(table);
            }
        }
    }

    browse = open_group(file, "/browse");
    if (browse >= 0) {
        browse_cache_summary browse_summary;
        if (!read_attr_u32(browse, "selected_feature_count", &browse_summary.selected_feature_count)
            || !read_attr_u32(browse, "sample_rows_per_part", &browse_summary.sample_rows_per_part)) {
            push_issue(&summary.issues, issue_severity::warning, "inspect", "failed to read browse cache header");
        } else if (browse_summary.selected_feature_count != 0u) {
            if (!read_dataset_vector(browse, "selected_feature_indices", H5T_NATIVE_UINT32, &browse_summary.selected_feature_indices)
                || !read_dataset_vector(browse, "gene_sum", H5T_NATIVE_FLOAT, &browse_summary.gene_sum)
                || !read_dataset_vector(browse, "gene_detected", H5T_NATIVE_FLOAT, &browse_summary.gene_detected)
                || !read_dataset_vector(browse, "gene_sq_sum", H5T_NATIVE_FLOAT, &browse_summary.gene_sq_sum)
                || !read_dataset_vector(browse, "dataset_feature_mean", H5T_NATIVE_FLOAT, &browse_summary.dataset_feature_mean)
                || !read_dataset_vector(browse, "shard_feature_mean", H5T_NATIVE_FLOAT, &browse_summary.shard_feature_mean)
                || !read_dataset_vector(browse, "part_sample_row_offsets", H5T_NATIVE_UINT32, &browse_summary.part_sample_row_offsets)
                || !read_dataset_vector(browse, "part_sample_global_rows", H5T_NATIVE_UINT64, &browse_summary.part_sample_global_rows)
                || !read_dataset_vector(browse, "part_sample_values", H5T_NATIVE_FLOAT, &browse_summary.part_sample_values)) {
                push_issue(&summary.issues, issue_severity::warning, "inspect", "failed to read browse cache payload");
            } else {
                browse_summary.selected_feature_names.reserve(browse_summary.selected_feature_indices.size());
                for (std::uint32_t feature_index : browse_summary.selected_feature_indices) {
                    if (feature_index < summary.feature_names.size()) browse_summary.selected_feature_names.push_back(summary.feature_names[feature_index]);
                    else browse_summary.selected_feature_names.push_back(std::string("g") + std::to_string(feature_index));
                }
                browse_summary.available = true;
                summary.browse = std::move(browse_summary);
            }
        }
    }

    summary.ok = !has_errors(summary.issues);

done:
    if (browse >= 0) H5Gclose(browse);
    if (embedded_metadata >= 0) H5Gclose(embedded_metadata);
    if (codecs >= 0) H5Gclose(codecs);
    if (provenance >= 0) H5Gclose(provenance);
    if (matrix >= 0) H5Gclose(matrix);
    if (datasets >= 0) H5Gclose(datasets);
    if (file >= 0) H5Fclose(file);
    return summary;
}

} // namespace cellerator::apps::workbench
