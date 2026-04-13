#include "series_workbench.hh"

#include <cuda_runtime.h>

#include "../ingest/common/barcode_table.cuh"
#include "../ingest/common/feature_table.cuh"
#include "../ingest/common/metadata_table.cuh"
#include "../ingest/mtx/mtx_reader.cuh"
#include "../ingest/series/series_partition.cuh"
#include "../../extern/CellShard/src/disk/matrix.cuh"

#include <hdf5.h>

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cellerator::apps::workbench {

namespace ccommon = ::cellerator::ingest::common;
namespace cmtx = ::cellerator::ingest::mtx;
namespace cscan = ::cellerator::ingest::scan;
namespace cseries = ::cellerator::ingest::series;
namespace fs = std::filesystem;

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

struct blocked_ell_candidate_stream {
    unsigned int block_size = 0u;
    unsigned long current_part = ULONG_MAX;
    unsigned long current_row_block = ULONG_MAX;
    unsigned int active_unique = 0u;
    std::uint32_t epoch = 1u;
    std::vector<std::uint32_t> marks;
    std::vector<unsigned long> total_unique_blocks;
    std::vector<unsigned int> ell_width;
};

inline void flush_candidate_row_block(blocked_ell_candidate_stream *stream) {
    if (stream == nullptr || stream->current_part == ULONG_MAX) return;
    stream->total_unique_blocks[stream->current_part] += (unsigned long) stream->active_unique;
    stream->ell_width[stream->current_part] = std::max(stream->ell_width[stream->current_part], stream->active_unique);
    stream->active_unique = 0u;
}

inline bool start_candidate_row_block(blocked_ell_candidate_stream *stream,
                                      unsigned long part_index,
                                      unsigned long row_block) {
    if (stream == nullptr) return false;
    flush_candidate_row_block(stream);
    stream->current_part = part_index;
    stream->current_row_block = row_block;
    ++stream->epoch;
    if (stream->epoch == 0u) {
        std::fill(stream->marks.begin(), stream->marks.end(), 0u);
        stream->epoch = 1u;
    }
    return true;
}

inline bool accumulate_candidate_entry(blocked_ell_candidate_stream *stream,
                                       unsigned long part_index,
                                       unsigned long local_row,
                                       unsigned long col) {
    if (stream == nullptr || stream->block_size == 0u) return false;
    const unsigned long row_block = local_row / stream->block_size;
    const unsigned long block_col = col / stream->block_size;
    if (block_col >= stream->marks.size() || part_index >= stream->total_unique_blocks.size()) return false;
    if (stream->current_part != part_index || stream->current_row_block != row_block) {
        if (!start_candidate_row_block(stream, part_index, row_block)) return false;
    }
    if (stream->marks[block_col] != stream->epoch) {
        stream->marks[block_col] = stream->epoch;
        ++stream->active_unique;
    }
    return true;
}

inline std::size_t choose_blocked_ell_candidate_count(const ingest_policy &policy) {
    return std::min<std::size_t>(
        policy.blocked_ell_candidate_count,
        sizeof(policy.blocked_ell_block_sizes) / sizeof(policy.blocked_ell_block_sizes[0]));
}

inline void choose_part_execution_layout(const ingest_policy &policy,
                                         unsigned long rows,
                                         std::size_t compressed_bytes,
                                         const std::vector<blocked_ell_candidate_stream> &candidates,
                                         unsigned long part_index,
                                         execution_format *preferred_format,
                                         unsigned int *block_size,
                                         double *fill_ratio,
                                         std::size_t *blocked_ell_bytes,
                                         std::size_t *execution_bytes) {
    execution_format chosen_format = execution_format::compressed;
    unsigned int chosen_block = 0u;
    double chosen_fill = 0.0;
    std::size_t chosen_blocked_ell_bytes = 0u;

    for (const blocked_ell_candidate_stream &candidate : candidates) {
        const unsigned long row_blocks = candidate.block_size == 0u
            ? 0ul
            : (rows + (unsigned long) candidate.block_size - 1ul) / (unsigned long) candidate.block_size;
        const unsigned int ell_width_blocks = part_index < candidate.ell_width.size() ? candidate.ell_width[part_index] : 0u;
        const double candidate_fill = row_blocks == 0ul || ell_width_blocks == 0u
            ? 1.0
            : (double) candidate.total_unique_blocks[part_index] / (double) (row_blocks * (unsigned long) ell_width_blocks);
        const std::size_t candidate_bytes = cellshard::packed_blocked_ell_bytes(
            (cellshard::types::dim_t) rows,
            ell_width_blocks * candidate.block_size,
            candidate.block_size,
            sizeof(::real::storage_t));

        if (candidate_fill + 1.0e-9 < policy.blocked_ell_min_fill_ratio) continue;
        if (chosen_block == 0u
            || candidate_fill > chosen_fill + 1.0e-9
            || (candidate_fill + 1.0e-9 >= chosen_fill && candidate_bytes < chosen_blocked_ell_bytes)
            || (candidate_fill + 1.0e-9 >= chosen_fill && candidate_bytes == chosen_blocked_ell_bytes && candidate.block_size > chosen_block)) {
            chosen_block = candidate.block_size;
            chosen_fill = candidate_fill;
            chosen_blocked_ell_bytes = candidate_bytes;
            chosen_format = execution_format::blocked_ell;
        }
    }

    if (preferred_format != nullptr) *preferred_format = chosen_format;
    if (block_size != nullptr) *block_size = chosen_block;
    if (fill_ratio != nullptr) *fill_ratio = chosen_fill;
    if (blocked_ell_bytes != nullptr) *blocked_ell_bytes = chosen_blocked_ell_bytes;
    if (execution_bytes != nullptr) {
        *execution_bytes = chosen_format == execution_format::blocked_ell && chosen_blocked_ell_bytes != 0u
            ? chosen_blocked_ell_bytes
            : compressed_bytes;
    }
}

inline bool collect_row_sorted_blocked_ell_metrics(const source_entry &source,
                                                   const cmtx::header &header,
                                                   const unsigned long *row_offsets,
                                                   unsigned long num_parts,
                                                   const ingest_policy &policy,
                                                   std::vector<blocked_ell_candidate_stream> *out_candidates) {
    cscan::buffered_file_reader reader;
    cmtx::header verify;
    int rc = 0;
    char *line = nullptr;
    std::size_t line_len = 0u;
    unsigned long row = 0ul;
    unsigned long col = 0ul;
    float value = 0.0f;
    std::vector<blocked_ell_candidate_stream> candidates;
    const std::size_t candidate_count = choose_blocked_ell_candidate_count(policy);

    if (out_candidates == nullptr || row_offsets == nullptr || num_parts == 0ul) return false;
    out_candidates->clear();
    if (!header.row_sorted || header.symmetric || candidate_count == 0u) return false;

    candidates.reserve(candidate_count);
    for (std::size_t i = 0; i < candidate_count; ++i) {
        const unsigned int block_size = policy.blocked_ell_block_sizes[i];
        const unsigned long col_blocks = block_size == 0u
            ? 0ul
            : (header.cols + (unsigned long) block_size - 1ul) / (unsigned long) block_size;
        blocked_ell_candidate_stream candidate;
        if (block_size == 0u || col_blocks == 0ul) continue;
        candidate.block_size = block_size;
        candidate.marks.assign(col_blocks, 0u);
        candidate.total_unique_blocks.assign(num_parts, 0ul);
        candidate.ell_width.assign(num_parts, 0u);
        candidates.push_back(std::move(candidate));
    }
    if (candidates.empty()) return false;

    cscan::init(&reader);
    cmtx::init(&verify);
    if (!cscan::open(&reader, source.matrix_path.c_str(), policy.reader_bytes)) {
        cscan::clear(&reader);
        return false;
    }
    if (!cmtx::read_header(&reader, &verify)
        || verify.rows != header.rows
        || verify.cols != header.cols
        || verify.nnz_file != header.nnz_file) {
        cscan::clear(&reader);
        return false;
    }

    while ((rc = cscan::next_line(&reader, &line, &line_len)) > 0) {
        if (line_len == 0u || line[0] == '%') continue;
        if (!cmtx::read_triplet(line, &verify, &row, &col, &value)) {
            cscan::clear(&reader);
            return false;
        }
        const unsigned long part_index = cellshard::find_offset_span(row, row_offsets, num_parts);
        if (part_index >= num_parts) {
            cscan::clear(&reader);
            return false;
        }
        const unsigned long local_row = row - row_offsets[part_index];
        for (blocked_ell_candidate_stream &candidate : candidates) {
            if (!accumulate_candidate_entry(&candidate, part_index, local_row, col)) {
                cscan::clear(&reader);
                return false;
            }
        }
    }
    cscan::clear(&reader);
    if (rc < 0) return false;
    for (blocked_ell_candidate_stream &candidate : candidates) flush_candidate_row_block(&candidate);
    *out_candidates = std::move(candidates);
    return true;
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

inline hid_t open_optional_group(hid_t parent, const char *path) {
    hid_t group = (hid_t) -1;
    H5E_BEGIN_TRY {
        group = H5Gopen2(parent, path, H5P_DEFAULT);
    } H5E_END_TRY;
    return group;
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

inline bool read_attr_string(hid_t obj, const char *name, std::string *value) {
    hid_t attr = (hid_t) -1;
    hid_t type = (hid_t) -1;
    std::size_t size = 0u;
    std::vector<char> buffer;

    if (value == nullptr) return false;
    attr = H5Aopen(obj, name, H5P_DEFAULT);
    if (attr < 0) return false;
    type = H5Aget_type(attr);
    if (type < 0) goto done;
    size = H5Tget_size(type);
    if (size == 0u) goto done;
    buffer.assign(size + 1u, '\0');
    if (H5Aread(attr, type, buffer.data()) < 0) goto done;
    *value = buffer.data();
    H5Tclose(type);
    H5Aclose(attr);
    return true;

done:
    if (type >= 0) H5Tclose(type);
    if (attr >= 0) H5Aclose(attr);
    return false;
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

inline unsigned long find_partition_index_for_row(const std::vector<std::uint64_t> &part_row_offsets,
                                                  std::uint64_t row_begin) {
    if (part_row_offsets.size() < 2u) return 0;
    const auto it = std::upper_bound(part_row_offsets.begin(), part_row_offsets.end(), row_begin);
    if (it == part_row_offsets.begin()) return 0;
    return (unsigned long) std::distance(part_row_offsets.begin(), it - 1);
}

inline unsigned long find_partition_end_for_row(const std::vector<std::uint64_t> &part_row_offsets,
                                                std::uint64_t row_end) {
    const unsigned long begin = find_partition_index_for_row(part_row_offsets, row_end);
    if (begin + 1u < part_row_offsets.size() && part_row_offsets[begin] < row_end) return begin + 1u;
    return begin;
}

std::string lower_copy(std::string value) {
    for (char &ch : value) ch = (char) std::tolower((unsigned char) ch);
    return value;
}

bool ends_with(const std::string &text, const std::string &suffix) {
    return text.size() >= suffix.size()
        && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void strip_suffix(std::string *text, const char *suffix) {
    if (text == nullptr || suffix == nullptr) return;
    const std::string suffix_text(suffix);
    if (ends_with(*text, suffix_text)) text->erase(text->size() - suffix_text.size());
}

void strip_known_extensions(std::string *text) {
    if (text == nullptr) return;
    bool changed = true;
    while (changed) {
        changed = false;
        const std::string before = *text;
        strip_suffix(text, ".gz");
        strip_suffix(text, ".bz2");
        strip_suffix(text, ".xz");
        strip_suffix(text, ".tsv");
        strip_suffix(text, ".txt");
        strip_suffix(text, ".csv");
        strip_suffix(text, ".mtx");
        changed = *text != before;
    }
}

void trim_role_tokens(std::string *text, builder_path_role role) {
    if (text == nullptr) return;
    static constexpr const char *matrix_tokens[] = {"_matrix", "-matrix", ".matrix", " matrix", "matrix"};
    static constexpr const char *feature_tokens[] = {"_features", "-features", ".features", " features",
                                                     "_feature", "-feature", ".feature", " feature",
                                                     "_genes", "-genes", ".genes", " genes",
                                                     "_gene", "-gene", ".gene", " gene"};
    static constexpr const char *barcode_tokens[] = {"_barcodes", "-barcodes", ".barcodes", " barcodes",
                                                     "_barcode", "-barcode", ".barcode", " barcode"};
    static constexpr const char *metadata_tokens[] = {"_metadata", "-metadata", ".metadata", " metadata",
                                                      "_meta", "-meta", ".meta", " meta"};
    const char *const *tokens = nullptr;
    std::size_t token_count = 0;
    switch (role) {
        case builder_path_role::matrix:
            tokens = matrix_tokens;
            token_count = sizeof(matrix_tokens) / sizeof(matrix_tokens[0]);
            break;
        case builder_path_role::features:
            tokens = feature_tokens;
            token_count = sizeof(feature_tokens) / sizeof(feature_tokens[0]);
            break;
        case builder_path_role::barcodes:
            tokens = barcode_tokens;
            token_count = sizeof(barcode_tokens) / sizeof(barcode_tokens[0]);
            break;
        case builder_path_role::metadata:
            tokens = metadata_tokens;
            token_count = sizeof(metadata_tokens) / sizeof(metadata_tokens[0]);
            break;
        case builder_path_role::none:
            return;
    }
    for (std::size_t i = 0; i < token_count; ++i) strip_suffix(text, tokens[i]);
}

std::string sanitize_dataset_key(std::string value) {
    value = lower_copy(std::move(value));
    for (char &ch : value) {
        const bool sep = ch == ' ' || ch == '-' || ch == '.' || ch == '/' || ch == '\\';
        if (sep) ch = '_';
    }
    std::string out;
    out.reserve(value.size());
    bool prev_sep = false;
    for (char ch : value) {
        const bool sep = ch == '_' || std::isspace((unsigned char) ch);
        if (sep) {
            if (!prev_sep) out.push_back('_');
            prev_sep = true;
            continue;
        }
        out.push_back(ch);
        prev_sep = false;
    }
    while (!out.empty() && out.front() == '_') out.erase(out.begin());
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out;
}

bool path_is_readable_file(const std::string &path) {
    if (path.empty()) return false;
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    std::fclose(f);
    return true;
}

bool path_is_readable_dir(const fs::path &path) {
    std::error_code ec;
    fs::directory_iterator it(path, ec);
    return !ec;
}

std::string dataset_key_for_builder_path(const std::string &path, builder_path_role role) {
    fs::path p(path);
    std::string key = lower_copy(p.filename().string());
    strip_known_extensions(&key);
    trim_role_tokens(&key, role);
    key = sanitize_dataset_key(key);
    if (!key.empty() && key != "matrix" && key != "features" && key != "genes"
        && key != "barcodes" && key != "metadata" && key != "meta") {
        return key;
    }
    key = sanitize_dataset_key(p.parent_path().filename().string());
    if (!key.empty()) return key;
    key = sanitize_dataset_key(p.stem().string());
    return key.empty() ? std::string("dataset") : key;
}

std::string default_dataset_id_for_draft(const draft_dataset &draft, std::size_t index) {
    if (!draft.dataset_id.empty()) return draft.dataset_id;
    const std::string path = !draft.matrix_path.empty() ? draft.matrix_path
                            : (!draft.feature_path.empty() ? draft.feature_path
                               : (!draft.barcode_path.empty() ? draft.barcode_path : draft.metadata_path));
    if (!path.empty()) {
        const builder_path_role role = infer_builder_path_role(path);
        const std::string key = dataset_key_for_builder_path(path, role);
        if (!key.empty()) return key;
    }
    return "dataset_" + std::to_string(index + 1u);
}

std::string tsv_clean(std::string text) {
    for (char &ch : text) {
        if (ch == '\t' || ch == '\n' || ch == '\r') ch = ' ';
    }
    return text;
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

std::string execution_format_name(execution_format format) {
    switch (format) {
        case execution_format::compressed: return "compressed";
        case execution_format::blocked_ell: return "blocked_ell";
        case execution_format::mixed: return "mixed";
        case execution_format::unknown: return "unknown";
    }
    return "unknown";
}

std::string builder_path_role_name(builder_path_role role) {
    switch (role) {
        case builder_path_role::matrix: return "matrix";
        case builder_path_role::features: return "features";
        case builder_path_role::barcodes: return "barcodes";
        case builder_path_role::metadata: return "metadata";
        case builder_path_role::none: return "none";
    }
    return "none";
}

builder_path_role infer_builder_path_role(const std::string &path) {
    const std::string name = lower_copy(fs::path(path).filename().string());
    if (name.find(".mtx") != std::string::npos) return builder_path_role::matrix;
    if (name.find("barcode") != std::string::npos) return builder_path_role::barcodes;
    if (name.find("feature") != std::string::npos || name.find("gene") != std::string::npos) return builder_path_role::features;
    if (name.find("metadata") != std::string::npos || name.find("_meta") != std::string::npos
        || name.rfind("meta", 0) == 0) return builder_path_role::metadata;
    return builder_path_role::none;
}

std::vector<filesystem_entry> list_filesystem_entries(const std::string &dir_path,
                                                      std::vector<issue> *issues) {
    std::vector<filesystem_entry> entries;
    std::error_code ec;
    const fs::path root = dir_path.empty() ? fs::current_path(ec) : fs::path(dir_path);
    if (ec) {
        push_issue(issues, issue_severity::error, "builder", "failed to resolve current directory");
        return entries;
    }
    if (!fs::exists(root, ec) || ec) {
        push_issue(issues, issue_severity::error, "builder", "directory does not exist: " + root.string());
        return entries;
    }
    if (!fs::is_directory(root, ec) || ec) {
        push_issue(issues, issue_severity::error, "builder", "path is not a directory: " + root.string());
        return entries;
    }
    for (fs::directory_iterator it(root, fs::directory_options::skip_permission_denied, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
        const fs::directory_entry &entry = *it;
        filesystem_entry out;
        out.name = entry.path().filename().string();
        out.path = entry.path().string();
        out.is_symlink = entry.is_symlink(ec);
        ec.clear();
        out.is_directory = entry.is_directory(ec);
        ec.clear();
        out.is_regular = entry.is_regular_file(ec);
        ec.clear();
        if (out.is_regular) {
            out.size = (std::uint64_t) entry.file_size(ec);
            ec.clear();
            out.readable = path_is_readable_file(out.path);
        } else if (out.is_directory) {
            out.readable = path_is_readable_dir(entry.path());
        }
        entries.push_back(std::move(out));
    }
    if (ec) {
        push_issue(issues, issue_severity::warning, "builder", "incomplete directory listing for " + root.string());
    }
    std::sort(entries.begin(),
              entries.end(),
              [](const filesystem_entry &lhs, const filesystem_entry &rhs) {
                  if (lhs.is_directory != rhs.is_directory) return lhs.is_directory > rhs.is_directory;
                  if (lhs.is_regular != rhs.is_regular) return lhs.is_regular > rhs.is_regular;
                  return lower_copy(lhs.name) < lower_copy(rhs.name);
              });
    return entries;
}

std::vector<draft_dataset> discover_dataset_drafts(const std::string &dir_path,
                                                   std::vector<issue> *issues) {
    std::vector<draft_dataset> drafts;
    std::vector<filesystem_entry> entries = list_filesystem_entries(dir_path, issues);
    std::vector<std::string> keys;
    std::vector<draft_dataset> grouped;
    for (const filesystem_entry &entry : entries) {
        if (!entry.is_regular || !entry.readable) continue;
        const builder_path_role role = infer_builder_path_role(entry.path);
        if (role == builder_path_role::none) continue;
        const std::string key = dataset_key_for_builder_path(entry.path, role);
        std::size_t index = 0;
        while (index < keys.size() && keys[index] != key) ++index;
        if (index == keys.size()) {
            keys.push_back(key);
            grouped.push_back({});
            grouped.back().dataset_id = key;
        }
        draft_dataset &draft = grouped[index];
        std::string *target = nullptr;
        switch (role) {
            case builder_path_role::matrix: target = &draft.matrix_path; break;
            case builder_path_role::features: target = &draft.feature_path; break;
            case builder_path_role::barcodes: target = &draft.barcode_path; break;
            case builder_path_role::metadata: target = &draft.metadata_path; break;
            case builder_path_role::none: break;
        }
        if (target == nullptr) continue;
        if (!target->empty() && *target != entry.path) {
            push_issue(issues,
                       issue_severity::warning,
                       draft.dataset_id,
                       "multiple candidate " + builder_path_role_name(role) + " files detected; keeping the first one");
            continue;
        }
        *target = entry.path;
    }
    for (draft_dataset &draft : grouped) {
        if (!draft.matrix_path.empty()) drafts.push_back(std::move(draft));
    }
    std::sort(drafts.begin(),
              drafts.end(),
              [](const draft_dataset &lhs, const draft_dataset &rhs) {
                  return lower_copy(lhs.dataset_id) < lower_copy(rhs.dataset_id);
              });
    if (drafts.empty()) {
        push_issue(issues, issue_severity::info, "builder", "no manifest candidates were inferred from the current directory");
    }
    return drafts;
}

std::vector<source_entry> sources_from_dataset_drafts(const std::vector<draft_dataset> &drafts) {
    std::vector<source_entry> sources;
    sources.reserve(drafts.size());
    for (std::size_t i = 0; i < drafts.size(); ++i) {
        const draft_dataset &draft = drafts[i];
        if (!draft.included) continue;
        source_entry source;
        source.included = draft.included;
        source.dataset_id = default_dataset_id_for_draft(draft, i);
        source.matrix_path = draft.matrix_path;
        source.feature_path = draft.feature_path;
        source.barcode_path = draft.barcode_path;
        source.metadata_path = draft.metadata_path;
        source.format = draft.format;
        sources.push_back(std::move(source));
    }
    return sources;
}

manifest_inspection inspect_source_entries(const std::vector<source_entry> &sources,
                                           const std::string &label,
                                           std::size_t reader_bytes) {
    manifest_inspection inspection;
    inspection.manifest_path = label;
    inspection.sources.reserve(sources.size());
    for (std::size_t i = 0; i < sources.size(); ++i) {
        const source_entry &source = sources[i];
        inspection.sources.push_back(
            probe_source_entry(source,
                               reader_bytes,
                               &inspection.issues,
                               source.dataset_id.empty() ? (label + "[" + std::to_string(i) + "]") : source.dataset_id));
    }
    inspection.ok = !has_errors(inspection.issues);
    return inspection;
}

bool export_manifest_tsv(const std::string &path,
                         const std::vector<draft_dataset> &drafts,
                         std::size_t reader_bytes,
                         std::vector<issue> *issues) {
    if (path.empty()) {
        push_issue(issues, issue_severity::error, "export", "manifest export path is empty");
        return false;
    }
    const std::vector<source_entry> draft_sources = sources_from_dataset_drafts(drafts);
    if (draft_sources.empty()) {
        push_issue(issues, issue_severity::error, "export", "no included draft datasets are available to export");
        return false;
    }
    const manifest_inspection inspection = inspect_source_entries(draft_sources, path, reader_bytes);
    for (const issue &entry : inspection.issues) push_issue(issues, entry.severity, entry.scope, entry.message);
    if (!inspection.ok) return false;

    std::FILE *fp = std::fopen(path.c_str(), "wb");
    if (fp == nullptr) {
        push_issue(issues, issue_severity::error, "export", "failed to open manifest path for writing");
        return false;
    }
    const char *header = "dataset\tpath\tformat\tfeatures\tbarcodes\tmetadata\trows\tcols\tnnz\n";
    if (std::fputs(header, fp) < 0) {
        std::fclose(fp);
        push_issue(issues, issue_severity::error, "export", "failed to write manifest header");
        return false;
    }
    for (const source_entry &source : inspection.sources) {
        if (std::fprintf(fp,
                         "%s\t%s\t%s\t%s\t%s\t%s\t%lu\t%lu\t%lu\n",
                         tsv_clean(source.dataset_id).c_str(),
                         tsv_clean(source.matrix_path).c_str(),
                         format_name(source.format).c_str(),
                         tsv_clean(source.feature_path).c_str(),
                         tsv_clean(source.barcode_path).c_str(),
                         tsv_clean(source.metadata_path).c_str(),
                         source.rows,
                         source.cols,
                         source.nnz) < 0) {
            std::fclose(fp);
            push_issue(issues, issue_severity::error, "export", "failed while writing manifest rows");
            return false;
        }
    }
    std::fclose(fp);
    return true;
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
        std::vector<blocked_ell_candidate_stream> blocked_candidates;
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

        const bool have_blocked_ell_metrics = collect_row_sorted_blocked_ell_metrics(
            source,
            header,
            row_offsets,
            num_parts,
            policy,
            &blocked_candidates);

        for (unsigned long local_part = 0; local_part < num_parts; ++local_part) {
            const unsigned long rows = row_offsets[local_part + 1u] - row_offsets[local_part];
            const unsigned long nnz = part_nnz_raw[local_part];
            const std::size_t compressed_bytes = estimate_standard_csr_bytes(rows, nnz);
            execution_format preferred_format = execution_format::compressed;
            unsigned int blocked_ell_block_size = 0u;
            double blocked_ell_fill_ratio = 0.0;
            std::size_t blocked_ell_bytes = 0u;
            std::size_t execution_bytes = compressed_bytes;
            planned_part part;
            if (have_blocked_ell_metrics) {
                choose_part_execution_layout(
                    policy,
                    rows,
                    compressed_bytes,
                    blocked_candidates,
                    local_part,
                    &preferred_format,
                    &blocked_ell_block_size,
                    &blocked_ell_fill_ratio,
                    &blocked_ell_bytes,
                    &execution_bytes);
            }
            part.part_id = (unsigned long) plan.parts.size();
            part.source_index = source_index;
            part.dataset_id = source.dataset_id;
            part.row_begin = global_row_begin + row_offsets[local_part];
            part.row_end = global_row_begin + row_offsets[local_part + 1u];
            part.rows = rows;
            part.nnz = nnz;
            part.estimated_bytes = compressed_bytes;
            part.execution_bytes = execution_bytes;
            part.blocked_ell_bytes = blocked_ell_bytes;
            part.blocked_ell_fill_ratio = blocked_ell_fill_ratio;
            part.blocked_ell_block_size = blocked_ell_block_size;
            part.preferred_format = preferred_format;
            plan.parts.push_back(part);
            part_rows.push_back(rows);
            part_nnz.push_back(nnz);
            part_bytes.push_back((unsigned long) execution_bytes);
            plan.total_estimated_bytes += execution_bytes;
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
                std::size_t blocked_ell_bytes = 0u;
                std::size_t execution_bytes = 0u;
                std::size_t fill_weight = 0u;
                double fill_weighted_sum = 0.0;
                unsigned int shard_block_size = 0u;
                execution_format shard_format = execution_format::unknown;
                shard.shard_id = shard_id;
                shard.part_begin = range.part_begin;
                shard.part_end = range.part_end;
                shard.row_begin = range.row_begin;
                shard.row_end = range.row_end;
                shard.rows = range.row_end - range.row_begin;
                shard.nnz = range.nnz;
                shard.estimated_bytes = (std::size_t) range.bytes;
                shard.preferred_pair = (std::uint32_t) (shard_id & 1ul);
                plan.shards.push_back(shard);
                for (unsigned long part_id = range.part_begin; part_id < range.part_end && part_id < plan.parts.size(); ++part_id) {
                    plan.parts[part_id].shard_id = shard_id;
                    execution_bytes += plan.parts[part_id].execution_bytes;
                    blocked_ell_bytes += plan.parts[part_id].blocked_ell_bytes;
                    fill_weight += (std::size_t) plan.parts[part_id].rows;
                    fill_weighted_sum += plan.parts[part_id].blocked_ell_fill_ratio * (double) plan.parts[part_id].rows;
                    if (shard_format == execution_format::unknown) {
                        shard_format = plan.parts[part_id].preferred_format;
                    } else if (shard_format != plan.parts[part_id].preferred_format) {
                        shard_format = execution_format::mixed;
                    }
                    if (shard_block_size == 0u) {
                        shard_block_size = plan.parts[part_id].blocked_ell_block_size;
                    } else if (shard_block_size != plan.parts[part_id].blocked_ell_block_size) {
                        shard_block_size = 0u;
                    }
                }
                plan.shards.back().execution_bytes = execution_bytes;
                plan.shards.back().blocked_ell_bytes = blocked_ell_bytes;
                plan.shards.back().blocked_ell_fill_ratio = fill_weight != 0u ? (fill_weighted_sum / (double) fill_weight) : 0.0;
                plan.shards.back().blocked_ell_block_size = shard_block_size;
                plan.shards.back().preferred_format = shard_format == execution_format::unknown
                    ? execution_format::compressed
                    : shard_format;
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
    hid_t observation_metadata = (hid_t) -1;
    hid_t browse = (hid_t) -1;
    hid_t execution = (hid_t) -1;
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
    std::vector<std::uint32_t> part_execution_formats;
    std::vector<std::uint32_t> part_block_sizes;
    std::vector<float> part_fill_ratios;
    std::vector<std::uint64_t> part_execution_bytes;
    std::vector<std::uint64_t> part_blocked_ell_bytes;
    std::vector<std::uint32_t> shard_execution_formats;
    std::vector<std::uint32_t> shard_block_sizes;
    std::vector<float> shard_fill_ratios;
    std::vector<std::uint64_t> shard_execution_bytes;
    std::vector<std::uint32_t> shard_pair_ids;

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
        || !read_attr_u64(file, "num_parts", &summary.num_partitions)
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

    summary.partitions.reserve(part_rows.size());
    for (std::size_t i = 0; i < part_rows.size(); ++i) {
        summary.partitions.push_back(series_partition_summary{
            (std::uint64_t) i,
            i < part_row_offsets.size() ? part_row_offsets[i] : 0ull,
            i + 1u < part_row_offsets.size() ? part_row_offsets[i + 1u] : 0ull,
            i < part_rows.size() ? part_rows[i] : 0ull,
            i < part_nnz.size() ? part_nnz[i] : 0ull,
            i < part_dataset_ids.size() ? part_dataset_ids[i] : 0u,
            i < part_axes.size() ? part_axes[i] : 0u,
            i < part_codec_ids.size() ? part_codec_ids[i] : 0u,
            0u,
            0u,
            0.0f,
            0ull,
            0ull
        });
    }

    summary.shards.reserve(shard_offsets.size() > 0 ? shard_offsets.size() - 1u : 0u);
    for (std::size_t i = 0; i + 1u < shard_offsets.size(); ++i) {
        const std::uint64_t row_begin = shard_offsets[i];
        const std::uint64_t row_end = shard_offsets[i + 1u];
        const unsigned long partition_begin = find_partition_index_for_row(part_row_offsets, row_begin);
        const unsigned long partition_end = find_partition_end_for_row(part_row_offsets, row_end);
        summary.shards.push_back(series_shard_summary{
            (std::uint64_t) i,
            partition_begin,
            row_end == row_begin ? partition_begin : std::max<unsigned long>(partition_begin, partition_end),
            row_begin,
            row_end,
            0u,
            0u,
            0.0f,
            0ull,
            0u
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

    embedded_metadata = open_optional_group(file, "/embedded_metadata");
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

    observation_metadata = open_optional_group(file, "/observation_metadata");
    if (observation_metadata >= 0) {
        observation_metadata_summary metadata_summary;
        if (!read_attr_u64(observation_metadata, "rows", &metadata_summary.rows)
            || !read_attr_u32(observation_metadata, "cols", &metadata_summary.cols)) {
            push_issue(&summary.issues, issue_severity::warning, "inspect", "failed to read observation metadata header");
        } else {
            metadata_summary.columns.reserve(metadata_summary.cols);
            for (std::uint32_t i = 0; i < metadata_summary.cols; ++i) {
                char column_name[64];
                hid_t column = (hid_t) -1;
                observation_metadata_column_summary column_summary;
                if (std::snprintf(column_name, sizeof(column_name), "column_%u", i) <= 0) continue;
                column = H5Gopen2(observation_metadata, column_name, H5P_DEFAULT);
                if (column < 0) continue;
                if (!read_attr_string(column, "name", &column_summary.name)
                    || !read_attr_u32(column, "type", &column_summary.type)) {
                    push_issue(&summary.issues, issue_severity::warning, "inspect", "failed to read one observation metadata column");
                    H5Gclose(column);
                    continue;
                }
                metadata_summary.columns.push_back(std::move(column_summary));
                H5Gclose(column);
            }
            metadata_summary.available = true;
            summary.observation_metadata = std::move(metadata_summary);
        }
    }

    browse = open_optional_group(file, "/browse");
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

    execution = open_optional_group(file, "/execution");
    if (execution >= 0) {
        if (!read_dataset_vector(execution, "part_execution_formats", H5T_NATIVE_UINT32, &part_execution_formats)
            || !read_dataset_vector(execution, "part_blocked_ell_block_sizes", H5T_NATIVE_UINT32, &part_block_sizes)
            || !read_dataset_vector(execution, "part_blocked_ell_fill_ratios", H5T_NATIVE_FLOAT, &part_fill_ratios)
            || !read_dataset_vector(execution, "part_execution_bytes", H5T_NATIVE_UINT64, &part_execution_bytes)
            || !read_dataset_vector(execution, "part_blocked_ell_bytes", H5T_NATIVE_UINT64, &part_blocked_ell_bytes)
            || !read_dataset_vector(execution, "shard_execution_formats", H5T_NATIVE_UINT32, &shard_execution_formats)
            || !read_dataset_vector(execution, "shard_blocked_ell_block_sizes", H5T_NATIVE_UINT32, &shard_block_sizes)
            || !read_dataset_vector(execution, "shard_blocked_ell_fill_ratios", H5T_NATIVE_FLOAT, &shard_fill_ratios)
            || !read_dataset_vector(execution, "shard_execution_bytes", H5T_NATIVE_UINT64, &shard_execution_bytes)
            || !read_dataset_vector(execution, "shard_preferred_pair_ids", H5T_NATIVE_UINT32, &shard_pair_ids)) {
            push_issue(&summary.issues, issue_severity::warning, "inspect", "failed to read execution layout metadata");
        } else {
            for (std::size_t i = 0; i < summary.partitions.size(); ++i) {
                summary.partitions[i].execution_format = i < part_execution_formats.size() ? part_execution_formats[i] : 0u;
                summary.partitions[i].blocked_ell_block_size = i < part_block_sizes.size() ? part_block_sizes[i] : 0u;
                summary.partitions[i].blocked_ell_fill_ratio = i < part_fill_ratios.size() ? part_fill_ratios[i] : 0.0f;
                summary.partitions[i].execution_bytes = i < part_execution_bytes.size() ? part_execution_bytes[i] : 0ull;
                summary.partitions[i].blocked_ell_bytes = i < part_blocked_ell_bytes.size() ? part_blocked_ell_bytes[i] : 0ull;
            }
            for (std::size_t i = 0; i < summary.shards.size(); ++i) {
                summary.shards[i].execution_format = i < shard_execution_formats.size() ? shard_execution_formats[i] : 0u;
                summary.shards[i].blocked_ell_block_size = i < shard_block_sizes.size() ? shard_block_sizes[i] : 0u;
                summary.shards[i].blocked_ell_fill_ratio = i < shard_fill_ratios.size() ? shard_fill_ratios[i] : 0.0f;
                summary.shards[i].execution_bytes = i < shard_execution_bytes.size() ? shard_execution_bytes[i] : 0ull;
                summary.shards[i].preferred_pair = i < shard_pair_ids.size() ? shard_pair_ids[i] : 0u;
            }
        }
    }

    summary.ok = !has_errors(summary.issues);

done:
    if (execution >= 0) H5Gclose(execution);
    if (browse >= 0) H5Gclose(browse);
    if (observation_metadata >= 0) H5Gclose(observation_metadata);
    if (embedded_metadata >= 0) H5Gclose(embedded_metadata);
    if (codecs >= 0) H5Gclose(codecs);
    if (provenance >= 0) H5Gclose(provenance);
    if (matrix >= 0) H5Gclose(matrix);
    if (datasets >= 0) H5Gclose(datasets);
    if (file >= 0) H5Fclose(file);
    return summary;
}

embedded_metadata_table load_embedded_metadata_table(const std::string &path,
                                                     std::size_t table_index) {
    embedded_metadata_table table_summary;
    hid_t file = (hid_t) -1;
    hid_t embedded_metadata = (hid_t) -1;
    hid_t table = (hid_t) -1;
    std::uint32_t metadata_count = 0;
    std::vector<std::uint32_t> dataset_indices;
    std::vector<std::uint64_t> global_row_begin;
    std::vector<std::uint64_t> global_row_end;
    char table_name[64];

    if (path.empty()) {
        table_summary.error = "series path is empty";
        return table_summary;
    }

    file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        table_summary.error = "failed to open series file";
        return table_summary;
    }

    embedded_metadata = open_group(file, "/embedded_metadata");
    if (embedded_metadata < 0) {
        table_summary.error = "embedded metadata section is missing";
        goto done;
    }

    if (!read_attr_u32(embedded_metadata, "count", &metadata_count)
        || !read_dataset_vector(embedded_metadata, "dataset_indices", H5T_NATIVE_UINT32, &dataset_indices)
        || !read_dataset_vector(embedded_metadata, "global_row_begin", H5T_NATIVE_UINT64, &global_row_begin)
        || !read_dataset_vector(embedded_metadata, "global_row_end", H5T_NATIVE_UINT64, &global_row_end)) {
        table_summary.error = "failed to read embedded metadata directory";
        goto done;
    }

    if (table_index >= (std::size_t) metadata_count) {
        table_summary.error = "embedded metadata table index is out of range";
        goto done;
    }

    if (std::snprintf(table_name, sizeof(table_name), "table_%zu", table_index) <= 0) {
        table_summary.error = "failed to format embedded metadata table name";
        goto done;
    }
    table = H5Gopen2(embedded_metadata, table_name, H5P_DEFAULT);
    if (table < 0) {
        table_summary.error = "failed to open embedded metadata table";
        goto done;
    }

    if (!read_attr_u32(table, "rows", &table_summary.rows)
        || !read_attr_u32(table, "cols", &table_summary.cols)
        || !read_text_column_strings(table, "column_names", &table_summary.column_names)
        || !read_text_column_strings(table, "field_values", &table_summary.field_values)
        || !read_dataset_vector(table, "row_offsets", H5T_NATIVE_UINT32, &table_summary.row_offsets)) {
        table_summary.error = "failed to read embedded metadata payload";
        goto done;
    }

    if (table_index < dataset_indices.size()) table_summary.dataset_index = dataset_indices[table_index];
    if (table_index < global_row_begin.size()) table_summary.row_begin = global_row_begin[table_index];
    if (table_index < global_row_end.size()) table_summary.row_end = global_row_end[table_index];
    table_summary.available = true;

done:
    if (table >= 0) H5Gclose(table);
    if (embedded_metadata >= 0) H5Gclose(embedded_metadata);
    if (file >= 0) H5Fclose(file);
    return table_summary;
}

observation_metadata_table load_observation_metadata_table(const std::string &path) {
    observation_metadata_table table_summary;
    hid_t file = (hid_t) -1;
    hid_t metadata = (hid_t) -1;
    bool ok = true;

    if (path.empty()) {
        table_summary.error = "series path is empty";
        return table_summary;
    }

    file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        table_summary.error = "failed to open series file";
        return table_summary;
    }

    metadata = open_group(file, "/observation_metadata");
    if (metadata < 0) {
        table_summary.error = "observation metadata section is missing";
        ok = false;
    }

    if (ok
        && (!read_attr_u64(metadata, "rows", &table_summary.rows)
            || !read_attr_u32(metadata, "cols", &table_summary.cols))) {
        table_summary.error = "failed to read observation metadata header";
        ok = false;
    }

    if (ok) {
        table_summary.columns.reserve(table_summary.cols);
        for (std::uint32_t i = 0; i < table_summary.cols; ++i) {
            char column_name[64];
            hid_t column = (hid_t) -1;
            observation_metadata_column column_data;

            if (std::snprintf(column_name, sizeof(column_name), "column_%u", i) <= 0) {
                table_summary.error = "failed to format observation metadata column name";
                ok = false;
                break;
            }
            column = H5Gopen2(metadata, column_name, H5P_DEFAULT);
            if (column < 0) {
                table_summary.error = "failed to open one observation metadata column";
                ok = false;
                break;
            }
            if (!read_attr_string(column, "name", &column_data.name)
                || !read_attr_u32(column, "type", &column_data.type)) {
                table_summary.error = "failed to read observation metadata column header";
                H5Gclose(column);
                ok = false;
                break;
            }

            if (column_data.type == cellshard::series_observation_metadata_type_text) {
                if (!read_text_column_strings(column, "values", &column_data.text_values)) {
                    table_summary.error = "failed to read observation metadata text payload";
                    H5Gclose(column);
                    ok = false;
                    break;
                }
            } else if (column_data.type == cellshard::series_observation_metadata_type_float32) {
                if (!read_dataset_vector(column, "values", H5T_NATIVE_FLOAT, &column_data.float32_values)) {
                    table_summary.error = "failed to read observation metadata float payload";
                    H5Gclose(column);
                    ok = false;
                    break;
                }
            } else if (column_data.type == cellshard::series_observation_metadata_type_uint8) {
                if (!read_dataset_vector(column, "values", H5T_NATIVE_UINT8, &column_data.uint8_values)) {
                    table_summary.error = "failed to read observation metadata uint8 payload";
                    H5Gclose(column);
                    ok = false;
                    break;
                }
            } else {
                table_summary.error = "observation metadata contains an unknown column type";
                H5Gclose(column);
                ok = false;
                break;
            }

            table_summary.columns.push_back(std::move(column_data));
            H5Gclose(column);
        }
    }

    if (ok) table_summary.available = true;

done:
    if (metadata >= 0) H5Gclose(metadata);
    if (file >= 0) H5Fclose(file);
    return table_summary;
}

} // namespace cellerator::apps::workbench
