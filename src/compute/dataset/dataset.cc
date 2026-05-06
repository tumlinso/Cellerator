#include <Cellerator/compute/dataset.hh>

#include <CellShard/io/csh5/api.cuh>

#include <algorithm>
#include <limits>
#include <map>
#include <random>
#include <utility>

namespace cellerator::compute::dataset {

namespace {

namespace ct = ::cellerator::core::types;

void set_error(std::string *error, const std::string &message) {
    if (error != nullptr) *error = message;
}

bool fits_u32(std::uint64_t value) {
    return value <= (std::uint64_t) std::numeric_limits<std::uint32_t>::max();
}

bool validate_row_labels(const stratified_sample_request &request, std::string *error) {
    if (request.row_count == 0u) {
        set_error(error, "stratified sampling requires at least one row");
        return false;
    }
    if (request.labels.labels == nullptr || request.labels.count != request.row_count) {
        set_error(error, "row label count does not match row count");
        return false;
    }
    if (request.max_rows_per_stratum == 0u) {
        set_error(error, "max_rows_per_stratum must be positive");
        return false;
    }
    for (std::uint64_t row = 0u; row < request.row_count; ++row) {
        const char *label = request.labels.labels[(std::size_t) row];
        if (label == nullptr || *label == '\0') {
            set_error(error, "row labels must be non-empty");
            return false;
        }
    }
    return true;
}

bool append_group_rows(const std::vector<std::uint64_t> &sampled,
                       const std::string &label,
                       std::uint64_t max_total_rows,
                       stratified_sample_plan *out) {
    if (sampled.empty()) return true;
    const std::uint64_t begin = (std::uint64_t) out->row_indices.size();
    std::uint64_t take = (std::uint64_t) sampled.size();
    if (max_total_rows != 0u) {
        if (begin >= max_total_rows) return false;
        const std::uint64_t remaining = max_total_rows - begin;
        if (take > remaining) take = remaining;
    }
    out->row_indices.insert(out->row_indices.end(), sampled.begin(), sampled.begin() + (std::ptrdiff_t) take);
    out->row_groups.push_back(row_group_span{label, begin, begin + take});
    return max_total_rows == 0u || begin + take < max_total_rows;
}

bool assign_compressed_from_csr(const cse::csr_matrix_export &csr, cm::compressed *out, std::string *error) {
    if (out == nullptr) {
        set_error(error, "output compressed matrix is null");
        return false;
    }
    if (!fits_u32(csr.rows) || !fits_u32(csr.cols) || !fits_u32(csr.data.size())) {
        set_error(error, "CSR export exceeds Cellerator compressed index limits");
        return false;
    }
    if (csr.indptr.size() != (std::size_t) csr.rows + 1u || csr.indices.size() != csr.data.size()) {
        set_error(error, "CSR export arrays are inconsistent");
        return false;
    }
    cm::clear(out);
    cm::init(out,
             (ct::dim_t) csr.rows,
             (ct::dim_t) csr.cols,
             (ct::nnz_t) csr.data.size(),
             cm::compressed_by_row);
    if (!cm::allocate(out)) {
        set_error(error, "failed to allocate compressed output");
        cm::init(out);
        return false;
    }
    for (std::size_t i = 0; i < csr.indptr.size(); ++i) {
        if (csr.indptr[i] < 0 || !fits_u32((std::uint64_t) csr.indptr[i])) {
            set_error(error, "CSR row pointer exceeds Cellerator index limits");
            cm::clear(out);
            return false;
        }
        out->majorPtr[i] = (ct::ptr_t) csr.indptr[i];
    }
    for (std::size_t i = 0; i < csr.indices.size(); ++i) {
        if (csr.indices[i] < 0 || !fits_u32((std::uint64_t) csr.indices[i])) {
            set_error(error, "CSR column index exceeds Cellerator index limits");
            cm::clear(out);
            return false;
        }
        out->minorIdx[i] = (ct::idx_t) csr.indices[i];
        out->val[i] = __float2half(csr.data[i]);
    }
    return true;
}

} // namespace

owned_dataset_artifact::owned_dataset_artifact() {
    cm::init(&matrix);
}

owned_dataset_artifact::~owned_dataset_artifact() {
    clear();
}

owned_dataset_artifact::owned_dataset_artifact(owned_dataset_artifact &&other) noexcept {
    cm::init(&matrix);
    *this = std::move(other);
}

owned_dataset_artifact &owned_dataset_artifact::operator=(owned_dataset_artifact &&other) noexcept {
    if (this == &other) return *this;
    clear();
    matrix = other.matrix;
    cm::init(&other.matrix);
    summary = other.summary;
    source_row_indices = std::move(other.source_row_indices);
    source_row_labels = std::move(other.source_row_labels);
    row_groups = std::move(other.row_groups);
    other.summary = dataset_summary_view{};
    return *this;
}

void owned_dataset_artifact::clear() {
    cm::clear(&matrix);
    summary = dataset_summary_view{};
    source_row_indices.clear();
    source_row_labels.clear();
    row_groups.clear();
}

dataset_matrix_handle make_cellshard_file_handle(const char *path) {
    dataset_matrix_handle out;
    out.kind = dataset_matrix_kind::cellshard_file;
    out.cellshard_path = path;
    return out;
}

dataset_matrix_handle make_compressed_handle(const cm::compressed *matrix) {
    dataset_matrix_handle out;
    out.kind = dataset_matrix_kind::owned_compressed;
    out.compressed = matrix;
    if (matrix != nullptr) {
        out.summary.rows = matrix->rows;
        out.summary.cols = matrix->cols;
        out.summary.nnz = matrix->nnz;
        out.summary.layout = matrix->axis == cm::compressed_by_row
            ? dataset_layout_kind::compressed_csr
            : dataset_layout_kind::unknown;
    }
    return out;
}

dataset_matrix_handle make_cellshard_sharded_compressed_handle(
    const ::cellshard::sharded< ::cellshard::sparse::compressed > *matrix) {
    dataset_matrix_handle out;
    out.kind = dataset_matrix_kind::cellshard_sharded_compressed;
    out.cellshard_compressed = matrix;
    if (matrix != nullptr) {
        out.summary.rows = matrix->rows;
        out.summary.cols = matrix->cols;
        out.summary.nnz = matrix->nnz;
        out.summary.layout = dataset_layout_kind::compressed_csr;
    }
    return out;
}

bool build_stratified_row_sample(const stratified_sample_request &request,
                                 stratified_sample_plan *out,
                                 std::string *error) {
    std::map<std::string, std::vector<std::uint64_t>> by_label;
    std::mt19937_64 rng(request.seed);
    if (out == nullptr) {
        set_error(error, "output stratified sample plan is null");
        return false;
    }
    out->row_indices.clear();
    out->row_groups.clear();
    if (!validate_row_labels(request, error)) return false;
    for (std::uint64_t row = 0u; row < request.row_count; ++row) {
        by_label[request.labels.labels[(std::size_t) row]].push_back(row);
    }
    for (auto &entry : by_label) {
        std::vector<std::uint64_t> rows = entry.second;
        const std::uint64_t take = std::min<std::uint64_t>(
            request.max_rows_per_stratum,
            (std::uint64_t) rows.size());
        if (take < rows.size()) {
            std::shuffle(rows.begin(), rows.end(), rng);
            rows.resize((std::size_t) take);
        }
        std::sort(rows.begin(), rows.end());
        if (!append_group_rows(rows, entry.first, request.max_total_rows, out)) break;
    }
    return !out->row_indices.empty();
}

bool load_cellshard_row_labels(const char *path,
                               const char *column_name,
                               std::vector<std::string> *out,
                               std::string *error) {
    std::vector<cse::observation_metadata_column> columns;
    if (path == nullptr || *path == '\0' || column_name == nullptr || *column_name == '\0' || out == nullptr) {
        set_error(error, "CellShard label load requires path, column name, and output vector");
        return false;
    }
    if (!cse::load_observation_metadata(path, &columns, error)) return false;
    for (const cse::observation_metadata_column &column : columns) {
        if (column.name != column_name) continue;
        if (column.type != ::cellshard::dataset_observation_metadata_type_text) {
            set_error(error, "stratified CellShard metadata column must be text");
            return false;
        }
        *out = column.text_values;
        return true;
    }
    set_error(error, std::string("missing CellShard observation metadata column: ") + column_name);
    return false;
}

bool rebuild_rows_as_compressed(const cm::compressed *source,
                                const std::uint64_t *row_indices,
                                std::uint64_t row_count,
                                cm::compressed *out,
                                std::string *error) {
    std::uint64_t nnz = 0u;
    std::uint64_t cursor = 0u;
    if (source == nullptr || out == nullptr || (row_count != 0u && row_indices == nullptr)) {
        set_error(error, "rebuild_rows_as_compressed received a null input");
        return false;
    }
    if (source->axis != cm::compressed_by_row) {
        set_error(error, "rebuild_rows_as_compressed requires row-compressed input");
        return false;
    }
    for (std::uint64_t i = 0u; i < row_count; ++i) {
        const std::uint64_t row = row_indices[(std::size_t) i];
        if (row >= source->rows) {
            set_error(error, "selected row is outside source matrix");
            return false;
        }
        nnz += (std::uint64_t) source->majorPtr[row + 1u] - (std::uint64_t) source->majorPtr[row];
    }
    if (!fits_u32(row_count) || !fits_u32(source->cols) || !fits_u32(nnz)) {
        set_error(error, "rebuilt matrix exceeds Cellerator compressed index limits");
        return false;
    }
    cm::clear(out);
    cm::init(out,
             (ct::dim_t) row_count,
             source->cols,
             (ct::nnz_t) nnz,
             cm::compressed_by_row);
    if (!cm::allocate(out)) {
        set_error(error, "failed to allocate rebuilt compressed matrix");
        cm::init(out);
        return false;
    }
    out->majorPtr[0] = 0u;
    for (std::uint64_t i = 0u; i < row_count; ++i) {
        const std::uint64_t row = row_indices[(std::size_t) i];
        const ct::ptr_t begin = source->majorPtr[row];
        const ct::ptr_t end = source->majorPtr[row + 1u];
        for (ct::ptr_t slot = begin; slot < end; ++slot) {
            out->minorIdx[(std::size_t) cursor] = source->minorIdx[slot];
            out->val[(std::size_t) cursor] = source->val[slot];
            ++cursor;
        }
        out->majorPtr[(std::size_t) i + 1u] = (ct::ptr_t) cursor;
    }
    return true;
}

bool rebuild_cellshard_rows_as_compressed(const char *path,
                                          const std::uint64_t *row_indices,
                                          std::uint64_t row_count,
                                          cm::compressed *out,
                                          std::string *error) {
    cse::csr_matrix_export csr;
    if (path == nullptr || *path == '\0' || out == nullptr || (row_count != 0u && row_indices == nullptr)) {
        set_error(error, "CellShard row rebuild requires path, rows, and output matrix");
        return false;
    }
    if (!cse::load_dataset_rows_as_csr(path, row_indices, (std::size_t) row_count, &csr, error)) return false;
    return assign_compressed_from_csr(csr, out, error);
}

bool build_stratified_downsample(const dataset_matrix_handle &source,
                                 const stratified_downsample_request &request,
                                 owned_dataset_artifact *out,
                                 std::string *error) {
    std::vector<std::string> owned_labels;
    std::vector<const char *> label_ptrs;
    stratified_sample_request sample_request;
    stratified_sample_plan plan;
    dataset_summary_view summary = source.summary;
    if (out == nullptr) {
        set_error(error, "output dataset artifact is null");
        return false;
    }
    out->clear();
    if (source.kind == dataset_matrix_kind::cellshard_file) {
        cse::dataset_summary cellshard_summary;
        if (source.cellshard_path == nullptr || *source.cellshard_path == '\0') {
            set_error(error, "CellShard dataset handle has no path");
            return false;
        }
        if (!cse::load_dataset_summary(source.cellshard_path, &cellshard_summary, error)) return false;
        summary.rows = cellshard_summary.rows;
        summary.cols = cellshard_summary.cols;
        summary.nnz = cellshard_summary.nnz;
        if (cellshard_summary.matrix_format == "blocked_ell") summary.layout = dataset_layout_kind::blocked_ell;
        else if (cellshard_summary.matrix_format == "sliced_ell") summary.layout = dataset_layout_kind::sliced_ell;
        else summary.layout = dataset_layout_kind::unknown;
        if (request.labels.labels != nullptr) sample_request.labels = request.labels;
        else {
            if (!load_cellshard_row_labels(source.cellshard_path, request.cellshard_label_column, &owned_labels, error)) return false;
            label_ptrs.reserve(owned_labels.size());
            for (const std::string &label : owned_labels) label_ptrs.push_back(label.c_str());
            sample_request.labels.labels = label_ptrs.empty() ? nullptr : label_ptrs.data();
            sample_request.labels.count = (std::uint64_t) label_ptrs.size();
        }
    } else {
        sample_request.labels = request.labels;
    }
    sample_request.row_count = summary.rows;
    sample_request.max_rows_per_stratum = request.max_rows_per_stratum;
    sample_request.max_total_rows = request.max_total_rows;
    sample_request.seed = request.seed;
    if (!build_stratified_row_sample(sample_request, &plan, error)) return false;
    if (source.kind == dataset_matrix_kind::owned_compressed) {
        if (!rebuild_rows_as_compressed(source.compressed,
                                        plan.row_indices.data(),
                                        (std::uint64_t) plan.row_indices.size(),
                                        &out->matrix,
                                        error)) return false;
    } else if (source.kind == dataset_matrix_kind::cellshard_file) {
        if (!rebuild_cellshard_rows_as_compressed(source.cellshard_path,
                                                  plan.row_indices.data(),
                                                  (std::uint64_t) plan.row_indices.size(),
                                                  &out->matrix,
                                                  error)) return false;
    } else {
        set_error(error, "dataset handle kind does not support in-memory rebuild in v1");
        return false;
    }
    out->source_row_indices = std::move(plan.row_indices);
    out->row_groups = std::move(plan.row_groups);
    out->source_row_labels.reserve(out->source_row_indices.size());
    for (std::uint64_t row : out->source_row_indices) {
        out->source_row_labels.emplace_back(sample_request.labels.labels[(std::size_t) row]);
    }
    out->summary = summary;
    out->summary.rows = out->matrix.rows;
    out->summary.cols = out->matrix.cols;
    out->summary.nnz = out->matrix.nnz;
    out->summary.layout = dataset_layout_kind::compressed_csr;
    return true;
}

} // namespace cellerator::compute::dataset
