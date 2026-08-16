#include <Cellerator/compute/sampling_materialization.hh>

#include <CellShard/export/dataset_export.hh>

#include <algorithm>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace cellerator::compute::sampling {

namespace {

namespace cm = ::cellerator::matrix;
namespace cse = ::cellshard::exporting;
namespace ct = ::cellerator::types;

struct staged_csr_structure {
    std::uint64_t sampled_row_count = 0u;
    std::uint64_t gene_count = 0u;
    std::uint64_t nnz = 0u;
    std::unique_ptr<ct::ptr_t[]> row_ptr;
    std::unique_ptr<ct::idx_t[]> column_indices;
    std::unique_ptr<std::uint64_t[]> sampled_position_to_global_row;
};

void set_error(std::string *error, const std::string &message) {
    if (error != nullptr) *error = message;
}

bool fits_canonical_ptr(std::uint64_t value) {
    return value <= (std::uint64_t) std::numeric_limits<ct::ptr_t>::max();
}

bool fits_canonical_index(std::uint64_t value) {
    return value <= (std::uint64_t) std::numeric_limits<ct::idx_t>::max();
}

bool validate_and_order_selection(std::uint64_t total_rows,
                                  const sample_plan &plan,
                                  std::vector<std::uint64_t> *ordered_rows,
                                  std::string *error) {
    const std::size_t selected = plan.global_row_indices.size();
    if (ordered_rows == nullptr) {
        set_error(error, "ordered sampled rows output is null");
        return false;
    }
    ordered_rows->clear();
    if (plan.provenance.hash_algorithm != splitmix64_algorithm_name
        || plan.provenance.hash_version != splitmix64_algorithm_version) {
        set_error(error, "sample plan uses an unsupported hash contract");
        return false;
    }
    if (plan.provenance.split_name.empty()) {
        set_error(error, "sample plan split identity is empty");
        return false;
    }
    if (plan.provenance.mode != selection_mode::hash_quantile_range
        && plan.provenance.mode != selection_mode::exact_lowest_hash
        && plan.provenance.mode != selection_mode::density_quantile_exact) {
        set_error(error, "sample plan selection mode is invalid");
        return false;
    }
    if (plan.provenance.cell_identity != cell_identity_kind::global_row_index
        && plan.provenance.cell_identity != cell_identity_kind::stable_cellshard_cell_id) {
        set_error(error, "sample plan cell identity kind is invalid");
        return false;
    }
    if (plan.provenance.total_rows != total_rows) {
        set_error(error, "sample population size does not match the source dataset");
        return false;
    }
    if (plan.provenance.selected_rows != (std::uint64_t) selected
        || plan.identity_hashes.size() != selected) {
        set_error(error, "sample plan arrays do not match provenance selected_rows");
        return false;
    }
    if ((!plan.row_strata.empty() && plan.row_strata.size() != selected)
        || (!plan.sampling_weights.empty() && plan.sampling_weights.size() != selected)) {
        set_error(error, "sample plan aligned metadata has an inconsistent size");
        return false;
    }
    if (plan.provenance.mode == selection_mode::density_quantile_exact
        && (plan.row_strata.size() != selected || plan.sampling_weights.size() != selected)) {
        set_error(error, "density sample plan is missing aligned strata or weights");
        return false;
    }
    if (!fits_canonical_ptr((std::uint64_t) selected)) {
        set_error(error, "sampled row count exceeds Cellerator CSR limits");
        return false;
    }

    ordered_rows->reserve(selected);
    for (std::size_t position = 0u; position < selected; ++position) {
        const std::uint64_t row = plan.global_row_indices[position];
        if (row >= total_rows) {
            set_error(error, "sampled global row is outside the source dataset");
            return false;
        }
        if (plan.provenance.cell_identity == cell_identity_kind::global_row_index
            && plan.identity_hashes[position] != hash_global_row_index(row, plan.provenance.seed)) {
            set_error(error, "sample row/hash alignment is invalid");
            return false;
        }
        ordered_rows->push_back(row);
    }
    std::sort(ordered_rows->begin(), ordered_rows->end());
    if (std::adjacent_find(ordered_rows->begin(), ordered_rows->end()) != ordered_rows->end()) {
        set_error(error, "sample plan contains duplicate global rows");
        ordered_rows->clear();
        return false;
    }
    return true;
}

bool allocate_structure(std::uint64_t sampled_rows,
                        std::uint64_t gene_count,
                        std::uint64_t nnz,
                        staged_csr_structure *out,
                        std::string *error) {
    if (out == nullptr) {
        set_error(error, "staged sampled CSR output is null");
        return false;
    }
    if (!fits_canonical_ptr(sampled_rows) || !fits_canonical_index(gene_count)
        || !fits_canonical_ptr(nnz)) {
        set_error(error, "sampled CSR dimensions exceed Cellerator index limits");
        return false;
    }
    if (sampled_rows == (std::uint64_t) std::numeric_limits<std::size_t>::max()) {
        set_error(error, "sampled CSR row-pointer length overflows size_t");
        return false;
    }
    const std::size_t row_ptr_count = (std::size_t) sampled_rows + 1u;
    if (row_ptr_count > std::numeric_limits<std::size_t>::max() / sizeof(ct::ptr_t)
        || (std::size_t) nnz > std::numeric_limits<std::size_t>::max() / sizeof(ct::idx_t)
        || (std::size_t) sampled_rows > std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t)) {
        set_error(error, "sampled CSR allocation size overflows size_t");
        return false;
    }

    staged_csr_structure staged;
    staged.sampled_row_count = sampled_rows;
    staged.gene_count = gene_count;
    staged.nnz = nnz;
    staged.row_ptr.reset(new (std::nothrow) ct::ptr_t[row_ptr_count]);
    if (nnz != 0u) staged.column_indices.reset(new (std::nothrow) ct::idx_t[(std::size_t) nnz]);
    if (sampled_rows != 0u) {
        staged.sampled_position_to_global_row.reset(
            new (std::nothrow) std::uint64_t[(std::size_t) sampled_rows]);
    }
    if (staged.row_ptr == nullptr
        || (nnz != 0u && staged.column_indices == nullptr)
        || (sampled_rows != 0u && staged.sampled_position_to_global_row == nullptr)) {
        set_error(error, "failed to allocate sampled CSR structure");
        return false;
    }
    *out = std::move(staged);
    return true;
}

bool validate_source_row_ptr(const cm::compressed *source, std::string *error) {
    if (source == nullptr) {
        set_error(error, "source CSR matrix is null");
        return false;
    }
    if (source->axis != cm::compressed_by_row) {
        set_error(error, "sampled structural materialization requires row-compressed CSR");
        return false;
    }
    if (source->majorPtr == nullptr || (source->nnz != 0u && source->minorIdx == nullptr)) {
        set_error(error, "source CSR structural arrays are null");
        return false;
    }
    if (source->majorPtr[0] != 0u) {
        set_error(error, "source CSR row pointers must begin at zero");
        return false;
    }
    ct::ptr_t previous = 0u;
    for (std::uint64_t row = 0u; row < source->rows; ++row) {
        const ct::ptr_t next = source->majorPtr[(std::size_t) row + 1u];
        if (next < previous || next > source->nnz) {
            set_error(error, "source CSR row pointers are non-monotonic or exceed nnz");
            return false;
        }
        previous = next;
    }
    if (previous != source->nnz) {
        set_error(error, "source CSR terminal row pointer does not equal nnz");
        return false;
    }
    return true;
}

bool gather_in_memory_structure(const cm::compressed *source,
                                const std::vector<std::uint64_t> &rows,
                                staged_csr_structure *out,
                                std::string *error) {
    std::uint64_t nnz = 0u;
    for (std::uint64_t row : rows) {
        const std::uint64_t begin = source->majorPtr[(std::size_t) row];
        const std::uint64_t end = source->majorPtr[(std::size_t) row + 1u];
        const std::uint64_t row_nnz = end - begin;
        if (row_nnz > (std::uint64_t) std::numeric_limits<ct::ptr_t>::max() - nnz) {
            set_error(error, "sampled CSR nnz accumulation overflowed Cellerator row pointers");
            return false;
        }
        nnz += row_nnz;
    }
    if (!allocate_structure((std::uint64_t) rows.size(), source->cols, nnz, out, error)) return false;

    std::uint64_t cursor = 0u;
    out->row_ptr[0] = 0u;
    for (std::size_t position = 0u; position < rows.size(); ++position) {
        const std::uint64_t row = rows[position];
        const ct::ptr_t begin = source->majorPtr[(std::size_t) row];
        const ct::ptr_t end = source->majorPtr[(std::size_t) row + 1u];
        out->sampled_position_to_global_row[position] = row;
        for (ct::ptr_t slot = begin; slot < end; ++slot) {
            const ct::idx_t column = source->minorIdx[slot];
            if ((std::uint64_t) column >= source->cols) {
                set_error(error, "source CSR column index is outside the gene dimension");
                return false;
            }
            out->column_indices[(std::size_t) cursor++] = column;
        }
        out->row_ptr[position + 1u] = (ct::ptr_t) cursor;
    }
    return true;
}

bool convert_cellshard_structure(const cse::csr_matrix_export &csr,
                                 const std::vector<std::uint64_t> &rows,
                                 std::uint64_t expected_gene_count,
                                 staged_csr_structure *out,
                                 std::string *error) {
    if (csr.rows != rows.size() || csr.cols != expected_gene_count) {
        set_error(error, "CellShard selected-row CSR shape does not match the request");
        return false;
    }
    if (csr.indptr.size() != rows.size() + 1u || csr.indices.size() != csr.data.size()) {
        set_error(error, "CellShard selected-row CSR arrays are inconsistent");
        return false;
    }
    if (csr.indptr.empty() || csr.indptr[0] != 0) {
        set_error(error, "CellShard selected-row CSR pointers must begin at zero");
        return false;
    }
    std::int64_t previous = 0;
    for (std::int64_t pointer : csr.indptr) {
        if (pointer < previous || pointer < 0
            || (std::uint64_t) pointer > csr.indices.size()
            || !fits_canonical_ptr((std::uint64_t) pointer)) {
            set_error(error, "CellShard selected-row CSR pointers are invalid");
            return false;
        }
        previous = pointer;
    }
    if ((std::uint64_t) previous != csr.indices.size()) {
        set_error(error, "CellShard selected-row CSR terminal pointer does not equal nnz");
        return false;
    }
    if (!allocate_structure((std::uint64_t) rows.size(), csr.cols,
                            (std::uint64_t) csr.indices.size(), out, error)) {
        return false;
    }
    for (std::size_t i = 0u; i < csr.indptr.size(); ++i) {
        out->row_ptr[i] = (ct::ptr_t) csr.indptr[i];
    }
    for (std::size_t slot = 0u; slot < csr.indices.size(); ++slot) {
        const std::int64_t column = csr.indices[slot];
        if (column < 0 || (std::uint64_t) column >= csr.cols
            || !fits_canonical_index((std::uint64_t) column)) {
            set_error(error, "CellShard selected-row column index is outside the gene dimension");
            return false;
        }
        out->column_indices[slot] = (ct::idx_t) column;
    }
    for (std::size_t position = 0u; position < rows.size(); ++position) {
        out->sampled_position_to_global_row[position] = rows[position];
    }
    return true;
}

} // namespace

owned_sampled_csr_structure::owned_sampled_csr_structure(
    std::uint64_t sampled_row_count,
    std::uint64_t gene_count,
    std::uint64_t nnz,
    std::unique_ptr<ct::ptr_t[]> row_ptr,
    std::unique_ptr<ct::idx_t[]> column_indices,
    std::unique_ptr<std::uint64_t[]> sampled_position_to_global_row,
    sample_provenance provenance) noexcept
    : sampled_row_count_(sampled_row_count),
      gene_count_(gene_count),
      nnz_(nnz),
      row_ptr_(std::move(row_ptr)),
      column_indices_(std::move(column_indices)),
      sampled_position_to_global_row_(std::move(sampled_position_to_global_row)),
      provenance_(std::move(provenance)) {}

sampled_csr_structure_view owned_sampled_csr_structure::view() const noexcept {
    return {
        sampled_row_count_,
        gene_count_,
        nnz_,
        row_ptr_.get(),
        column_indices_.get(),
        sampled_position_to_global_row_.get(),
        &provenance_
    };
}

const sample_provenance &owned_sampled_csr_structure::sampling_provenance() const noexcept {
    return provenance_;
}

bool materialize_sampled_csr_structure(const cm::compressed *source,
                                       const sample_plan &plan,
                                       owned_sampled_csr_structure *out,
                                       std::string *error) {
    std::vector<std::uint64_t> ordered_rows;
    staged_csr_structure staged;
    if (out == nullptr) {
        set_error(error, "owned sampled CSR output is null");
        return false;
    }
    if (!validate_source_row_ptr(source, error)
        || !validate_and_order_selection(source->rows, plan, &ordered_rows, error)
        || !gather_in_memory_structure(source, ordered_rows, &staged, error)) {
        return false;
    }
    *out = owned_sampled_csr_structure(
        staged.sampled_row_count,
        staged.gene_count,
        staged.nnz,
        std::move(staged.row_ptr),
        std::move(staged.column_indices),
        std::move(staged.sampled_position_to_global_row),
        plan.provenance);
    return true;
}

bool materialize_cellshard_sampled_csr_structure(const char *path,
                                                 const sample_plan &plan,
                                                 owned_sampled_csr_structure *out,
                                                 std::string *error) {
    cse::dataset_summary summary;
    cse::csr_matrix_export csr;
    std::vector<std::uint64_t> ordered_rows;
    staged_csr_structure staged;
    if (path == nullptr || *path == '\0' || out == nullptr) {
        set_error(error, "CellShard sampled CSR materialization requires a path and output");
        return false;
    }
    if (!cse::load_dataset_summary(path, &summary, error)
        || !validate_and_order_selection(summary.rows, plan, &ordered_rows, error)) {
        return false;
    }
    if (!cse::load_dataset_rows_as_csr(
            path,
            ordered_rows.empty() ? nullptr : ordered_rows.data(),
            ordered_rows.size(),
            &csr,
            error)) {
        return false;
    }
    if (!convert_cellshard_structure(csr, ordered_rows, summary.cols, &staged, error)) return false;
    *out = owned_sampled_csr_structure(
        staged.sampled_row_count,
        staged.gene_count,
        staged.nnz,
        std::move(staged.row_ptr),
        std::move(staged.column_indices),
        std::move(staged.sampled_position_to_global_row),
        plan.provenance);
    return true;
}

} // namespace cellerator::compute::sampling
