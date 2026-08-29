#include <Cellerator/compute/sampling_materialization.hh>

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace cellerator::compute::sampling {

namespace {

namespace cm = ::cellerator::matrix;
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

bool image_layout(std::uint64_t sampled_rows, std::uint64_t nnz,
                  std::size_t *total, std::size_t *row_offset,
                  std::size_t *column_offset, std::size_t *mapping_offset) noexcept {
    if (total == nullptr || row_offset == nullptr || column_offset == nullptr
        || mapping_offset == nullptr || sampled_rows > SIZE_MAX - 1u) return false;
    const auto add = [](std::size_t *cursor, std::size_t alignment,
                        std::size_t count, std::size_t width, std::size_t *offset) {
        if (count > SIZE_MAX / width) return false;
        const std::size_t mask = alignment - 1u;
        if (*cursor > SIZE_MAX - mask) return false;
        const std::size_t aligned = (*cursor + mask) & ~mask;
        const std::size_t bytes = count * width;
        if (bytes > SIZE_MAX - aligned) return false;
        *offset = aligned;
        *cursor = aligned + bytes;
        return true;
    };
    std::size_t cursor = sizeof(sampled_csr_image_header);
    if (!add(&cursor, alignof(ct::ptr_t), static_cast<std::size_t>(sampled_rows) + 1u,
             sizeof(ct::ptr_t), row_offset)
        || !add(&cursor, alignof(ct::idx_t), static_cast<std::size_t>(nnz),
                sizeof(ct::idx_t), column_offset)
        || !add(&cursor, alignof(std::uint64_t), static_cast<std::size_t>(sampled_rows),
                sizeof(std::uint64_t), mapping_offset)) return false;
    *total = cursor;
    return true;
}

bool prepare_image(std::uint64_t sampled_rows, std::uint64_t gene_count,
                   std::uint64_t nnz, std::uint64_t selection_identity,
                   ::cellerator::memory::image_buffer image,
                   sampled_csr_image_view *out, std::string *error) {
    std::size_t bytes = 0u, row_offset = 0u, column_offset = 0u, mapping_offset = 0u;
    if (out == nullptr || !fits_canonical_ptr(sampled_rows)
        || !fits_canonical_ptr(nnz) || !fits_canonical_index(gene_count)
        || !image_layout(sampled_rows, nnz, &bytes, &row_offset, &column_offset, &mapping_offset)
        || image.base == nullptr || image.bytes < bytes) {
        set_error(error, "sampled CSR image capacity or dimensions are invalid");
        return false;
    }
    std::memset(image.base, 0, bytes);
    auto *header = static_cast<sampled_csr_image_header *>(image.base);
    auto *base = static_cast<unsigned char *>(image.base);
    header->common.magic = sampled_csr_image_magic;
    header->common.schema_version = sampled_csr_image_schema_version;
    header->common.total_bytes = bytes;
    header->common.required_alignment = alignof(sampled_csr_image_header);
    header->common.section_count = 3u;
    header->common.identity = selection_identity;
    header->sampled_row_count = sampled_rows;
    header->gene_count = gene_count;
    header->nnz = nnz;
    header->sample_selection_identity = selection_identity;
    header->row_ptr.byte_offset = row_offset;
    header->column_indices.byte_offset = column_offset;
    header->sampled_position_to_global_row.byte_offset = mapping_offset;
    *out = {header,
            reinterpret_cast<ct::ptr_t *>(base + row_offset),
            reinterpret_cast<ct::idx_t *>(base + column_offset),
            reinterpret_cast<std::uint64_t *>(base + mapping_offset)};
    return true;
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
        && plan.provenance.cell_identity != cell_identity_kind::stable_item_id) {
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

bool copy_selected_structure(const selected_csr_structure_view &source,
                             const std::vector<std::uint64_t> &rows,
                             staged_csr_structure *out,
                             std::string *error) {
    if (source.selected_row_count != rows.size()) {
        set_error(error, "selected-row CSR shape does not match the sample plan");
        return false;
    }
    if (source.row_ptr == nullptr
        || (source.nnz != 0u && source.column_indices == nullptr)
        || (source.selected_row_count != 0u && source.global_row_indices == nullptr)) {
        set_error(error, "selected-row CSR arrays are null");
        return false;
    }
    if (source.row_ptr[0] != 0u) {
        set_error(error, "selected-row CSR pointers must begin at zero");
        return false;
    }
    ct::ptr_t previous = 0u;
    for (std::size_t i = 0u; i <= rows.size(); ++i) {
        const ct::ptr_t pointer = source.row_ptr[i];
        if (pointer < previous || (std::uint64_t) pointer > source.nnz) {
            set_error(error, "selected-row CSR pointers are invalid");
            return false;
        }
        previous = pointer;
    }
    if ((std::uint64_t) previous != source.nnz) {
        set_error(error, "selected-row CSR terminal pointer does not equal nnz");
        return false;
    }
    if (!allocate_structure(source.selected_row_count, source.gene_count,
                            source.nnz, out, error)) {
        return false;
    }
    for (std::size_t i = 0u; i <= rows.size(); ++i) {
        out->row_ptr[i] = source.row_ptr[i];
    }
    for (std::size_t slot = 0u; slot < (std::size_t) source.nnz; ++slot) {
        const ct::idx_t column = source.column_indices[slot];
        if ((std::uint64_t) column >= source.gene_count) {
            set_error(error, "selected-row column index is outside the gene dimension");
            return false;
        }
        out->column_indices[slot] = column;
    }
    for (std::size_t position = 0u; position < rows.size(); ++position) {
        if (source.global_row_indices[position] != rows[position]) {
            set_error(error, "selected-row CSR rows are not aligned to the sample plan");
            return false;
        }
        out->sampled_position_to_global_row[position] = rows[position];
    }
    return true;
}

} // namespace

std::size_t sampled_csr_image_bytes(std::uint64_t sampled_rows,
                                    std::uint64_t nnz) noexcept {
    std::size_t bytes = 0u, row_offset = 0u, column_offset = 0u, mapping_offset = 0u;
    return image_layout(sampled_rows, nnz, &bytes, &row_offset, &column_offset,
                        &mapping_offset) ? bytes : 0u;
}

bool materialize_sampled_csr_image(const cm::compressed *source,
                                   const sample_selection_view &selection,
                                   ::cellerator::memory::image_buffer image,
                                   sampled_csr_image_view *out,
                                   std::string *error) {
    if (selection.header == nullptr || selection.selected_global_rows == nullptr
        || !validate_source_row_ptr(source, error)) {
        if (selection.header == nullptr || selection.selected_global_rows == nullptr)
            set_error(error, "sample selection view is incomplete");
        return false;
    }
    std::uint64_t nnz = 0u;
    for (std::uint64_t position = 0u; position < selection.header->selected_rows; ++position) {
        const std::uint64_t row = selection.selected_global_rows[position];
        if (row >= source->rows || (position != 0u && row <= selection.selected_global_rows[position - 1u])) {
            set_error(error, "sample selection rows are not canonical for source CSR");
            return false;
        }
        const std::uint64_t row_nnz = source->majorPtr[row + 1u] - source->majorPtr[row];
        if (row_nnz > std::numeric_limits<std::uint64_t>::max() - nnz) {
            set_error(error, "sampled CSR nnz overflows");
            return false;
        }
        nnz += row_nnz;
    }
    if (!prepare_image(selection.header->selected_rows, source->cols, nnz,
                       selection.header->common.identity, image, out, error)) return false;
    auto *row_ptr = const_cast<ct::ptr_t *>(out->row_ptr);
    auto *columns = const_cast<ct::idx_t *>(out->column_indices);
    auto *mapping = const_cast<std::uint64_t *>(out->sampled_position_to_global_row);
    std::uint64_t cursor = 0u;
    row_ptr[0] = 0u;
    for (std::uint64_t position = 0u; position < selection.header->selected_rows; ++position) {
        const std::uint64_t row = selection.selected_global_rows[position];
        mapping[position] = row;
        for (ct::ptr_t slot = source->majorPtr[row]; slot < source->majorPtr[row + 1u]; ++slot) {
            if (source->minorIdx[slot] >= source->cols) {
                set_error(error, "sampled CSR source column is out of range");
                return false;
            }
            columns[cursor++] = source->minorIdx[slot];
        }
        row_ptr[position + 1u] = static_cast<ct::ptr_t>(cursor);
    }
    return true;
}

bool materialize_selected_csr_image(const selected_csr_structure_view &source,
                                    const sample_selection_view &selection,
                                    ::cellerator::memory::image_buffer image,
                                    sampled_csr_image_view *out,
                                    std::string *error) {
    if (selection.header == nullptr || selection.selected_global_rows == nullptr
        || source.selected_row_count != selection.header->selected_rows
        || source.total_row_count != selection.header->population_rows
        || source.row_ptr == nullptr
        || (source.nnz != 0u && source.column_indices == nullptr)
        || (source.selected_row_count != 0u && source.global_row_indices == nullptr)) {
        set_error(error, "selected CSR and sample selection contract mismatch");
        return false;
    }
    if (source.row_ptr[0] != 0u || source.row_ptr[source.selected_row_count] != source.nnz) {
        set_error(error, "selected CSR row pointers do not span nnz");
        return false;
    }
    for (std::uint64_t i = 0u; i < source.selected_row_count; ++i) {
        if (source.global_row_indices[i] != selection.selected_global_rows[i]
            || source.row_ptr[i + 1u] < source.row_ptr[i]) {
            set_error(error, "selected CSR row mapping or pointers are invalid");
            return false;
        }
    }
    if (!prepare_image(source.selected_row_count, source.gene_count, source.nnz,
                       selection.header->common.identity, image, out, error)) return false;
    auto *row_ptr = const_cast<ct::ptr_t *>(out->row_ptr);
    auto *columns = const_cast<ct::idx_t *>(out->column_indices);
    auto *mapping = const_cast<std::uint64_t *>(out->sampled_position_to_global_row);
    std::memcpy(row_ptr, source.row_ptr,
                (static_cast<std::size_t>(source.selected_row_count) + 1u) * sizeof(ct::ptr_t));
    std::memcpy(mapping, source.global_row_indices,
                static_cast<std::size_t>(source.selected_row_count) * sizeof(std::uint64_t));
    for (std::uint64_t slot = 0u; slot < source.nnz; ++slot) {
        if (source.column_indices[slot] >= source.gene_count) {
            set_error(error, "selected CSR column is out of range");
            return false;
        }
        columns[slot] = source.column_indices[slot];
    }
    return true;
}

bool resolve_sampled_csr_image(::cellerator::memory::const_image_view image,
                               sampled_csr_image_view *out,
                               std::string *error) {
    if (out == nullptr || image.base == nullptr || image.bytes < sizeof(sampled_csr_image_header)) {
        set_error(error, "sampled CSR image is truncated or output is null");
        return false;
    }
    *out = {};
    const auto *header = static_cast<const sampled_csr_image_header *>(image.base);
    std::size_t expected = 0u, row_offset = 0u, column_offset = 0u, mapping_offset = 0u;
    if (header->common.magic != sampled_csr_image_magic
        || header->common.schema_version != sampled_csr_image_schema_version
        || !image_layout(header->sampled_row_count, header->nnz, &expected,
                         &row_offset, &column_offset, &mapping_offset)
        || header->common.total_bytes != expected || expected > image.bytes
        || header->row_ptr.byte_offset != row_offset
        || header->column_indices.byte_offset != column_offset
        || header->sampled_position_to_global_row.byte_offset != mapping_offset) {
        set_error(error, "sampled CSR image header or sections are invalid");
        return false;
    }
    const auto *base = static_cast<const unsigned char *>(image.base);
    const auto *row_ptr = reinterpret_cast<const ct::ptr_t *>(base + row_offset);
    const auto *columns = reinterpret_cast<const ct::idx_t *>(base + column_offset);
    const auto *mapping = reinterpret_cast<const std::uint64_t *>(base + mapping_offset);
    if (row_ptr[0] != 0u || row_ptr[header->sampled_row_count] != header->nnz) {
        set_error(error, "sampled CSR image row pointers do not span nnz");
        return false;
    }
    for (std::uint64_t row = 0u; row < header->sampled_row_count; ++row) {
        if (row_ptr[row + 1u] < row_ptr[row]
            || mapping[row] >= std::numeric_limits<std::uint64_t>::max()
            || (row != 0u && mapping[row] <= mapping[row - 1u])) {
            set_error(error, "sampled CSR image rows are invalid");
            return false;
        }
    }
    for (std::uint64_t slot = 0u; slot < header->nnz; ++slot) {
        if (columns[slot] >= header->gene_count) {
            set_error(error, "sampled CSR image column is invalid");
            return false;
        }
    }
    *out = {header, row_ptr, columns, mapping};
    return true;
}

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

bool materialize_selected_csr_structure(const selected_csr_structure_view &source,
                                        const sample_plan &plan,
                                        owned_sampled_csr_structure *out,
                                        std::string *error) {
    std::vector<std::uint64_t> ordered_rows;
    staged_csr_structure staged;
    if (out == nullptr) {
        set_error(error, "owned sampled CSR output is null");
        return false;
    }
    if (!validate_and_order_selection(source.total_row_count, plan, &ordered_rows, error)
        || !copy_selected_structure(source, ordered_rows, &staged, error)) {
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

} // namespace cellerator::compute::sampling
