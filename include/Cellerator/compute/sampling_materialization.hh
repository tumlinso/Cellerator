#pragma once

#include <Cellerator/compute/sampling.hh>
#include <Cellerator/matrix/compressed.cuh>

#include <cstdint>
#include <memory>
#include <string>

namespace cellerator::compute::sampling {

inline constexpr std::uint32_t sampled_csr_image_magic = 0x31524353u; // SCR1
inline constexpr std::uint16_t sampled_csr_image_schema_version = 1u;

struct selected_csr_structure_view;

struct sampled_csr_image_header {
    ::cellerator::memory::image_header common{};
    std::uint64_t sampled_row_count = 0u;
    std::uint64_t gene_count = 0u;
    std::uint64_t nnz = 0u;
    std::uint64_t sample_selection_identity = 0u;
    ::cellerator::memory::rel64 row_ptr{};
    ::cellerator::memory::rel64 column_indices{};
    ::cellerator::memory::rel64 sampled_position_to_global_row{};
};

struct sampled_csr_image_view {
    const sampled_csr_image_header *header = nullptr;
    const ::cellerator::types::ptr_t *row_ptr = nullptr;
    const ::cellerator::types::idx_t *column_indices = nullptr;
    const std::uint64_t *sampled_position_to_global_row = nullptr;
};

std::size_t sampled_csr_image_bytes(std::uint64_t sampled_rows,
                                    std::uint64_t nnz) noexcept;
bool materialize_sampled_csr_image(
    const ::cellerator::matrix::compressed *source,
    const sample_selection_view &selection,
    ::cellerator::memory::image_buffer image,
    sampled_csr_image_view *out,
    std::string *error = nullptr);
bool materialize_selected_csr_image(
    const selected_csr_structure_view &source,
    const sample_selection_view &selection,
    ::cellerator::memory::image_buffer image,
    sampled_csr_image_view *out,
    std::string *error = nullptr);
bool resolve_sampled_csr_image(::cellerator::memory::const_image_view image,
                               sampled_csr_image_view *out,
                               std::string *error = nullptr);

struct sampled_csr_structure_view {
    std::uint64_t sampled_row_count = 0u;
    std::uint64_t gene_count = 0u;
    std::uint64_t nnz = 0u;
    const ::cellerator::types::ptr_t *row_ptr = nullptr;
    const ::cellerator::types::idx_t *column_indices = nullptr;
    const std::uint64_t *sampled_position_to_global_row = nullptr;
    const sample_provenance *provenance = nullptr;
};

// Storage-neutral handoff for rows selected and loaded by an external owner.
// Rows and their CSR segments must be aligned in ascending global-row order.
struct selected_csr_structure_view {
    std::uint64_t total_row_count = 0u;
    std::uint64_t selected_row_count = 0u;
    std::uint64_t gene_count = 0u;
    std::uint64_t nnz = 0u;
    const ::cellerator::types::ptr_t *row_ptr = nullptr;
    const ::cellerator::types::idx_t *column_indices = nullptr;
    const std::uint64_t *global_row_indices = nullptr;
};

class owned_sampled_csr_structure {
public:
    owned_sampled_csr_structure() = default;
    ~owned_sampled_csr_structure() = default;
    owned_sampled_csr_structure(const owned_sampled_csr_structure &) = delete;
    owned_sampled_csr_structure &operator=(const owned_sampled_csr_structure &) = delete;
    owned_sampled_csr_structure(owned_sampled_csr_structure &&) noexcept = default;
    owned_sampled_csr_structure &operator=(owned_sampled_csr_structure &&) noexcept = default;

    sampled_csr_structure_view view() const noexcept;
    const sample_provenance &sampling_provenance() const noexcept;

private:
    owned_sampled_csr_structure(
        std::uint64_t sampled_row_count,
        std::uint64_t gene_count,
        std::uint64_t nnz,
        std::unique_ptr< ::cellerator::types::ptr_t[]> row_ptr,
        std::unique_ptr< ::cellerator::types::idx_t[]> column_indices,
        std::unique_ptr<std::uint64_t[]> sampled_position_to_global_row,
        sample_provenance provenance) noexcept;

    std::uint64_t sampled_row_count_ = 0u;
    std::uint64_t gene_count_ = 0u;
    std::uint64_t nnz_ = 0u;
    std::unique_ptr< ::cellerator::types::ptr_t[]> row_ptr_;
    std::unique_ptr< ::cellerator::types::idx_t[]> column_indices_;
    std::unique_ptr<std::uint64_t[]> sampled_position_to_global_row_;
    sample_provenance provenance_;

    friend bool materialize_sampled_csr_structure(
        const ::cellerator::matrix::compressed *,
        const sample_plan &,
        owned_sampled_csr_structure *,
        std::string *);
    friend bool materialize_selected_csr_structure(
        const selected_csr_structure_view &,
        const sample_plan &,
        owned_sampled_csr_structure *,
        std::string *);
};

// Sample positions are canonicalized to ascending global row index even if a
// caller supplies the selected rows in another order. Each complete source row
// is copied structurally; expression values are neither retained nor exposed.
bool materialize_sampled_csr_structure(
    const ::cellerator::matrix::compressed *source,
    const sample_plan &plan,
    owned_sampled_csr_structure *out,
    std::string *error = nullptr);

// Converts already-loaded selected CSR rows into Cellerator's owned structural
// representation. This performs no storage access and retains no values.
bool materialize_selected_csr_structure(
    const selected_csr_structure_view &source,
    const sample_plan &plan,
    owned_sampled_csr_structure *out,
    std::string *error = nullptr);

} // namespace cellerator::compute::sampling
