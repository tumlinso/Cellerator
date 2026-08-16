#pragma once

#include <Cellerator/compute/sampling.hh>
#include <Cellerator/matrix/compressed.cuh>

#include <cstdint>
#include <memory>
#include <string>

namespace cellerator::compute::sampling {

struct sampled_csr_structure_view {
    std::uint64_t sampled_row_count = 0u;
    std::uint64_t gene_count = 0u;
    std::uint64_t nnz = 0u;
    const ::cellerator::types::ptr_t *row_ptr = nullptr;
    const ::cellerator::types::idx_t *column_indices = nullptr;
    const std::uint64_t *sampled_position_to_global_row = nullptr;
    const sample_provenance *provenance = nullptr;
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
    friend bool materialize_cellshard_sampled_csr_structure(
        const char *,
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

// CellShard owns .csh5 access and partition decoding. This adapter delegates to
// its selected-row CSR export and discards any incidentally decoded values.
bool materialize_cellshard_sampled_csr_structure(
    const char *path,
    const sample_plan &plan,
    owned_sampled_csr_structure *out,
    std::string *error = nullptr);

} // namespace cellerator::compute::sampling
