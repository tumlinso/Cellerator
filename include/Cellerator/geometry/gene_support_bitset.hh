#pragma once

#include <Cellerator/compute/sampling_materialization.hh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace cellerator::compute::gene_support {

using support_word_t = ::cellerator::types::u32;

inline constexpr std::size_t cells_per_support_word = 32u;

struct gene_support_layout {
    std::uint64_t sampled_cell_count = 0u;
    std::uint64_t gene_count = 0u;
    std::size_t words_per_gene = 0u;
    std::size_t support_word_count = 0u;
    std::size_t support_bytes = 0u;
    std::size_t detection_count_bytes = 0u;
};

// Pure checked sizing. It does not allocate, so callers can validate the
// 65,536-cell production shape without committing the corresponding memory.
bool calculate_gene_support_layout(std::uint64_t sampled_cell_count,
                                   std::uint64_t gene_count,
                                   gene_support_layout *out,
                                   std::string *error = nullptr);

struct gene_support_bitset_view {
    gene_support_layout layout;
    // Gene-major contiguous storage:
    // gene_support[gene * words_per_gene + sampled_position / 32].
    const support_word_t *gene_support = nullptr;
    const ::cellerator::types::count_value_t *detected_cell_counts = nullptr;
    const std::uint64_t *sampled_position_to_global_row = nullptr;
    const ::cellerator::compute::sampling::sample_provenance *provenance = nullptr;
};

class owned_gene_support_bitsets {
public:
    owned_gene_support_bitsets() = default;
    ~owned_gene_support_bitsets() = default;
    owned_gene_support_bitsets(const owned_gene_support_bitsets &) = delete;
    owned_gene_support_bitsets &operator=(const owned_gene_support_bitsets &) = delete;
    owned_gene_support_bitsets(owned_gene_support_bitsets &&) noexcept = default;
    owned_gene_support_bitsets &operator=(owned_gene_support_bitsets &&) noexcept = default;

    owned_gene_support_bitsets(
        gene_support_layout layout,
        std::unique_ptr<support_word_t[]> gene_support,
        std::unique_ptr< ::cellerator::types::count_value_t[]> detected_cell_counts,
        std::unique_ptr<std::uint64_t[]> sampled_position_to_global_row,
        ::cellerator::compute::sampling::sample_provenance provenance) noexcept;

    gene_support_bitset_view view() const noexcept;
    const ::cellerator::compute::sampling::sample_provenance &sampling_provenance() const noexcept;

private:
    gene_support_layout layout_;
    std::unique_ptr<support_word_t[]> gene_support_;
    std::unique_ptr< ::cellerator::types::count_value_t[]> detected_cell_counts_;
    std::unique_ptr<std::uint64_t[]> sampled_position_to_global_row_;
    ::cellerator::compute::sampling::sample_provenance provenance_;

    friend bool build_gene_support_bitsets_cpu(
        const ::cellerator::compute::sampling::sampled_csr_structure_view &,
        owned_gene_support_bitsets *,
        std::string *);
    friend bool build_gene_support_bitsets_cuda(
        const ::cellerator::compute::sampling::sampled_csr_structure_view &,
        int,
        owned_gene_support_bitsets *,
        std::string *);
};

// Reference implementation. Duplicate columns within one sampled row are
// idempotent and contribute one detected cell to that gene.
bool build_gene_support_bitsets_cpu(
    const ::cellerator::compute::sampling::sampled_csr_structure_view &sampled,
    owned_gene_support_bitsets *out,
    std::string *error = nullptr);

// Uploads only sampled CSR row pointers and column indices. The bitset is
// constructed on the requested CUDA device, copied into the owned result, and
// retains the immutable provenance and sampled-position/global-row mapping.
bool build_gene_support_bitsets_cuda(
    const ::cellerator::compute::sampling::sampled_csr_structure_view &sampled,
    int device,
    owned_gene_support_bitsets *out,
    std::string *error = nullptr);

} // namespace cellerator::compute::gene_support
