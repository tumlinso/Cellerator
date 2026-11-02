#pragma once

#include <cstddef>
#include <cstdint>

namespace cellerator::tensor::quantized {

namespace blocked_ell {

enum : std::uint32_t {
    decode_policy_unknown = 0u,
    decode_policy_per_gene_affine = 1u,
    decode_policy_column_scale_row_offset = 2u
};

inline constexpr std::uint32_t invalid_block_col = 0xffffffffu;

} // namespace blocked_ell

// Borrowed view of a block-ELL matrix whose values are packed independently
// per row. Bits is a compile-time storage policy; Metadata describes how codes
// reconstruct values and remains borrowed alongside the payload buffers.
template<int Bits, typename Real, typename Metadata>
struct alignas(16) BlockedEllView {
    static_assert(Bits == 1 || Bits == 2 || Bits == 4 || Bits == 8,
                  "Bits must be 1, 2, 4, or 8");

    using real_type = Real;
    using metadata_type = Metadata;

    int rows;
    int cols;
    int nnz;
    int block_size;
    int ell_cols;
    int row_stride_bytes;
    const std::uint32_t* block_col_idx;
    const unsigned char* packed_values;
    Metadata metadata;
};

// Runtime-erased encoding view for APIs that dispatch on bit width and affine
// policy. It borrows exactly the same row-packed payload as BlockedEllView;
// dispatch code binds this descriptor to a compile-time Bits/Metadata view.
struct RuntimeBlockedEllView {
    std::uint32_t rows = 0u;
    std::uint32_t cols = 0u;
    std::uint32_t nnz = 0u;
    std::uint32_t block_size = 0u;
    std::uint32_t ell_cols = 0u;
    std::uint32_t row_stride_bytes = 0u;
    std::uint32_t bits = 0u;
    std::uint32_t decode_policy = blocked_ell::decode_policy_unknown;
    const std::uint32_t* block_col_idx = nullptr;
    const std::uint8_t* packed_values = nullptr;
    const float* column_scales = nullptr;
    const float* column_offsets = nullptr;
    const float* row_offsets = nullptr;
};

template<int Bits, typename Real, typename Metadata>
inline BlockedEllView<Bits, Real, Metadata> make_blocked_ell_view(
    int rows,
    int cols,
    int nnz,
    int block_size,
    int ell_cols,
    int row_stride_bytes,
    const std::uint32_t* block_col_idx,
    const unsigned char* packed_values,
    Metadata metadata) {
    return BlockedEllView<Bits, Real, Metadata>{
        rows,
        cols,
        nnz,
        block_size,
        ell_cols,
        row_stride_bytes,
        block_col_idx,
        packed_values,
        metadata};
}

template<int Bits, typename Metadata>
inline BlockedEllView<Bits, typename Metadata::real_type, Metadata> make_blocked_ell_view(
    int rows,
    int cols,
    int nnz,
    int block_size,
    int ell_cols,
    int row_stride_bytes,
    const std::uint32_t* block_col_idx,
    const unsigned char* packed_values,
    Metadata metadata) {
    using real_type = typename Metadata::real_type;
    return make_blocked_ell_view<Bits, real_type, Metadata>(
        rows,
        cols,
        nnz,
        block_size,
        ell_cols,
        row_stride_bytes,
        block_col_idx,
        packed_values,
        metadata);
}

} // namespace cellerator::tensor::quantized
