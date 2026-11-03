#pragma once

#include <cstddef>

namespace cellerator::tensor::quantized {

inline constexpr int storage_format_csr = 1;

// Borrowed view of low-bit values following a conventional CSR row/column
// structure. The row and column indices are not quantized or owned here.
template<int Bits, typename Real, typename Metadata>
struct alignas(16) CsrView {
    static_assert(Bits == 1 || Bits == 2 || Bits == 4 || Bits == 8,
                  "Bits must be 1, 2, 4, or 8");

    using real_type = Real;
    using metadata_type = Metadata;

    static constexpr int bits_per_value = Bits;
    static constexpr int codes_per_byte = 8 / Bits;
    static constexpr unsigned int code_mask = (1u << Bits) - 1u;

    int rows;
    int cols;
    int nnz;
    int format;
    int block_count;
    const int* row_ptr;
    const int* packed_row_ptr;
    const int* col_idx;
    const int* block_row_ptr;
    unsigned char* packed_values;
    Metadata metadata;
};

struct alignas(32) CsrBlock {
    int block_index;
    int row_begin;
    int row_end;
    int nnz_begin;
    int nnz_end;
    int packed_begin;
    int packed_end;
};

template<int Bits, typename Real, typename Metadata>
inline CsrView<Bits, Real, Metadata> make_csr_view(
    int rows,
    int cols,
    int nnz,
    int block_count,
    const int* row_ptr,
    const int* packed_row_ptr,
    const int* col_idx,
    const int* block_row_ptr,
    unsigned char* packed_values,
    Metadata metadata) {
    return CsrView<Bits, Real, Metadata>{
        rows,
        cols,
        nnz,
        storage_format_csr,
        block_count,
        row_ptr,
        packed_row_ptr,
        col_idx,
        block_row_ptr,
        packed_values,
        metadata};
}

template<int Bits, typename Metadata>
inline CsrView<Bits, typename Metadata::real_type, Metadata> make_csr_view(
    int rows,
    int cols,
    int nnz,
    int block_count,
    const int* row_ptr,
    const int* packed_row_ptr,
    const int* col_idx,
    const int* block_row_ptr,
    unsigned char* packed_values,
    Metadata metadata) {
    using real_type = typename Metadata::real_type;
    return make_csr_view<Bits, real_type, Metadata>(
        rows,
        cols,
        nnz,
        block_count,
        row_ptr,
        packed_row_ptr,
        col_idx,
        block_row_ptr,
        packed_values,
        metadata);
}

} // namespace cellerator::tensor::quantized
