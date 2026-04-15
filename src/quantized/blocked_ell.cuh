#pragma once

#include <cstddef>
#include <cstdint>

#include "access.cuh"

namespace cellerator::quantized::blocked_ell {

enum : std::uint32_t {
    decode_policy_unknown = 0u,
    decode_policy_per_gene_affine = 1u,
    decode_policy_column_scale_row_offset = 2u
};

inline constexpr std::uint32_t invalid_block_col = 0xffffffffu;

template<int Bits, typename Real, typename Metadata>
struct alignas(16) matrix {
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
    const std::uint32_t *block_col_idx;
    unsigned char *packed_values;
    Metadata metadata;
};

template<int Bits>
__host__ __device__ __forceinline__ int row_bytes(int ell_cols) {
    return format_traits<Bits>::row_bytes(ell_cols);
}

template<int Bits>
__host__ __device__ __forceinline__ int aligned_row_bytes(
    int ell_cols,
    int alignment = packed_storage_alignment) {
    return format_traits<Bits>::aligned_row_bytes(ell_cols, alignment);
}

template<int Bits, typename Real, typename Metadata>
__host__ __device__ __forceinline__ matrix<Bits, Real, Metadata> make_matrix(
    int rows,
    int cols,
    int nnz,
    int block_size,
    int ell_cols,
    int row_stride_bytes,
    const std::uint32_t *block_col_idx,
    unsigned char *packed_values,
    Metadata metadata) {
    matrix<Bits, Real, Metadata> out;

    out.rows = rows;
    out.cols = cols;
    out.nnz = nnz;
    out.block_size = block_size;
    out.ell_cols = ell_cols;
    out.row_stride_bytes = row_stride_bytes;
    out.block_col_idx = block_col_idx;
    out.packed_values = packed_values;
    out.metadata = metadata;
    return out;
}

template<int Bits, typename Metadata>
__host__ __device__ __forceinline__ matrix<Bits, typename Metadata::real_type, Metadata> make_matrix(
    int rows,
    int cols,
    int nnz,
    int block_size,
    int ell_cols,
    int row_stride_bytes,
    const std::uint32_t *block_col_idx,
    unsigned char *packed_values,
    Metadata metadata) {
    using real_type = typename Metadata::real_type;

    return make_matrix<Bits, real_type, Metadata>(
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

template<int Bits, typename Real, typename Metadata>
__host__ __device__ __forceinline__ int ell_width_blocks(const matrix<Bits, Real, Metadata> *src) {
    if (src == nullptr || src->block_size <= 0 || src->ell_cols <= 0) return 0;
    return src->ell_cols / src->block_size;
}

template<int Bits, typename Real, typename Metadata>
__host__ __device__ __forceinline__ const unsigned char *row_packed_base(
    const matrix<Bits, Real, Metadata> *src,
    int row) {
    return src->packed_values + static_cast<std::size_t>(row) * static_cast<std::size_t>(src->row_stride_bytes);
}

template<int Bits, typename Real, typename Metadata>
__host__ __device__ __forceinline__ unsigned int get_code(
    const matrix<Bits, Real, Metadata> *src,
    int row,
    int slot) {
    return format_traits<Bits>::unpack(row_packed_base(src, row), slot);
}

template<int Bits, typename Real, typename Metadata>
__host__ __device__ __forceinline__ int column_for_slot(
    const matrix<Bits, Real, Metadata> *src,
    int row,
    int slot) {
    if (src == nullptr || row < 0 || row >= src->rows || slot < 0 || slot >= src->ell_cols || src->block_size <= 0) {
        return -1;
    }

    const int row_block = row / src->block_size;
    const int slot_block = slot / src->block_size;
    const int slot_offset = slot % src->block_size;
    const int width = ell_width_blocks(src);
    const std::uint32_t block_col = load_scalar(
        src->block_col_idx + static_cast<std::size_t>(row_block) * static_cast<std::size_t>(width) + slot_block);

    if (block_col == invalid_block_col) return -1;
    return static_cast<int>(block_col) * src->block_size + slot_offset;
}

template<int Bits, typename Real, typename Metadata>
__host__ __device__ __forceinline__ Real get_value(
    const matrix<Bits, Real, Metadata> *src,
    int row,
    int slot,
    typename Metadata::row_cache row_cache) {
    const int column = column_for_slot(src, row, slot);

    if (column < 0 || column >= src->cols) return from_float<Real>(0.0f);
    return dequantize_code<Bits>(get_code(src, row, slot), src->metadata, row_cache, column);
}

template<int Bits, typename Real, typename Metadata>
__host__ __device__ __forceinline__ Real get_value(
    const matrix<Bits, Real, Metadata> *src,
    int row,
    int slot) {
    const auto row_cache = src->metadata.prepare_row(row);

    return get_value(src, row, slot, row_cache);
}

template<int Bits, typename Real, typename Metadata>
inline int pack_row_major_values(
    matrix<Bits, Real, Metadata> *dst,
    const Real *values_by_slot) {
    if (dst == nullptr || values_by_slot == nullptr || dst->packed_values == nullptr || dst->rows < 0
        || dst->ell_cols < 0 || dst->row_stride_bytes < row_bytes<Bits>(dst->ell_cols)) {
        return -1;
    }

    for (int row = 0; row < dst->rows; ++row) {
        unsigned char *out = dst->packed_values + static_cast<std::size_t>(row) * static_cast<std::size_t>(dst->row_stride_bytes);
        const auto row_cache = dst->metadata.prepare_row(row);

        if constexpr (Bits == 8) {
            for (int slot = 0; slot < dst->ell_cols; ++slot) {
                const int column = column_for_slot(dst, row, slot);
                const int safe_column = column >= 0 ? column : 0;
                out[slot] = static_cast<unsigned char>(quantize_code<Bits>(
                    values_by_slot[static_cast<std::size_t>(row) * static_cast<std::size_t>(dst->ell_cols) + slot],
                    dst->metadata,
                    row_cache,
                    safe_column));
            }
            for (int pad = dst->ell_cols; pad < dst->row_stride_bytes; ++pad) out[pad] = 0u;
        } else {
            int slot = 0;
            int out_byte = 0;

            while (slot < dst->ell_cols) {
                unsigned int packed = 0u;
                #pragma unroll
                for (int lane = 0; lane < format_traits<Bits>::codes_per_byte; ++lane) {
                    const int current = slot + lane;
                    if (current < dst->ell_cols) {
                        const int column = column_for_slot(dst, row, current);
                        const int safe_column = column >= 0 ? column : 0;
                        const unsigned int code = quantize_code<Bits>(
                            values_by_slot[static_cast<std::size_t>(row) * static_cast<std::size_t>(dst->ell_cols) + current],
                            dst->metadata,
                            row_cache,
                            safe_column);
                        packed |= (code & static_cast<unsigned int>(format_traits<Bits>::code_mask)) << (lane * Bits);
                    }
                }
                out[out_byte++] = format_traits<Bits>::pack_byte(packed);
                slot += format_traits<Bits>::codes_per_byte;
            }
            while (out_byte < dst->row_stride_bytes) out[out_byte++] = 0u;
        }
    }
    return 0;
}

template<int Bits, typename Real, typename Metadata>
inline int unpack_row_major_values(
    const matrix<Bits, Real, Metadata> *src,
    Real *values_by_slot) {
    if (src == nullptr || values_by_slot == nullptr || src->packed_values == nullptr || src->rows < 0 || src->ell_cols < 0) {
        return -1;
    }

    for (int row = 0; row < src->rows; ++row) {
        const auto row_cache = src->metadata.prepare_row(row);
        for (int slot = 0; slot < src->ell_cols; ++slot) {
            values_by_slot[static_cast<std::size_t>(row) * static_cast<std::size_t>(src->ell_cols) + slot] =
                get_value(src, row, slot, row_cache);
        }
    }
    return 0;
}

} // namespace cellerator::quantized::blocked_ell
