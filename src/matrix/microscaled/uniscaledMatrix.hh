#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(__CUDACC__)
#define USCSR_HD __host__ __device__ __forceinline__
#define USCSR_RESTRICT __restrict__
#define USCSR_LAUNCH_BOUNDS __launch_bounds__(128, 4)
#else
#define USCSR_HD inline
#define USCSR_RESTRICT
#define USCSR_LAUNCH_BOUNDS
#endif

template<int Bits>
struct uscsr_traits;

template<>
struct uscsr_traits<8> {
    enum {
        bits = 8,
        codes_per_byte = 1,
        code_mask = 0xff
    };

    USCSR_HD static int row_bytes(int row_nnz) {
        return row_nnz;
    }

    USCSR_HD static unsigned int unpack(const unsigned char* packed, int local_index) {
        return static_cast<unsigned int>(packed[local_index]);
    }
};

template<>
struct uscsr_traits<4> {
    enum {
        bits = 4,
        codes_per_byte = 2,
        code_mask = 0x0f
    };

    USCSR_HD static int row_bytes(int row_nnz) {
        return (row_nnz + 1) >> 1;
    }

    USCSR_HD static unsigned int unpack(const unsigned char* packed, int local_index) {
        int shift;

        shift = (local_index & 1) << 2;
        return (static_cast<unsigned int>(packed[local_index >> 1]) >> shift) & 0x0fu;
    }
};

template<>
struct uscsr_traits<2> {
    enum {
        bits = 2,
        codes_per_byte = 4,
        code_mask = 0x03
    };

    USCSR_HD static int row_bytes(int row_nnz) {
        return (row_nnz + 3) >> 2;
    }

    USCSR_HD static unsigned int unpack(const unsigned char* packed, int local_index) {
        int shift;

        shift = (local_index & 3) << 1;
        return (static_cast<unsigned int>(packed[local_index >> 2]) >> shift) & 0x03u;
    }
};

template<typename Real>
USCSR_HD float uscsr_to_float(Real value) {
    return static_cast<float>(value);
}

template<>
USCSR_HD float uscsr_to_float<__half>(__half value) {
    return __half2float(value);
}

template<typename Real>
USCSR_HD Real uscsr_from_float(float value) {
    return static_cast<Real>(value);
}

template<>
USCSR_HD __half uscsr_from_float<__half>(float value) {
    return __float2half(value);
}

template<typename T>
USCSR_HD T uscsr_ldg(const T* ptr) {
#if defined(__CUDA_ARCH__)
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

USCSR_HD float uscsr_fast_div(float num, float den) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 700)
    return __fdividef(num, den);
#else
    return num / den;
#endif
}

USCSR_HD float uscsr_fast_fma(float a, float b, float c) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    return __fmaf_rn(a, b, c);
#else
    return a * b + c;
#endif
}

template<int Bits, typename Real>
struct uniscaled_csr_matrix {
    static_assert(Bits == 8 || Bits == 4 || Bits == 2, "Bits must be 8, 4, or 2");

    enum {
        bits_per_value = Bits,
        codes_per_byte = uscsr_traits<Bits>::codes_per_byte,
        code_mask = uscsr_traits<Bits>::code_mask
    };

    int rows;
    int cols;
    int nnz;
    int block_count;
    const int* row_ptr;
    const int* packed_row_ptr;
    const int* col_idx;
    const int* block_row_ptr;
    unsigned char* packed_values;
    const Real* gene_scales;
};

struct uniscaled_csr_block {
    int block_index;
    int row_begin;
    int row_end;
    int nnz_begin;
    int nnz_end;
    int packed_begin;
    int packed_end;
};

template<int Bits, typename Real>
USCSR_HD uniscaled_csr_matrix<Bits, Real> uscsr_make_matrix(
    int rows,
    int cols,
    int nnz,
    int block_count,
    const int* row_ptr,
    const int* packed_row_ptr,
    const int* col_idx,
    const int* block_row_ptr,
    unsigned char* packed_values,
    const Real* gene_scales) {
    uniscaled_csr_matrix<Bits, Real> matrix;

    matrix.rows = rows;
    matrix.cols = cols;
    matrix.nnz = nnz;
    matrix.block_count = block_count;
    matrix.row_ptr = row_ptr;
    matrix.packed_row_ptr = packed_row_ptr;
    matrix.col_idx = col_idx;
    matrix.block_row_ptr = block_row_ptr;
    matrix.packed_values = packed_values;
    matrix.gene_scales = gene_scales;
    return matrix;
}

template<int Bits>
USCSR_HD int uscsr_row_packed_nbytes(int row_nnz) {
    return uscsr_traits<Bits>::row_bytes(row_nnz);
}

template<int Bits>
int uscsr_build_packed_row_ptr(const int* row_ptr, int rows, int* packed_row_ptr) {
    int row;
    int row_nnz;

    if (row_ptr == nullptr || packed_row_ptr == nullptr || rows < 0) {
        return -1;
    }

    packed_row_ptr[0] = 0;
    row = 0;

next_row:
    if (row >= rows) {
        return packed_row_ptr[rows];
    }

    row_nnz = row_ptr[row + 1] - row_ptr[row];
    packed_row_ptr[row + 1] = packed_row_ptr[row] + uscsr_row_packed_nbytes<Bits>(row_nnz);
    ++row;
    goto next_row;
}

USCSR_HD int uscsr_block_count_for_rows(int rows, int rows_per_block) {
    if (rows <= 0 || rows_per_block <= 0) {
        return 0;
    }
    return (rows + rows_per_block - 1) / rows_per_block;
}

int uscsr_build_uniform_block_row_ptr(int rows, int rows_per_block, int* block_row_ptr) {
    int block_count;
    int block;
    int row;

    if (block_row_ptr == nullptr || rows < 0 || rows_per_block <= 0) {
        return -1;
    }

    block_count = uscsr_block_count_for_rows(rows, rows_per_block);
    block = 0;
    row = 0;

next_block:
    if (block >= block_count) {
        block_row_ptr[block_count] = rows;
        return block_count;
    }

    block_row_ptr[block] = row;
    row += rows_per_block;
    if (row > rows) {
        row = rows;
    }
    ++block;
    goto next_block;
}

template<int Bits, typename Real>
USCSR_HD unsigned int uscsr_quantize_code(Real value, Real gene_scale) {
    float scale;
    float q;
    int rounded;

    scale = uscsr_to_float(gene_scale);
    if (scale == 0.0f) {
        return 0u;
    }

    q = uscsr_fast_div(uscsr_to_float(value), scale);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    rounded = __float2int_rn(q);
#else
    rounded = (q >= 0.0f) ? static_cast<int>(q + 0.5f) : static_cast<int>(q - 0.5f);
#endif
    if (rounded < 0) {
        return 0u;
    }
    if (rounded > uscsr_traits<Bits>::code_mask) {
        rounded = uscsr_traits<Bits>::code_mask;
    }
    return static_cast<unsigned int>(rounded);
}

template<int Bits, typename Real>
USCSR_HD Real uscsr_dequantize_code(unsigned int code, Real gene_scale) {
    return uscsr_from_float<Real>(uscsr_fast_fma(static_cast<float>(code), uscsr_to_float(gene_scale), 0.0f));
}

template<int Bits, typename Real>
USCSR_HD uniscaled_csr_block uscsr_get_block(
    const uniscaled_csr_matrix<Bits, Real>* matrix,
    int block_index) {
    uniscaled_csr_block block;

    block.block_index = block_index;
    block.row_begin = 0;
    block.row_end = 0;
    block.nnz_begin = 0;
    block.nnz_end = 0;
    block.packed_begin = 0;
    block.packed_end = 0;

    if (matrix == nullptr) {
        return block;
    }

    if (matrix->block_row_ptr == nullptr || matrix->block_count <= 0) {
        block.block_index = 0;
        block.row_begin = 0;
        block.row_end = matrix->rows;
    } else {
        if (block_index < 0 || block_index >= matrix->block_count) {
            return block;
        }
        block.row_begin = uscsr_ldg(matrix->block_row_ptr + block_index);
        block.row_end = uscsr_ldg(matrix->block_row_ptr + block_index + 1);
    }

    block.nnz_begin = uscsr_ldg(matrix->row_ptr + block.row_begin);
    block.nnz_end = uscsr_ldg(matrix->row_ptr + block.row_end);
    block.packed_begin = uscsr_ldg(matrix->packed_row_ptr + block.row_begin);
    block.packed_end = uscsr_ldg(matrix->packed_row_ptr + block.row_end);
    return block;
}

template<int Bits, typename Real>
USCSR_HD int uscsr_find_in_row(const uniscaled_csr_matrix<Bits, Real>* matrix, int row, int column) {
    int begin;
    int end;
    int cursor;

    if (matrix == nullptr || row < 0 || row >= matrix->rows) {
        return -1;
    }

    begin = uscsr_ldg(matrix->row_ptr + row);
    end = uscsr_ldg(matrix->row_ptr + row + 1);
    cursor = begin;

scan_next:
    if (cursor >= end) {
        return -1;
    }
    if (uscsr_ldg(matrix->col_idx + cursor) == column) {
        return cursor;
    }
    ++cursor;
    goto scan_next;
}

template<int Bits, typename Real>
USCSR_HD Real uscsr_get_value(const uniscaled_csr_matrix<Bits, Real>* matrix, int row, int column) {
    int index;
    int local_index;
    int row_nnz_begin;
    const unsigned char* row_packed_values;
    unsigned int code;

    if (matrix == nullptr || row < 0 || row >= matrix->rows || column < 0 || column >= matrix->cols) {
        return uscsr_from_float<Real>(0.0f);
    }

    index = uscsr_find_in_row(matrix, row, column);
    if (index < 0) {
        return uscsr_from_float<Real>(0.0f);
    }

    row_nnz_begin = uscsr_ldg(matrix->row_ptr + row);
    local_index = index - row_nnz_begin;
    row_packed_values = matrix->packed_values + uscsr_ldg(matrix->packed_row_ptr + row);
    code = uscsr_traits<Bits>::unpack(row_packed_values, local_index);
    return uscsr_dequantize_code<Bits>(code, uscsr_ldg(matrix->gene_scales + column));
}

template<int Bits, typename Real>
USCSR_HD void uscsr_pack_row_values(
    const uniscaled_csr_matrix<Bits, Real>* matrix,
    int row,
    int value_base,
    const Real* values_by_nnz) {
    int index;
    int end;
    unsigned char* out;

    index = uscsr_ldg(matrix->row_ptr + row);
    end = uscsr_ldg(matrix->row_ptr + row + 1);
    out = matrix->packed_values + uscsr_ldg(matrix->packed_row_ptr + row);

    if constexpr (Bits == 8) {
pack8_next:
        if (index >= end) {
            return;
        }
        *out = static_cast<unsigned char>(uscsr_quantize_code<Bits>(
            uscsr_ldg(values_by_nnz + (index - value_base)),
            uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index))));
        ++index;
        ++out;
        goto pack8_next;
    }

    if constexpr (Bits == 4) {
        unsigned int q0;
        unsigned int q1;

pack4_pair:
        if (index + 1 < end) {
            q0 = uscsr_quantize_code<Bits>(
                uscsr_ldg(values_by_nnz + (index - value_base)),
                uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index)));
            q1 = uscsr_quantize_code<Bits>(
                uscsr_ldg(values_by_nnz + (index + 1 - value_base)),
                uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index + 1)));
            *out = static_cast<unsigned char>((q0 & 0x0fu) | ((q1 & 0x0fu) << 4));
            index += 2;
            ++out;
            goto pack4_pair;
        }

        if (index < end) {
            q0 = uscsr_quantize_code<Bits>(
                uscsr_ldg(values_by_nnz + (index - value_base)),
                uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index)));
            *out = static_cast<unsigned char>(q0 & 0x0fu);
        }
        return;
    }

    if constexpr (Bits == 2) {
        unsigned int q0;
        unsigned int q1;
        unsigned int q2;
        unsigned int q3;

pack2_quad:
        if (index + 3 < end) {
            q0 = uscsr_quantize_code<Bits>(
                uscsr_ldg(values_by_nnz + (index - value_base)),
                uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index)));
            q1 = uscsr_quantize_code<Bits>(
                uscsr_ldg(values_by_nnz + (index + 1 - value_base)),
                uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index + 1)));
            q2 = uscsr_quantize_code<Bits>(
                uscsr_ldg(values_by_nnz + (index + 2 - value_base)),
                uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index + 2)));
            q3 = uscsr_quantize_code<Bits>(
                uscsr_ldg(values_by_nnz + (index + 3 - value_base)),
                uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index + 3)));
            *out = static_cast<unsigned char>(
                (q0 & 0x03u) |
                ((q1 & 0x03u) << 2) |
                ((q2 & 0x03u) << 4) |
                ((q3 & 0x03u) << 6));
            index += 4;
            ++out;
            goto pack2_quad;
        }

        if (index < end) {
            unsigned int packed_byte;

            packed_byte = 0u;
            if (index < end) {
                q0 = uscsr_quantize_code<Bits>(
                    uscsr_ldg(values_by_nnz + (index - value_base)),
                    uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index)));
                packed_byte |= (q0 & 0x03u);
                ++index;
            }
            if (index < end) {
                q1 = uscsr_quantize_code<Bits>(
                    uscsr_ldg(values_by_nnz + (index - value_base)),
                    uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index)));
                packed_byte |= ((q1 & 0x03u) << 2);
                ++index;
            }
            if (index < end) {
                q2 = uscsr_quantize_code<Bits>(
                    uscsr_ldg(values_by_nnz + (index - value_base)),
                    uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index)));
                packed_byte |= ((q2 & 0x03u) << 4);
                ++index;
            }
            if (index < end) {
                q3 = uscsr_quantize_code<Bits>(
                    uscsr_ldg(values_by_nnz + (index - value_base)),
                    uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index)));
                packed_byte |= ((q3 & 0x03u) << 6);
            }
            *out = static_cast<unsigned char>(packed_byte);
        }
    }
}

template<int Bits, typename Real>
USCSR_HD void uscsr_unpack_row_values(
    const uniscaled_csr_matrix<Bits, Real>* matrix,
    int row,
    int value_base,
    Real* values_by_nnz) {
    int index;
    int end;
    const unsigned char* in;

    index = uscsr_ldg(matrix->row_ptr + row);
    end = uscsr_ldg(matrix->row_ptr + row + 1);
    in = matrix->packed_values + uscsr_ldg(matrix->packed_row_ptr + row);

    if constexpr (Bits == 8) {
unpack8_next:
        if (index >= end) {
            return;
        }
        values_by_nnz[index - value_base] = uscsr_dequantize_code<Bits>(
            static_cast<unsigned int>(*in),
            uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index)));
        ++index;
        ++in;
        goto unpack8_next;
    }

    if constexpr (Bits == 4) {
        unsigned int packed_byte;

unpack4_pair:
        if (index + 1 < end) {
            packed_byte = static_cast<unsigned int>(*in);
            values_by_nnz[index - value_base] = uscsr_dequantize_code<Bits>(
                packed_byte & 0x0fu,
                uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index)));
            values_by_nnz[index + 1 - value_base] = uscsr_dequantize_code<Bits>(
                (packed_byte >> 4) & 0x0fu,
                uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index + 1)));
            index += 2;
            ++in;
            goto unpack4_pair;
        }

        if (index < end) {
            packed_byte = static_cast<unsigned int>(*in);
            values_by_nnz[index - value_base] = uscsr_dequantize_code<Bits>(
                packed_byte & 0x0fu,
                uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index)));
        }
        return;
    }

    if constexpr (Bits == 2) {
        unsigned int packed_byte;

unpack2_quad:
        if (index + 3 < end) {
            packed_byte = static_cast<unsigned int>(*in);
            values_by_nnz[index - value_base] = uscsr_dequantize_code<Bits>(
                packed_byte & 0x03u,
                uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index)));
            values_by_nnz[index + 1 - value_base] = uscsr_dequantize_code<Bits>(
                (packed_byte >> 2) & 0x03u,
                uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index + 1)));
            values_by_nnz[index + 2 - value_base] = uscsr_dequantize_code<Bits>(
                (packed_byte >> 4) & 0x03u,
                uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index + 2)));
            values_by_nnz[index + 3 - value_base] = uscsr_dequantize_code<Bits>(
                (packed_byte >> 6) & 0x03u,
                uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index + 3)));
            index += 4;
            ++in;
            goto unpack2_quad;
        }

        if (index < end) {
            packed_byte = static_cast<unsigned int>(*in);
            if (index < end) {
                values_by_nnz[index - value_base] = uscsr_dequantize_code<Bits>(
                    packed_byte & 0x03u,
                    uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index)));
                ++index;
            }
            if (index < end) {
                values_by_nnz[index - value_base] = uscsr_dequantize_code<Bits>(
                    (packed_byte >> 2) & 0x03u,
                    uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index)));
                ++index;
            }
            if (index < end) {
                values_by_nnz[index - value_base] = uscsr_dequantize_code<Bits>(
                    (packed_byte >> 4) & 0x03u,
                    uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index)));
                ++index;
            }
            if (index < end) {
                values_by_nnz[index - value_base] = uscsr_dequantize_code<Bits>(
                    (packed_byte >> 6) & 0x03u,
                    uscsr_ldg(matrix->gene_scales + uscsr_ldg(matrix->col_idx + index)));
            }
        }
    }
}

template<int Bits, typename Real>
int uscsr_pack_nnz_values(const uniscaled_csr_matrix<Bits, Real>* matrix, const Real* values_by_nnz) {
    int row;

    if (matrix == nullptr || matrix->packed_values == nullptr || values_by_nnz == nullptr) {
        return -1;
    }

    row = 0;

pack_row_loop:
    if (row >= matrix->rows) {
        return 0;
    }

    uscsr_pack_row_values(matrix, row, 0, values_by_nnz);
    ++row;
    goto pack_row_loop;
}

template<int Bits, typename Real>
int uscsr_unpack_nnz_values(const uniscaled_csr_matrix<Bits, Real>* matrix, Real* values_by_nnz) {
    int row;

    if (matrix == nullptr || values_by_nnz == nullptr) {
        return -1;
    }

    row = 0;

unpack_row_loop:
    if (row >= matrix->rows) {
        return 0;
    }

    uscsr_unpack_row_values(matrix, row, 0, values_by_nnz);
    ++row;
    goto unpack_row_loop;
}

template<int Bits, typename Real>
int uscsr_pack_block_nnz_values(
    const uniscaled_csr_matrix<Bits, Real>* matrix,
    uniscaled_csr_block block,
    const Real* values_by_nnz_block) {
    int row;

    if (matrix == nullptr || values_by_nnz_block == nullptr) {
        return -1;
    }

    row = block.row_begin;

pack_block_row_loop:
    if (row >= block.row_end) {
        return 0;
    }

    uscsr_pack_row_values(matrix, row, block.nnz_begin, values_by_nnz_block);
    ++row;
    goto pack_block_row_loop;
}

template<int Bits, typename Real>
int uscsr_unpack_block_nnz_values(
    const uniscaled_csr_matrix<Bits, Real>* matrix,
    uniscaled_csr_block block,
    Real* values_by_nnz_block) {
    int row;

    if (matrix == nullptr || values_by_nnz_block == nullptr) {
        return -1;
    }

    row = block.row_begin;

unpack_block_row_loop:
    if (row >= block.row_end) {
        return 0;
    }

    uscsr_unpack_row_values(matrix, row, block.nnz_begin, values_by_nnz_block);
    ++row;
    goto unpack_block_row_loop;
}
