#pragma once

#include <cstddef>
#include <cstdint>

#include "../matrix.cuh"

template<int Bits>
struct mulcsr_traits;

template<>
struct mulcsr_traits<8> {
    enum {
        bits = 8,
        codes_per_byte = 1,
        code_mask = 0xff
    };

    __host__ __device__ __forceinline__ static int row_bytes(int row_nnz) {
        return row_nnz;
    }

    __host__ __device__ __forceinline__ static unsigned int unpack(const unsigned char* packed, int local_index) {
        return static_cast<unsigned int>(packed[local_index]);
    }
};

template<>
struct mulcsr_traits<4> {
    enum {
        bits = 4,
        codes_per_byte = 2,
        code_mask = 0x0f
    };

    __host__ __device__ __forceinline__ static int row_bytes(int row_nnz) {
        return (row_nnz + 1) >> 1;
    }

    __host__ __device__ __forceinline__ static unsigned int unpack(const unsigned char* packed, int local_index) {
        int shift;

        shift = (local_index & 1) << 2;
        return (static_cast<unsigned int>(packed[local_index >> 1]) >> shift) & 0x0fu;
    }
};

template<>
struct mulcsr_traits<2> {
    enum {
        bits = 2,
        codes_per_byte = 4,
        code_mask = 0x03
    };

    __host__ __device__ __forceinline__ static int row_bytes(int row_nnz) {
        return (row_nnz + 3) >> 2;
    }

    __host__ __device__ __forceinline__ static unsigned int unpack(const unsigned char* packed, int local_index) {
        int shift;

        shift = (local_index & 3) << 1;
        return (static_cast<unsigned int>(packed[local_index >> 2]) >> shift) & 0x03u;
    }
};

template<typename Real>
__host__ __device__ __forceinline__ float mulcsr_to_float(Real value) {
    return static_cast<float>(value);
}

template<>
__host__ __device__ __forceinline__ float mulcsr_to_float<__half>(__half value) {
    return __half2float(value);
}

template<typename Real>
__host__ __device__ __forceinline__ Real mulcsr_from_float(float value) {
    return static_cast<Real>(value);
}

template<>
__host__ __device__ __forceinline__ __half mulcsr_from_float<__half>(float value) {
    return __float2half(value);
}

__host__ __device__ __forceinline__ float mulcsr_fast_div(float num, float den) {
    return num / den;
}

__host__ __device__ __forceinline__ float mulcsr_fast_fma(float a, float b, float c) {
    return a * b + c;
}

template<int Bits, typename Real>
struct multiscaled_csr_matrix : public matrix::sparse::csr_base<int, const int*, const int*> {
    static_assert(Bits == 8 || Bits == 4 || Bits == 2, "Bits must be 8, 4, or 2");

    enum {
        bits_per_value = Bits,
        codes_per_byte = mulcsr_traits<Bits>::codes_per_byte,
        code_mask = mulcsr_traits<Bits>::code_mask
    };

    int block_count;
    const int* packed_row_ptr;
    const int* block_row_ptr;
    unsigned char* packed_values;
    const Real* column_scales;
    const Real* row_offsets;
};

struct multiscaled_csr_block {
    int block_index;
    int row_begin;
    int row_end;
    int nnz_begin;
    int nnz_end;
    int packed_begin;
    int packed_end;
};

template<int Bits, typename Real>
__host__ __device__ __forceinline__ multiscaled_csr_matrix<Bits, Real> mulcsr_make_matrix(
    int rows,
    int cols,
    int nnz,
    int block_count,
    const int* row_ptr,
    const int* packed_row_ptr,
    const int* col_idx,
    const int* block_row_ptr,
    unsigned char* packed_values,
    const Real* column_scales,
    const Real* row_offsets) {
    multiscaled_csr_matrix<Bits, Real> matrix;

    matrix.rows = rows;
    matrix.cols = cols;
    matrix.nnz = nnz;
    matrix.format = matrix::format_csr;
    matrix.block_count = block_count;
    matrix.rowPtr = row_ptr;
    matrix.packed_row_ptr = packed_row_ptr;
    matrix.colIdx = col_idx;
    matrix.block_row_ptr = block_row_ptr;
    matrix.packed_values = packed_values;
    matrix.column_scales = column_scales;
    matrix.row_offsets = row_offsets;
    return matrix;
}

template<int Bits>
__host__ __device__ __forceinline__ int mulcsr_row_packed_nbytes(int row_nnz) {
    return mulcsr_traits<Bits>::row_bytes(row_nnz);
}

template<int Bits>
int mulcsr_build_packed_row_ptr(const int* row_ptr, int rows, int* packed_row_ptr) {
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
    packed_row_ptr[row + 1] = packed_row_ptr[row] + mulcsr_row_packed_nbytes<Bits>(row_nnz);
    ++row;
    goto next_row;
}

__host__ __device__ __forceinline__ int mulcsr_block_count_for_rows(int rows, int rows_per_block) {
    if (rows <= 0 || rows_per_block <= 0) {
        return 0;
    }
    return (rows + rows_per_block - 1) / rows_per_block;
}

int mulcsr_build_uniform_block_row_ptr(int rows, int rows_per_block, int* block_row_ptr) {
    int block_count;
    int block;
    int row;

    if (block_row_ptr == nullptr || rows < 0 || rows_per_block <= 0) {
        return -1;
    }

    block_count = mulcsr_block_count_for_rows(rows, rows_per_block);
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
__host__ __device__ __forceinline__ unsigned int mulcsr_quantize_code(Real value, Real column_scale, Real row_offset) {
    float scale;
    float q;
    int rounded;

    scale = mulcsr_to_float(column_scale);
    if (scale == 0.0f) {
        return 0u;
    }

    q = mulcsr_fast_div(mulcsr_to_float(value) - mulcsr_to_float(row_offset), scale);
    rounded = (q >= 0.0f) ? static_cast<int>(q + 0.5f) : static_cast<int>(q - 0.5f);
    if (rounded < 0) {
        return 0u;
    }
    if (rounded > mulcsr_traits<Bits>::code_mask) {
        rounded = mulcsr_traits<Bits>::code_mask;
    }
    return static_cast<unsigned int>(rounded);
}

template<int Bits, typename Real>
__host__ __device__ __forceinline__ Real mulcsr_dequantize_code(unsigned int code, Real column_scale, Real row_offset) {
    return mulcsr_from_float<Real>(
        mulcsr_fast_fma(static_cast<float>(code), mulcsr_to_float(column_scale), mulcsr_to_float(row_offset)));
}

template<int Bits, typename Real>
__host__ __device__ __forceinline__ multiscaled_csr_block mulcsr_get_block(
    const multiscaled_csr_matrix<Bits, Real>* matrix,
    int block_index) {
    multiscaled_csr_block block;

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
        block.row_begin = matrix->block_row_ptr[block_index];
        block.row_end = matrix->block_row_ptr[block_index + 1];
    }

    block.nnz_begin = matrix->rowPtr[block.row_begin];
    block.nnz_end = matrix->rowPtr[block.row_end];
    block.packed_begin = matrix->packed_row_ptr[block.row_begin];
    block.packed_end = matrix->packed_row_ptr[block.row_end];
    return block;
}

template<int Bits, typename Real>
__host__ __device__ __forceinline__ int mulcsr_find_in_row(const multiscaled_csr_matrix<Bits, Real>* matrix, int row, int column) {
    int begin;
    int end;
    int cursor;

    if (matrix == nullptr || row < 0 || row >= matrix->rows) {
        return -1;
    }

    begin = matrix->rowPtr[row];
    end = matrix->rowPtr[row + 1];
    cursor = begin;

scan_next:
    if (cursor >= end) {
        return -1;
    }
    if (matrix->colIdx[cursor] == column) {
        return cursor;
    }
    ++cursor;
    goto scan_next;
}

template<int Bits, typename Real>
__host__ __device__ __forceinline__ Real mulcsr_get_value(const multiscaled_csr_matrix<Bits, Real>* matrix, int row, int column) {
    int index;
    int local_index;
    int row_nnz_begin;
    const unsigned char* row_packed_values;
    unsigned int code;

    if (matrix == nullptr || row < 0 || row >= matrix->rows || column < 0 || column >= matrix->cols) {
        return mulcsr_from_float<Real>(0.0f);
    }

    index = mulcsr_find_in_row(matrix, row, column);
    if (index < 0) {
        return mulcsr_from_float<Real>(0.0f);
    }

    row_nnz_begin = matrix->rowPtr[row];
    local_index = index - row_nnz_begin;
    row_packed_values = matrix->packed_values + matrix->packed_row_ptr[row];
    code = mulcsr_traits<Bits>::unpack(row_packed_values, local_index);
    return mulcsr_dequantize_code<Bits>(
        code,
        matrix->column_scales[column],
        matrix->row_offsets[row]);
}

template<int Bits, typename Real>
__host__ __device__ __forceinline__ void mulcsr_pack_row_values(
    const multiscaled_csr_matrix<Bits, Real>* matrix,
    int row,
    int value_base,
    const Real* values_by_nnz) {
    int index;
    int end;
    unsigned char* out;
    Real row_offset;

    index = matrix->rowPtr[row];
    end = matrix->rowPtr[row + 1];
    out = matrix->packed_values + matrix->packed_row_ptr[row];
    row_offset = matrix->row_offsets[row];

    if constexpr (Bits == 8) {
pack8_next:
        if (index >= end) {
            return;
        }
        *out = static_cast<unsigned char>(mulcsr_quantize_code<Bits>(
            values_by_nnz[index - value_base],
            matrix->column_scales[matrix->colIdx[index]],
            row_offset));
        ++index;
        ++out;
        goto pack8_next;
    }

    if constexpr (Bits == 4) {
        unsigned int q0;
        unsigned int q1;

pack4_pair:
        if (index + 1 < end) {
            q0 = mulcsr_quantize_code<Bits>(
                values_by_nnz[index - value_base],
                matrix->column_scales[matrix->colIdx[index]],
                row_offset);
            q1 = mulcsr_quantize_code<Bits>(
                values_by_nnz[index + 1 - value_base],
                matrix->column_scales[matrix->colIdx[index + 1]],
                row_offset);
            *out = static_cast<unsigned char>((q0 & 0x0fu) | ((q1 & 0x0fu) << 4));
            index += 2;
            ++out;
            goto pack4_pair;
        }

        if (index < end) {
            q0 = mulcsr_quantize_code<Bits>(
                values_by_nnz[index - value_base],
                matrix->column_scales[matrix->colIdx[index]],
                row_offset);
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
            q0 = mulcsr_quantize_code<Bits>(
                values_by_nnz[index - value_base],
                matrix->column_scales[matrix->colIdx[index]],
                row_offset);
            q1 = mulcsr_quantize_code<Bits>(
                values_by_nnz[index + 1 - value_base],
                matrix->column_scales[matrix->colIdx[index + 1]],
                row_offset);
            q2 = mulcsr_quantize_code<Bits>(
                values_by_nnz[index + 2 - value_base],
                matrix->column_scales[matrix->colIdx[index + 2]],
                row_offset);
            q3 = mulcsr_quantize_code<Bits>(
                values_by_nnz[index + 3 - value_base],
                matrix->column_scales[matrix->colIdx[index + 3]],
                row_offset);
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
                q0 = mulcsr_quantize_code<Bits>(
                    values_by_nnz[index - value_base],
                    matrix->column_scales[matrix->colIdx[index]],
                    row_offset);
                packed_byte |= (q0 & 0x03u);
                ++index;
            }
            if (index < end) {
                q1 = mulcsr_quantize_code<Bits>(
                    values_by_nnz[index - value_base],
                    matrix->column_scales[matrix->colIdx[index]],
                    row_offset);
                packed_byte |= ((q1 & 0x03u) << 2);
                ++index;
            }
            if (index < end) {
                q2 = mulcsr_quantize_code<Bits>(
                    values_by_nnz[index - value_base],
                    matrix->column_scales[matrix->colIdx[index]],
                    row_offset);
                packed_byte |= ((q2 & 0x03u) << 4);
                ++index;
            }
            if (index < end) {
                q3 = mulcsr_quantize_code<Bits>(
                    values_by_nnz[index - value_base],
                    matrix->column_scales[matrix->colIdx[index]],
                    row_offset);
                packed_byte |= ((q3 & 0x03u) << 6);
            }
            *out = static_cast<unsigned char>(packed_byte);
        }
    }
}

template<int Bits, typename Real>
__host__ __device__ __forceinline__ void mulcsr_unpack_row_values(
    const multiscaled_csr_matrix<Bits, Real>* matrix,
    int row,
    int value_base,
    Real* values_by_nnz) {
    int index;
    int end;
    const unsigned char* in;
    Real row_offset;

    index = matrix->rowPtr[row];
    end = matrix->rowPtr[row + 1];
    in = matrix->packed_values + matrix->packed_row_ptr[row];
    row_offset = matrix->row_offsets[row];

    if constexpr (Bits == 8) {
unpack8_next:
        if (index >= end) {
            return;
        }
        values_by_nnz[index - value_base] = mulcsr_dequantize_code<Bits>(
            static_cast<unsigned int>(*in),
            matrix->column_scales[matrix->colIdx[index]],
            row_offset);
        ++index;
        ++in;
        goto unpack8_next;
    }

    if constexpr (Bits == 4) {
        unsigned int packed_byte;

unpack4_pair:
        if (index + 1 < end) {
            packed_byte = static_cast<unsigned int>(*in);
            values_by_nnz[index - value_base] = mulcsr_dequantize_code<Bits>(
                packed_byte & 0x0fu,
                matrix->column_scales[matrix->colIdx[index]],
                row_offset);
            values_by_nnz[index + 1 - value_base] = mulcsr_dequantize_code<Bits>(
                (packed_byte >> 4) & 0x0fu,
                matrix->column_scales[matrix->colIdx[index + 1]],
                row_offset);
            index += 2;
            ++in;
            goto unpack4_pair;
        }

        if (index < end) {
            packed_byte = static_cast<unsigned int>(*in);
            values_by_nnz[index - value_base] = mulcsr_dequantize_code<Bits>(
                packed_byte & 0x0fu,
                matrix->column_scales[matrix->colIdx[index]],
                row_offset);
        }
        return;
    }

    if constexpr (Bits == 2) {
        unsigned int packed_byte;

unpack2_quad:
        if (index + 3 < end) {
            packed_byte = static_cast<unsigned int>(*in);
            values_by_nnz[index - value_base] = mulcsr_dequantize_code<Bits>(
                packed_byte & 0x03u,
                matrix->column_scales[matrix->colIdx[index]],
                row_offset);
            values_by_nnz[index + 1 - value_base] = mulcsr_dequantize_code<Bits>(
                (packed_byte >> 2) & 0x03u,
                matrix->column_scales[matrix->colIdx[index + 1]],
                row_offset);
            values_by_nnz[index + 2 - value_base] = mulcsr_dequantize_code<Bits>(
                (packed_byte >> 4) & 0x03u,
                matrix->column_scales[matrix->colIdx[index + 2]],
                row_offset);
            values_by_nnz[index + 3 - value_base] = mulcsr_dequantize_code<Bits>(
                (packed_byte >> 6) & 0x03u,
                matrix->column_scales[matrix->colIdx[index + 3]],
                row_offset);
            index += 4;
            ++in;
            goto unpack2_quad;
        }

        if (index < end) {
            packed_byte = static_cast<unsigned int>(*in);
            if (index < end) {
                values_by_nnz[index - value_base] = mulcsr_dequantize_code<Bits>(
                    packed_byte & 0x03u,
                    matrix->column_scales[matrix->colIdx[index]],
                    row_offset);
                ++index;
            }
            if (index < end) {
                values_by_nnz[index - value_base] = mulcsr_dequantize_code<Bits>(
                    (packed_byte >> 2) & 0x03u,
                    matrix->column_scales[matrix->colIdx[index]],
                    row_offset);
                ++index;
            }
            if (index < end) {
                values_by_nnz[index - value_base] = mulcsr_dequantize_code<Bits>(
                    (packed_byte >> 4) & 0x03u,
                    matrix->column_scales[matrix->colIdx[index]],
                    row_offset);
                ++index;
            }
            if (index < end) {
                values_by_nnz[index - value_base] = mulcsr_dequantize_code<Bits>(
                    (packed_byte >> 6) & 0x03u,
                    matrix->column_scales[matrix->colIdx[index]],
                    row_offset);
            }
        }
    }
}

template<int Bits, typename Real>
int mulcsr_pack_nnz_values(const multiscaled_csr_matrix<Bits, Real>* matrix, const Real* values_by_nnz) {
    int row;

    if (matrix == nullptr || matrix->packed_values == nullptr || values_by_nnz == nullptr) {
        return -1;
    }

    row = 0;

pack_row_loop:
    if (row >= matrix->rows) {
        return 0;
    }

    mulcsr_pack_row_values(matrix, row, 0, values_by_nnz);
    ++row;
    goto pack_row_loop;
}

template<int Bits, typename Real>
int mulcsr_unpack_nnz_values(const multiscaled_csr_matrix<Bits, Real>* matrix, Real* values_by_nnz) {
    int row;

    if (matrix == nullptr || values_by_nnz == nullptr) {
        return -1;
    }

    row = 0;

unpack_row_loop:
    if (row >= matrix->rows) {
        return 0;
    }

    mulcsr_unpack_row_values(matrix, row, 0, values_by_nnz);
    ++row;
    goto unpack_row_loop;
}

template<int Bits, typename Real>
int mulcsr_pack_block_nnz_values(
    const multiscaled_csr_matrix<Bits, Real>* matrix,
    multiscaled_csr_block block,
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

    mulcsr_pack_row_values(matrix, row, block.nnz_begin, values_by_nnz_block);
    ++row;
    goto pack_block_row_loop;
}

template<int Bits, typename Real>
int mulcsr_unpack_block_nnz_values(
    const multiscaled_csr_matrix<Bits, Real>* matrix,
    multiscaled_csr_block block,
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

    mulcsr_unpack_row_values(matrix, row, block.nnz_begin, values_by_nnz_block);
    ++row;
    goto unpack_block_row_loop;
}

template<int Bits, typename Real>
__global__ __launch_bounds__(128, 4) void mulcsr_quantize_block_kernel(
    multiscaled_csr_matrix<Bits, Real> matrix,
    multiscaled_csr_block block,
    const Real* __restrict__ values_by_nnz_block) {
    int row;

    row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + block.row_begin;
    if (row >= block.row_end) {
        return;
    }

    mulcsr_pack_row_values(&matrix, row, block.nnz_begin, values_by_nnz_block);
}

template<int Bits, typename Real>
__global__ __launch_bounds__(128, 4) void mulcsr_dequantize_block_kernel(
    multiscaled_csr_matrix<Bits, Real> matrix,
    multiscaled_csr_block block,
    Real* __restrict__ values_by_nnz_block) {
    int row;

    row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + block.row_begin;
    if (row >= block.row_end) {
        return;
    }

    mulcsr_unpack_row_values(&matrix, row, block.nnz_begin, values_by_nnz_block);
}

template<int Bits, typename Real>
cudaError_t mulcsr_launch_quantize_block(
    const multiscaled_csr_matrix<Bits, Real>* matrix,
    int block_index,
    const Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    multiscaled_csr_block block;
    int rows_in_block;
    int threads;
    int blocks;

    if (matrix == nullptr || values_by_nnz_block == nullptr) {
        return cudaErrorInvalidValue;
    }

    block = mulcsr_get_block(matrix, block_index);
    rows_in_block = block.row_end - block.row_begin;
    if (rows_in_block <= 0) {
        return cudaErrorInvalidValue;
    }

    threads = 128;
    blocks = (rows_in_block + threads - 1) >> 7;
    mulcsr_quantize_block_kernel<Bits><<<blocks, threads, 0, stream>>>(*matrix, block, values_by_nnz_block);
    return cudaGetLastError();
}

template<int Bits, typename Real>
cudaError_t mulcsr_launch_dequantize_block(
    const multiscaled_csr_matrix<Bits, Real>* matrix,
    int block_index,
    Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    multiscaled_csr_block block;
    int rows_in_block;
    int threads;
    int blocks;

    if (matrix == nullptr || values_by_nnz_block == nullptr) {
        return cudaErrorInvalidValue;
    }

    block = mulcsr_get_block(matrix, block_index);
    rows_in_block = block.row_end - block.row_begin;
    if (rows_in_block <= 0) {
        return cudaErrorInvalidValue;
    }

    threads = 128;
    blocks = (rows_in_block + threads - 1) >> 7;
    mulcsr_dequantize_block_kernel<Bits><<<blocks, threads, 0, stream>>>(*matrix, block, values_by_nnz_block);
    return cudaGetLastError();
}

template<int Bits, typename Real>
cudaError_t mulcsr_launch_quantize_block_v100(
    const multiscaled_csr_matrix<Bits, Real>* matrix,
    int block_index,
    const Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    cudaError_t status;

    status = cudaFuncSetCacheConfig(mulcsr_quantize_block_kernel<Bits, Real>, cudaFuncCachePreferL1);
    if (status != cudaSuccess) {
        return status;
    }
    return mulcsr_launch_quantize_block(matrix, block_index, values_by_nnz_block, stream);
}

template<int Bits, typename Real>
cudaError_t mulcsr_launch_dequantize_block_v100(
    const multiscaled_csr_matrix<Bits, Real>* matrix,
    int block_index,
    Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    cudaError_t status;

    status = cudaFuncSetCacheConfig(mulcsr_dequantize_block_kernel<Bits, Real>, cudaFuncCachePreferL1);
    if (status != cudaSuccess) {
        return status;
    }
    return mulcsr_launch_dequantize_block(matrix, block_index, values_by_nnz_block, stream);
}
