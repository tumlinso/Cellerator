#pragma once

#include "layout.cuh"

namespace cellerator::microscaled::quantized {

template<int Bits, typename Real, typename Metadata>
__host__ __device__ __forceinline__ csr_block get_block(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    int block_index) {
    csr_block block;

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

    block.nnz_begin = load_scalar(matrix->rowPtr + block.row_begin);
    block.nnz_end = load_scalar(matrix->rowPtr + block.row_end);
    block.packed_begin = load_scalar(matrix->packed_row_ptr + block.row_begin);
    block.packed_end = load_scalar(matrix->packed_row_ptr + block.row_end);
    return block;
}

template<int Bits, typename Real, typename Metadata>
__host__ __device__ __forceinline__ int find_in_row(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    int row,
    int column) {
    int cursor = 0;
    int end = 0;

    if (matrix == nullptr || row < 0 || row >= matrix->rows) {
        return -1;
    }

    cursor = load_scalar(matrix->rowPtr + row);
    end = load_scalar(matrix->rowPtr + row + 1);
    while (cursor < end) {
        if (load_scalar(matrix->colIdx + cursor) == column) {
            return cursor;
        }
        ++cursor;
    }
    return -1;
}

template<int Bits, typename Real, typename Metadata>
__host__ __device__ __forceinline__ unsigned int get_code(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    int row,
    int local_index) {
    const int packed_begin = load_scalar(matrix->packed_row_ptr + row);

    return format_traits<Bits>::unpack(matrix->packed_values + packed_begin, local_index);
}

template<int Bits, typename Real, typename Metadata>
__host__ __device__ __forceinline__ Real dequantize_code(
    unsigned int code,
    const Metadata& metadata,
    typename Metadata::row_cache row_cache,
    int column) {
    const float scale = to_float(metadata.scale_for(row_cache, column));
    const float offset = to_float(metadata.offset_for(row_cache, column));

    return from_float<Real>(fast_fma(static_cast<float>(code), scale, offset));
}

template<int Bits, typename Metadata>
__host__ __device__ __forceinline__ typename Metadata::real_type dequantize_code(
    unsigned int code,
    const Metadata& metadata,
    typename Metadata::row_cache row_cache,
    int column) {
    using real_type = typename Metadata::real_type;

    return dequantize_code<Bits, real_type, Metadata>(code, metadata, row_cache, column);
}

template<int Bits, typename Real, typename Metadata>
__host__ __device__ __forceinline__ unsigned int quantize_code(
    Real value,
    const Metadata& metadata,
    typename Metadata::row_cache row_cache,
    int column) {
    const float scale = to_float(metadata.scale_for(row_cache, column));
    const float offset = to_float(metadata.offset_for(row_cache, column));
    const float centered = to_float(value) - offset;
    float q = 0.0f;
    int rounded = 0;

    if (scale == 0.0f) {
        return 0u;
    }

    q = fast_div(centered, scale);
    rounded = q >= 0.0f ? static_cast<int>(q + 0.5f) : static_cast<int>(q - 0.5f);
    if (rounded < 0) {
        return 0u;
    }
    if (rounded > format_traits<Bits>::code_mask) {
        rounded = format_traits<Bits>::code_mask;
    }
    return static_cast<unsigned int>(rounded);
}

template<int Bits, typename Real, typename Metadata>
__host__ __device__ __forceinline__ Real get_value(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    int row,
    int column) {
    int index = 0;
    int local_index = 0;
    const auto row_cache = matrix->metadata.prepare_row(row);

    if (matrix == nullptr || row < 0 || row >= matrix->rows || column < 0 || column >= matrix->cols) {
        return from_float<Real>(0.0f);
    }

    index = find_in_row(matrix, row, column);
    if (index < 0) {
        return from_float<Real>(0.0f);
    }

    local_index = index - load_scalar(matrix->rowPtr + row);
    return dequantize_code<Bits>(get_code(matrix, row, local_index), matrix->metadata, row_cache, column);
}

} // namespace cellerator::microscaled::quantized
