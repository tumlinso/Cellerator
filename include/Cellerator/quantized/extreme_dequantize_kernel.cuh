#pragma once

#include "packing.cuh"
#include "extreme_metadata_access.cuh"

namespace cellerator::quantized::extreme_backend {

template<int Bits, typename Real, typename Metadata>
__device__ __forceinline__ Real dequantize_code_sm70_extreme(
    unsigned int code,
    const Metadata& metadata,
    typename Metadata::row_cache row_cache,
    int column) {
    const auto prepared = prepare_metadata_sm70(metadata, row_cache);
    return dequantize_code_sm70_extreme_prepared<Bits, decltype(prepared), Real>(code, prepared, column);
}

template<int Bits, typename Real, typename Metadata>
__device__ __forceinline__ void unpack_row_values_sm70_extreme(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    int row,
    int value_base,
    Real* values_by_nnz) {
    int index = load_scalar(matrix->rowPtr + row);
    const int end = load_scalar(matrix->rowPtr + row + 1);
    const unsigned char* in = matrix->packed_values + load_scalar(matrix->packed_row_ptr + row);
    const auto row_cache = matrix->metadata.prepare_row(row);
    const auto prepared = prepare_metadata_sm70(matrix->metadata, row_cache);

    if constexpr (Bits == 8) {
        while (index < end && !is_aligned_16(in)) {
            const int column = load_scalar(matrix->colIdx + index);
            values_by_nnz[index - value_base] = dequantize_code_sm70_extreme_prepared<Bits, decltype(prepared), Real>(
                static_cast<unsigned int>(*in),
                prepared,
                column);
            ++index;
            ++in;
        }

        while (index + 15 < end) {
            const uint4 packed = *reinterpret_cast<const uint4*>(in);
            const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};

            #pragma unroll
            for (int group = 0; group < 4; ++group) {
                #pragma unroll
                for (int lane = 0; lane < 4; ++lane) {
                    const int current = index + group * 4 + lane;
                    const unsigned int code = (words[group] >> (lane * 8)) & 0xffu;

                    values_by_nnz[current - value_base] = dequantize_code_sm70_extreme_prepared<Bits, decltype(prepared), Real>(
                        code,
                        prepared,
                        load_scalar(matrix->colIdx + current));
                }
            }

            index += 16;
            in += 16;
        }

        while (index + 3 < end) {
            const int c0 = load_scalar(matrix->colIdx + index);
            const int c1 = load_scalar(matrix->colIdx + index + 1);
            const int c2 = load_scalar(matrix->colIdx + index + 2);
            const int c3 = load_scalar(matrix->colIdx + index + 3);

            values_by_nnz[index - value_base] = dequantize_code_sm70_extreme_prepared<Bits, decltype(prepared), Real>(
                static_cast<unsigned int>(in[0]),
                prepared,
                c0);
            values_by_nnz[index + 1 - value_base] = dequantize_code_sm70_extreme_prepared<Bits, decltype(prepared), Real>(
                static_cast<unsigned int>(in[1]),
                prepared,
                c1);
            values_by_nnz[index + 2 - value_base] = dequantize_code_sm70_extreme_prepared<Bits, decltype(prepared), Real>(
                static_cast<unsigned int>(in[2]),
                prepared,
                c2);
            values_by_nnz[index + 3 - value_base] = dequantize_code_sm70_extreme_prepared<Bits, decltype(prepared), Real>(
                static_cast<unsigned int>(in[3]),
                prepared,
                c3);
            index += 4;
            in += 4;
        }

        while (index < end) {
            const int column = load_scalar(matrix->colIdx + index);
            values_by_nnz[index - value_base] = dequantize_code_sm70_extreme_prepared<Bits, decltype(prepared), Real>(
                static_cast<unsigned int>(*in),
                prepared,
                column);
            ++index;
            ++in;
        }
    } else {
        while (index < end) {
            const unsigned int packed_byte = static_cast<unsigned int>(*in);

            #pragma unroll
            for (int lane = 0; lane < format_traits<Bits>::codes_per_byte; ++lane) {
                const int current = index + lane;

                if (current < end) {
                    const int column = load_scalar(matrix->colIdx + current);
                    const unsigned int code =
                        (packed_byte >> (lane * Bits)) & static_cast<unsigned int>(format_traits<Bits>::code_mask);

                    values_by_nnz[current - value_base] = dequantize_code_sm70_extreme_prepared<Bits, decltype(prepared), Real>(
                        code,
                        prepared,
                        column);
                }
            }

            index += format_traits<Bits>::codes_per_byte;
            ++in;
        }
    }
}

template<int Bits, typename Real, typename Metadata>
__global__ __launch_bounds__(launch_policy::threads, launch_policy::min_blocks_per_sm)
void dequantize_block_kernel(
    csr_matrix<Bits, Real, Metadata> matrix,
    csr_block block,
    Real* __restrict__ values_by_nnz_block) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + block.row_begin;

    if (row >= block.row_end) {
        return;
    }
    unpack_row_values_sm70_extreme(&matrix, row, block.nnz_begin, values_by_nnz_block);
}

} // namespace cellerator::quantized::extreme_backend
