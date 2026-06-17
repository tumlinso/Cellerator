#pragma once

#include "packing.cuh"
#include "extreme_metadata_access.cuh"

namespace cellerator::quantized::extreme_backend {

template<int Bits, typename Real, typename Metadata>
__device__ __forceinline__ unsigned int quantize_code_sm70_extreme(
    Real value,
    const Metadata& metadata,
    typename Metadata::row_cache row_cache,
    int column) {
    const auto prepared = prepare_metadata_sm70(metadata, row_cache);
    return quantize_code_sm70_extreme_prepared<Bits>(value, prepared, column);
}

template<int Bits, typename Real, typename Metadata>
__device__ __forceinline__ void pack_row_values_sm70_extreme(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    int row,
    int value_base,
    const Real* values_by_nnz) {
    int index = load_scalar(matrix->rowPtr + row);
    const int end = load_scalar(matrix->rowPtr + row + 1);
    unsigned char* out = matrix->packed_values + load_scalar(matrix->packed_row_ptr + row);
    const auto row_cache = matrix->metadata.prepare_row(row);
    const auto prepared = prepare_metadata_sm70(matrix->metadata, row_cache);

    if constexpr (Bits == 8) {
        while (index < end && !is_aligned_16(out)) {
            const int column = load_scalar(matrix->colIdx + index);
            *out = static_cast<unsigned char>(quantize_code_sm70_extreme_prepared<Bits>(
                values_by_nnz[index - value_base],
                prepared,
                column));
            ++index;
            ++out;
        }

        while (index + 15 < end) {
            unsigned int w0 = 0u;
            unsigned int w1 = 0u;
            unsigned int w2 = 0u;
            unsigned int w3 = 0u;

            #pragma unroll
            for (int lane = 0; lane < 4; ++lane) {
                const int i0 = index + lane;
                const int i1 = index + 4 + lane;
                const int i2 = index + 8 + lane;
                const int i3 = index + 12 + lane;

                w0 |= quantize_code_sm70_extreme_prepared<Bits>(
                    values_by_nnz[i0 - value_base],
                    prepared,
                    load_scalar(matrix->colIdx + i0)) << (lane * 8);
                w1 |= quantize_code_sm70_extreme_prepared<Bits>(
                    values_by_nnz[i1 - value_base],
                    prepared,
                    load_scalar(matrix->colIdx + i1)) << (lane * 8);
                w2 |= quantize_code_sm70_extreme_prepared<Bits>(
                    values_by_nnz[i2 - value_base],
                    prepared,
                    load_scalar(matrix->colIdx + i2)) << (lane * 8);
                w3 |= quantize_code_sm70_extreme_prepared<Bits>(
                    values_by_nnz[i3 - value_base],
                    prepared,
                    load_scalar(matrix->colIdx + i3)) << (lane * 8);
            }

            *reinterpret_cast<uint4*>(out) = make_uint4(w0, w1, w2, w3);
            index += 16;
            out += 16;
        }

        while (index + 3 < end) {
            const int c0 = load_scalar(matrix->colIdx + index);
            const int c1 = load_scalar(matrix->colIdx + index + 1);
            const int c2 = load_scalar(matrix->colIdx + index + 2);
            const int c3 = load_scalar(matrix->colIdx + index + 3);

            out[0] = static_cast<unsigned char>(quantize_code_sm70_extreme_prepared<Bits>(
                values_by_nnz[index - value_base],
                prepared,
                c0));
            out[1] = static_cast<unsigned char>(quantize_code_sm70_extreme_prepared<Bits>(
                values_by_nnz[index + 1 - value_base],
                prepared,
                c1));
            out[2] = static_cast<unsigned char>(quantize_code_sm70_extreme_prepared<Bits>(
                values_by_nnz[index + 2 - value_base],
                prepared,
                c2));
            out[3] = static_cast<unsigned char>(quantize_code_sm70_extreme_prepared<Bits>(
                values_by_nnz[index + 3 - value_base],
                prepared,
                c3));
            index += 4;
            out += 4;
        }

        while (index < end) {
            const int column = load_scalar(matrix->colIdx + index);
            *out = static_cast<unsigned char>(quantize_code_sm70_extreme_prepared<Bits>(
                values_by_nnz[index - value_base],
                prepared,
                column));
            ++index;
            ++out;
        }
    } else {
        while (index < end) {
            unsigned int packed_byte = 0u;

            #pragma unroll
            for (int lane = 0; lane < format_traits<Bits>::codes_per_byte; ++lane) {
                const int current = index + lane;

                if (current < end) {
                    const int column = load_scalar(matrix->colIdx + current);
                    const unsigned int code = quantize_code_sm70_extreme_prepared<Bits>(
                        values_by_nnz[current - value_base],
                        prepared,
                        column);

                    packed_byte |= (code & static_cast<unsigned int>(format_traits<Bits>::code_mask)) << (lane * Bits);
                }
            }

            *out = format_traits<Bits>::pack_byte(packed_byte);
            index += format_traits<Bits>::codes_per_byte;
            ++out;
        }
    }
}

template<int Bits, typename Real, typename Metadata>
__global__ __launch_bounds__(launch_policy::threads, launch_policy::min_blocks_per_sm)
void quantize_block_kernel(
    csr_matrix<Bits, Real, Metadata> matrix,
    csr_block block,
    const Real* __restrict__ values_by_nnz_block) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + block.row_begin;

    if (row >= block.row_end) {
        return;
    }
    pack_row_values_sm70_extreme(&matrix, row, block.nnz_begin, values_by_nnz_block);
}

} // namespace cellerator::quantized::extreme_backend
