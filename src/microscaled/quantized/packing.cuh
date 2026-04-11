#pragma once

#include <cstdint>

#include "access.cuh"

namespace cellerator::microscaled::quantized {

__host__ __device__ __forceinline__ bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<std::uintptr_t>(ptr) & 0x0fu) == 0u;
}

template<int Bits, typename Real, typename Metadata>
__host__ __device__ __forceinline__ void pack_row_values(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    int row,
    int value_base,
    const Real* values_by_nnz) {
    int index = load_scalar(matrix->rowPtr + row);
    const int end = load_scalar(matrix->rowPtr + row + 1);
    unsigned char* out = matrix->packed_values + load_scalar(matrix->packed_row_ptr + row);
    const auto row_cache = matrix->metadata.prepare_row(row);

    if constexpr (Bits == 8) {
        // The 8-bit path earns a wider vectorized fast path on aligned output
        // because one code maps cleanly to one byte.
        while (index < end && !is_aligned_16(out)) {
            const int column = load_scalar(matrix->colIdx + index);

            *out = static_cast<unsigned char>(quantize_code<Bits>(
                values_by_nnz[index - value_base],
                matrix->metadata,
                row_cache,
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

                w0 |= quantize_code<Bits>(
                    values_by_nnz[i0 - value_base],
                    matrix->metadata,
                    row_cache,
                    load_scalar(matrix->colIdx + i0)) << (lane * 8);
                w1 |= quantize_code<Bits>(
                    values_by_nnz[i1 - value_base],
                    matrix->metadata,
                    row_cache,
                    load_scalar(matrix->colIdx + i1)) << (lane * 8);
                w2 |= quantize_code<Bits>(
                    values_by_nnz[i2 - value_base],
                    matrix->metadata,
                    row_cache,
                    load_scalar(matrix->colIdx + i2)) << (lane * 8);
                w3 |= quantize_code<Bits>(
                    values_by_nnz[i3 - value_base],
                    matrix->metadata,
                    row_cache,
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

            out[0] = static_cast<unsigned char>(quantize_code<Bits>(
                values_by_nnz[index - value_base],
                matrix->metadata,
                row_cache,
                c0));
            out[1] = static_cast<unsigned char>(quantize_code<Bits>(
                values_by_nnz[index + 1 - value_base],
                matrix->metadata,
                row_cache,
                c1));
            out[2] = static_cast<unsigned char>(quantize_code<Bits>(
                values_by_nnz[index + 2 - value_base],
                matrix->metadata,
                row_cache,
                c2));
            out[3] = static_cast<unsigned char>(quantize_code<Bits>(
                values_by_nnz[index + 3 - value_base],
                matrix->metadata,
                row_cache,
                c3));
            index += 4;
            out += 4;
        }
        while (index < end) {
            const int column = load_scalar(matrix->colIdx + index);

            *out = static_cast<unsigned char>(quantize_code<Bits>(
                values_by_nnz[index - value_base],
                matrix->metadata,
                row_cache,
                column));
            ++index;
            ++out;
        }
    } else {
        // Sub-byte formats trade a little extra bit packing work for much
        // smaller payloads. The loop stays row-local to avoid extra staging.
        while (index < end) {
            unsigned int packed_byte = 0u;

            #pragma unroll
            for (int lane = 0; lane < format_traits<Bits>::codes_per_byte; ++lane) {
                const int current = index + lane;

                if (current < end) {
                    const int column = load_scalar(matrix->colIdx + current);
                    const unsigned int code = quantize_code<Bits>(
                        values_by_nnz[current - value_base],
                        matrix->metadata,
                        row_cache,
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
__host__ __device__ __forceinline__ void unpack_row_values(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    int row,
    int value_base,
    Real* values_by_nnz) {
    int index = load_scalar(matrix->rowPtr + row);
    const int end = load_scalar(matrix->rowPtr + row + 1);
    const unsigned char* in = matrix->packed_values + load_scalar(matrix->packed_row_ptr + row);
    const auto row_cache = matrix->metadata.prepare_row(row);

    if constexpr (Bits == 8) {
        while (index < end && !is_aligned_16(in)) {
            const int column = load_scalar(matrix->colIdx + index);

            values_by_nnz[index - value_base] = dequantize_code<Bits>(
                static_cast<unsigned int>(*in),
                matrix->metadata,
                row_cache,
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

                    values_by_nnz[current - value_base] = dequantize_code<Bits>(
                        code,
                        matrix->metadata,
                        row_cache,
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

            values_by_nnz[index - value_base] = dequantize_code<Bits>(
                static_cast<unsigned int>(in[0]),
                matrix->metadata,
                row_cache,
                c0);
            values_by_nnz[index + 1 - value_base] = dequantize_code<Bits>(
                static_cast<unsigned int>(in[1]),
                matrix->metadata,
                row_cache,
                c1);
            values_by_nnz[index + 2 - value_base] = dequantize_code<Bits>(
                static_cast<unsigned int>(in[2]),
                matrix->metadata,
                row_cache,
                c2);
            values_by_nnz[index + 3 - value_base] = dequantize_code<Bits>(
                static_cast<unsigned int>(in[3]),
                matrix->metadata,
                row_cache,
                c3);
            index += 4;
            in += 4;
        }
        while (index < end) {
            const int column = load_scalar(matrix->colIdx + index);

            values_by_nnz[index - value_base] = dequantize_code<Bits>(
                static_cast<unsigned int>(*in),
                matrix->metadata,
                row_cache,
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

                    values_by_nnz[current - value_base] = dequantize_code<Bits>(
                        code,
                        matrix->metadata,
                        row_cache,
                        column);
                }
            }

            index += format_traits<Bits>::codes_per_byte;
            ++in;
        }
    }
}

template<int Bits, typename Real, typename Metadata>
inline int pack_nnz_values(const csr_matrix<Bits, Real, Metadata>* matrix, const Real* values_by_nnz) {
    int row = 0;

    if (matrix == nullptr || matrix->packed_values == nullptr || values_by_nnz == nullptr) {
        return -1;
    }

    // Host helper for export/build code. Steady-state device work should use
    // the block launchers instead of walking rows serially on the host.
    while (row < matrix->rows) {
        pack_row_values(matrix, row, 0, values_by_nnz);
        ++row;
    }
    return 0;
}

template<int Bits, typename Real, typename Metadata>
inline int unpack_nnz_values(const csr_matrix<Bits, Real, Metadata>* matrix, Real* values_by_nnz) {
    int row = 0;

    if (matrix == nullptr || values_by_nnz == nullptr) {
        return -1;
    }

    while (row < matrix->rows) {
        unpack_row_values(matrix, row, 0, values_by_nnz);
        ++row;
    }
    return 0;
}

template<int Bits, typename Real, typename Metadata>
inline int pack_block_nnz_values(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    csr_block block,
    const Real* values_by_nnz_block) {
    int row = block.row_begin;

    if (matrix == nullptr || values_by_nnz_block == nullptr) {
        return -1;
    }

    while (row < block.row_end) {
        pack_row_values(matrix, row, block.nnz_begin, values_by_nnz_block);
        ++row;
    }
    return 0;
}

template<int Bits, typename Real, typename Metadata>
inline int unpack_block_nnz_values(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    csr_block block,
    Real* values_by_nnz_block) {
    int row = block.row_begin;

    if (matrix == nullptr || values_by_nnz_block == nullptr) {
        return -1;
    }

    while (row < block.row_end) {
        unpack_row_values(matrix, row, block.nnz_begin, values_by_nnz_block);
        ++row;
    }
    return 0;
}

} // namespace cellerator::microscaled::quantized
