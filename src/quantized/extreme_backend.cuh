#pragma once

#include "packing.cuh"

namespace cellerator::quantized::extreme_backend {

struct launch_policy {
    enum {
        // Extreme mode leans harder into row residency and tries to keep more
        // independent rows in flight per launch on V100.
        threads = 256,
        min_blocks_per_sm = 3
    };
};

__device__ __forceinline__ float ptx_mul_f32(float lhs, float rhs) {
    float out = 0.0f;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(lhs), "f"(rhs));
    return out;
}

__device__ __forceinline__ float ptx_sub_f32(float lhs, float rhs) {
    float out = 0.0f;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(lhs), "f"(rhs));
    return out;
}

__device__ __forceinline__ float ptx_fma_f32(float a, float b, float c) {
    float out = 0.0f;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(out) : "f"(a), "f"(b), "f"(c));
    return out;
}

__device__ __forceinline__ float ptx_rcp_nr1_f32(float den) {
    float recip = 0.0f;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(recip) : "f"(den));
    const float err = ptx_sub_f32(1.0f, ptx_mul_f32(den, recip));
    return ptx_fma_f32(recip, err, recip);
}

__device__ __forceinline__ int ptx_round_rni_s32(float value) {
    int out = 0;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(out) : "f"(value));
    return out;
}

__device__ __forceinline__ float ptx_cvt_f32_u32(unsigned int value) {
    float out = 0.0f;
    asm("cvt.rn.f32.u32 %0, %1;" : "=f"(out) : "r"(value));
    return out;
}

__device__ __forceinline__ unsigned int ptx_clamp_code(unsigned int max_code, int rounded) {
    int lower_bounded = 0;
    int upper_bounded = 0;

    asm(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.lt.s32 p, %1, 0;\n\t"
        "selp.s32 %0, 0, %1, p;\n\t"
        "}"
        : "=r"(lower_bounded)
        : "r"(rounded));

    asm(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.gt.s32 p, %1, %2;\n\t"
        "selp.s32 %0, %2, %1, p;\n\t"
        "}"
        : "=r"(upper_bounded)
        : "r"(lower_bounded), "r"(static_cast<int>(max_code)));

    return static_cast<unsigned int>(upper_bounded);
}

template<int Bits, typename Real, typename Metadata>
__device__ __forceinline__ unsigned int quantize_code_sm70_extreme(
    Real value,
    const Metadata& metadata,
    typename Metadata::row_cache row_cache,
    int column) {
    const float scale = to_float(metadata.scale_for(row_cache, column));

    if (scale == 0.0f) {
        return 0u;
    }

    const float offset = to_float(metadata.offset_for(row_cache, column));
    const float centered = ptx_sub_f32(to_float(value), offset);
    const float q = ptx_mul_f32(centered, ptx_rcp_nr1_f32(scale));
    return ptx_clamp_code(static_cast<unsigned int>(format_traits<Bits>::code_mask), ptx_round_rni_s32(q));
}

template<int Bits, typename Real, typename Metadata>
__device__ __forceinline__ Real dequantize_code_sm70_extreme(
    unsigned int code,
    const Metadata& metadata,
    typename Metadata::row_cache row_cache,
    int column) {
    const float scale = to_float(metadata.scale_for(row_cache, column));
    const float offset = to_float(metadata.offset_for(row_cache, column));
    const float reconstructed = ptx_fma_f32(ptx_cvt_f32_u32(code), scale, offset);
    return from_float<Real>(reconstructed);
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

    if constexpr (Bits == 8) {
        while (index < end && !is_aligned_16(out)) {
            const int column = load_scalar(matrix->colIdx + index);
            *out = static_cast<unsigned char>(quantize_code_sm70_extreme<Bits>(
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

                w0 |= quantize_code_sm70_extreme<Bits>(
                    values_by_nnz[i0 - value_base],
                    matrix->metadata,
                    row_cache,
                    load_scalar(matrix->colIdx + i0)) << (lane * 8);
                w1 |= quantize_code_sm70_extreme<Bits>(
                    values_by_nnz[i1 - value_base],
                    matrix->metadata,
                    row_cache,
                    load_scalar(matrix->colIdx + i1)) << (lane * 8);
                w2 |= quantize_code_sm70_extreme<Bits>(
                    values_by_nnz[i2 - value_base],
                    matrix->metadata,
                    row_cache,
                    load_scalar(matrix->colIdx + i2)) << (lane * 8);
                w3 |= quantize_code_sm70_extreme<Bits>(
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

            out[0] = static_cast<unsigned char>(quantize_code_sm70_extreme<Bits>(
                values_by_nnz[index - value_base],
                matrix->metadata,
                row_cache,
                c0));
            out[1] = static_cast<unsigned char>(quantize_code_sm70_extreme<Bits>(
                values_by_nnz[index + 1 - value_base],
                matrix->metadata,
                row_cache,
                c1));
            out[2] = static_cast<unsigned char>(quantize_code_sm70_extreme<Bits>(
                values_by_nnz[index + 2 - value_base],
                matrix->metadata,
                row_cache,
                c2));
            out[3] = static_cast<unsigned char>(quantize_code_sm70_extreme<Bits>(
                values_by_nnz[index + 3 - value_base],
                matrix->metadata,
                row_cache,
                c3));
            index += 4;
            out += 4;
        }

        while (index < end) {
            const int column = load_scalar(matrix->colIdx + index);
            *out = static_cast<unsigned char>(quantize_code_sm70_extreme<Bits>(
                values_by_nnz[index - value_base],
                matrix->metadata,
                row_cache,
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
                    const unsigned int code = quantize_code_sm70_extreme<Bits>(
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
__device__ __forceinline__ void unpack_row_values_sm70_extreme(
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
            values_by_nnz[index - value_base] = dequantize_code_sm70_extreme<Bits, Real, Metadata>(
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

                    values_by_nnz[current - value_base] = dequantize_code_sm70_extreme<Bits, Real, Metadata>(
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

            values_by_nnz[index - value_base] = dequantize_code_sm70_extreme<Bits, Real, Metadata>(
                static_cast<unsigned int>(in[0]),
                matrix->metadata,
                row_cache,
                c0);
            values_by_nnz[index + 1 - value_base] = dequantize_code_sm70_extreme<Bits, Real, Metadata>(
                static_cast<unsigned int>(in[1]),
                matrix->metadata,
                row_cache,
                c1);
            values_by_nnz[index + 2 - value_base] = dequantize_code_sm70_extreme<Bits, Real, Metadata>(
                static_cast<unsigned int>(in[2]),
                matrix->metadata,
                row_cache,
                c2);
            values_by_nnz[index + 3 - value_base] = dequantize_code_sm70_extreme<Bits, Real, Metadata>(
                static_cast<unsigned int>(in[3]),
                matrix->metadata,
                row_cache,
                c3);
            index += 4;
            in += 4;
        }

        while (index < end) {
            const int column = load_scalar(matrix->colIdx + index);
            values_by_nnz[index - value_base] = dequantize_code_sm70_extreme<Bits, Real, Metadata>(
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

                    values_by_nnz[current - value_base] = dequantize_code_sm70_extreme<Bits, Real, Metadata>(
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

template<int Bits, typename Real, typename Metadata>
inline cudaError_t launch_quantize_block(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    csr_block block,
    const Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    const int rows_in_block = block.row_end - block.row_begin;
    const int blocks = (rows_in_block + launch_policy::threads - 1) / launch_policy::threads;

    if (matrix == nullptr || values_by_nnz_block == nullptr || rows_in_block <= 0) {
        return cudaErrorInvalidValue;
    }

    quantize_block_kernel<Bits><<<blocks, launch_policy::threads, 0, stream>>>(
        *matrix,
        block,
        values_by_nnz_block);
    return cudaGetLastError();
}

template<int Bits, typename Real, typename Metadata>
inline cudaError_t launch_dequantize_block(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    csr_block block,
    Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    const int rows_in_block = block.row_end - block.row_begin;
    const int blocks = (rows_in_block + launch_policy::threads - 1) / launch_policy::threads;

    if (matrix == nullptr || values_by_nnz_block == nullptr || rows_in_block <= 0) {
        return cudaErrorInvalidValue;
    }

    dequantize_block_kernel<Bits><<<blocks, launch_policy::threads, 0, stream>>>(
        *matrix,
        block,
        values_by_nnz_block);
    return cudaGetLastError();
}

template<int Bits, typename Real, typename Metadata>
inline cudaError_t launch_quantize_block_v100(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    csr_block block,
    const Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    cudaError_t status = cudaFuncSetCacheConfig(quantize_block_kernel<Bits, Real, Metadata>, cudaFuncCachePreferL1);

    if (status != cudaSuccess) {
        return status;
    }
    return extreme_backend::launch_quantize_block<Bits, Real, Metadata>(
        matrix,
        block,
        values_by_nnz_block,
        stream);
}

template<int Bits, typename Real, typename Metadata>
inline cudaError_t launch_dequantize_block_v100(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    csr_block block,
    Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    cudaError_t status = cudaFuncSetCacheConfig(dequantize_block_kernel<Bits, Real, Metadata>, cudaFuncCachePreferL1);

    if (status != cudaSuccess) {
        return status;
    }
    return extreme_backend::launch_dequantize_block<Bits, Real, Metadata>(
        matrix,
        block,
        values_by_nnz_block,
        stream);
}

} // namespace cellerator::quantized::extreme_backend
