#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace cellerator::microscaled::quantized {

// Align packed rows so repeated row starts stay friendly to Volta global
// transactions and packed-allocation slicing.
inline constexpr int packed_storage_alignment = 16;

template<int Bits>
struct format_traits {
    static_assert(Bits == 1 || Bits == 2 || Bits == 4 || Bits == 8,
                  "Bits must be 1, 2, 4, or 8");

    enum {
        bits = Bits,
        codes_per_byte = 8 / Bits,
        code_mask = (1u << Bits) - 1u
    };

    __host__ __device__ __forceinline__ static constexpr int row_bytes(int row_nnz) {
        return row_nnz <= 0 ? 0 : (row_nnz + codes_per_byte - 1) / codes_per_byte;
    }

    __host__ __device__ __forceinline__ static constexpr int aligned_row_bytes(
        int row_nnz,
        int alignment = packed_storage_alignment) {
        const int bytes = row_bytes(row_nnz);

        // Padding costs a few bytes per row, but it avoids ragged starts that
        // make row-local decode and writeback harder to coalesce.
        return alignment <= 1 ? bytes : ((bytes + alignment - 1) / alignment) * alignment;
    }

    __host__ __device__ __forceinline__ static unsigned int unpack(
        const unsigned char* packed,
        int local_index) {
        const int slot = local_index / codes_per_byte;
        const int lane = local_index & (codes_per_byte - 1);
        const int shift = lane * Bits;

        // Scalar unpack is the cheap row-local helper, not a vector decode path.
        return (static_cast<unsigned int>(packed[slot]) >> shift) & static_cast<unsigned int>(code_mask);
    }

    __host__ __device__ __forceinline__ static unsigned char pack_byte(unsigned int packed_byte) {
        // Callers assemble the packed byte explicitly; this helper just clamps
        // the final store value.
        return static_cast<unsigned char>(packed_byte & 0xffu);
    }
};

} // namespace cellerator::microscaled::quantized
