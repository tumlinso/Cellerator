#pragma once

#include <Cellerator/common/compiler_annotations.hh>
#include <Cellerator/ops/sequence/dna2.hh>

namespace cellerator::ops::sequence::detail {
namespace s = tensor::sequence;

inline Status validate_mask(s::BitPlanesView input, uint8_t code, const uint32_t *out,
                            uint32_t capacity) noexcept {
    if (code > 3u || !s::valid_view(input) || (input.base_count && !out)) return Status::invalid_argument;
    return capacity < s::word_count32(input.base_count) ? Status::insufficient_capacity : Status::ok;
}

inline uint32_t window_count(s::PackedView input, motif32_exact motif) noexcept {
    return input.base_count >= motif.length ? input.base_count - motif.length + 1u : 0u;
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE uint8_t match_mismatches(s::PackedView input, uint32_t start, motif32_exact motif) noexcept {
    return s::valid_window(input, start, motif.length) ? motif_mismatches(s::packed_window(input, start), motif) : UINT8_MAX;
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE uint32_t code_mask(uint32_t low, uint32_t high, uint8_t code) noexcept {
    return ((code & 1u) ? low : ~low) & ((code & 2u) ? high : ~high);
}

enum class Mask { base, gc, cpg };

template<Mask mask>
CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE uint32_t mask_word(s::BitPlanesView input, uint32_t i, uint8_t code) noexcept {
    uint32_t lo = input.low[i], hi = input.high[i], valid = s::validity_word32(input, i);
    if constexpr (mask == Mask::base) return valid & code_mask(lo, hi, code);
    if constexpr (mask == Mask::gc) return valid & (lo ^ hi);
    uint32_t c = valid & lo & ~hi, g = valid & hi & ~lo;
    if (i + 1u < s::word_count32(input.base_count)) {
        uint32_t next_g = s::validity_word32(input, i + 1u) & input.high[i + 1u] & ~input.low[i + 1u];
        return c & ((g >> 1u) | (next_g << 31u));
    }
    return c & (g >> 1u);
}

}  // namespace cellerator::ops::sequence::detail
