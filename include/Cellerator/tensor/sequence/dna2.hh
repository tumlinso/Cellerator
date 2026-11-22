#pragma once

#include <stdint.h>

#if defined(__CUDACC__)
#define CELLERATOR_DNA2_HD __host__ __device__ __forceinline__
#else
#define CELLERATOR_DNA2_HD inline
#endif

namespace cellerator::tensor::sequence {

inline constexpr uint32_t max_chunk_bases = 0x7fffffffu;

// Two bits per base and, when present, one validity bit per base.
struct PackedView {
    const uint64_t *words;
    const uint64_t *validity;
    uint32_t word_count, base_count, validity_word_count;
};

// A null validity plane denotes all in-range bases valid. Tail bits are absent.
struct BitPlanesView {
    const uint32_t *low;
    const uint32_t *high;
    const uint32_t *validity;
    uint32_t word_count, base_count;
};

struct MutableBitPlanesView {
    uint32_t *low;
    uint32_t *high;
    uint32_t *validity;
    uint32_t word_capacity;
};

CELLERATOR_DNA2_HD constexpr uint32_t word_count32(uint32_t n) noexcept {
    return n/32u + (n%32u != 0u);
}

CELLERATOR_DNA2_HD constexpr uint32_t word_count64(uint32_t n) noexcept {
    return n/64u + (n%64u != 0u);
}

CELLERATOR_DNA2_HD constexpr uint32_t low_mask32(uint32_t n) noexcept {
    return n >= 32u ? UINT32_MAX : (uint32_t{1} << n) - 1u;
}

CELLERATOR_DNA2_HD uint32_t validity_word32(PackedView v, uint32_t word) noexcept {
    if (word >= word_count32(v.base_count)) return 0;
    uint32_t tail = low_mask32(v.base_count - 32u*word);
    return v.validity ? tail & uint32_t(v.validity[word/2u] >> (32u*(word%2u))) : tail;
}

CELLERATOR_DNA2_HD uint32_t validity_word32(BitPlanesView v, uint32_t word) noexcept {
    if (word >= word_count32(v.base_count)) return 0;
    uint32_t tail = low_mask32(v.base_count - 32u*word);
    return v.validity ? tail & v.validity[word] : tail;
}

inline bool valid_view(PackedView v) noexcept {
    return v.base_count <= max_chunk_bases && v.word_count >= word_count32(v.base_count)
        && (!v.word_count || v.words)
        && (v.validity ? v.validity_word_count >= word_count64(v.base_count) : !v.validity_word_count);
}

inline bool valid_view(BitPlanesView v) noexcept {
    return v.base_count <= max_chunk_bases && v.word_count >= word_count32(v.base_count)
        && (!v.word_count || (v.low && v.high));
}

// Unchecked position access on a validated view, using logical extents at seams.
CELLERATOR_DNA2_HD uint8_t base_code(PackedView v, uint32_t pos) noexcept {
    return uint8_t((v.words[pos/32u] >> (2u*(pos%32u))) & 3u);
}

CELLERATOR_DNA2_HD uint64_t packed_window(PackedView v, uint32_t start) noexcept {
    uint32_t word = start/32u, shift = 2u*(start%32u);
    uint64_t lo = v.words[word];
    if (!shift) return lo;
    uint64_t hi = word + 1u < word_count32(v.base_count) ? v.words[word + 1u] : 0u;
    return (lo >> shift) | (hi << (64u - shift));
}

CELLERATOR_DNA2_HD uint32_t plane_window(const uint32_t *src, uint32_t n, uint32_t start) noexcept {
    uint32_t word = start/32u, shift = start%32u;
    uint32_t lo = src[word];
    if (!shift) return lo;
    uint32_t hi = word + 1u < n ? src[word + 1u] : 0u;
    return (lo >> shift) | (hi << (32u - shift));
}

// Windows have at most 32 bases. Empty windows at the end are valid.
CELLERATOR_DNA2_HD bool valid_window(PackedView v, uint32_t start, uint32_t length) noexcept {
    if (start > v.base_count || length > 32u || length > v.base_count - start) return false;
    if (!length || !v.validity) return true;
    uint32_t word = start/64u, shift = start%64u;
    uint64_t mask = v.validity[word] >> shift;
    if (length > 64u - shift) mask |= v.validity[word + 1u] << (64u - shift);
    uint64_t active = low_mask32(length);
    return (mask & active) == active;
}

CELLERATOR_DNA2_HD bool valid_window(BitPlanesView v, uint32_t start, uint32_t length) noexcept {
    if (start > v.base_count || length > 32u || length > v.base_count - start) return false;
    if (!length || !v.validity) return true;
    uint32_t active = low_mask32(length);
    return (plane_window(v.validity, word_count32(v.base_count), start) & active) == active;
}

CELLERATOR_DNA2_HD uint32_t gather_even_bits(uint64_t x) noexcept {
    x &= 0x5555555555555555ULL;
    x = (x | (x >> 1u)) & 0x3333333333333333ULL;
    x = (x | (x >> 2u)) & 0x0f0f0f0f0f0f0f0fULL;
    x = (x | (x >> 4u)) & 0x00ff00ff00ff00ffULL;
    x = (x | (x >> 8u)) & 0x0000ffff0000ffffULL;
    return uint32_t(x | (x >> 16u));
}

CELLERATOR_DNA2_HD void split_word(uint64_t packed, uint32_t *low, uint32_t *high) noexcept {
    *low = gather_even_bits(packed);
    *high = gather_even_bits(packed >> 1u);
}

}  // namespace cellerator::tensor::sequence

#undef CELLERATOR_DNA2_HD
