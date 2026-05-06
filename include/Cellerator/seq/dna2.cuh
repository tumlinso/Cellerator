#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#if defined(__CUDACC__)
#define CELLERATOR_SEQ_HD __host__ __device__ __forceinline__
#define CELLERATOR_SEQ_DEVICE __device__ __forceinline__
#else
#define CELLERATOR_SEQ_HD inline
#endif

namespace cellerator::seq {

// Canonical 2-bit sequence encoding for packed DNA/RNA primitives:
// A = 00, C = 01, G = 10, T/U = 11. The ASCII packer below accepts uppercase
// A/C/G/T only; non-canonical bytes assert in debug builds and pack as A.
enum class dna2_base : std::uint8_t {
    A = 0,
    C = 1,
    G = 2,
    T = 3,
};

struct dna2_word {
    std::uint64_t bits;
};

struct dna2_planes {
    std::uint64_t lo;
    std::uint64_t hi;
};

struct dna2_word64 {
    std::uint64_t packed;
};

struct dna2_planes32 {
    std::uint32_t lo;
    std::uint32_t hi;
};

struct dna2_window32 {
    dna2_planes32 planes;
    std::uint32_t valid_mask;
    std::uint64_t genomic_offset;
};

namespace detail {

constexpr std::uint32_t all_bases_mask = 0xffffffffu;
constexpr std::uint64_t packed_field_lo_mask = 0x5555555555555555ULL;

CELLERATOR_SEQ_HD std::uint32_t active_mask_from_length(int length) {
    if (length <= 0) return 0u;
    if (length >= 32) return all_bases_mask;
    return (1u << static_cast<unsigned int>(length)) - 1u;
}

CELLERATOR_SEQ_HD int popcount32(std::uint32_t value) {
#if defined(__CUDA_ARCH__)
    return __popc(value);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(value);
#else
    int count = 0;
    while (value != 0u) {
        count += static_cast<int>(value & 1u);
        value >>= 1u;
    }
    return count;
#endif
}

CELLERATOR_SEQ_HD int popcount64(std::uint64_t value) {
#if defined(__CUDA_ARCH__)
    return __popcll(static_cast<unsigned long long>(value));
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(static_cast<unsigned long long>(value));
#else
    int count = 0;
    while (value != 0u) {
        count += static_cast<int>(value & 1u);
        value >>= 1u;
    }
    return count;
#endif
}

CELLERATOR_SEQ_HD std::uint64_t spread_active_mask_to_packed_fields(std::uint32_t active_mask) {
    std::uint64_t bits = active_mask;
    bits = (bits | (bits << 16u)) & 0x0000ffff0000ffffULL;
    bits = (bits | (bits << 8u)) & 0x00ff00ff00ff00ffULL;
    bits = (bits | (bits << 4u)) & 0x0f0f0f0f0f0f0f0fULL;
    bits = (bits | (bits << 2u)) & 0x3333333333333333ULL;
    bits = (bits | (bits << 1u)) & packed_field_lo_mask;
    return bits;
}

CELLERATOR_SEQ_HD int word64_mismatches_packed_count_fields(
    dna2_word64 a,
    dna2_word64 b,
    std::uint64_t active_fields) {
    const std::uint64_t x = a.packed ^ b.packed;
    const std::uint64_t y = x | (x >> 1u);
    const std::uint64_t mismatch_fields = y & packed_field_lo_mask;
    return popcount64(mismatch_fields & active_fields);
}

} // namespace detail

CELLERATOR_SEQ_HD std::uint8_t make_base(char c) {
    return (c == 'C' || c == 'c') ? static_cast<std::uint8_t>(dna2_base::C)
        : (c == 'G' || c == 'g') ? static_cast<std::uint8_t>(dna2_base::G)
        : (c == 'T' || c == 't' || c == 'U' || c == 'u') ? static_cast<std::uint8_t>(dna2_base::T)
        : static_cast<std::uint8_t>(dna2_base::A);
}

CELLERATOR_SEQ_HD char base_to_char(std::uint8_t b) {
    const std::uint8_t base = b & 0x3u;
    return base == static_cast<std::uint8_t>(dna2_base::C) ? 'C'
        : base == static_cast<std::uint8_t>(dna2_base::G) ? 'G'
        : base == static_cast<std::uint8_t>(dna2_base::T) ? 'T'
        : 'A';
}

CELLERATOR_SEQ_HD std::uint8_t get_base(dna2_word64 w, int i) {
    const unsigned int slot = static_cast<unsigned int>(i) & 31u;
    return static_cast<std::uint8_t>((w.packed >> (slot * 2u)) & 0x3ULL);
}

CELLERATOR_SEQ_HD void set_base(dna2_word64& w, int i, std::uint8_t b) {
    const unsigned int slot = static_cast<unsigned int>(i) & 31u;
    const unsigned int shift = slot * 2u;
    const std::uint64_t mask = 0x3ULL << shift;
    w.packed = (w.packed & ~mask) | ((static_cast<std::uint64_t>(b & 0x3u)) << shift);
}

CELLERATOR_SEQ_HD dna2_planes32 unpack_word64_to_planes32(dna2_word64 w) {
    dna2_planes32 p{0u, 0u};
    for (int i = 0; i < 32; ++i) {
        const std::uint8_t base = get_base(w, i);
        p.lo |= static_cast<std::uint32_t>(base & 0x1u) << i;
        p.hi |= static_cast<std::uint32_t>((base >> 1u) & 0x1u) << i;
    }
    return p;
}

CELLERATOR_SEQ_HD dna2_word64 pack_planes32_to_word64(dna2_planes32 p) {
    dna2_word64 w{0ULL};
    for (int i = 0; i < 32; ++i) {
        const std::uint8_t base = static_cast<std::uint8_t>(
            ((((p.hi >> i) & 0x1u) << 1u) | ((p.lo >> i) & 0x1u)) & 0x3u);
        set_base(w, i, base);
    }
    return w;
}

CELLERATOR_SEQ_HD std::uint32_t planes32_mismatch_mask(dna2_planes32 a, dna2_planes32 b, std::uint32_t active_mask) {
    return ((a.lo ^ b.lo) | (a.hi ^ b.hi)) & active_mask;
}

CELLERATOR_SEQ_HD int planes32_mismatches(dna2_planes32 a, dna2_planes32 b, std::uint32_t active_mask) {
    return detail::popcount32(planes32_mismatch_mask(a, b, active_mask));
}

CELLERATOR_SEQ_HD bool planes32_exact_match(dna2_planes32 a, dna2_planes32 b, std::uint32_t active_mask) {
    return planes32_mismatch_mask(a, b, active_mask) == 0u;
}

CELLERATOR_SEQ_HD std::uint32_t word64_mismatch_mask(dna2_word64 a, dna2_word64 b, std::uint32_t active_mask_32bases) {
    const std::uint64_t x = a.packed ^ b.packed;
    const std::uint64_t y = x | (x >> 1u);
    const std::uint64_t mismatch_fields = y & detail::packed_field_lo_mask;
    std::uint32_t mask = 0u;
    for (int i = 0; i < 32; ++i) {
        mask |= static_cast<std::uint32_t>((mismatch_fields >> (2u * static_cast<unsigned int>(i))) & 0x1ULL) << i;
    }
    return mask & active_mask_32bases;
}

CELLERATOR_SEQ_HD int word64_mismatches(dna2_word64 a, dna2_word64 b, std::uint32_t active_mask_32bases) {
    return detail::popcount32(word64_mismatch_mask(a, b, active_mask_32bases));
}

CELLERATOR_SEQ_HD int word64_mismatches_packed_count(dna2_word64 a, dna2_word64 b, std::uint32_t active_mask_32bases) {
    const std::uint64_t active_fields = detail::spread_active_mask_to_packed_fields(active_mask_32bases);
    return detail::word64_mismatches_packed_count_fields(a, b, active_fields);
}

dna2_word dna2_pack_ascii_32(const char* bases, std::size_t n);
void dna2_unpack_ascii_32(dna2_word word, char* out, std::size_t n);

dna2_planes dna2_to_planes(dna2_word word);
dna2_word planes_to_dna2(dna2_planes planes);

std::uint32_t planes_base_mask(dna2_planes planes, char base);
std::uint32_t planes_gc_mask(dna2_planes planes);
std::uint32_t planes_cpg_start_mask(dna2_planes planes);

std::uint32_t dna2_hamming_distance(dna2_word a, dna2_word b, std::size_t n);

void dna2_pack_ascii_batch_scalar(
    const char* input,
    std::size_t stride,
    dna2_word* output,
    std::size_t count,
    std::size_t n_per_seq);

void dna2_unpack_ascii_batch_scalar(
    const dna2_word* input,
    char* output,
    std::size_t stride,
    std::size_t count,
    std::size_t n_per_seq);

void dna2_to_planes_batch_scalar(
    const dna2_word* input,
    dna2_planes* output,
    std::size_t count);

void planes_gc_mask_batch_scalar(
    const dna2_planes* input,
    std::uint32_t* output_masks,
    std::size_t count);

void planes_cpg_start_mask_batch_scalar(
    const dna2_planes* input,
    std::uint32_t* output_masks,
    std::size_t count);

void dna2_pack_ascii_batch_highway(
    const char* input,
    std::size_t stride,
    dna2_word* output,
    std::size_t count,
    std::size_t n_per_seq);

void dna2_unpack_ascii_batch_highway(
    const dna2_word* input,
    char* output,
    std::size_t stride,
    std::size_t count,
    std::size_t n_per_seq);

void dna2_to_planes_batch_highway(
    const dna2_word* input,
    dna2_planes* output,
    std::size_t count);

void planes_gc_mask_batch_highway(
    const dna2_planes* input,
    std::uint32_t* output_masks,
    std::size_t count);

void planes_cpg_start_mask_batch_highway(
    const dna2_planes* input,
    std::uint32_t* output_masks,
    std::size_t count);

void dna2_pack_ascii_batch(
    const char* input,
    std::size_t stride,
    dna2_word* output,
    std::size_t count,
    std::size_t n_per_seq);

void dna2_unpack_ascii_batch(
    const dna2_word* input,
    char* output,
    std::size_t stride,
    std::size_t count,
    std::size_t n_per_seq);

void dna2_to_planes_batch(
    const dna2_word* input,
    dna2_planes* output,
    std::size_t count);

void planes_gc_mask_batch(
    const dna2_planes* input,
    std::uint32_t* output_masks,
    std::size_t count);

void planes_cpg_start_mask_batch(
    const dna2_planes* input,
    std::uint32_t* output_masks,
    std::size_t count);

CELLERATOR_SEQ_HD dna2_word64 reverse_complement_word64(dna2_word64 w, int length) {
    const int n = length <= 0 ? 0 : (length > 32 ? 32 : length);
    dna2_word64 out{0ULL};
    for (int i = 0; i < n; ++i) {
        const std::uint8_t base = get_base(w, n - 1 - i) ^ 0x3u;
        set_base(out, i, base);
    }
    return out;
}

CELLERATOR_SEQ_HD dna2_planes32 reverse_complement_planes32(dna2_planes32 p, int length) {
    return unpack_word64_to_planes32(reverse_complement_word64(pack_planes32_to_word64(p), length));
}

#if defined(__CUDACC__)
CELLERATOR_SEQ_DEVICE dna2_planes32 warp_encode_base_lanes(std::uint8_t base, unsigned mask = 0xffffffffu) {
    dna2_planes32 p;
    p.lo = __ballot_sync(mask, (base & 0x1u) != 0u);
    p.hi = __ballot_sync(mask, (base & 0x2u) != 0u);
    return p;
}

__global__ void scan_motif_warp32_unpacked(
    const std::uint8_t* seq_bases,
    int n_bases,
    dna2_planes32 motif,
    int motif_len,
    int max_mismatches,
    std::uint8_t* hits);

__global__ void convert_word64_to_planes32_warp(
    const std::uint64_t* packed_words,
    dna2_planes32* planes,
    int word_count);

__global__ void scan_motif_word64_reference(
    const std::uint64_t* packed_seq,
    int n_bases,
    dna2_word64 motif_word,
    int motif_len,
    int max_mismatches,
    std::uint8_t* hits);

__global__ void scan_motif_word64_shifted_count(
    const std::uint64_t* packed_seq,
    int n_bases,
    dna2_word64 motif_word,
    int motif_len,
    int max_mismatches,
    unsigned long long* hit_count);
#endif

} // namespace cellerator::seq

#undef CELLERATOR_SEQ_HD
#if defined(__CUDACC__)
#undef CELLERATOR_SEQ_DEVICE
#endif
