#include <Cellerator/seq/dna2.cuh>

#include <cassert>

#ifndef CELLERATOR_ENABLE_HIGHWAY
#define CELLERATOR_ENABLE_HIGHWAY 0
#endif

namespace cellerator::seq {
namespace {

constexpr std::uint64_t packed_lo_mask = 0x5555555555555555ULL;
constexpr std::uint64_t all_base_positions = 0xffffffffULL;

std::uint32_t active_mask(std::size_t n) {
    if (n == 0u) return 0u;
    if (n >= 32u) return 0xffffffffu;
    return (1u << static_cast<unsigned int>(n)) - 1u;
}

std::uint8_t pack_ascii_base(char base) {
#ifndef NDEBUG
    assert(base == 'A' || base == 'C' || base == 'G' || base == 'T');
#endif
    return base == 'C' ? 1u : base == 'G' ? 2u : base == 'T' ? 3u : 0u;
}

std::uint64_t compact_even_bits(std::uint64_t bits) {
    bits &= packed_lo_mask;
    bits = (bits | (bits >> 1u)) & 0x3333333333333333ULL;
    bits = (bits | (bits >> 2u)) & 0x0f0f0f0f0f0f0f0fULL;
    bits = (bits | (bits >> 4u)) & 0x00ff00ff00ff00ffULL;
    bits = (bits | (bits >> 8u)) & 0x0000ffff0000ffffULL;
    bits = (bits | (bits >> 16u)) & 0x00000000ffffffffULL;
    return bits;
}

std::uint64_t spread_bits32(std::uint64_t bits) {
    bits &= all_base_positions;
    bits = (bits | (bits << 16u)) & 0x0000ffff0000ffffULL;
    bits = (bits | (bits << 8u)) & 0x00ff00ff00ff00ffULL;
    bits = (bits | (bits << 4u)) & 0x0f0f0f0f0f0f0f0fULL;
    bits = (bits | (bits << 2u)) & 0x3333333333333333ULL;
    bits = (bits | (bits << 1u)) & packed_lo_mask;
    return bits;
}

std::uint32_t base_mask_from_code(dna2_planes planes, std::uint8_t code) {
    const std::uint64_t lo = planes.lo & all_base_positions;
    const std::uint64_t hi = planes.hi & all_base_positions;
    const std::uint64_t mask = code == 0u ? (~lo & ~hi)
        : code == 1u ? (lo & ~hi)
        : code == 2u ? (~lo & hi)
        : (lo & hi);
    return static_cast<std::uint32_t>(mask & all_base_positions);
}

} // namespace

dna2_word dna2_pack_ascii_32(const char* bases, std::size_t n) {
    assert(n <= 32u);
    assert(bases != nullptr || n == 0u);
    dna2_word out{0ULL};
    const std::size_t limit = n > 32u ? 32u : n;
    for (std::size_t i = 0u; i < limit; ++i) {
        out.bits |= static_cast<std::uint64_t>(pack_ascii_base(bases[i])) << (2u * static_cast<unsigned int>(i));
    }
    return out;
}

void dna2_unpack_ascii_32(dna2_word word, char* out, std::size_t n) {
    assert(n <= 32u);
    assert(out != nullptr || n == 0u);
    const std::size_t limit = n > 32u ? 32u : n;
    for (std::size_t i = 0u; i < limit; ++i) {
        const std::uint8_t base = static_cast<std::uint8_t>((word.bits >> (2u * static_cast<unsigned int>(i))) & 0x3ULL);
        out[i] = base_to_char(base);
    }
}

dna2_planes dna2_to_planes(dna2_word word) {
    return dna2_planes{
        compact_even_bits(word.bits),
        compact_even_bits(word.bits >> 1u)
    };
}

dna2_word planes_to_dna2(dna2_planes planes) {
    return dna2_word{spread_bits32(planes.lo) | (spread_bits32(planes.hi) << 1u)};
}

std::uint32_t planes_base_mask(dna2_planes planes, char base) {
    return base_mask_from_code(planes, pack_ascii_base(base));
}

std::uint32_t planes_gc_mask(dna2_planes planes) {
    return static_cast<std::uint32_t>((planes.lo ^ planes.hi) & all_base_positions);
}

std::uint32_t planes_cpg_start_mask(dna2_planes planes) {
    const std::uint32_t c_mask = base_mask_from_code(planes, 1u);
    const std::uint32_t g_mask = base_mask_from_code(planes, 2u);
    return (c_mask & (g_mask >> 1u)) & 0x7fffffffu;
}

std::uint32_t dna2_hamming_distance(dna2_word a, dna2_word b, std::size_t n) {
    const std::uint32_t mask = word64_mismatch_mask(dna2_word64{a.bits}, dna2_word64{b.bits}, active_mask(n));
    return static_cast<std::uint32_t>(detail::popcount32(mask));
}

void dna2_pack_ascii_batch_scalar(
    const char* input,
    std::size_t stride,
    dna2_word* output,
    std::size_t count,
    std::size_t n_per_seq) {
    assert(input != nullptr || count == 0u || n_per_seq == 0u);
    assert(output != nullptr || count == 0u);
    for (std::size_t i = 0u; i < count; ++i) {
        output[i] = dna2_pack_ascii_32(input + i * stride, n_per_seq);
    }
}

void dna2_unpack_ascii_batch_scalar(
    const dna2_word* input,
    char* output,
    std::size_t stride,
    std::size_t count,
    std::size_t n_per_seq) {
    assert(input != nullptr || count == 0u);
    assert(output != nullptr || count == 0u || n_per_seq == 0u);
    for (std::size_t i = 0u; i < count; ++i) {
        dna2_unpack_ascii_32(input[i], output + i * stride, n_per_seq);
    }
}

void dna2_to_planes_batch_scalar(const dna2_word* input, dna2_planes* output, std::size_t count) {
    assert(input != nullptr || count == 0u);
    assert(output != nullptr || count == 0u);
    for (std::size_t i = 0u; i < count; ++i) {
        output[i] = dna2_to_planes(input[i]);
    }
}

void planes_gc_mask_batch_scalar(const dna2_planes* input, std::uint32_t* output_masks, std::size_t count) {
    assert(input != nullptr || count == 0u);
    assert(output_masks != nullptr || count == 0u);
    for (std::size_t i = 0u; i < count; ++i) {
        output_masks[i] = planes_gc_mask(input[i]);
    }
}

void planes_cpg_start_mask_batch_scalar(const dna2_planes* input, std::uint32_t* output_masks, std::size_t count) {
    assert(input != nullptr || count == 0u);
    assert(output_masks != nullptr || count == 0u);
    for (std::size_t i = 0u; i < count; ++i) {
        output_masks[i] = planes_cpg_start_mask(input[i]);
    }
}

#if !CELLERATOR_ENABLE_HIGHWAY
void dna2_pack_ascii_batch_highway(
    const char* input,
    std::size_t stride,
    dna2_word* output,
    std::size_t count,
    std::size_t n_per_seq) {
    dna2_pack_ascii_batch_scalar(input, stride, output, count, n_per_seq);
}

void dna2_unpack_ascii_batch_highway(
    const dna2_word* input,
    char* output,
    std::size_t stride,
    std::size_t count,
    std::size_t n_per_seq) {
    dna2_unpack_ascii_batch_scalar(input, output, stride, count, n_per_seq);
}

void dna2_to_planes_batch_highway(const dna2_word* input, dna2_planes* output, std::size_t count) {
    dna2_to_planes_batch_scalar(input, output, count);
}

void planes_gc_mask_batch_highway(const dna2_planes* input, std::uint32_t* output_masks, std::size_t count) {
    planes_gc_mask_batch_scalar(input, output_masks, count);
}

void planes_cpg_start_mask_batch_highway(const dna2_planes* input, std::uint32_t* output_masks, std::size_t count) {
    planes_cpg_start_mask_batch_scalar(input, output_masks, count);
}
#endif

void dna2_pack_ascii_batch(
    const char* input,
    std::size_t stride,
    dna2_word* output,
    std::size_t count,
    std::size_t n_per_seq) {
#if CELLERATOR_ENABLE_HIGHWAY
    dna2_pack_ascii_batch_highway(input, stride, output, count, n_per_seq);
#else
    dna2_pack_ascii_batch_scalar(input, stride, output, count, n_per_seq);
#endif
}

void dna2_unpack_ascii_batch(
    const dna2_word* input,
    char* output,
    std::size_t stride,
    std::size_t count,
    std::size_t n_per_seq) {
#if CELLERATOR_ENABLE_HIGHWAY
    dna2_unpack_ascii_batch_highway(input, output, stride, count, n_per_seq);
#else
    dna2_unpack_ascii_batch_scalar(input, output, stride, count, n_per_seq);
#endif
}

void dna2_to_planes_batch(const dna2_word* input, dna2_planes* output, std::size_t count) {
#if CELLERATOR_ENABLE_HIGHWAY
    dna2_to_planes_batch_highway(input, output, count);
#else
    dna2_to_planes_batch_scalar(input, output, count);
#endif
}

void planes_gc_mask_batch(const dna2_planes* input, std::uint32_t* output_masks, std::size_t count) {
#if CELLERATOR_ENABLE_HIGHWAY
    planes_gc_mask_batch_highway(input, output_masks, count);
#else
    planes_gc_mask_batch_scalar(input, output_masks, count);
#endif
}

void planes_cpg_start_mask_batch(const dna2_planes* input, std::uint32_t* output_masks, std::size_t count) {
#if CELLERATOR_ENABLE_HIGHWAY
    planes_cpg_start_mask_batch_highway(input, output_masks, count);
#else
    planes_cpg_start_mask_batch_scalar(input, output_masks, count);
#endif
}

} // namespace cellerator::seq
