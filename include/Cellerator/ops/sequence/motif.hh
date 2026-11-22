#pragma once

#include <Cellerator/common/compiler_annotations.hh>
#include <Cellerator/tensor/sequence/dna2.hh>

#include <stdint.h>

namespace cellerator::ops::sequence {

struct motif32_exact {
    uint64_t packed;
    uint64_t active_fields;
    uint8_t length;
    uint8_t max_mismatches;
    uint16_t motif_id;
};

enum class dna2_strand_mode : uint8_t { forward = 0, reverse = 1, both = 2 };
enum class dna2_palindrome_policy : uint8_t { deduplicate = 0, report_both = 1 };

inline constexpr uint8_t dna2_forward_strand_bit = 0x1u;
inline constexpr uint8_t dna2_reverse_strand_bit = 0x2u;

struct motif32_allowed {
    uint32_t allow_A, allow_C, allow_G, allow_T;
    uint32_t active_mask;
    uint8_t length;
    uint8_t max_mismatches;
    uint16_t motif_id;
    dna2_strand_mode strand_mode;
    dna2_palindrome_policy palindrome_policy;
};

struct dna2_allowed_match_result {
    uint32_t forward_mismatch_mask, reverse_mismatch_mask;
    uint8_t strand_hits;
    bool palindrome;
};

bool dna2_normalize_motif32_exact(uint64_t packed, uint8_t length,
    uint8_t max_mismatches, uint16_t motif_id, motif32_exact *output) noexcept;
bool dna2_valid_motif32_exact(const motif32_exact &motif) noexcept;

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE uint8_t motif_mismatches(
    uint64_t word, motif32_exact motif) noexcept {
    const uint64_t x = word ^ motif.packed;
    const uint64_t different = (x | (x >> 1)) & motif.active_fields;
#if defined(__CUDA_ARCH__)
    return uint8_t(__popcll(different));
#else
    return uint8_t(__builtin_popcountll(different));
#endif
}

// The view and motif must have passed their admission checks. A start past the
// final full window or a window with any invalid base does not match.
CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE bool match_exact_at(
    tensor::sequence::BitPlanesView input, uint32_t anchor, motif32_exact motif) noexcept {
    namespace ts = tensor::sequence;
    if (!ts::valid_window(input, anchor, motif.length)) return false;
    const uint32_t n = ts::word_count32(input.base_count), active = ts::low_mask32(motif.length);
    uint32_t lo, hi;
    ts::split_word(motif.packed, &lo, &hi);
    const uint32_t dl = ts::plane_window(input.low, n, anchor) ^ lo;
    const uint32_t dh = ts::plane_window(input.high, n, anchor) ^ hi;
#if defined(__CUDA_ARCH__)
    return __popc((dl | dh) & active) <= motif.max_mismatches;
#else
    return __builtin_popcount((dl | dh) & active) <= motif.max_mismatches;
#endif
}

// Each allowed_sets entry is a nonempty four-bit A/C/G/T subset.
bool dna2_make_motif32_allowed(const uint8_t *allowed_sets, uint8_t length,
    uint8_t max_mismatches, uint16_t motif_id, dna2_strand_mode strand_mode,
    dna2_palindrome_policy palindrome_policy, motif32_allowed *output) noexcept;
bool dna2_normalize_motif32_allowed(uint32_t allow_A, uint32_t allow_C,
    uint32_t allow_G, uint32_t allow_T, uint8_t length, uint8_t max_mismatches,
    uint16_t motif_id, dna2_strand_mode strand_mode, dna2_palindrome_policy palindrome_policy,
    motif32_allowed *output) noexcept;
bool dna2_valid_motif32_allowed(const motif32_allowed &motif) noexcept;
bool dna2_reverse_complement_allowed(const motif32_allowed &motif, motif32_allowed *output) noexcept;
bool dna2_allowed_is_palindrome(const motif32_allowed &motif) noexcept;

// Invalid active positions consume the same mismatch budget as a disallowed base.
bool dna2_allowed_mismatch_mask(uint32_t low, uint32_t high, uint32_t validity_mask,
    const motif32_allowed &motif, uint32_t *mismatch_mask) noexcept;
bool dna2_match_allowed_window(uint32_t low, uint32_t high, uint32_t validity_mask,
    const motif32_allowed &motif, dna2_allowed_match_result *result) noexcept;

} // namespace cellerator::ops::sequence
