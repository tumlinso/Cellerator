#include <Cellerator/sequence/motif.hh>

namespace cellerator::sequence {
namespace {

bool valid_motif_length(uint8_t length, uint8_t max_mismatches) noexcept {
    return length && length <= 32u && max_mismatches <= length;
}

bool valid_allowed_policy(dna2_strand_mode strand, dna2_palindrome_policy palindrome) noexcept {
    return (strand == dna2_strand_mode::forward || strand == dna2_strand_mode::reverse
            || strand == dna2_strand_mode::both)
        && (palindrome == dna2_palindrome_policy::deduplicate
            || palindrome == dna2_palindrome_policy::report_both);
}

uint64_t active_fields(uint8_t length) noexcept {
    const uint64_t mask = length == 32u ? UINT64_MAX : (uint64_t(1) << (2u*length)) - 1u;
    return mask & 0x5555555555555555ULL;
}

uint32_t reverse_active_bits(uint32_t mask, uint8_t length) noexcept {
    uint32_t reversed = 0;
    for (uint8_t i = 0; i < length; ++i) reversed = (reversed << 1) | ((mask >> i) & 1u);
    return reversed;
}

motif32_allowed reverse_complement(const motif32_allowed &motif) noexcept {
    const auto n = motif.length;
    return {reverse_active_bits(motif.allow_T, n), reverse_active_bits(motif.allow_G, n),
        reverse_active_bits(motif.allow_C, n), reverse_active_bits(motif.allow_A, n),
        motif.active_mask, n, motif.max_mismatches, motif.motif_id, motif.strand_mode,
        motif.palindrome_policy};
}

bool same_allowed_bases(const motif32_allowed &a, const motif32_allowed &b) noexcept {
    return a.allow_A == b.allow_A && a.allow_C == b.allow_C
        && a.allow_G == b.allow_G && a.allow_T == b.allow_T;
}

uint32_t allowed_mismatches(uint32_t lo, uint32_t hi, uint32_t valid,
    const motif32_allowed &motif) noexcept {
    const uint32_t accepted = (~lo & ~hi & motif.allow_A) | (lo & ~hi & motif.allow_C)
        | (~lo & hi & motif.allow_G) | (lo & hi & motif.allow_T);
    return motif.active_mask & ~(accepted & valid);
}

} // namespace

bool dna2_normalize_motif32_exact(uint64_t packed, uint8_t length,
    uint8_t max_mismatches, uint16_t motif_id, motif32_exact *output) noexcept {
    if (!output || !valid_motif_length(length, max_mismatches)) return false;
    const uint64_t fields = active_fields(length);
    *output = {packed & (fields | (fields << 1)), fields, length, max_mismatches, motif_id};
    return true;
}

bool dna2_valid_motif32_exact(const motif32_exact &motif) noexcept {
    if (!valid_motif_length(motif.length, motif.max_mismatches)) return false;
    const uint64_t fields = active_fields(motif.length);
    return motif.active_fields == fields && (motif.packed & ~(fields | (fields << 1))) == 0u;
}

bool dna2_make_motif32_allowed(const uint8_t *allowed_sets, uint8_t length,
    uint8_t max_mismatches, uint16_t motif_id, dna2_strand_mode strand_mode,
    dna2_palindrome_policy palindrome_policy, motif32_allowed *output) noexcept {
    if (!allowed_sets || !output || !valid_motif_length(length, max_mismatches)) return false;
    uint32_t allowed[4]{};
    for (uint8_t i = 0; i < length; ++i) {
        const uint8_t set = allowed_sets[i];
        if (!set || (set & 0xf0u)) return false;
        for (uint8_t b = 0; b < 4u; ++b) allowed[b] |= uint32_t((set >> b) & 1u) << i;
    }
    return dna2_normalize_motif32_allowed(allowed[0], allowed[1], allowed[2],
        allowed[3], length, max_mismatches, motif_id, strand_mode, palindrome_policy, output);
}

bool dna2_normalize_motif32_allowed(uint32_t allow_A, uint32_t allow_C,
    uint32_t allow_G, uint32_t allow_T, uint8_t length, uint8_t max_mismatches,
    uint16_t motif_id, dna2_strand_mode strand_mode, dna2_palindrome_policy palindrome_policy,
    motif32_allowed *output) noexcept {
    if (!output || !valid_motif_length(length, max_mismatches)
        || !valid_allowed_policy(strand_mode, palindrome_policy)) return false;
    const uint32_t active = low_mask32(length);
    if (((allow_A | allow_C | allow_G | allow_T) & active) != active) return false;
    *output = {allow_A & active, allow_C & active, allow_G & active, allow_T & active,
        active, length, max_mismatches, motif_id, strand_mode, palindrome_policy};
    return true;
}

bool dna2_valid_motif32_allowed(const motif32_allowed &motif) noexcept {
    if (!valid_motif_length(motif.length, motif.max_mismatches)
        || !valid_allowed_policy(motif.strand_mode, motif.palindrome_policy)) return false;
    const uint32_t active = low_mask32(motif.length);
    return motif.active_mask == active
        && (motif.allow_A | motif.allow_C | motif.allow_G | motif.allow_T) == active;
}

bool dna2_reverse_complement_allowed(const motif32_allowed &motif, motif32_allowed *output) noexcept {
    if (!output || !dna2_valid_motif32_allowed(motif)) return false;
    *output = reverse_complement(motif);
    return true;
}

bool dna2_allowed_is_palindrome(const motif32_allowed &motif) noexcept {
    return dna2_valid_motif32_allowed(motif) && same_allowed_bases(motif, reverse_complement(motif));
}

bool dna2_allowed_mismatch_mask(uint32_t low, uint32_t high, uint32_t validity_mask,
    const motif32_allowed &motif, uint32_t *mismatch_mask) noexcept {
    if (!mismatch_mask || !dna2_valid_motif32_allowed(motif)) return false;
    *mismatch_mask = allowed_mismatches(low, high, validity_mask, motif);
    return true;
}

bool dna2_match_allowed_window(uint32_t low, uint32_t high, uint32_t validity_mask,
    const motif32_allowed &motif, dna2_allowed_match_result *result) noexcept {
    if (!result || !dna2_valid_motif32_allowed(motif)) return false;

    const motif32_allowed reverse = reverse_complement(motif);
    dna2_allowed_match_result match{};
    if (motif.strand_mode != dna2_strand_mode::reverse) {
        match.forward_mismatch_mask = allowed_mismatches(low, high, validity_mask, motif);
        if (__builtin_popcount(match.forward_mismatch_mask) <= motif.max_mismatches)
            match.strand_hits |= dna2_forward_strand_bit;
    }
    if (motif.strand_mode != dna2_strand_mode::forward) {
        match.reverse_mismatch_mask = allowed_mismatches(low, high, validity_mask, reverse);
        if (__builtin_popcount(match.reverse_mismatch_mask) <= motif.max_mismatches)
            match.strand_hits |= dna2_reverse_strand_bit;
    }
    match.palindrome = same_allowed_bases(motif, reverse);
    if (motif.strand_mode == dna2_strand_mode::both && match.palindrome && match.strand_hits
        && motif.palindrome_policy == dna2_palindrome_policy::deduplicate)
        match.strand_hits = dna2_forward_strand_bit;
    *result = match;
    return true;
}

} // namespace cellerator::sequence
