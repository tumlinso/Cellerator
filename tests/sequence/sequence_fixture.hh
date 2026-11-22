#pragma once

#include <Cellerator/layout/sequence/dna2.hh>
#include <Cellerator/ops/sequence/dna2.hh>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace sequence_test {

namespace layout = cellerator::layout::sequence;
namespace seq = cellerator::ops::sequence;
namespace tensor = cellerator::tensor::sequence;

inline void require(bool ok, const char *message) {
    if (ok) return;
    fprintf(stderr, "%s\n", message);
    exit(1);
}

inline int code(char c) {
    return c == 'A' || c == 'a' ? 0 : c == 'C' || c == 'c' ? 1 :
           c == 'G' || c == 'g' ? 2 : c == 'T' || c == 't' || c == 'U' || c == 'u' ? 3 : -1;
}

inline constexpr uint32_t base_count = 70, word_count = 3;

struct Fixture {
    char bases[base_count];
    uint64_t words[word_count], validity[2];
    uint32_t low[word_count], high[word_count], valid[word_count];
    tensor::PackedView packed{};
    seq::motif32_exact motif{};
};

inline void make_fixture(Fixture *f) {
    for (uint32_t i = 0; i < base_count; ++i) f->bases[i] = 'T';
    f->bases[1] = 'u';
    f->bases[4] = 'c'; f->bases[5] = 'G';
    f->bases[30] = 'A'; f->bases[31] = 'C'; f->bases[32] = 'G';
    f->bases[45] = 'A'; f->bases[46] = 'N'; f->bases[47] = 'G';
    f->bases[62] = 'A'; f->bases[63] = 'C'; f->bases[64] = 'G';
    f->bases[68] = 'C';
    require(layout::pack_ascii(f->bases, base_count, f->words, word_count,
                              f->validity, 2u, &f->packed) == layout::Status::ok, "pack reference fixture");
    require(layout::pack_planes(f->packed, {f->low, f->high, f->valid, word_count})
                == layout::Status::ok, "pack reference planes");
    // Two-bit A/C/G = 0/1/2, least significant field first.
    require(seq::dna2_normalize_motif32_exact(0x24u, 3u, 1u, 17u, &f->motif), "normalize fixture motif");
}

inline int mismatches(const Fixture &f, uint32_t start) {
    int mismatch = 0;
    for (uint32_t i = 0; i < 3u; ++i) {
        const int b = code(f.bases[start + i]);
        if (b < 0) return -1;
        mismatch += b != int(i);
    }
    return mismatch;
}

inline uint32_t oracle_hits(const Fixture &f, uint32_t *positions) {
    uint32_t count = 0;
    for (uint32_t i = 0; i + 3u <= base_count; ++i) {
        const int d = mismatches(f, i);
        if (d >= 0 && d <= 1) positions[count++] = i;
    }
    return count;
}

inline void oracle_masks(const Fixture &f, uint32_t *cytosine, uint32_t *gc, uint32_t *cpg) {
    for (uint32_t i = 0; i < word_count; ++i) cytosine[i] = gc[i] = cpg[i] = 0u;
    for (uint32_t i = 0; i < base_count; ++i) {
        const int b = code(f.bases[i]);
        const uint32_t word = i / 32u, bit = 1u << (i % 32u);
        if (b == 1) cytosine[word] |= bit;
        if (b == 1 || b == 2) gc[word] |= bit;
        if (b == 1 && i + 1u < base_count && code(f.bases[i + 1u]) == 2) cpg[word] |= bit;
    }
}

inline void check_counts(seq::sequence_emit_counts counts, uint32_t expected, uint32_t capacity) {
    const uint32_t n = capacity < expected ? capacity : expected;
    require(seq::sequence_emit_counts_valid(seq::sequence_output_mode::stable_emit, capacity, counts)
        && counts.total_matches == expected && counts.required_capacity == expected
        && counts.stored_records == n && counts.dropped_records == expected - n,
        "total, stored, dropped, or required motif capacity");
}

} // namespace sequence_test
