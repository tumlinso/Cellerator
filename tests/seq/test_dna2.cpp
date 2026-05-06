#include <Cellerator/seq/dna2.cuh>

#include "dna2_test_helpers.hh"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CELLERATOR_ENABLE_HIGHWAY
#define CELLERATOR_ENABLE_HIGHWAY 0
#endif

namespace seq = ::cellerator::seq;
namespace seq_test = ::cellerator::seq::test;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint32_t active_mask_ref(int length) {
    if (length <= 0) return 0u;
    if (length >= 32) return 0xffffffffu;
    return (1u << static_cast<unsigned int>(length)) - 1u;
}

std::uint8_t make_base_ref(char c) {
    return (c == 'C' || c == 'c') ? 1u
        : (c == 'G' || c == 'g') ? 2u
        : (c == 'T' || c == 't' || c == 'U' || c == 'u') ? 3u
        : 0u;
}

char base_to_char_ref(std::uint8_t b) {
    switch (b & 0x3u) {
        case 1u: return 'C';
        case 2u: return 'G';
        case 3u: return 'T';
        default: return 'A';
    }
}

std::uint64_t pack_ref(const std::vector<std::uint8_t>& bases) {
    std::uint64_t packed = 0ULL;
    for (int i = 0; i < 32; ++i) {
        packed |= static_cast<std::uint64_t>(bases[static_cast<std::size_t>(i)] & 0x3u) << (2u * static_cast<unsigned int>(i));
    }
    return packed;
}

std::uint8_t get_ref(std::uint64_t packed, int i) {
    return static_cast<std::uint8_t>((packed >> (2u * static_cast<unsigned int>(i))) & 0x3ULL);
}

seq::dna2_planes32 unpack_ref(std::uint64_t packed) {
    seq::dna2_planes32 p{0u, 0u};
    for (int i = 0; i < 32; ++i) {
        const std::uint8_t base = get_ref(packed, i);
        p.lo |= static_cast<std::uint32_t>(base & 0x1u) << i;
        p.hi |= static_cast<std::uint32_t>((base >> 1u) & 0x1u) << i;
    }
    return p;
}

std::uint32_t mismatch_mask_ref(std::uint64_t a, std::uint64_t b, std::uint32_t active_mask) {
    std::uint32_t mask = 0u;
    for (int i = 0; i < 32; ++i) {
        mask |= static_cast<std::uint32_t>(get_ref(a, i) != get_ref(b, i)) << i;
    }
    return mask & active_mask;
}

int popcount_ref(std::uint32_t value) {
    int count = 0;
    while (value != 0u) {
        count += static_cast<int>(value & 1u);
        value >>= 1u;
    }
    return count;
}

std::uint64_t reverse_complement_ref(std::uint64_t packed, int length) {
    const int n = length <= 0 ? 0 : (length > 32 ? 32 : length);
    std::uint64_t out = 0ULL;
    for (int i = 0; i < n; ++i) {
        const std::uint8_t base = get_ref(packed, n - 1 - i) ^ 0x3u;
        out |= static_cast<std::uint64_t>(base) << (2u * static_cast<unsigned int>(i));
    }
    return out;
}

std::vector<std::uint8_t> bases_from_pattern(const std::string& pattern) {
    std::vector<std::uint8_t> bases(32u, 0u);
    for (int i = 0; i < 32; ++i) {
        bases[static_cast<std::size_t>(i)] = make_base_ref(pattern[static_cast<std::size_t>(i % static_cast<int>(pattern.size()))]);
    }
    return bases;
}

void check_case(const std::vector<std::uint8_t>& bases) {
    const std::uint64_t ref_word = pack_ref(bases);
    seq::dna2_word64 word{0ULL};
    for (int i = 0; i < 32; ++i) {
        seq::set_base(word, i, bases[static_cast<std::size_t>(i)]);
        require(seq::get_base(word, i) == bases[static_cast<std::size_t>(i)], "get/set base mismatch");
    }
    require(word.packed == ref_word, "packed word layout mismatch");

    const seq::dna2_planes32 ref_planes = unpack_ref(ref_word);
    const seq::dna2_planes32 planes = seq::unpack_word64_to_planes32(word);
    require(planes.lo == ref_planes.lo && planes.hi == ref_planes.hi, "unpack word64 to planes32 mismatch");
    require(seq::pack_planes32_to_word64(planes).packed == ref_word, "pack planes32 to word64 mismatch");
}

void test_pack_roundtrip() {
    check_case(bases_from_pattern("A"));
    check_case(bases_from_pattern("C"));
    check_case(bases_from_pattern("G"));
    check_case(bases_from_pattern("T"));
    check_case(bases_from_pattern("ACGT"));
    check_case(bases_from_pattern("TGCA"));

    for (int trial = 0; trial < 256; ++trial) {
        check_case(seq_test::random_window32(1337u + static_cast<std::uint32_t>(trial)));
    }
}

void test_planes_word_equivalence_and_masks() {
    const std::uint64_t a = pack_ref(bases_from_pattern("ACGT"));
    const std::uint64_t b = pack_ref(bases_from_pattern("AGGT"));
    const seq::dna2_word64 wa{a};
    const seq::dna2_word64 wb{b};
    const seq::dna2_planes32 pa = seq::unpack_word64_to_planes32(wa);
    const seq::dna2_planes32 pb = seq::unpack_word64_to_planes32(wb);
    const int lengths[] = {1, 2, 15, 16, 31, 32};
    for (int length : lengths) {
        const std::uint32_t active = active_mask_ref(length);
        const std::uint32_t expected_mask = mismatch_mask_ref(a, b, active);
        require(seq::planes32_mismatch_mask(pa, pb, active) == expected_mask, "planes mismatch mask mismatch");
        require(seq::word64_mismatch_mask(wa, wb, active) == expected_mask, "word mismatch mask mismatch");
        require(seq::planes32_mismatches(pa, pb, active) == popcount_ref(expected_mask), "planes mismatch count mismatch");
        require(seq::word64_mismatches(wa, wb, active) == popcount_ref(expected_mask), "word mismatch count mismatch");
        require(seq::planes32_mismatches(pa, pb, active) == seq::word64_mismatches(wa, wb, active),
                "planes/word mismatch counts diverged");
    }
}

void test_exact_and_approximate_match() {
    const seq::dna2_word64 motif{pack_ref(bases_from_pattern("ACGT"))};
    seq::dna2_word64 one = motif;
    seq::dna2_word64 two = motif;
    seq::set_base(one, 3, static_cast<std::uint8_t>(seq::get_base(one, 3) ^ 0x1u));
    seq::set_base(two, 3, static_cast<std::uint8_t>(seq::get_base(two, 3) ^ 0x1u));
    seq::set_base(two, 9, static_cast<std::uint8_t>(seq::get_base(two, 9) ^ 0x2u));
    const std::uint32_t active = active_mask_ref(16);

    const seq::dna2_planes32 motif_p = seq::unpack_word64_to_planes32(motif);
    require(seq::planes32_exact_match(motif_p, motif_p, active), "exact planes match failed");
    require(seq::planes32_mismatches(seq::unpack_word64_to_planes32(one), motif_p, active) == 1,
            "one-mismatch case failed");
    require(seq::word64_mismatches(two, motif, active) == 2, "two-mismatch case failed");
    require(seq::word64_mismatches(one, motif, active) <= 1, "one mismatch should pass threshold 1");
    require(!(seq::word64_mismatches(two, motif, active) <= 1), "two mismatches should fail threshold 1");
    require(seq::word64_mismatches(two, motif, active) <= 2, "two mismatches should pass threshold 2");
}

void test_reverse_complement() {
    const std::uint64_t packed = pack_ref(bases_from_pattern("ACGTTAACCGGT"));
    const seq::dna2_word64 word{packed};
    for (int length = 1; length <= 32; ++length) {
        const std::uint64_t expected = reverse_complement_ref(packed, length);
        const seq::dna2_word64 rc = seq::reverse_complement_word64(word, length);
        require(rc.packed == expected, "reverse-complement word mismatch");
        for (int i = length; i < 32; ++i) {
            require(seq::get_base(rc, i) == 0u, "reverse-complement leaked inactive word bits");
        }
        const seq::dna2_planes32 rc_planes = seq::reverse_complement_planes32(seq::unpack_word64_to_planes32(word), length);
        require(seq::pack_planes32_to_word64(rc_planes).packed == expected, "reverse-complement planes mismatch");
    }

    seq::dna2_word64 acgt{0ULL};
    seq::set_base(acgt, 0, static_cast<std::uint8_t>(seq::dna2_base::A));
    seq::set_base(acgt, 1, static_cast<std::uint8_t>(seq::dna2_base::C));
    seq::set_base(acgt, 2, static_cast<std::uint8_t>(seq::dna2_base::G));
    seq::set_base(acgt, 3, static_cast<std::uint8_t>(seq::dna2_base::T));
    const seq::dna2_word64 rc = seq::reverse_complement_word64(acgt, 4);
    require(seq::get_base(rc, 0) == static_cast<std::uint8_t>(seq::dna2_base::A), "T complement should be A");
    require(seq::get_base(rc, 1) == static_cast<std::uint8_t>(seq::dna2_base::C), "G complement should be C");
    require(seq::get_base(rc, 2) == static_cast<std::uint8_t>(seq::dna2_base::G), "C complement should be G");
    require(seq::get_base(rc, 3) == static_cast<std::uint8_t>(seq::dna2_base::T), "A complement should be T");
}

void test_char_helpers() {
    const char chars[] = {'A', 'C', 'G', 'T', 'a', 'c', 'g', 't', 'U'};
    for (char c : chars) {
        require(seq::make_base(c) == make_base_ref(c), "make_base mismatch");
    }
    for (std::uint8_t b = 0; b < 4; ++b) {
        require(seq::base_to_char(b) == base_to_char_ref(b), "base_to_char mismatch");
    }
}

std::string repeat_pattern(const std::string& pattern, std::size_t n) {
    std::string out(n, 'A');
    for (std::size_t i = 0u; i < n; ++i) {
        out[i] = pattern[i % pattern.size()];
    }
    return out;
}

std::string random_ascii(std::size_t n, std::uint32_t seed) {
    std::string out(n, 'A');
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> base_dist(0, 3);
    const char alphabet[4] = {'A', 'C', 'G', 'T'};
    for (char& c : out) {
        c = alphabet[base_dist(rng)];
    }
    return out;
}

std::uint32_t gc_mask_ref(const std::string& bases, std::size_t n) {
    std::uint32_t mask = 0u;
    for (std::size_t i = 0u; i < n; ++i) {
        mask |= static_cast<std::uint32_t>(bases[i] == 'C' || bases[i] == 'G') << static_cast<unsigned int>(i);
    }
    return mask;
}

std::uint32_t base_mask_ref(const std::string& bases, std::size_t n, char base) {
    std::uint32_t mask = 0u;
    for (std::size_t i = 0u; i < n; ++i) {
        mask |= static_cast<std::uint32_t>(bases[i] == base) << static_cast<unsigned int>(i);
    }
    return mask;
}

std::uint32_t cpg_mask_ref(const std::string& bases, std::size_t n) {
    std::uint32_t mask = 0u;
    if (n == 0u) return mask;
    for (std::size_t i = 0u; i + 1u < n; ++i) {
        mask |= static_cast<std::uint32_t>(bases[i] == 'C' && bases[i + 1u] == 'G') << static_cast<unsigned int>(i);
    }
    return mask;
}

std::uint32_t hamming_ref(const std::string& a, const std::string& b, std::size_t n) {
    std::uint32_t distance = 0u;
    for (std::size_t i = 0u; i < n; ++i) {
        distance += a[i] != b[i] ? 1u : 0u;
    }
    return distance;
}

void check_ascii_case(const std::string& bases, std::size_t n) {
    const seq::dna2_word word = seq::dna2_pack_ascii_32(bases.data(), n);
    std::string unpacked(n, '\0');
    seq::dna2_unpack_ascii_32(word, unpacked.data(), n);
    require(unpacked == bases.substr(0u, n), "ascii pack/unpack roundtrip failed");

    const seq::dna2_planes planes = seq::dna2_to_planes(word);
    require(seq::planes_to_dna2(planes).bits == word.bits, "dna2 planes roundtrip failed");
    require(seq::planes_gc_mask(planes) == gc_mask_ref(bases, n), "GC mask mismatch");
    require((seq::planes_cpg_start_mask(planes) & active_mask_ref(static_cast<int>(n))) == cpg_mask_ref(bases, n),
            "CpG start mask mismatch");

    std::string shifted = bases;
    const char next_base[4] = {'C', 'G', 'T', 'A'};
    for (std::size_t i = 0u; i < n; i += 5u) {
        shifted[i] = next_base[make_base_ref(shifted[i])];
    }
    const seq::dna2_word shifted_word = seq::dna2_pack_ascii_32(shifted.data(), n);
    require(seq::dna2_hamming_distance(word, shifted_word, n) == hamming_ref(bases, shifted, n),
            "hamming distance mismatch");

    const std::uint32_t active = active_mask_ref(static_cast<int>(n));
    require((seq::planes_base_mask(planes, 'A') & active) == base_mask_ref(bases, n, 'A'), "A base mask mismatch");
    require((seq::planes_base_mask(planes, 'C') & active) == base_mask_ref(bases, n, 'C'), "C base mask mismatch");
    require((seq::planes_base_mask(planes, 'G') & active) == base_mask_ref(bases, n, 'G'), "G base mask mismatch");
    require((seq::planes_base_mask(planes, 'T') & active) == base_mask_ref(bases, n, 'T'), "T base mask mismatch");
}

void check_batch_case(const std::string& seed_bases, std::size_t n, std::size_t count) {
    constexpr std::size_t stride = 32u;
    std::vector<char> input(count * stride, 'A');
    std::vector<char> unpacked_scalar(count * stride, '\0');
    std::vector<char> unpacked_backend(count * stride, '\0');
    std::vector<seq::dna2_word> scalar_words(count);
    std::vector<seq::dna2_word> backend_words(count);
    std::vector<seq::dna2_planes> scalar_planes(count);
    std::vector<seq::dna2_planes> backend_planes(count);
    std::vector<std::uint32_t> scalar_masks(count, 0u);
    std::vector<std::uint32_t> backend_masks(count, 0u);

    const char alphabet[4] = {'A', 'C', 'G', 'T'};
    for (std::size_t row = 0u; row < count; ++row) {
        for (std::size_t col = 0u; col < stride; ++col) {
            input[row * stride + col] = seed_bases[col % seed_bases.size()];
        }
        if (n != 0u) {
            input[row * stride + (row % n)] = alphabet[(row + n) & 0x3u];
        }
    }

    seq::dna2_pack_ascii_batch_scalar(input.data(), stride, scalar_words.data(), count, n);
    seq::dna2_pack_ascii_batch(input.data(), stride, backend_words.data(), count, n);
    for (std::size_t i = 0u; i < count; ++i) {
        require(scalar_words[i].bits == backend_words[i].bits, "batch pack backend mismatch");
    }

    seq::dna2_unpack_ascii_batch_scalar(scalar_words.data(), unpacked_scalar.data(), stride, count, n);
    seq::dna2_unpack_ascii_batch(backend_words.data(), unpacked_backend.data(), stride, count, n);
    for (std::size_t row = 0u; row < count; ++row) {
        for (std::size_t col = 0u; col < n; ++col) {
            require(unpacked_scalar[row * stride + col] == input[row * stride + col], "scalar batch unpack roundtrip failed");
            require(unpacked_backend[row * stride + col] == input[row * stride + col], "backend batch unpack roundtrip failed");
        }
    }

    seq::dna2_to_planes_batch_scalar(scalar_words.data(), scalar_planes.data(), count);
    seq::dna2_to_planes_batch(backend_words.data(), backend_planes.data(), count);
    for (std::size_t i = 0u; i < count; ++i) {
        require(scalar_planes[i].lo == backend_planes[i].lo && scalar_planes[i].hi == backend_planes[i].hi,
                "batch planes backend mismatch");
        require(seq::planes_to_dna2(backend_planes[i]).bits == backend_words[i].bits, "batch planes roundtrip failed");
        require(seq::dna2_hamming_distance(scalar_words[i], backend_words[i], n) == 0u, "batch hamming mismatch");
    }

    seq::planes_gc_mask_batch_scalar(scalar_planes.data(), scalar_masks.data(), count);
    seq::planes_gc_mask_batch(backend_planes.data(), backend_masks.data(), count);
    for (std::size_t i = 0u; i < count; ++i) {
        require(scalar_masks[i] == backend_masks[i], "batch GC mask backend mismatch");
    }

    seq::planes_cpg_start_mask_batch_scalar(scalar_planes.data(), scalar_masks.data(), count);
    seq::planes_cpg_start_mask_batch(backend_planes.data(), backend_masks.data(), count);
    for (std::size_t i = 0u; i < count; ++i) {
        require(scalar_masks[i] == backend_masks[i], "batch CpG mask backend mismatch");
    }
}

void test_ascii_bit_primitives_and_batches() {
    const std::size_t batch_sizes[] = {1u, 2u, 3u, 7u, 32u, 1024u};
    const std::string cases[] = {
        repeat_pattern("A", 32u),
        repeat_pattern("C", 32u),
        repeat_pattern("G", 32u),
        repeat_pattern("T", 32u),
        repeat_pattern("AC", 32u),
        repeat_pattern("GT", 32u),
        repeat_pattern("ACGT", 32u),
        random_ascii(32u, 90125u),
        random_ascii(32u, 90126u),
    };
    for (const std::string& test_case : cases) {
        for (std::size_t n = 0u; n <= 32u; ++n) {
            check_ascii_case(test_case, n);
            for (std::size_t batch_size : batch_sizes) {
                check_batch_case(test_case, n, batch_size);
            }
        }
    }
}

} // namespace

int main() {
    try {
        test_char_helpers();
        test_pack_roundtrip();
        test_planes_word_equivalence_and_masks();
        test_exact_and_approximate_match();
        test_reverse_complement();
        test_ascii_bit_primitives_and_batches();
    } catch (const std::exception& e) {
        std::cerr << "test_dna2 failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
