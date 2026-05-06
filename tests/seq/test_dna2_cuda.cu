#include <Cellerator/seq/dna2.cuh>

#include "dna2_test_helpers.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace seq = ::cellerator::seq;
namespace seq_test = ::cellerator::seq::test;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_require(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
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

std::uint8_t get_ref(std::uint64_t packed, int i) {
    return static_cast<std::uint8_t>((packed >> (2u * static_cast<unsigned int>(i))) & 0x3ULL);
}

void set_ref(std::uint64_t& packed, int i, std::uint8_t base) {
    const unsigned int shift = 2u * static_cast<unsigned int>(i);
    packed = (packed & ~(0x3ULL << shift)) | (static_cast<std::uint64_t>(base & 0x3u) << shift);
}

std::uint64_t pack_ref(const std::vector<std::uint8_t>& bases, int offset = 0) {
    std::uint64_t packed = 0ULL;
    for (int i = 0; i < 32; ++i) {
        const int source = offset + i;
        const std::uint8_t base = source >= 0 && source < static_cast<int>(bases.size())
            ? bases[static_cast<std::size_t>(source)] : 0u;
        set_ref(packed, i, base);
    }
    return packed;
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
        set_ref(out, i, static_cast<std::uint8_t>(get_ref(packed, n - 1 - i) ^ 0x3u));
    }
    return out;
}

std::vector<std::uint8_t> encode_string(const std::string& text) {
    std::vector<std::uint8_t> bases(text.size(), 0u);
    for (std::size_t i = 0; i < text.size(); ++i) bases[i] = make_base_ref(text[i]);
    return bases;
}

std::vector<std::uint8_t> bases_from_pattern(const std::string& pattern) {
    std::vector<std::uint8_t> bases(32u, 0u);
    for (int i = 0; i < 32; ++i) {
        bases[static_cast<std::size_t>(i)] = make_base_ref(pattern[static_cast<std::size_t>(i % static_cast<int>(pattern.size()))]);
    }
    return bases;
}

std::vector<std::uint64_t> pack_sequence_ref(const std::vector<std::uint8_t>& bases) {
    const std::size_t words = (bases.size() + 31u) / 32u;
    std::vector<std::uint64_t> packed(words + 1u, 0ULL);
    for (std::size_t i = 0; i < bases.size(); ++i) {
        set_ref(packed[i >> 5u], static_cast<int>(i & 31u), bases[i]);
    }
    return packed;
}

std::vector<std::uint8_t> scan_ref(
    const std::vector<std::uint8_t>& seq_bases,
    const std::vector<std::uint8_t>& motif,
    int max_mismatches) {
    std::vector<std::uint8_t> hits(seq_bases.size(), 0u);
    const int windows = static_cast<int>(seq_bases.size()) >= static_cast<int>(motif.size())
        ? static_cast<int>(seq_bases.size() - motif.size() + 1u) : 0;
    for (int start = 0; start < windows; ++start) {
        int mismatches = 0;
        for (int i = 0; i < static_cast<int>(motif.size()); ++i) {
            mismatches += seq_bases[static_cast<std::size_t>(start + i)] != motif[static_cast<std::size_t>(i)] ? 1 : 0;
        }
        hits[static_cast<std::size_t>(start)] = static_cast<std::uint8_t>(mismatches <= max_mismatches ? 1u : 0u);
    }
    return hits;
}

std::uint64_t count_hits_ref(
    const std::vector<std::uint8_t>& seq_bases,
    const std::vector<std::uint8_t>& motif,
    int max_mismatches) {
    const std::vector<std::uint8_t> hits = scan_ref(seq_bases, motif, max_mismatches);
    std::uint64_t count = 0ULL;
    for (std::uint8_t hit : hits) count += hit != 0u ? 1ULL : 0ULL;
    return count;
}

struct helper_case {
    std::uint64_t a;
    std::uint64_t b;
    std::uint32_t active;
    int length;
    int slot;
    std::uint8_t set_value;
    char input_char;
    std::uint8_t char_base;
};

struct helper_result {
    std::uint8_t made_base;
    char base_char;
    std::uint8_t got_base;
    std::uint64_t set_packed;
    std::uint32_t planes_lo;
    std::uint32_t planes_hi;
    std::uint64_t repacked;
    std::uint32_t planes_mask;
    int planes_mismatches;
    std::uint8_t planes_exact;
    std::uint32_t word_mask;
    int word_mismatches;
    std::uint64_t rc_word;
    std::uint32_t rc_planes_lo;
    std::uint32_t rc_planes_hi;
};

__global__ void helper_kernel(const helper_case* cases, helper_result* results, int count) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= count) return;

    const helper_case c = cases[idx];
    const seq::dna2_word64 a{c.a};
    const seq::dna2_word64 b{c.b};
    const seq::dna2_planes32 ap = seq::unpack_word64_to_planes32(a);
    const seq::dna2_planes32 bp = seq::unpack_word64_to_planes32(b);
    seq::dna2_word64 set_word = a;
    seq::set_base(set_word, c.slot, c.set_value);
    const seq::dna2_word64 rc_word = seq::reverse_complement_word64(a, c.length);
    const seq::dna2_planes32 rc_planes = seq::reverse_complement_planes32(ap, c.length);

    helper_result r{};
    r.made_base = seq::make_base(c.input_char);
    r.base_char = seq::base_to_char(c.char_base);
    r.got_base = seq::get_base(a, c.slot);
    r.set_packed = set_word.packed;
    r.planes_lo = ap.lo;
    r.planes_hi = ap.hi;
    r.repacked = seq::pack_planes32_to_word64(ap).packed;
    r.planes_mask = seq::planes32_mismatch_mask(ap, bp, c.active);
    r.planes_mismatches = seq::planes32_mismatches(ap, bp, c.active);
    r.planes_exact = static_cast<std::uint8_t>(seq::planes32_exact_match(ap, bp, c.active) ? 1u : 0u);
    r.word_mask = seq::word64_mismatch_mask(a, b, c.active);
    r.word_mismatches = seq::word64_mismatches(a, b, c.active);
    r.rc_word = rc_word.packed;
    r.rc_planes_lo = rc_planes.lo;
    r.rc_planes_hi = rc_planes.hi;
    results[idx] = r;
}

__global__ void warp_encode_kernel(const std::uint8_t* bases, seq::dna2_planes32* out) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const seq::dna2_planes32 planes = seq::warp_encode_base_lanes(bases[lane] & 0x3u);
    if (lane == 0) *out = planes;
}

void test_device_helpers() {
    const char chars[] = {'A', 'C', 'G', 'T', 'a', 'c', 'g', 't', 'U', 'N'};

    std::vector<helper_case> cases(96u);
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const std::vector<std::uint8_t> a_bases = seq_test::random_window32(2026u + static_cast<std::uint32_t>(i));
        const std::vector<std::uint8_t> b_bases = seq_test::random_window32(9090u + static_cast<std::uint32_t>(i));
        const int length = 1 + static_cast<int>((i * 17u) % 32u);
        cases[i] = helper_case{
            pack_ref(a_bases),
            pack_ref(b_bases),
            active_mask_ref(length),
            length,
            static_cast<int>((i * 7u) % 32u),
            static_cast<std::uint8_t>((i * 5u) & 0x3u),
            chars[i % (sizeof(chars) / sizeof(chars[0]))],
            static_cast<std::uint8_t>((i * 3u) & 0x3u)
        };
    }

    helper_case* d_cases = nullptr;
    helper_result* d_results = nullptr;
    std::vector<helper_result> results(cases.size());
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_cases), cases.size() * sizeof(helper_case)), "cudaMalloc cases");
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_results), results.size() * sizeof(helper_result)), "cudaMalloc results");
    cuda_require(cudaMemcpy(d_cases, cases.data(), cases.size() * sizeof(helper_case), cudaMemcpyHostToDevice), "copy cases");
    helper_kernel<<<static_cast<int>((cases.size() + 127u) / 128u), 128>>>(d_cases, d_results, static_cast<int>(cases.size()));
    cuda_require(cudaGetLastError(), "helper kernel launch");
    cuda_require(cudaDeviceSynchronize(), "helper kernel sync");
    cuda_require(cudaMemcpy(results.data(), d_results, results.size() * sizeof(helper_result), cudaMemcpyDeviceToHost), "copy results");
    cudaFree(d_cases);
    cudaFree(d_results);

    for (std::size_t i = 0; i < cases.size(); ++i) {
        const helper_case& c = cases[i];
        const helper_result& r = results[i];
        std::uint64_t set_packed = c.a;
        set_ref(set_packed, c.slot, c.set_value);
        const seq::dna2_planes32 ap = unpack_ref(c.a);
        const seq::dna2_planes32 rc_planes = unpack_ref(reverse_complement_ref(c.a, c.length));
        const std::uint32_t expected_mask = mismatch_mask_ref(c.a, c.b, c.active);

        require(r.made_base == make_base_ref(c.input_char), "device make_base mismatch");
        require(r.base_char == base_to_char_ref(c.char_base), "device base_to_char mismatch");
        require(r.got_base == get_ref(c.a, c.slot), "device get_base mismatch");
        require(r.set_packed == set_packed, "device set_base mismatch");
        require(r.planes_lo == ap.lo && r.planes_hi == ap.hi, "device unpack mismatch");
        require(r.repacked == c.a, "device repack mismatch");
        require(r.planes_mask == expected_mask, "device planes mismatch mask mismatch");
        require(r.planes_mismatches == popcount_ref(expected_mask), "device planes mismatch count mismatch");
        require(r.planes_exact == static_cast<std::uint8_t>(expected_mask == 0u ? 1u : 0u), "device exact match mismatch");
        require(r.word_mask == expected_mask, "device word mismatch mask mismatch");
        require(r.word_mismatches == popcount_ref(expected_mask), "device word mismatch count mismatch");
        require(r.rc_word == reverse_complement_ref(c.a, c.length), "device reverse-complement word mismatch");
        require(r.rc_planes_lo == rc_planes.lo && r.rc_planes_hi == rc_planes.hi,
                "device reverse-complement planes mismatch");
    }
}

void test_warp_encoder() {
    const std::vector<std::uint8_t> bases = encode_string("ACGTTGCAAAAACCCCGGGGTTTTACGTACGT");
    const seq::dna2_planes32 expected = unpack_ref(pack_ref(bases));
    std::uint8_t* d_bases = nullptr;
    seq::dna2_planes32* d_out = nullptr;
    seq::dna2_planes32 out{};
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_bases), 32u * sizeof(std::uint8_t)), "cudaMalloc warp bases");
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(seq::dna2_planes32)), "cudaMalloc warp out");
    cuda_require(cudaMemcpy(d_bases, bases.data(), 32u * sizeof(std::uint8_t), cudaMemcpyHostToDevice), "copy warp bases");
    warp_encode_kernel<<<1, 32>>>(d_bases, d_out);
    cuda_require(cudaGetLastError(), "warp encode launch");
    cuda_require(cudaDeviceSynchronize(), "warp encode sync");
    cuda_require(cudaMemcpy(&out, d_out, sizeof(seq::dna2_planes32), cudaMemcpyDeviceToHost), "copy warp out");
    cudaFree(d_bases);
    cudaFree(d_out);
    require(out.lo == expected.lo && out.hi == expected.hi, "warp encoder planes mismatch");
}

void test_warp_word64_to_planes_conversion() {
    std::vector<std::uint64_t> packed;
    packed.reserve(134u);
    packed.push_back(pack_ref(bases_from_pattern("A")));
    packed.push_back(pack_ref(bases_from_pattern("C")));
    packed.push_back(pack_ref(bases_from_pattern("G")));
    packed.push_back(pack_ref(bases_from_pattern("T")));
    packed.push_back(pack_ref(bases_from_pattern("ACGT")));
    packed.push_back(pack_ref(bases_from_pattern("TGCA")));
    for (int trial = 0; trial < 128; ++trial) {
        packed.push_back(pack_ref(seq_test::random_window32(5150u + static_cast<std::uint32_t>(trial))));
    }

    std::vector<seq::dna2_planes32> planes(packed.size());
    std::uint64_t* d_packed = nullptr;
    seq::dna2_planes32* d_planes = nullptr;
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_packed), packed.size() * sizeof(std::uint64_t)),
                 "cudaMalloc conversion packed words");
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_planes), planes.size() * sizeof(seq::dna2_planes32)),
                 "cudaMalloc conversion planes");
    cuda_require(cudaMemcpy(d_packed, packed.data(), packed.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice),
                 "copy conversion packed words");

    const int threads = 128;
    const int blocks = (static_cast<int>(packed.size()) * 32 + threads - 1) / threads;
    seq::convert_word64_to_planes32_warp<<<blocks, threads>>>(
        d_packed, d_planes, static_cast<int>(packed.size()));
    cuda_require(cudaGetLastError(), "warp word64 conversion launch");
    cuda_require(cudaDeviceSynchronize(), "warp word64 conversion sync");
    cuda_require(cudaMemcpy(planes.data(), d_planes, planes.size() * sizeof(seq::dna2_planes32), cudaMemcpyDeviceToHost),
                 "copy conversion planes");
    cudaFree(d_packed);
    cudaFree(d_planes);

    for (std::size_t i = 0u; i < packed.size(); ++i) {
        const seq::dna2_planes32 expected = unpack_ref(packed[i]);
        require(planes[i].lo == expected.lo && planes[i].hi == expected.hi,
                "warp word64 conversion planes mismatch");
    }
}

void run_warp_scan_case(const std::vector<std::uint8_t>& sequence, const std::vector<std::uint8_t>& motif, int max_mismatches) {
    const std::vector<std::uint8_t> expected = scan_ref(sequence, motif, max_mismatches);
    const seq::dna2_planes32 motif_planes = unpack_ref(pack_ref(motif));
    std::uint8_t* d_seq = nullptr;
    std::uint8_t* d_hits = nullptr;
    std::vector<std::uint8_t> hits(sequence.size(), 0u);
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_seq), sequence.size() * sizeof(std::uint8_t)), "cudaMalloc scan seq");
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_hits), sequence.size() * sizeof(std::uint8_t)), "cudaMalloc scan hits");
    cuda_require(cudaMemcpy(d_seq, sequence.data(), sequence.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice), "copy scan seq");
    cuda_require(cudaMemset(d_hits, 0, sequence.size() * sizeof(std::uint8_t)), "clear scan hits");

    const int windows = static_cast<int>(sequence.size() - motif.size() + 1u);
    const int threads = 128;
    const int blocks = (windows * 32 + threads - 1) / threads;
    seq::scan_motif_warp32_unpacked<<<blocks, threads>>>(
        d_seq, static_cast<int>(sequence.size()), motif_planes, static_cast<int>(motif.size()), max_mismatches, d_hits);
    cuda_require(cudaGetLastError(), "warp scan launch");
    cuda_require(cudaDeviceSynchronize(), "warp scan sync");
    cuda_require(cudaMemcpy(hits.data(), d_hits, hits.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost), "copy warp hits");
    cudaFree(d_seq);
    cudaFree(d_hits);
    require(hits == expected, "warp motif scan did not match CPU reference");
}

void run_word_scan_case(const std::vector<std::uint8_t>& sequence, const std::vector<std::uint8_t>& motif, int max_mismatches) {
    const std::vector<std::uint8_t> expected = scan_ref(sequence, motif, max_mismatches);
    const std::vector<std::uint64_t> packed = pack_sequence_ref(sequence);
    const seq::dna2_word64 motif_word{pack_ref(motif)};
    std::uint64_t* d_seq = nullptr;
    std::uint8_t* d_hits = nullptr;
    std::vector<std::uint8_t> hits(sequence.size(), 0u);
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_seq), packed.size() * sizeof(std::uint64_t)), "cudaMalloc packed seq");
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_hits), sequence.size() * sizeof(std::uint8_t)), "cudaMalloc packed hits");
    cuda_require(cudaMemcpy(d_seq, packed.data(), packed.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice), "copy packed seq");
    cuda_require(cudaMemset(d_hits, 0, sequence.size() * sizeof(std::uint8_t)), "clear packed hits");

    const int windows = static_cast<int>(sequence.size() - motif.size() + 1u);
    const int threads = 128;
    const int blocks = (windows + threads - 1) / threads;
    seq::scan_motif_word64_reference<<<blocks, threads>>>(
        d_seq, static_cast<int>(sequence.size()), motif_word, static_cast<int>(motif.size()), max_mismatches, d_hits);
    cuda_require(cudaGetLastError(), "packed scan launch");
    cuda_require(cudaDeviceSynchronize(), "packed scan sync");
    cuda_require(cudaMemcpy(hits.data(), d_hits, hits.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost), "copy packed hits");
    cudaFree(d_seq);
    cudaFree(d_hits);
    require(hits == expected, "packed motif scan did not match CPU reference");
}

void run_shifted_count_case(const std::vector<std::uint8_t>& sequence, const std::vector<std::uint8_t>& motif, int max_mismatches) {
    const std::uint64_t expected = count_hits_ref(sequence, motif, max_mismatches);
    const std::vector<std::uint64_t> packed = pack_sequence_ref(sequence);
    const seq::dna2_word64 motif_word{pack_ref(motif)};
    std::uint64_t* d_seq = nullptr;
    unsigned long long* d_count = nullptr;
    unsigned long long count = 0ULL;
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_seq), packed.size() * sizeof(std::uint64_t)), "cudaMalloc shifted packed seq");
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_count), sizeof(unsigned long long)), "cudaMalloc shifted count");
    cuda_require(cudaMemcpy(d_seq, packed.data(), packed.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice), "copy shifted packed seq");
    cuda_require(cudaMemset(d_count, 0, sizeof(unsigned long long)), "clear shifted count");

    const int windows = static_cast<int>(sequence.size() - motif.size() + 1u);
    const int threads = 256;
    const int blocks = std::min(4096, (windows + threads - 1) / threads);
    seq::scan_motif_word64_shifted_count<<<blocks, threads, threads * sizeof(unsigned int)>>>(
        d_seq, static_cast<int>(sequence.size()), motif_word, static_cast<int>(motif.size()), max_mismatches, d_count);
    cuda_require(cudaGetLastError(), "shifted count scan launch");
    cuda_require(cudaDeviceSynchronize(), "shifted count scan sync");
    cuda_require(cudaMemcpy(&count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "copy shifted count");
    cudaFree(d_seq);
    cudaFree(d_count);
    require(count == expected, "shifted packed count scan did not match CPU reference");
}

void test_motif_scan() {
    const std::vector<std::uint8_t> sequence = seq_test::random_bases(256u, 44001u);
    run_warp_scan_case(sequence, encode_string("ACGTACGT"), 0);
    run_warp_scan_case(sequence, seq_test::motif_from_sequence(sequence, 41u, 16u), 0);
    std::vector<std::uint8_t> near_motif = seq_test::motif_from_sequence(sequence, 89u, 17u);
    seq_test::force_mismatch(near_motif, 3u, 1u);
    seq_test::force_mismatch(near_motif, 11u, 2u);
    run_warp_scan_case(sequence, near_motif, 2);
    run_word_scan_case(sequence, encode_string("ACGTACGT"), 0);
    run_word_scan_case(sequence, seq_test::motif_from_sequence(sequence, 41u, 16u), 0);
    run_word_scan_case(sequence, near_motif, 2);
    run_shifted_count_case(sequence, encode_string("ACGTACGT"), 0);
    run_shifted_count_case(sequence, seq_test::motif_from_sequence(sequence, 41u, 16u), 0);
    run_shifted_count_case(sequence, near_motif, 2);
    run_shifted_count_case(sequence, seq_test::motif_from_sequence(sequence, 7u, 32u), 1);
}

} // namespace

int main() {
    try {
        int device_count = 0;
        cuda_require(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        require(device_count > 0, "test_dna2_cuda requires a CUDA device");
        test_device_helpers();
        test_warp_encoder();
        test_warp_word64_to_planes_conversion();
        test_motif_scan();
    } catch (const std::exception& e) {
        std::cerr << "test_dna2_cuda failed: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
