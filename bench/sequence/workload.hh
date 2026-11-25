#pragma once

#include <Cellerator/sequence/dna2.hh>
#include <Cellerator/sequence/operations.hh>
#include <Cellerator/sequence/predicate_plan.hh>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

namespace sequence_bench {

namespace seq = cellerator::sequence;

inline constexpr char alphabet[] = "ACGT";
inline constexpr char motif_text[] = "ACGTTGCAAGTCTCGA";
inline constexpr uint32_t motif_size = sizeof(motif_text) - 1u;
inline constexpr uint32_t seed = 20260819u;

struct Settings {
    uint32_t bases = 1u << 20u, warmup = 3u, repeats = 20u;
};

struct Input {
    char *bases;
    uint64_t *words;
    uint64_t *validity;
    uint32_t word_count, validity_count;
    uint64_t expected;
};

inline void require(bool ok, const char *message) {
    if (ok) return;
    fputs(message, stderr); fputc('\n', stderr);
    exit(1);
}

inline uint32_t parse(const char *arg) {
    require(*arg >= '0' && *arg <= '9', "expected unsigned decimal argument");
    char *end;
    errno = 0;
    const auto n = strtoull(arg, &end, 10);
    require(errno != ERANGE && !*end && n <= INT32_MAX, "argument outside 32-bit count range");
    return uint32_t(n);
}

inline Settings settings(int argc, char **argv) {
    require(argc <= 4, "usage: sequence_bench [bases [warmup [repeats]]]");
    Settings s{};
    if (argc > 1) s.bases = parse(argv[1]);
    if (argc > 2) s.warmup = parse(argv[2]);
    if (argc > 3) s.repeats = parse(argv[3]);
    require(s.bases >= motif_size && s.repeats > 0u, "motif requires 16 bases and at least one repeat");
    return s;
}

inline double now_ms() {
    timespec t;
    require(clock_gettime(CLOCK_MONOTONIC, &t) == 0, "read monotonic clock");
    return 1000.0*double(t.tv_sec) + 1.0e-6*double(t.tv_nsec);
}

inline Input make_input(uint32_t n) {
    const uint32_t words = (n + 31u)/32u, validity = (n + 63u)/64u;
    auto *bases = static_cast<char *>(malloc(n));
    auto *packed = static_cast<uint64_t *>(malloc((size_t(words) + validity)*sizeof(uint64_t)));
    require(bases && packed, "allocate benchmark input");
    uint32_t x = seed;
    for (uint32_t i = 0; i < n; ++i) {
        x ^= x << 13u; x ^= x >> 17u; x ^= x << 5u;
        bases[i] = alphabet[x & 3u];
    }
    const uint32_t windows = n - motif_size + 1u;
    for (uint32_t i = 0; i < windows; i += 4096u) __builtin_memcpy(bases + i, motif_text, motif_size);

    uint64_t expected = 0u;
    for (uint32_t i = 0; i < windows; ++i) {
        uint32_t j = 0u;
        while (j < motif_size && bases[i + j] == motif_text[j]) ++j;
        expected += j == motif_size;
    }
    return {bases, packed, packed + words, words, validity, expected};
}

inline seq::PackedView pack(const Input &input, uint32_t n) {
    seq::PackedView view{};
    require(seq::pack_ascii(input.bases, n, input.words, input.word_count,
                               input.validity, input.validity_count, &view)
                == seq::Status::ok, "pack benchmark input");
    return view;
}

inline seq::motif32_exact prepare_predicate() {
    uint64_t packed = 0u;
    for (uint32_t i = 0; i < motif_size; ++i) {
        const char c = motif_text[i];
        const uint64_t b = c == 'A' ? 0u : c == 'C' ? 1u : c == 'G' ? 2u : 3u;
        packed |= b << (2u*i);
    }
    seq::sequence_predicate_program program{};
    program.version = seq::sequence_program_version;
    program.node_count = program.output_count = program.exact_motif_count = 1u;
    require(seq::dna2_normalize_motif32_exact(packed, motif_size, 0u, 17u, &program.exact_motifs[0]),
            "normalize count motif");
    const uint16_t none = seq::sequence_program_no_input;
    program.nodes[0] = {seq::predicate_opcode::exact_motif, seq::predicate_value_kind::mask,
                        0u, 0u, none, none, 0u, 0u};
    program.outputs[0] = {0u, 17u, seq::sequence_output_mode::count, 0u, 0u, 0u};
    seq::prepared_predicate_plan plan{};
    require(seq::prepare_sequence_predicate_program(program, &plan) == seq::predicate_plan_status::ok,
            "prepare count predicate");
    return program.exact_motifs[0];
}

inline void print_result(Settings s, const char *backend, uint64_t matches,
                         double pack_ms, double prepare_ms, double upload_ms, double repeat_ms) {
    printf("{\"backend\":\"%s\",\"bases\":%u,\"windows\":%u,\"motif_bases\":%u,"
           "\"seed\":%u,\"warmup\":%u,\"repeats\":%u,\"matches\":%llu,"
           "\"pack_ms\":%.6f,\"predicate_prepare_ms\":%.6f,\"upload_ms\":%.6f,"
           "\"resident_batch_ms\":%.6f,\"resident_mean_ms\":%.6f}\n",
           backend, s.bases, s.bases - motif_size + 1u, motif_size,
           seed, s.warmup, s.repeats, static_cast<unsigned long long>(matches),
           pack_ms, prepare_ms, upload_ms, repeat_ms, repeat_ms/double(s.repeats));
}

} // namespace sequence_bench
