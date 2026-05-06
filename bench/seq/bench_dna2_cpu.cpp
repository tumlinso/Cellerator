#include <Cellerator/seq/dna2.cuh>

#include <bench/benchmark_mutex.hh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace seq = ::cellerator::seq;

#ifndef CELLERATOR_ENABLE_HIGHWAY
#define CELLERATOR_ENABLE_HIGHWAY 0
#endif

namespace {

template <class Fn>
double time_seconds(Fn&& fn) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

template <class Fn>
void parallel_chunks(std::size_t count, unsigned int requested_threads, Fn&& fn) {
    const unsigned int available = std::thread::hardware_concurrency();
    const unsigned int fallback = available == 0u ? 1u : available;
    const unsigned int thread_count = requested_threads == 0u ? fallback : requested_threads;
    const unsigned int workers = static_cast<unsigned int>(std::min<std::size_t>(count == 0u ? 1u : count, thread_count));
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (unsigned int worker = 0u; worker < workers; ++worker) {
        const std::size_t begin = (count * worker) / workers;
        const std::size_t end = (count * (worker + 1u)) / workers;
        threads.emplace_back([&, begin, end] {
            fn(begin, end);
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
}

std::uint64_t checksum_words(const seq::dna2_word* words, std::size_t count) {
    std::uint64_t sum = 0ULL;
    for (std::size_t i = 0u; i < count; ++i) {
        sum ^= words[i].bits + 0x9e3779b97f4a7c15ULL + (sum << 6u) + (sum >> 2u);
    }
    return sum;
}

std::uint64_t checksum_planes(const seq::dna2_planes* planes, std::size_t count) {
    std::uint64_t sum = 0ULL;
    for (std::size_t i = 0u; i < count; ++i) {
        sum ^= planes[i].lo + 0x9e3779b97f4a7c15ULL + (sum << 6u) + (sum >> 2u);
        sum ^= planes[i].hi + 0xbf58476d1ce4e5b9ULL + (sum << 6u) + (sum >> 2u);
    }
    return sum;
}

std::uint64_t checksum_masks(const std::uint32_t* masks, std::size_t count) {
    std::uint64_t sum = 0ULL;
    for (std::size_t i = 0u; i < count; ++i) {
        sum ^= static_cast<std::uint64_t>(masks[i]) + 0x94d049bb133111ebULL + (sum << 6u) + (sum >> 2u);
    }
    return sum;
}

void report(const char* name, std::size_t count, int iterations, double seconds, std::uint64_t checksum) {
    const double chunks_per_sec = seconds > 0.0 ? (static_cast<double>(count) * iterations) / seconds : 0.0;
    const double bases_per_sec = chunks_per_sec * 32.0;
    std::printf("%s_seconds=%.6f\n", name, seconds);
    std::printf("%s_chunks_per_sec=%.3f\n", name, chunks_per_sec);
    std::printf("%s_bases_per_sec=%.3f\n", name, bases_per_sec);
    std::printf("%s_checksum=%llu\n", name, static_cast<unsigned long long>(checksum));
}

std::uint64_t pack_window_from_ascii(const char* bases, std::size_t offset, int motif_length) {
    seq::dna2_word64 word{0ULL};
    for (int i = 0; i < motif_length; ++i) {
        const std::uint8_t base = seq::make_base(bases[offset + static_cast<std::size_t>(i)]);
        seq::set_base(word, i, base);
    }
    return word.packed;
}

std::uint64_t shifted_window_word64(const seq::dna2_word* words, std::size_t start) {
    const std::size_t word_index = start >> 5u;
    const unsigned int shift = static_cast<unsigned int>(start & 31u) * 2u;
    const std::uint64_t lo = words[word_index].bits;
    if (shift == 0u) return lo;
    const std::uint64_t hi = words[word_index + 1u].bits;
    return (lo >> shift) | (hi << (64u - shift));
}

std::uint64_t scan_motif_packed_shifted_count(
    const seq::dna2_word* words,
    std::size_t windows,
    seq::dna2_word64 motif_word,
    int motif_length,
    int max_mismatches,
    unsigned int requested_threads,
    int iterations) {
    const unsigned int available = std::thread::hardware_concurrency();
    const unsigned int fallback = available == 0u ? 1u : available;
    const unsigned int thread_count = requested_threads == 0u ? fallback : requested_threads;
    const unsigned int workers = static_cast<unsigned int>(std::min<std::size_t>(windows == 0u ? 1u : windows, thread_count));
    std::vector<std::thread> threads;
    std::vector<std::uint64_t> partial(static_cast<std::size_t>(workers), 0ULL);
    threads.reserve(workers);
    const std::uint32_t active_mask = seq::detail::active_mask_from_length(motif_length);
    const std::uint64_t active_fields = seq::detail::spread_active_mask_to_packed_fields(active_mask);
    for (unsigned int worker = 0u; worker < workers; ++worker) {
        const std::size_t begin = (windows * worker) / workers;
        const std::size_t end = (windows * (worker + 1u)) / workers;
        threads.emplace_back([&, worker, begin, end] {
            std::uint64_t local_hits = 0ULL;
            for (int iter = 0; iter < iterations; ++iter) {
                std::uint64_t iter_hits = 0ULL;
                for (std::size_t start = begin; start < end; ++start) {
                    const seq::dna2_word64 window{shifted_window_word64(words, start)};
                    const int mismatches = seq::detail::word64_mismatches_packed_count_fields(window, motif_word, active_fields);
                    iter_hits += mismatches <= max_mismatches ? 1ULL : 0ULL;
                }
                local_hits = iter_hits;
            }
            partial[worker] = local_hits;
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    std::uint64_t hits = 0ULL;
    for (std::uint64_t value : partial) hits += value;
    return hits;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t count = argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : (1ULL << 20u);
        const int iterations = argc > 2 ? std::atoi(argv[2]) : 20;
        const std::uint32_t seed = argc > 3 ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 20260504u;
        const unsigned int threads = argc > 4 ? static_cast<unsigned int>(std::strtoul(argv[4], nullptr, 10)) : 0u;
        const int motif_length = argc > 5 ? std::atoi(argv[5]) : 16;
        const int max_mismatches = argc > 6 ? std::atoi(argv[6]) : 1;
        if (iterations < 1 || motif_length < 1 || motif_length > 32) {
            throw std::runtime_error(
                "usage: sequenceDna2CpuBench [count] [iterations] [seed] [threads, 0=hardware] "
                "[motif_length 1..32] [max_mismatches]");
        }

        const cellerator::bench::benchmark_mutex_guard guard("sequence-dna2-cpu");
        const unsigned int hardware_threads = std::thread::hardware_concurrency();
        const unsigned int active_threads = threads == 0u ? (hardware_threads == 0u ? 1u : hardware_threads) : threads;
        std::printf("sequence_chunks=%llu\n", static_cast<unsigned long long>(count));
        std::printf("sequence_bases=%llu\n", static_cast<unsigned long long>(count * 32u));
        std::printf("iterations=%d\n", iterations);
        std::printf("seed=%u\n", seed);
        std::printf("backend=%s\n", CELLERATOR_ENABLE_HIGHWAY ? "highway" : "scalar");
        std::printf("threads=%u\n", active_threads);
        std::printf("hardware_threads=%u\n", hardware_threads);
        std::printf("motif_length=%d\n", motif_length);
        std::printf("max_mismatches=%d\n", max_mismatches);

        constexpr std::size_t stride = 32u;
        std::unique_ptr<char[]> input(new char[count * stride]);
        std::unique_ptr<seq::dna2_word[]> words(new seq::dna2_word[count + 1u]);
        std::unique_ptr<seq::dna2_planes[]> planes(new seq::dna2_planes[count]);
        std::unique_ptr<std::uint32_t[]> masks(new std::uint32_t[count]);
        words[count].bits = 0ULL;

        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> base_dist(0, 3);
        const char alphabet[4] = {'A', 'C', 'G', 'T'};
        for (std::size_t i = 0u; i < count * stride; ++i) {
            input[i] = alphabet[base_dist(rng)];
        }

        seq::dna2_pack_ascii_batch(input.get(), stride, words.get(), count, stride);
        words[count].bits = 0ULL;
        seq::dna2_to_planes_batch(words.get(), planes.get(), count);

        const double pack_seconds = time_seconds([&] {
            parallel_chunks(count, active_threads, [&](std::size_t begin, std::size_t end) {
                for (int iter = 0; iter < iterations; ++iter) {
                    seq::dna2_pack_ascii_batch(input.get() + begin * stride, stride, words.get() + begin, end - begin, stride);
                }
            });
        });
        report("pack_ascii", count, iterations, pack_seconds, checksum_words(words.get(), count));

        const double planes_seconds = time_seconds([&] {
            parallel_chunks(count, active_threads, [&](std::size_t begin, std::size_t end) {
                for (int iter = 0; iter < iterations; ++iter) {
                    seq::dna2_to_planes_batch(words.get() + begin, planes.get() + begin, end - begin);
                }
            });
        });
        report("to_planes", count, iterations, planes_seconds, checksum_planes(planes.get(), count));

        const double gc_seconds = time_seconds([&] {
            parallel_chunks(count, active_threads, [&](std::size_t begin, std::size_t end) {
                for (int iter = 0; iter < iterations; ++iter) {
                    seq::planes_gc_mask_batch(planes.get() + begin, masks.get() + begin, end - begin);
                }
            });
        });
        report("gc_mask", count, iterations, gc_seconds, checksum_masks(masks.get(), count));

        const double cpg_seconds = time_seconds([&] {
            parallel_chunks(count, active_threads, [&](std::size_t begin, std::size_t end) {
                for (int iter = 0; iter < iterations; ++iter) {
                    seq::planes_cpg_start_mask_batch(planes.get() + begin, masks.get() + begin, end - begin);
                }
            });
        });
        report("cpg_mask", count, iterations, cpg_seconds, checksum_masks(masks.get(), count));

        const std::size_t sequence_bases = count * stride;
        const std::size_t windows = sequence_bases >= static_cast<std::size_t>(motif_length)
            ? sequence_bases - static_cast<std::size_t>(motif_length) + 1u : 0u;
        const std::size_t motif_start = sequence_bases > 4096u ? 4096u : 0u;
        const seq::dna2_word64 motif_word{pack_window_from_ascii(input.get(), motif_start, motif_length)};
        std::uint64_t scan_hits = 0ULL;
        const double scan_seconds = time_seconds([&] {
            scan_hits = scan_motif_packed_shifted_count(
                words.get(), windows, motif_word, motif_length, max_mismatches, active_threads, iterations);
        });
        const double windows_per_sec = scan_seconds > 0.0
            ? (static_cast<double>(windows) * iterations) / scan_seconds : 0.0;
        const double bases_per_sec = scan_seconds > 0.0
            ? (static_cast<double>(sequence_bases) * iterations) / scan_seconds : 0.0;
        std::printf("motif_scan_packed_shifted_seconds=%.6f\n", scan_seconds);
        std::printf("motif_scan_packed_shifted_windows_per_sec=%.3f\n", windows_per_sec);
        std::printf("motif_scan_packed_shifted_bases_per_sec=%.3f\n", bases_per_sec);
        std::printf("motif_scan_packed_shifted_hits=%llu\n", static_cast<unsigned long long>(scan_hits));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sequenceDna2CpuBench failed: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
