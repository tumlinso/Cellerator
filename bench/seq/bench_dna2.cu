#include <Cellerator/seq/dna2.cuh>

#include <bench/benchmark_mutex.hh>

#include <cuda_runtime.h>

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace seq = ::cellerator::seq;

namespace {

void cuda_require(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

void set_ref(std::uint64_t& packed, int i, std::uint8_t base) {
    const unsigned int shift = 2u * static_cast<unsigned int>(i);
    packed = (packed & ~(0x3ULL << shift)) | (static_cast<std::uint64_t>(base & 0x3u) << shift);
}

std::vector<std::uint64_t> pack_sequence(const std::vector<std::uint8_t>& bases) {
    std::vector<std::uint64_t> packed(((bases.size() + 31u) / 32u) + 1u, 0ULL);
    for (std::size_t i = 0; i < bases.size(); ++i) {
        set_ref(packed[i >> 5u], static_cast<int>(i & 31u), bases[i]);
    }
    return packed;
}

std::uint64_t pack_window(const std::vector<std::uint8_t>& bases) {
    std::uint64_t packed = 0ULL;
    for (int i = 0; i < 32; ++i) {
        const std::uint8_t base = i < static_cast<int>(bases.size()) ? bases[static_cast<std::size_t>(i)] : 0u;
        set_ref(packed, i, base);
    }
    return packed;
}

std::uint64_t count_hits(const std::vector<std::uint8_t>& hits) {
    std::uint64_t count = 0ULL;
    for (std::uint8_t hit : hits) count += hit != 0u ? 1ULL : 0ULL;
    return count;
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cuda_require(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    return ms;
}

void report_result(
    int sequence_length,
    int motif_length,
    int max_mismatches,
    int iterations,
    std::uint32_t seed,
    const char* representation,
    int cuda_devices,
    std::uint64_t hits,
    float elapsed_kernel_ms,
    int windows) {
    const double seconds = static_cast<double>(elapsed_kernel_ms) / 1000.0;
    const double bases_per_sec = seconds > 0.0 ? static_cast<double>(sequence_length) / seconds : 0.0;
    const double windows_per_sec = seconds > 0.0 ? static_cast<double>(windows) / seconds : 0.0;
    std::printf("sequence_length=%d\n", sequence_length);
    std::printf("motif_length=%d\n", motif_length);
    std::printf("max_mismatches=%d\n", max_mismatches);
    std::printf("iterations=%d\n", iterations);
    std::printf("seed=%u\n", seed);
    std::printf("representation=%s\n", representation);
    std::printf("cuda_devices=%d\n", cuda_devices);
    std::printf("hits=%llu\n", static_cast<unsigned long long>(hits));
    std::printf("elapsed_kernel_ms=%.3f\n", elapsed_kernel_ms);
    std::printf("bases_per_sec=%.3f\n", bases_per_sec);
    std::printf("windows_per_sec=%.3f\n\n", windows_per_sec);
}

void report_kernel_attributes_once() {
    static bool reported = false;
    if (reported) return;
    reported = true;
    cudaFuncAttributes packed_attr{};
    cudaFuncAttributes shifted_attr{};
    cudaFuncAttributes warp_attr{};
    cuda_require(cudaFuncGetAttributes(&packed_attr, seq::scan_motif_word64_reference), "packed kernel attributes");
    cuda_require(cudaFuncGetAttributes(&shifted_attr, seq::scan_motif_word64_shifted_count), "shifted packed kernel attributes");
    cuda_require(cudaFuncGetAttributes(&warp_attr, seq::scan_motif_warp32_unpacked), "warp kernel attributes");
    std::printf("packed_regs_per_thread=%d\n", packed_attr.numRegs);
    std::printf("packed_local_bytes_per_thread=%zu\n", packed_attr.localSizeBytes);
    std::printf("packed_shared_bytes_static=%zu\n", packed_attr.sharedSizeBytes);
    std::printf("packed_max_threads_per_block=%d\n", packed_attr.maxThreadsPerBlock);
    std::printf("shifted_count_regs_per_thread=%d\n", shifted_attr.numRegs);
    std::printf("shifted_count_local_bytes_per_thread=%zu\n", shifted_attr.localSizeBytes);
    std::printf("shifted_count_shared_bytes_static=%zu\n", shifted_attr.sharedSizeBytes);
    std::printf("shifted_count_max_threads_per_block=%d\n", shifted_attr.maxThreadsPerBlock);
    std::printf("warp_regs_per_thread=%d\n", warp_attr.numRegs);
    std::printf("warp_local_bytes_per_thread=%zu\n", warp_attr.localSizeBytes);
    std::printf("warp_shared_bytes_static=%zu\n", warp_attr.sharedSizeBytes);
    std::printf("warp_max_threads_per_block=%d\n", warp_attr.maxThreadsPerBlock);
}

std::vector<std::uint8_t> make_window_segment(
    const std::vector<std::uint8_t>& sequence,
    std::size_t start_window,
    int window_count,
    int motif_length) {
    std::vector<std::uint8_t> segment(static_cast<std::size_t>(window_count + motif_length - 1), 0u);
    for (std::size_t i = 0; i < segment.size(); ++i) {
        segment[i] = sequence[start_window + i];
    }
    return segment;
}

double wall_ms_since(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

void run_warp_planes_bench(
    const std::vector<std::uint8_t>& sequence,
    const std::vector<std::uint8_t>& motif,
    int max_mismatches,
    int iterations,
    std::uint32_t seed) {
    const int sequence_length = static_cast<int>(sequence.size());
    const int motif_length = static_cast<int>(motif.size());
    const int windows = sequence_length - motif_length + 1;
    const int threads = 128;
    const int blocks = (windows * 32 + threads - 1) / threads;
    const seq::dna2_planes32 motif_planes = seq::unpack_word64_to_planes32(seq::dna2_word64{pack_window(motif)});
    std::uint8_t* d_sequence = nullptr;
    std::uint8_t* d_hits = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::vector<std::uint8_t> hits(sequence.size(), 0u);

    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_sequence), sequence.size() * sizeof(std::uint8_t)), "cudaMalloc sequence");
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_hits), sequence.size() * sizeof(std::uint8_t)), "cudaMalloc hits");
    cuda_require(cudaMemcpy(d_sequence, sequence.data(), sequence.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice), "copy sequence");
    cuda_require(cudaEventCreate(&start), "cudaEventCreate start");
    cuda_require(cudaEventCreate(&stop), "cudaEventCreate stop");

    seq::scan_motif_warp32_unpacked<<<blocks, threads>>>(d_sequence, sequence_length, motif_planes, motif_length, max_mismatches, d_hits);
    cuda_require(cudaGetLastError(), "warmup warp scan");
    cuda_require(cudaDeviceSynchronize(), "warmup warp scan sync");

    cuda_require(cudaMemset(d_hits, 0, sequence.size() * sizeof(std::uint8_t)), "clear warp hits");
    cuda_require(cudaEventRecord(start), "record warp start");
    for (int i = 0; i < iterations; ++i) {
        seq::scan_motif_warp32_unpacked<<<blocks, threads>>>(d_sequence, sequence_length, motif_planes, motif_length, max_mismatches, d_hits);
    }
    cuda_require(cudaEventRecord(stop), "record warp stop");
    cuda_require(cudaEventSynchronize(stop), "sync warp stop");
    cuda_require(cudaGetLastError(), "warp scan bench");
    cuda_require(cudaMemcpy(hits.data(), d_hits, hits.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost), "copy warp hits");

    report_result(
        sequence_length,
        motif_length,
        max_mismatches,
        iterations,
        seed,
        "warp_planes32",
        1,
        count_hits(hits),
        elapsed_ms(start, stop) / static_cast<float>(iterations),
        windows);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_sequence);
    cudaFree(d_hits);
}

void run_packed_word_bench(
    const std::vector<std::uint8_t>& sequence,
    const std::vector<std::uint8_t>& motif,
    int max_mismatches,
    int iterations,
    std::uint32_t seed) {
    const int sequence_length = static_cast<int>(sequence.size());
    const int motif_length = static_cast<int>(motif.size());
    const int windows = sequence_length - motif_length + 1;
    const int threads = 128;
    const int blocks = (windows + threads - 1) / threads;
    const std::vector<std::uint64_t> packed = pack_sequence(sequence);
    const seq::dna2_word64 motif_word{pack_window(motif)};
    std::uint64_t* d_sequence = nullptr;
    std::uint8_t* d_hits = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::vector<std::uint8_t> hits(sequence.size(), 0u);

    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_sequence), packed.size() * sizeof(std::uint64_t)), "cudaMalloc packed");
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_hits), sequence.size() * sizeof(std::uint8_t)), "cudaMalloc packed hits");
    cuda_require(cudaMemcpy(d_sequence, packed.data(), packed.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice), "copy packed");
    cuda_require(cudaEventCreate(&start), "cudaEventCreate packed start");
    cuda_require(cudaEventCreate(&stop), "cudaEventCreate packed stop");

    seq::scan_motif_word64_reference<<<blocks, threads>>>(d_sequence, sequence_length, motif_word, motif_length, max_mismatches, d_hits);
    cuda_require(cudaGetLastError(), "warmup packed scan");
    cuda_require(cudaDeviceSynchronize(), "warmup packed scan sync");

    cuda_require(cudaMemset(d_hits, 0, sequence.size() * sizeof(std::uint8_t)), "clear packed hits");
    cuda_require(cudaEventRecord(start), "record packed start");
    for (int i = 0; i < iterations; ++i) {
        seq::scan_motif_word64_reference<<<blocks, threads>>>(d_sequence, sequence_length, motif_word, motif_length, max_mismatches, d_hits);
    }
    cuda_require(cudaEventRecord(stop), "record packed stop");
    cuda_require(cudaEventSynchronize(stop), "sync packed stop");
    cuda_require(cudaGetLastError(), "packed scan bench");
    cuda_require(cudaMemcpy(hits.data(), d_hits, hits.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost), "copy packed hits");

    report_result(
        sequence_length,
        motif_length,
        max_mismatches,
        iterations,
        seed,
        "packed_word64",
        1,
        count_hits(hits),
        elapsed_ms(start, stop) / static_cast<float>(iterations),
        windows);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_sequence);
    cudaFree(d_hits);
}

void run_packed_word_shifted_count_bench(
    const std::vector<std::uint8_t>& sequence,
    const std::vector<std::uint8_t>& motif,
    int max_mismatches,
    int iterations,
    std::uint32_t seed) {
    const int sequence_length = static_cast<int>(sequence.size());
    const int motif_length = static_cast<int>(motif.size());
    const int windows = sequence_length - motif_length + 1;
    const int threads = 256;
    const int blocks = std::min(8192, (windows + threads - 1) / threads);
    const std::vector<std::uint64_t> packed = pack_sequence(sequence);
    const seq::dna2_word64 motif_word{pack_window(motif)};
    std::uint64_t* d_sequence = nullptr;
    unsigned long long* d_hit_count = nullptr;
    unsigned long long hits = 0ULL;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_sequence), packed.size() * sizeof(std::uint64_t)), "cudaMalloc shifted packed");
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&d_hit_count), sizeof(unsigned long long)), "cudaMalloc shifted hit count");
    cuda_require(cudaMemcpy(d_sequence, packed.data(), packed.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice), "copy shifted packed");
    cuda_require(cudaEventCreate(&start), "cudaEventCreate shifted start");
    cuda_require(cudaEventCreate(&stop), "cudaEventCreate shifted stop");

    cuda_require(cudaMemset(d_hit_count, 0, sizeof(unsigned long long)), "clear shifted warmup count");
    seq::scan_motif_word64_shifted_count<<<blocks, threads, threads * sizeof(unsigned int)>>>(
        d_sequence, sequence_length, motif_word, motif_length, max_mismatches, d_hit_count);
    cuda_require(cudaGetLastError(), "warmup shifted packed scan");
    cuda_require(cudaDeviceSynchronize(), "warmup shifted packed scan sync");

    cuda_require(cudaEventRecord(start), "record shifted start");
    for (int i = 0; i < iterations; ++i) {
        cuda_require(cudaMemset(d_hit_count, 0, sizeof(unsigned long long)), "clear shifted count");
        seq::scan_motif_word64_shifted_count<<<blocks, threads, threads * sizeof(unsigned int)>>>(
            d_sequence, sequence_length, motif_word, motif_length, max_mismatches, d_hit_count);
    }
    cuda_require(cudaEventRecord(stop), "record shifted stop");
    cuda_require(cudaEventSynchronize(stop), "sync shifted stop");
    cuda_require(cudaGetLastError(), "shifted packed scan bench");
    cuda_require(cudaMemcpy(&hits, d_hit_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "copy shifted count");

    report_result(
        sequence_length,
        motif_length,
        max_mismatches,
        iterations,
        seed,
        "packed_word64_shifted_count",
        1,
        hits,
        elapsed_ms(start, stop) / static_cast<float>(iterations),
        windows);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_sequence);
    cudaFree(d_hit_count);
}

void run_warp_planes_all_gpus_bench(
    const std::vector<std::uint8_t>& sequence,
    const std::vector<std::uint8_t>& motif,
    int max_mismatches,
    int iterations,
    std::uint32_t seed,
    int device_count) {
    const int sequence_length = static_cast<int>(sequence.size());
    const int motif_length = static_cast<int>(motif.size());
    const int windows = sequence_length - motif_length + 1;
    const int threads = 128;
    const seq::dna2_planes32 motif_planes = seq::unpack_word64_to_planes32(seq::dna2_word64{pack_window(motif)});
    struct part {
        int device;
        int windows;
        int bases;
        std::uint8_t* d_sequence;
        std::uint8_t* d_hits;
    };
    std::vector<part> parts;
    for (int device = 0; device < device_count; ++device) {
        const std::size_t begin = (static_cast<std::size_t>(windows) * static_cast<std::size_t>(device))
            / static_cast<std::size_t>(device_count);
        const std::size_t end = (static_cast<std::size_t>(windows) * static_cast<std::size_t>(device + 1))
            / static_cast<std::size_t>(device_count);
        const int part_windows = static_cast<int>(end - begin);
        if (part_windows <= 0) continue;
        const std::vector<std::uint8_t> segment = make_window_segment(sequence, begin, part_windows, motif_length);
        part p{device, part_windows, static_cast<int>(segment.size()), nullptr, nullptr};
        cuda_require(cudaSetDevice(device), "set device");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&p.d_sequence), segment.size() * sizeof(std::uint8_t)), "cudaMalloc all-gpu sequence");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&p.d_hits), segment.size() * sizeof(std::uint8_t)), "cudaMalloc all-gpu hits");
        cuda_require(cudaMemcpy(p.d_sequence, segment.data(), segment.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice), "copy all-gpu sequence");
        parts.push_back(p);
    }

    for (part& p : parts) {
        cuda_require(cudaSetDevice(p.device), "set warmup device");
        cuda_require(cudaMemset(p.d_hits, 0, static_cast<std::size_t>(p.bases) * sizeof(std::uint8_t)), "clear all-gpu warmup hits");
        const int blocks = (p.windows * 32 + threads - 1) / threads;
        seq::scan_motif_warp32_unpacked<<<blocks, threads>>>(p.d_sequence, p.bases, motif_planes, motif_length, max_mismatches, p.d_hits);
        cuda_require(cudaGetLastError(), "warmup all-gpu warp scan");
    }
    for (part& p : parts) {
        cuda_require(cudaSetDevice(p.device), "set warmup sync device");
        cuda_require(cudaDeviceSynchronize(), "sync all-gpu warmup");
        cuda_require(cudaMemset(p.d_hits, 0, static_cast<std::size_t>(p.bases) * sizeof(std::uint8_t)), "clear all-gpu hits");
    }

    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        for (part& p : parts) {
            cuda_require(cudaSetDevice(p.device), "set all-gpu warp device");
            const int blocks = (p.windows * 32 + threads - 1) / threads;
            seq::scan_motif_warp32_unpacked<<<blocks, threads>>>(p.d_sequence, p.bases, motif_planes, motif_length, max_mismatches, p.d_hits);
            cuda_require(cudaGetLastError(), "launch all-gpu warp scan");
        }
        for (part& p : parts) {
            cuda_require(cudaSetDevice(p.device), "set all-gpu warp sync device");
            cuda_require(cudaDeviceSynchronize(), "sync all-gpu warp scan");
        }
    }
    const auto stop = std::chrono::steady_clock::now();

    std::uint64_t hits = 0ULL;
    for (part& p : parts) {
        cuda_require(cudaSetDevice(p.device), "set all-gpu copy device");
        std::vector<std::uint8_t> host_hits(static_cast<std::size_t>(p.windows), 0u);
        cuda_require(cudaMemcpy(host_hits.data(), p.d_hits, host_hits.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost), "copy all-gpu warp hits");
        hits += count_hits(host_hits);
        cudaFree(p.d_sequence);
        cudaFree(p.d_hits);
    }

    report_result(
        sequence_length,
        motif_length,
        max_mismatches,
        iterations,
        seed,
        "warp_planes32_all_gpus",
        static_cast<int>(parts.size()),
        hits,
        static_cast<float>(wall_ms_since(start, stop) / static_cast<double>(iterations)),
        windows);
}

void run_packed_word_all_gpus_bench(
    const std::vector<std::uint8_t>& sequence,
    const std::vector<std::uint8_t>& motif,
    int max_mismatches,
    int iterations,
    std::uint32_t seed,
    int device_count) {
    const int sequence_length = static_cast<int>(sequence.size());
    const int motif_length = static_cast<int>(motif.size());
    const int windows = sequence_length - motif_length + 1;
    const int threads = 128;
    const seq::dna2_word64 motif_word{pack_window(motif)};
    struct part {
        int device;
        int windows;
        int bases;
        std::uint64_t* d_sequence;
        std::uint8_t* d_hits;
    };
    std::vector<part> parts;
    for (int device = 0; device < device_count; ++device) {
        const std::size_t begin = (static_cast<std::size_t>(windows) * static_cast<std::size_t>(device))
            / static_cast<std::size_t>(device_count);
        const std::size_t end = (static_cast<std::size_t>(windows) * static_cast<std::size_t>(device + 1))
            / static_cast<std::size_t>(device_count);
        const int part_windows = static_cast<int>(end - begin);
        if (part_windows <= 0) continue;
        const std::vector<std::uint8_t> segment = make_window_segment(sequence, begin, part_windows, motif_length);
        const std::vector<std::uint64_t> packed = pack_sequence(segment);
        part p{device, part_windows, static_cast<int>(segment.size()), nullptr, nullptr};
        cuda_require(cudaSetDevice(device), "set packed device");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&p.d_sequence), packed.size() * sizeof(std::uint64_t)), "cudaMalloc all-gpu packed");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&p.d_hits), segment.size() * sizeof(std::uint8_t)), "cudaMalloc all-gpu packed hits");
        cuda_require(cudaMemcpy(p.d_sequence, packed.data(), packed.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice), "copy all-gpu packed");
        parts.push_back(p);
    }

    for (part& p : parts) {
        cuda_require(cudaSetDevice(p.device), "set packed warmup device");
        cuda_require(cudaMemset(p.d_hits, 0, static_cast<std::size_t>(p.bases) * sizeof(std::uint8_t)), "clear all-gpu packed warmup hits");
        const int blocks = (p.windows + threads - 1) / threads;
        seq::scan_motif_word64_reference<<<blocks, threads>>>(p.d_sequence, p.bases, motif_word, motif_length, max_mismatches, p.d_hits);
        cuda_require(cudaGetLastError(), "warmup all-gpu packed scan");
    }
    for (part& p : parts) {
        cuda_require(cudaSetDevice(p.device), "set packed warmup sync device");
        cuda_require(cudaDeviceSynchronize(), "sync all-gpu packed warmup");
        cuda_require(cudaMemset(p.d_hits, 0, static_cast<std::size_t>(p.bases) * sizeof(std::uint8_t)), "clear all-gpu packed hits");
    }

    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        for (part& p : parts) {
            cuda_require(cudaSetDevice(p.device), "set all-gpu packed device");
            const int blocks = (p.windows + threads - 1) / threads;
            seq::scan_motif_word64_reference<<<blocks, threads>>>(p.d_sequence, p.bases, motif_word, motif_length, max_mismatches, p.d_hits);
            cuda_require(cudaGetLastError(), "launch all-gpu packed scan");
        }
        for (part& p : parts) {
            cuda_require(cudaSetDevice(p.device), "set all-gpu packed sync device");
            cuda_require(cudaDeviceSynchronize(), "sync all-gpu packed scan");
        }
    }
    const auto stop = std::chrono::steady_clock::now();

    std::uint64_t hits = 0ULL;
    for (part& p : parts) {
        cuda_require(cudaSetDevice(p.device), "set all-gpu packed copy device");
        std::vector<std::uint8_t> host_hits(static_cast<std::size_t>(p.windows), 0u);
        cuda_require(cudaMemcpy(host_hits.data(), p.d_hits, host_hits.size() * sizeof(std::uint8_t), cudaMemcpyDeviceToHost), "copy all-gpu packed hits");
        hits += count_hits(host_hits);
        cudaFree(p.d_sequence);
        cudaFree(p.d_hits);
    }

    report_result(
        sequence_length,
        motif_length,
        max_mismatches,
        iterations,
        seed,
        "packed_word64_all_gpus",
        static_cast<int>(parts.size()),
        hits,
        static_cast<float>(wall_ms_since(start, stop) / static_cast<double>(iterations)),
        windows);
}

void run_packed_word_shifted_count_all_gpus_bench(
    const std::vector<std::uint8_t>& sequence,
    const std::vector<std::uint8_t>& motif,
    int max_mismatches,
    int iterations,
    std::uint32_t seed,
    int device_count) {
    const int sequence_length = static_cast<int>(sequence.size());
    const int motif_length = static_cast<int>(motif.size());
    const int windows = sequence_length - motif_length + 1;
    const int threads = 256;
    const seq::dna2_word64 motif_word{pack_window(motif)};
    struct part {
        int device;
        int windows;
        int bases;
        std::uint64_t* d_sequence;
        unsigned long long* d_hit_count;
    };
    std::vector<part> parts;
    for (int device = 0; device < device_count; ++device) {
        const std::size_t begin = (static_cast<std::size_t>(windows) * static_cast<std::size_t>(device))
            / static_cast<std::size_t>(device_count);
        const std::size_t end = (static_cast<std::size_t>(windows) * static_cast<std::size_t>(device + 1))
            / static_cast<std::size_t>(device_count);
        const int part_windows = static_cast<int>(end - begin);
        if (part_windows <= 0) continue;
        const std::vector<std::uint8_t> segment = make_window_segment(sequence, begin, part_windows, motif_length);
        const std::vector<std::uint64_t> packed = pack_sequence(segment);
        part p{device, part_windows, static_cast<int>(segment.size()), nullptr, nullptr};
        cuda_require(cudaSetDevice(device), "set shifted packed device");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&p.d_sequence), packed.size() * sizeof(std::uint64_t)), "cudaMalloc all-gpu shifted packed");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&p.d_hit_count), sizeof(unsigned long long)), "cudaMalloc all-gpu shifted count");
        cuda_require(cudaMemcpy(p.d_sequence, packed.data(), packed.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice), "copy all-gpu shifted packed");
        parts.push_back(p);
    }

    for (part& p : parts) {
        cuda_require(cudaSetDevice(p.device), "set shifted warmup device");
        cuda_require(cudaMemset(p.d_hit_count, 0, sizeof(unsigned long long)), "clear all-gpu shifted warmup count");
        const int blocks = std::min(8192, (p.windows + threads - 1) / threads);
        seq::scan_motif_word64_shifted_count<<<blocks, threads, threads * sizeof(unsigned int)>>>(
            p.d_sequence, p.bases, motif_word, motif_length, max_mismatches, p.d_hit_count);
        cuda_require(cudaGetLastError(), "warmup all-gpu shifted scan");
    }
    for (part& p : parts) {
        cuda_require(cudaSetDevice(p.device), "set shifted warmup sync device");
        cuda_require(cudaDeviceSynchronize(), "sync all-gpu shifted warmup");
    }

    const auto start = std::chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        for (part& p : parts) {
            cuda_require(cudaSetDevice(p.device), "set all-gpu shifted device");
            cuda_require(cudaMemset(p.d_hit_count, 0, sizeof(unsigned long long)), "clear all-gpu shifted count");
            const int blocks = std::min(8192, (p.windows + threads - 1) / threads);
            seq::scan_motif_word64_shifted_count<<<blocks, threads, threads * sizeof(unsigned int)>>>(
                p.d_sequence, p.bases, motif_word, motif_length, max_mismatches, p.d_hit_count);
            cuda_require(cudaGetLastError(), "launch all-gpu shifted scan");
        }
        for (part& p : parts) {
            cuda_require(cudaSetDevice(p.device), "set all-gpu shifted sync device");
            cuda_require(cudaDeviceSynchronize(), "sync all-gpu shifted scan");
        }
    }
    const auto stop = std::chrono::steady_clock::now();

    std::uint64_t hits = 0ULL;
    for (part& p : parts) {
        cuda_require(cudaSetDevice(p.device), "set all-gpu shifted copy device");
        unsigned long long part_hits = 0ULL;
        cuda_require(cudaMemcpy(&part_hits, p.d_hit_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "copy all-gpu shifted count");
        hits += part_hits;
        cudaFree(p.d_sequence);
        cudaFree(p.d_hit_count);
    }

    report_result(
        sequence_length,
        motif_length,
        max_mismatches,
        iterations,
        seed,
        "packed_word64_shifted_count_all_gpus",
        static_cast<int>(parts.size()),
        hits,
        static_cast<float>(wall_ms_since(start, stop) / static_cast<double>(iterations)),
        windows);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const int sequence_length = argc > 1 ? std::atoi(argv[1]) : (1 << 20);
        const int motif_length = argc > 2 ? std::atoi(argv[2]) : 16;
        const int max_mismatches = argc > 3 ? std::atoi(argv[3]) : 1;
        const int iterations = argc > 4 ? std::atoi(argv[4]) : 50;
        const char* representation = argc > 5 ? argv[5] : "both";
        const std::uint32_t seed = argc > 6 ? static_cast<std::uint32_t>(std::strtoul(argv[6], nullptr, 10)) : 20260504u;
        const char* device_mode = argc > 7 ? argv[7] : "single_gpu";
        if (sequence_length < motif_length || motif_length < 1 || motif_length > 32 || iterations < 1) {
            throw std::runtime_error(
                "usage: sequenceDna2Bench [sequence_length] [motif_length 1..32] "
                "[max_mismatches] [iterations] [both|packed_word64|packed_word64_shifted_count|warp_planes32] "
                "[seed] [single_gpu|all_gpus]");
        }
        const bool run_packed = std::strcmp(representation, "both") == 0 || std::strcmp(representation, "packed_word64") == 0;
        const bool run_shifted = std::strcmp(representation, "both") == 0
            || std::strcmp(representation, "packed_word64_shifted_count") == 0;
        const bool run_warp = std::strcmp(representation, "both") == 0 || std::strcmp(representation, "warp_planes32") == 0;
        if (!run_packed && !run_shifted && !run_warp) {
            throw std::runtime_error("representation must be one of: both, packed_word64, packed_word64_shifted_count, warp_planes32");
        }
        const bool all_gpus = std::strcmp(device_mode, "all_gpus") == 0;
        if (!all_gpus && std::strcmp(device_mode, "single_gpu") != 0) {
            throw std::runtime_error("device mode must be one of: single_gpu, all_gpus");
        }

        int device = 0;
        cuda_require(cudaGetDevice(&device), "cudaGetDevice");
        int device_count = 0;
        cuda_require(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        std::vector<int> benchmark_devices;
        if (all_gpus) {
            benchmark_devices.resize(static_cast<std::size_t>(device_count));
            for (int i = 0; i < device_count; ++i) benchmark_devices[static_cast<std::size_t>(i)] = i;
        } else {
            benchmark_devices.push_back(device);
        }
        const cellerator::bench::benchmark_mutex_guard guard(
            "sequence-dna2",
            benchmark_devices.data(),
            benchmark_devices.size());
        report_kernel_attributes_once();

        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> base_dist(0, 3);
        std::vector<std::uint8_t> sequence(static_cast<std::size_t>(sequence_length), 0u);
        for (std::uint8_t& base : sequence) {
            base = static_cast<std::uint8_t>(base_dist(rng));
        }
        const int motif_start = sequence_length > 4096 ? 4096 : 0;
        std::vector<std::uint8_t> motif(static_cast<std::size_t>(motif_length), 0u);
        for (int i = 0; i < motif_length; ++i) {
            motif[static_cast<std::size_t>(i)] = sequence[static_cast<std::size_t>((motif_start + i) % sequence_length)];
        }

        if (all_gpus) {
            if (run_packed) run_packed_word_all_gpus_bench(sequence, motif, max_mismatches, iterations, seed, device_count);
            if (run_shifted) run_packed_word_shifted_count_all_gpus_bench(sequence, motif, max_mismatches, iterations, seed, device_count);
            if (run_warp) run_warp_planes_all_gpus_bench(sequence, motif, max_mismatches, iterations, seed, device_count);
        } else {
            if (run_packed) run_packed_word_bench(sequence, motif, max_mismatches, iterations, seed);
            if (run_shifted) run_packed_word_shifted_count_bench(sequence, motif, max_mismatches, iterations, seed);
            if (run_warp) run_warp_planes_bench(sequence, motif, max_mismatches, iterations, seed);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_dna2 failed: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
