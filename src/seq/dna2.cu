/*
 * SequenceBits CUDA benchmark note, 2026-05-04:
 * Compared the original packed reference scanner against
 * scan_motif_word64_shifted_count on 4x Tesla V100-SXM2-16GB, sm_70.
 * Commands:
 *   ./build/sequenceDna2Bench 67108864 16 1 30 packed_word64 20260504 single_gpu
 *   ./build/sequenceDna2Bench 67108864 16 1 30 packed_word64_shifted_count 20260504 single_gpu
 *   ./build/sequenceDna2Bench 1073741824 16 1 20 packed_word64_shifted_count 20260504 all_gpus
 * Results: old packed reference 3.155 ms / 21.3G windows/s on one V100;
 * shifted count 0.222 ms / 301.9G windows/s on one V100 for 67M bases;
 * 1.073B bases reached 3.644 ms / 294.7G windows/s on one V100 and
 * 0.952 ms / 1.128T windows/s across four V100s. Correctness was validated
 * against CPU references by sequenceDna2CudaTest.
 */

#include <Cellerator/seq/dna2.cuh>

namespace cellerator::seq {

namespace {

__device__ __forceinline__ std::uint64_t shifted_window_word64(const std::uint64_t* packed_seq, int start) {
    const unsigned int word_index = static_cast<unsigned int>(start) >> 5u;
    const unsigned int shift = (static_cast<unsigned int>(start) & 31u) * 2u;
    const std::uint64_t lo = packed_seq[word_index];
    if (shift == 0u) return lo;
    const std::uint64_t hi = packed_seq[word_index + 1u];
    return (lo >> shift) | (hi << (64u - shift));
}

} // namespace

__global__ void scan_motif_warp32_unpacked(
    const std::uint8_t* seq_bases,
    int n_bases,
    dna2_planes32 motif,
    int motif_len,
    int max_mismatches,
    std::uint8_t* hits) {
    const int global_thread = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int start = global_thread >> 5;
    const int windows = n_bases >= motif_len ? n_bases - motif_len + 1 : 0;
    if (start >= windows) return;

    std::uint8_t base = 0u;
    if (lane < motif_len) {
        base = seq_bases[start + lane] & 0x3u;
    }

    const dna2_planes32 window = warp_encode_base_lanes(base);
    if (lane == 0) {
        const std::uint32_t active_mask = detail::active_mask_from_length(motif_len);
        const int mismatches = planes32_mismatches(window, motif, active_mask);
        hits[start] = static_cast<std::uint8_t>(mismatches <= max_mismatches ? 1u : 0u);
    }
}

__global__ void convert_word64_to_planes32_warp(
    const std::uint64_t* packed_words,
    dna2_planes32* planes,
    int word_count) {
    if (packed_words == nullptr || planes == nullptr || word_count <= 0 || (blockDim.x & 31u) != 0u) return;

    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int warp_in_block = static_cast<int>(threadIdx.x >> 5u);
    const int warps_per_block = static_cast<int>(blockDim.x >> 5u);
    const int word_index = static_cast<int>(blockIdx.x) * warps_per_block + warp_in_block;
    if (word_index >= word_count) return;

    const std::uint64_t packed = packed_words[word_index];
    const std::uint8_t base = static_cast<std::uint8_t>((packed >> (2u * static_cast<unsigned int>(lane))) & 0x3ULL);
    const dna2_planes32 word_planes = warp_encode_base_lanes(base);
    if (lane == 0) {
        planes[word_index] = word_planes;
    }
}

__global__ void scan_motif_word64_reference(
    const std::uint64_t* packed_seq,
    int n_bases,
    dna2_word64 motif_word,
    int motif_len,
    int max_mismatches,
    std::uint8_t* hits) {
    const int start = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int windows = n_bases >= motif_len ? n_bases - motif_len + 1 : 0;
    if (start >= windows) return;

    dna2_word64 window{0ULL};
    for (int i = 0; i < motif_len; ++i) {
        const int seq_index = start + i;
        const dna2_word64 source{packed_seq[static_cast<unsigned int>(seq_index) >> 5u]};
        set_base(window, i, get_base(source, seq_index & 31));
    }

    const std::uint32_t active_mask = detail::active_mask_from_length(motif_len);
    const int mismatches = word64_mismatches(window, motif_word, active_mask);
    hits[start] = static_cast<std::uint8_t>(mismatches <= max_mismatches ? 1u : 0u);
}

__global__ void scan_motif_word64_shifted_count(
    const std::uint64_t* packed_seq,
    int n_bases,
    dna2_word64 motif_word,
    int motif_len,
    int max_mismatches,
    unsigned long long* hit_count) {
    extern __shared__ unsigned int block_counts[];
    const int tid = static_cast<int>(threadIdx.x);
    const int global_thread = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(gridDim.x * blockDim.x);
    const int windows = n_bases >= motif_len ? n_bases - motif_len + 1 : 0;
    const std::uint32_t active_mask = detail::active_mask_from_length(motif_len);
    const std::uint64_t active_fields = detail::spread_active_mask_to_packed_fields(active_mask);
    unsigned int local_hits = 0u;

    for (int start = global_thread; start < windows; start += stride) {
        const dna2_word64 window{shifted_window_word64(packed_seq, start)};
        const int mismatches = detail::word64_mismatches_packed_count_fields(window, motif_word, active_fields);
        local_hits += mismatches <= max_mismatches ? 1u : 0u;
    }

    block_counts[tid] = local_hits;
    __syncthreads();
    for (int offset = static_cast<int>(blockDim.x) >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) block_counts[tid] += block_counts[tid + offset];
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(hit_count, static_cast<unsigned long long>(block_counts[0]));
    }
}

} // namespace cellerator::seq
