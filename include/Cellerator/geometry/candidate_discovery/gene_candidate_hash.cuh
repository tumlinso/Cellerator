#pragma once

#include <cstdint>

namespace cellerator::compute::gene_candidates::detail {

inline constexpr std::uint64_t splitmix_gamma = 0x9e3779b97f4a7c15ull;
inline constexpr std::uint64_t splitmix_multiplier_1 = 0xbf58476d1ce4e5b9ull;
inline constexpr std::uint64_t splitmix_multiplier_2 = 0x94d049bb133111ebull;
inline constexpr std::uint64_t sketch_stride = 0xd2b74407b1ce6e93ull;
inline constexpr std::uint64_t minhash_domain = 0x4d494e4841534851ull;
inline constexpr std::uint64_t lsh_band_domain = 0x4c53485f42414e44ull;
inline constexpr std::uint64_t bucket_cap_domain = 0x4255434b45545f31ull;

// Prefer these helpers over copying the fixed hash expressions into kernels.
__host__ __device__ inline std::uint64_t splitmix64_v1(std::uint64_t value) noexcept {
    std::uint64_t z = value + splitmix_gamma;
    z = (z ^ (z >> 30u)) * splitmix_multiplier_1;
    z = (z ^ (z >> 27u)) * splitmix_multiplier_2;
    return z ^ (z >> 31u);
}

__host__ __device__ inline std::uint64_t minhash_value_v1(
    std::uint64_t global_row_index,
    std::uint64_t seed,
    std::uint32_t sketch_index) noexcept {
    const std::uint64_t salt = splitmix64_v1(
        seed ^ minhash_domain ^ ((std::uint64_t) sketch_index * sketch_stride));
    return splitmix64_v1(global_row_index ^ salt);
}

__host__ __device__ inline std::uint64_t lsh_band_key_v1(
    const std::uint64_t *sketch_values,
    std::uint32_t rows_per_band,
    std::uint64_t seed,
    std::uint32_t band) noexcept {
    std::uint64_t state = splitmix64_v1(seed ^ lsh_band_domain ^ (std::uint64_t) band);
    for (std::uint32_t row = 0u; row < rows_per_band; ++row) {
        state = splitmix64_v1(
            state ^ sketch_values[row] ^ ((std::uint64_t) row * splitmix_gamma));
    }
    return state;
}

__host__ __device__ inline std::uint64_t oversized_bucket_window_start_v1(
    std::uint64_t lsh_key,
    std::uint64_t seed,
    std::uint32_t band,
    std::uint64_t bucket_size) noexcept {
    if (bucket_size == 0u) return 0u;
    return splitmix64_v1(
        lsh_key ^ seed ^ bucket_cap_domain ^ ((std::uint64_t) band * sketch_stride)) % bucket_size;
}

} // namespace cellerator::compute::gene_candidates::detail
