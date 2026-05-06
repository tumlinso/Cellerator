#pragma once

#include <cstdint>

namespace cellerator::core::sequence {

enum class base : std::uint8_t {
    a = 0u,
    c = 1u,
    g = 2u,
    t = 3u
};

enum {
    bits_per_base = 2u,
    bases_per_u64 = 32u,
    packed_base_mask = 0x3u
};

__host__ __device__ __forceinline__ constexpr std::uint8_t pack_base(base value) {
    return static_cast<std::uint8_t>(value) & packed_base_mask;
}

__host__ __device__ __forceinline__ constexpr base unpack_base(std::uint8_t packed) {
    return static_cast<base>(packed & packed_base_mask);
}

__host__ __device__ __forceinline__ constexpr base complement(base value) {
    return unpack_base(pack_base(value) ^ packed_base_mask);
}

__host__ __device__ __forceinline__ constexpr bool is_defined_base_char(char value) {
    return value == 'A' || value == 'C' || value == 'G' || value == 'T' || value == 'U'
        || value == 'a' || value == 'c' || value == 'g' || value == 't' || value == 'u';
}

__host__ __device__ __forceinline__ constexpr base base_from_char(char value) {
    return value == 'C' || value == 'c' ? base::c
        : value == 'G' || value == 'g' ? base::g
        : value == 'T' || value == 't' || value == 'U' || value == 'u' ? base::t
        : base::a;
}

__host__ __device__ __forceinline__ constexpr char char_from_base(base value) {
    return value == base::c ? 'C'
        : value == base::g ? 'G'
        : value == base::t ? 'T'
        : 'A';
}

__host__ __device__ __forceinline__ constexpr std::uint64_t store_base(std::uint64_t word, unsigned int slot, base value) {
    const unsigned int shift = (slot % bases_per_u64) * bits_per_base;
    const std::uint64_t mask = static_cast<std::uint64_t>(packed_base_mask) << shift;
    return (word & ~mask) | (static_cast<std::uint64_t>(pack_base(value)) << shift);
}

__host__ __device__ __forceinline__ constexpr base load_base(std::uint64_t word, unsigned int slot) {
    const unsigned int shift = (slot % bases_per_u64) * bits_per_base;
    return unpack_base(static_cast<std::uint8_t>((word >> shift) & packed_base_mask));
}

} // namespace cellerator::core::sequence
