#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace cellerator::seq::test {

inline std::vector<std::uint8_t> random_bases(std::size_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> base_dist(0, 3);
    std::vector<std::uint8_t> bases(count, 0u);
    for (std::uint8_t& base : bases) {
        base = static_cast<std::uint8_t>(base_dist(rng));
    }
    return bases;
}

inline std::vector<std::uint8_t> random_window32(std::uint32_t seed) {
    return random_bases(32u, seed);
}

inline std::vector<std::uint8_t> motif_from_sequence(
    const std::vector<std::uint8_t>& sequence,
    std::size_t start,
    std::size_t length) {
    std::vector<std::uint8_t> motif(length, 0u);
    for (std::size_t i = 0; i < length; ++i) {
        motif[i] = sequence[(start + i) % sequence.size()] & 0x3u;
    }
    return motif;
}

inline void force_mismatch(std::vector<std::uint8_t>& bases, std::size_t index, std::uint8_t delta = 1u) {
    bases[index] = static_cast<std::uint8_t>((bases[index] ^ (delta & 0x3u)) & 0x3u);
}

inline std::string bases_to_string(const std::vector<std::uint8_t>& bases) {
    std::string out;
    out.reserve(bases.size());
    for (std::uint8_t base : bases) {
        switch (base & 0x3u) {
            case 1u: out.push_back('C'); break;
            case 2u: out.push_back('G'); break;
            case 3u: out.push_back('T'); break;
            default: out.push_back('A'); break;
        }
    }
    return out;
}

} // namespace cellerator::seq::test
