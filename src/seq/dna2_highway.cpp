#include <Cellerator/seq/dna2.cuh>

#include <hwy/highway.h>

#include <cassert>

namespace cellerator::seq {
namespace {

namespace hn = hwy::HWY_NAMESPACE;

constexpr std::size_t lane_cap64 = 8u;
constexpr std::size_t lane_cap32 = 16u;
constexpr std::size_t bases_per_word = 32u;

dna2_word dna2_pack_ascii_32_highway(const char* bases) {
    const hn::CappedTag<std::uint8_t, bases_per_word> d;
    const auto chars = hn::LoadU(d, reinterpret_cast<const std::uint8_t*>(bases));
    const std::uint64_t c_mask = hn::BitsFromMask(d, hn::Eq(chars, hn::Set(d, static_cast<std::uint8_t>('C'))));
    const std::uint64_t g_mask = hn::BitsFromMask(d, hn::Eq(chars, hn::Set(d, static_cast<std::uint8_t>('G'))));
    const std::uint64_t t_mask = hn::BitsFromMask(d, hn::Eq(chars, hn::Set(d, static_cast<std::uint8_t>('T'))));
    const dna2_planes planes{c_mask | t_mask, g_mask | t_mask};
    return planes_to_dna2(planes);
}

void dna2_unpack_ascii_32_highway(dna2_word input, char* output) {
    const hn::CappedTag<std::uint8_t, bases_per_word> d;
    const dna2_planes planes = dna2_to_planes(input);
    const std::uint64_t lo = planes.lo & 0xffffffffULL;
    const std::uint64_t hi = planes.hi & 0xffffffffULL;
    const std::uint32_t c_bits = static_cast<std::uint32_t>(lo & ~hi);
    const std::uint32_t g_bits = static_cast<std::uint32_t>(~lo & hi);
    const std::uint32_t t_bits = static_cast<std::uint32_t>(lo & hi);
    const std::uint8_t c_bytes[8] = {
        static_cast<std::uint8_t>(c_bits),
        static_cast<std::uint8_t>(c_bits >> 8u),
        static_cast<std::uint8_t>(c_bits >> 16u),
        static_cast<std::uint8_t>(c_bits >> 24u),
        0u, 0u, 0u, 0u
    };
    const std::uint8_t g_bytes[8] = {
        static_cast<std::uint8_t>(g_bits),
        static_cast<std::uint8_t>(g_bits >> 8u),
        static_cast<std::uint8_t>(g_bits >> 16u),
        static_cast<std::uint8_t>(g_bits >> 24u),
        0u, 0u, 0u, 0u
    };
    const std::uint8_t t_bytes[8] = {
        static_cast<std::uint8_t>(t_bits),
        static_cast<std::uint8_t>(t_bits >> 8u),
        static_cast<std::uint8_t>(t_bits >> 16u),
        static_cast<std::uint8_t>(t_bits >> 24u),
        0u, 0u, 0u, 0u
    };
    const auto c_mask = hn::LoadMaskBits(d, c_bytes);
    const auto g_mask = hn::LoadMaskBits(d, g_bytes);
    const auto t_mask = hn::LoadMaskBits(d, t_bytes);
    auto chars = hn::Set(d, static_cast<std::uint8_t>('A'));
    chars = hn::IfThenElse(c_mask, hn::Set(d, static_cast<std::uint8_t>('C')), chars);
    chars = hn::IfThenElse(g_mask, hn::Set(d, static_cast<std::uint8_t>('G')), chars);
    chars = hn::IfThenElse(t_mask, hn::Set(d, static_cast<std::uint8_t>('T')), chars);
    hn::StoreU(chars, d, reinterpret_cast<std::uint8_t*>(output));
}

} // namespace

void dna2_pack_ascii_batch_highway(
    const char* input,
    std::size_t stride,
    dna2_word* output,
    std::size_t count,
    std::size_t n_per_seq) {
    assert(input != nullptr || count == 0u || n_per_seq == 0u);
    assert(output != nullptr || count == 0u);
    std::size_t i = 0u;
    if (n_per_seq == bases_per_word && stride >= bases_per_word) {
        for (; i < count; ++i) {
            output[i] = dna2_pack_ascii_32_highway(input + i * stride);
        }
        return;
    }
    for (; i < count; ++i) {
        output[i] = dna2_pack_ascii_32(input + i * stride, n_per_seq);
    }
}

void dna2_unpack_ascii_batch_highway(
    const dna2_word* input,
    char* output,
    std::size_t stride,
    std::size_t count,
    std::size_t n_per_seq) {
    assert(input != nullptr || count == 0u);
    assert(output != nullptr || count == 0u || n_per_seq == 0u);
    std::size_t i = 0u;
    if (n_per_seq == bases_per_word && stride >= bases_per_word) {
        for (; i < count; ++i) {
            dna2_unpack_ascii_32_highway(input[i], output + i * stride);
        }
        return;
    }
    for (; i < count; ++i) {
        dna2_unpack_ascii_32(input[i], output + i * stride, n_per_seq);
    }
}

void dna2_to_planes_batch_highway(const dna2_word* input, dna2_planes* output, std::size_t count) {
    assert(input != nullptr || count == 0u);
    assert(output != nullptr || count == 0u);
    const hn::CappedTag<std::uint64_t, lane_cap64> d;
    const std::size_t lanes = hn::Lanes(d);
    alignas(64) std::uint64_t lane_words[lane_cap64];
    alignas(64) std::uint64_t lane_lo[lane_cap64];
    alignas(64) std::uint64_t lane_hi[lane_cap64];
    std::size_t i = 0u;
    for (; i + lanes <= count; i += lanes) {
        for (std::size_t lane = 0u; lane < lanes; ++lane) {
            lane_words[lane] = input[i + lane].bits;
        }
        const auto words = hn::LoadU(d, lane_words);
        auto lo = words & hn::Set(d, 0x5555555555555555ULL);
        lo = (lo | hn::ShiftRight<1>(lo)) & hn::Set(d, 0x3333333333333333ULL);
        lo = (lo | hn::ShiftRight<2>(lo)) & hn::Set(d, 0x0f0f0f0f0f0f0f0fULL);
        lo = (lo | hn::ShiftRight<4>(lo)) & hn::Set(d, 0x00ff00ff00ff00ffULL);
        lo = (lo | hn::ShiftRight<8>(lo)) & hn::Set(d, 0x0000ffff0000ffffULL);
        lo = (lo | hn::ShiftRight<16>(lo)) & hn::Set(d, 0x00000000ffffffffULL);
        auto hi = hn::ShiftRight<1>(words) & hn::Set(d, 0x5555555555555555ULL);
        hi = (hi | hn::ShiftRight<1>(hi)) & hn::Set(d, 0x3333333333333333ULL);
        hi = (hi | hn::ShiftRight<2>(hi)) & hn::Set(d, 0x0f0f0f0f0f0f0f0fULL);
        hi = (hi | hn::ShiftRight<4>(hi)) & hn::Set(d, 0x00ff00ff00ff00ffULL);
        hi = (hi | hn::ShiftRight<8>(hi)) & hn::Set(d, 0x0000ffff0000ffffULL);
        hi = (hi | hn::ShiftRight<16>(hi)) & hn::Set(d, 0x00000000ffffffffULL);
        hn::StoreU(lo, d, lane_lo);
        hn::StoreU(hi, d, lane_hi);
        for (std::size_t lane = 0u; lane < lanes; ++lane) {
            output[i + lane] = dna2_planes{lane_lo[lane], lane_hi[lane]};
        }
    }
    for (; i < count; ++i) {
        output[i] = dna2_to_planes(input[i]);
    }
}

void planes_gc_mask_batch_highway(const dna2_planes* input, std::uint32_t* output_masks, std::size_t count) {
    assert(input != nullptr || count == 0u);
    assert(output_masks != nullptr || count == 0u);
    const hn::CappedTag<std::uint32_t, lane_cap32> d;
    const std::size_t lanes = hn::Lanes(d);
    alignas(64) std::uint32_t lane_lo[lane_cap32];
    alignas(64) std::uint32_t lane_hi[lane_cap32];
    alignas(64) std::uint32_t lane_out[lane_cap32];
    std::size_t i = 0u;
    for (; i + lanes <= count; i += lanes) {
        for (std::size_t lane = 0u; lane < lanes; ++lane) {
            lane_lo[lane] = static_cast<std::uint32_t>(input[i + lane].lo);
            lane_hi[lane] = static_cast<std::uint32_t>(input[i + lane].hi);
        }
        const auto lo = hn::LoadU(d, lane_lo);
        const auto hi = hn::LoadU(d, lane_hi);
        hn::StoreU(lo ^ hi, d, lane_out);
        for (std::size_t lane = 0u; lane < lanes; ++lane) {
            output_masks[i + lane] = lane_out[lane];
        }
    }
    for (; i < count; ++i) {
        output_masks[i] = planes_gc_mask(input[i]);
    }
}

void planes_cpg_start_mask_batch_highway(const dna2_planes* input, std::uint32_t* output_masks, std::size_t count) {
    assert(input != nullptr || count == 0u);
    assert(output_masks != nullptr || count == 0u);
    const hn::CappedTag<std::uint32_t, lane_cap32> d;
    const std::size_t lanes = hn::Lanes(d);
    alignas(64) std::uint32_t lane_lo[lane_cap32];
    alignas(64) std::uint32_t lane_hi[lane_cap32];
    alignas(64) std::uint32_t lane_out[lane_cap32];
    std::size_t i = 0u;
    for (; i + lanes <= count; i += lanes) {
        for (std::size_t lane = 0u; lane < lanes; ++lane) {
            lane_lo[lane] = static_cast<std::uint32_t>(input[i + lane].lo);
            lane_hi[lane] = static_cast<std::uint32_t>(input[i + lane].hi);
        }
        const auto lo = hn::LoadU(d, lane_lo);
        const auto hi = hn::LoadU(d, lane_hi);
        const auto c_mask = lo & hn::Not(hi);
        const auto g_mask = hn::Not(lo) & hi;
        const auto starts = c_mask & hn::ShiftRight<1>(g_mask) & hn::Set(d, 0x7fffffffu);
        hn::StoreU(starts, d, lane_out);
        for (std::size_t lane = 0u; lane < lanes; ++lane) {
            output_masks[i + lane] = lane_out[lane];
        }
    }
    for (; i < count; ++i) {
        output_masks[i] = planes_cpg_start_mask(input[i]);
    }
}

} // namespace cellerator::seq
