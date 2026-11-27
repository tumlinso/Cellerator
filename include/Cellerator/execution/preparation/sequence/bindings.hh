#pragma once

#include <Cellerator/sequence/chunk.hh>
#include <Cellerator/execution/validation.hh>
#include <Cellerator/sequence/dna2.hh>

#include <stdint.h>

namespace cellerator::execution::sequence {

// The persistent genome is resolved by the caller to genome_domain. Compact
// launch metadata preserves the existing biological operand ABI.
inline bool bind_sequence_chunk(const ::cellerator::sequence::SequenceChunk &source,
    domain_handle genome_domain, sequence_domain *out) noexcept {
    if (!out || !::cellerator::sequence::valid_sequence_chunk(source)
        || source.contig > UINT32_MAX || source.chunk > UINT32_MAX
        || source.halo_left > UINT16_MAX || source.halo_right > UINT16_MAX)
        return false;
    const sequence_domain domain{genome_domain,
        uint32_t(source.contig), uint32_t(source.chunk), source.global_base_begin,
        source.base_count, source.owned_begin, source.owned_end,
        uint16_t(source.halo_left), uint16_t(source.halo_right)};
    if (validate_sequence_domain(domain) != biological_validation_code::ok)
        return false;
    *out = domain;
    return true;
}

// Borrow the same payload with launch identity and placement. The runtime
// requires explicit validity for every nonempty sequence.
inline bool bind_sequence_planes(::cellerator::sequence::BitPlanesView planes,
    axis_identity axis, device_location location,
    biological_operand_view *out) noexcept {
    if (!out || planes.base_count > ::cellerator::sequence::max_chunk_bases) return false;
    const bit_plane_view bits{axis, planes.low, planes.high, planes.validity,
        location, planes.word_count, planes.base_count};
    if (validate_bit_plane(bits) != biological_validation_code::ok) return false;
    *out = {};
    out->kind = operand_kind::bit_plane;
    out->storage.bits = bits;
    return true;
}

constexpr ::cellerator::sequence::BitPlanesView sequence_planes(
    const bit_plane_view &view) noexcept {
    return {view.low, view.high, view.validity, view.word_count, view.base_count};
}

} // namespace cellerator::execution::sequence
