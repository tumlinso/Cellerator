#pragma once

#include <Cellerator/structure/identity.hh>
#include <stdint.h>

namespace cellerator::sequence {

enum class SequenceStrand : uint8_t { forward = 1, reverse = 2 };

struct EncodedNucleotide {
    uint8_t code;
    bool valid;
};

constexpr EncodedNucleotide encode_nucleotide(char base) noexcept {
    switch (base) {
    case 'A': case 'a': return {0u, true};
    case 'C': case 'c': return {1u, true};
    case 'G': case 'g': return {2u, true};
    case 'T': case 't': case 'U': case 'u': return {3u, true};
    default: return {0u, false};
    }
}

// Positions in a chunk increase in the original genomic coordinate system.
struct SequenceChunk {
    structure::domain_id genome;
    uint64_t contig, chunk, global_base_begin;
    uint32_t base_count, owned_begin, owned_end, halo_left, halo_right;
};

constexpr bool valid_sequence_chunk(const SequenceChunk &chunk) noexcept {
    return structure::valid_identity(chunk.genome) && chunk.base_count <= INT32_MAX
        && chunk.owned_begin <= chunk.owned_end && chunk.owned_end <= chunk.base_count
        && chunk.halo_left == chunk.owned_begin
        && chunk.halo_right == chunk.base_count - chunk.owned_end
        && (chunk.base_count == 0u
            || chunk.global_base_begin <= UINT64_MAX - (chunk.base_count - 1u));
}

} // namespace cellerator::sequence
