#pragma once

#include <Cellerator/sequence/motif.hh>
#include <Cellerator/sequence/sequence_event.hh>

namespace cellerator::sequence {

// Forward scan in increasing local coordinate order. A window containing an
// invalid base cannot match. Input and output arrays are disjoint.
Status count_motif(PackedView input, motif32_exact motif, uint64_t *out) noexcept;
// Emit the prefix that fits and account for all matches, including at capacity
// zero. The input, records and counters occupy disjoint storage.
Status emit_motif(PackedView input, motif32_exact motif, sequence_event *out,
                  uint32_t capacity, sequence_emit_counts *counts) noexcept;

// Mask words follow input coordinate order in disjoint output storage. Invalid
// positions and tail bits are clear. CpG requires both bases valid, including
// across a word seam.
Status base_mask(BitPlanesView input, uint8_t code, uint32_t *out, uint32_t capacity) noexcept;
Status gc_mask(BitPlanesView input, uint32_t *out, uint32_t capacity) noexcept;
Status cpg_start_mask(BitPlanesView input, uint32_t *out, uint32_t capacity) noexcept;

}  // namespace cellerator::sequence
