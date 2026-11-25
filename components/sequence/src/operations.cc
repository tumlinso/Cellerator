#include "operations_detail.hh"

namespace cellerator::sequence {
namespace {

template<detail::Mask mask>
Status mask_cpu(BitPlanesView input, uint8_t code, uint32_t *out, uint32_t capacity) noexcept {
    Status status = detail::validate_mask(input, code, out, capacity);
    if (status != Status::ok) return status;
    const uint32_t n = word_count32(input.base_count);
    for (uint32_t i = 0; i < n; ++i) out[i] = detail::mask_word<mask>(input, i, code);
    return Status::ok;
}

}  // namespace

Status count_motif(PackedView input, motif32_exact motif, uint64_t *out) noexcept {
    if (!out || !valid_view(input) || !dna2_valid_motif32_exact(motif)) return Status::invalid_argument;
    uint64_t hits = 0;
    const uint32_t n = detail::window_count(input, motif);
    for (uint32_t i = 0; i < n; ++i) hits += detail::match_mismatches(input, i, motif) <= motif.max_mismatches;
    *out = hits;
    return Status::ok;
}

Status emit_motif(PackedView input, motif32_exact motif, sequence_event *out,
                  uint32_t capacity, sequence_emit_counts *counts) noexcept {
    if ((capacity && !out) || !counts || !valid_view(input) || !dna2_valid_motif32_exact(motif))
        return Status::invalid_argument;
    uint32_t hits = 0;
    const uint32_t n = detail::window_count(input, motif);
    for (uint32_t i = 0; i < n; ++i) {
        uint8_t errors = detail::match_mismatches(input, i, motif);
        if (errors > motif.max_mismatches) continue;
        if (hits < capacity) out[hits] = {i, motif.motif_id, errors, 0u};
        ++hits;
    }
    uint32_t stored = hits < capacity ? hits : capacity;
    *counts = {hits, stored, hits - stored, hits};
    return Status::ok;
}

Status base_mask(BitPlanesView input, uint8_t code, uint32_t *out, uint32_t capacity) noexcept {
    return mask_cpu<detail::Mask::base>(input, code, out, capacity);
}

Status gc_mask(BitPlanesView input, uint32_t *out, uint32_t capacity) noexcept {
    return mask_cpu<detail::Mask::gc>(input, 0, out, capacity);
}

Status cpg_start_mask(BitPlanesView input, uint32_t *out, uint32_t capacity) noexcept {
    return mask_cpu<detail::Mask::cpg>(input, 0, out, capacity);
}

}  // namespace cellerator::sequence
