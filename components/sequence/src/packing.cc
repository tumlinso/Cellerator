#include "packing_detail.hh"

#include <Cellerator/sequence/chunk.hh>

namespace cellerator::sequence {

Status pack_ascii(const char *bases, uint32_t n, uint64_t *words, uint32_t word_capacity,
                  uint64_t *validity, uint32_t validity_capacity, PackedView *out) noexcept {
    if (!out || n > max_chunk_bases || (n && !bases)) return Status::invalid_argument;
    uint32_t nw = word_count32(n), nv = word_count64(n);
    if (nw > word_capacity || nv > validity_capacity) return Status::insufficient_capacity;
    if (n && (!words || !validity)) return Status::invalid_argument;
    for (uint32_t w = 0; w < nw; ++w) {
        uint32_t start = 32u*w, size = n - start < 32u ? n - start : 32u;
        uint64_t word = 0, valid = 0;
        for (uint32_t j = 0; j < size; ++j) {
            auto base = encode_nucleotide(bases[start + j]);
            word |= uint64_t(base.code) << (2u*j);
            valid |= uint64_t(base.valid) << j;
        }
        words[w] = word;
        if (w%2u) validity[w/2u] |= valid << 32u;
        else validity[w/2u] = valid;
    }
    *out = {words, validity, nw, n, nv};
    return Status::ok;
}

Status pack_planes(PackedView input, MutableBitPlanesView output) noexcept {
    Status status = detail::validate_output(input, output);
    if (status != Status::ok) return status;
    uint32_t n = word_count32(input.base_count);
    for (uint32_t i = 0; i < n; ++i) detail::pack_word(input, output, i);
    return Status::ok;
}

}  // namespace cellerator::sequence
