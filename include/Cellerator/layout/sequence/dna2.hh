#pragma once

#include <Cellerator/tensor/sequence/dna2.hh>

namespace cellerator::layout::sequence {

enum class Status { ok, invalid_argument, insufficient_capacity };

// Input text, output arrays and the result view occupy disjoint storage.
// Invalid input bytes clear validity.
// Query output extents with tensor::sequence::word_count32 and word_count64.
Status pack_ascii(const char *bases, uint32_t base_count, uint64_t *words, uint32_t word_capacity,
                  uint64_t *validity, uint32_t validity_capacity, tensor::sequence::PackedView *out) noexcept;

// Input storage and all output arrays are mutually disjoint. Explicit input
// validity requires an output validity buffer. Conversion retains base order.
Status pack_planes(tensor::sequence::PackedView input, tensor::sequence::MutableBitPlanesView output) noexcept;

}  // namespace cellerator::layout::sequence
