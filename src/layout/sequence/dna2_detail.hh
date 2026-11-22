#pragma once

#include <Cellerator/common/compiler_annotations.hh>
#include <Cellerator/layout/sequence/dna2.hh>

namespace cellerator::layout::sequence::detail {

inline Status validate_output(tensor::sequence::PackedView input, tensor::sequence::MutableBitPlanesView output) noexcept {
    const uint32_t n = tensor::sequence::word_count32(input.base_count);
    if (!tensor::sequence::valid_view(input)) return Status::invalid_argument;
    if (n > output.word_capacity) return Status::insufficient_capacity;
    if (n && (!output.low || !output.high || (input.validity && !output.validity))) return Status::invalid_argument;
    return Status::ok;
}

CELLERATOR_HOST_DEVICE CELLERATOR_CUDA_FORCEINLINE void pack_word(tensor::sequence::PackedView input,
                                      tensor::sequence::MutableBitPlanesView output, uint32_t i) noexcept {
    namespace s = tensor::sequence;
    uint32_t lo, hi;
    s::split_word(input.words[i], &lo, &hi);
    uint32_t tail = s::low_mask32(input.base_count - 32u*i);
    output.low[i] = lo & tail;
    output.high[i] = hi & tail;
    if (output.validity) output.validity[i] = s::validity_word32(input, i);
}

}  // namespace cellerator::layout::sequence::detail
