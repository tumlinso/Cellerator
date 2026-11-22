#pragma once

#include <Cellerator/layout/sequence/dna2.hh>
#include <cuda_runtime_api.h>

namespace cellerator::layout::sequence {

// Same layout contract as pack_planes. Borrow device storage through completion.
cudaError_t pack_planes_cuda(tensor::sequence::PackedView input, tensor::sequence::MutableBitPlanesView output,
                            cudaStream_t stream) noexcept;

}  // namespace cellerator::layout::sequence
