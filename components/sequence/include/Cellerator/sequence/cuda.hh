#pragma once

#include <Cellerator/sequence/operations.hh>
#include <cuda_runtime_api.h>

namespace cellerator::sequence {

// Same layout contract as pack_planes. Borrow device storage through completion.
cudaError_t pack_planes_cuda(PackedView input, MutableBitPlanesView output,
                            cudaStream_t stream) noexcept;

// The caller retains all device arrays and counters through stream completion.
// Inputs and writable arrays follow the scalar operations' disjoint contracts.
cudaError_t count_motif_cuda(PackedView input, motif32_exact motif,
                             unsigned long long *out, cudaStream_t stream) noexcept;
// Emission order is unspecified. A short output can retain any subset of the
// requested size. Counts cover the full scan, even when capacity is zero.
cudaError_t emit_motif_cuda(PackedView input, motif32_exact motif,
                            sequence_event *out, uint32_t capacity, sequence_emit_counts *counts,
                            cudaStream_t stream) noexcept;
cudaError_t base_mask_cuda(BitPlanesView input, uint8_t code,
                           uint32_t *out, uint32_t capacity, cudaStream_t stream) noexcept;
cudaError_t gc_mask_cuda(BitPlanesView input, uint32_t *out,
                         uint32_t capacity, cudaStream_t stream) noexcept;
cudaError_t cpg_start_mask_cuda(BitPlanesView input, uint32_t *out,
                                uint32_t capacity, cudaStream_t stream) noexcept;

}  // namespace cellerator::sequence
