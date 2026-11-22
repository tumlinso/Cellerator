#include <Cellerator/ops/sequence/dna2_cuda.hh>

#include "dna2_detail.hh"

namespace cellerator::ops::sequence {
namespace s = tensor::sequence;
namespace {

constexpr uint32_t threads = 256;

uint32_t scan_blocks(uint32_t n) noexcept {
    uint32_t blocks = (n + threads - 1u)/threads;
    return blocks < 8192u ? blocks : 8192u;
}

__global__ void count_kernel(s::PackedView input, motif32_exact motif, uint32_t n, unsigned long long *out) {
    __shared__ uint32_t block_count[threads];
    uint32_t lane = threadIdx.x, hits = 0;
    uint32_t stride = gridDim.x*blockDim.x;
    for (uint32_t i = blockIdx.x*blockDim.x + lane; i < n; i += stride)
        hits += detail::match_mismatches(input, i, motif) <= motif.max_mismatches;
    block_count[lane] = hits;
    __syncthreads();
    for (uint32_t k = threads/2; k; k /= 2u) {
        if (lane < k) block_count[lane] += block_count[lane + k];
        __syncthreads();
    }
    if (!lane) atomicAdd(out, block_count[0]);
}

__global__ void emit_kernel(s::PackedView input, motif32_exact motif, uint32_t n, sequence_event *out,
                            uint32_t capacity, sequence_emit_counts *counts) {
    uint32_t stride = gridDim.x*blockDim.x;
    for (uint32_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride) {
        uint8_t errors = detail::match_mismatches(input, i, motif);
        if (errors > motif.max_mismatches) continue;
        unsigned long long slot = atomicAdd(&counts->total_matches, 1ULL);
        if (slot < capacity) out[slot] = {i, motif.motif_id, errors, 0u};
    }
}

__global__ void finish_emit(sequence_emit_counts *counts, uint32_t capacity) {
    unsigned long long hits = counts->total_matches, stored = hits < capacity ? hits : capacity;
    *counts = {hits, stored, hits - stored, hits};
}

template<detail::Mask mask>
__global__ void mask_kernel(s::BitPlanesView input, uint8_t code, uint32_t *out, uint32_t n) {
    uint32_t stride = gridDim.x*blockDim.x;
    for (uint32_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride)
        out[i] = detail::mask_word<mask>(input, i, code);
}

template<detail::Mask mask>
cudaError_t mask_cuda(s::BitPlanesView input, uint8_t code, uint32_t *out, uint32_t capacity,
                      cudaStream_t stream) noexcept {
    if (detail::validate_mask(input, code, out, capacity) != Status::ok) return cudaErrorInvalidValue;
    uint32_t n = s::word_count32(input.base_count);
    if (!n) return cudaSuccess;
    mask_kernel<mask><<<scan_blocks(n), threads, 0, stream>>>(input, code, out, n);
    return cudaGetLastError();
}

}  // namespace

cudaError_t count_motif_cuda(s::PackedView input, motif32_exact motif, unsigned long long *out,
                             cudaStream_t stream) noexcept {
    if (!out || !s::valid_view(input) || !dna2_valid_motif32_exact(motif)) return cudaErrorInvalidValue;
    uint32_t n = detail::window_count(input, motif);
    cudaError_t error = cudaMemsetAsync(out, 0, sizeof(*out), stream);
    if (error != cudaSuccess || !n) return error;
    count_kernel<<<scan_blocks(n), threads, 0, stream>>>(input, motif, n, out);
    return cudaGetLastError();
}

cudaError_t emit_motif_cuda(s::PackedView input, motif32_exact motif, sequence_event *out, uint32_t capacity,
                            sequence_emit_counts *counts, cudaStream_t stream) noexcept {
    if ((capacity && !out) || !counts || !s::valid_view(input) || !dna2_valid_motif32_exact(motif))
        return cudaErrorInvalidValue;
    uint32_t n = detail::window_count(input, motif);
    cudaError_t error = cudaMemsetAsync(counts, 0, sizeof(*counts), stream);
    if (error != cudaSuccess || !n) return error;
    emit_kernel<<<scan_blocks(n), threads, 0, stream>>>(input, motif, n, out, capacity, counts);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    finish_emit<<<1, 1, 0, stream>>>(counts, capacity);
    return cudaGetLastError();
}

cudaError_t base_mask_cuda(s::BitPlanesView input, uint8_t code, uint32_t *out, uint32_t capacity,
                           cudaStream_t stream) noexcept {
    return mask_cuda<detail::Mask::base>(input, code, out, capacity, stream);
}

cudaError_t gc_mask_cuda(s::BitPlanesView input, uint32_t *out, uint32_t capacity, cudaStream_t stream) noexcept {
    return mask_cuda<detail::Mask::gc>(input, 0, out, capacity, stream);
}

cudaError_t cpg_start_mask_cuda(s::BitPlanesView input, uint32_t *out, uint32_t capacity, cudaStream_t stream) noexcept {
    return mask_cuda<detail::Mask::cpg>(input, 0, out, capacity, stream);
}

}  // namespace cellerator::ops::sequence
