#include <Cellerator/sequence/cuda.hh>

#include "packing_detail.hh"

namespace cellerator::sequence {
namespace {

__global__ void pack_planes_kernel(PackedView input, MutableBitPlanesView output, uint32_t n) {
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) detail::pack_word(input, output, i);
}

}  // namespace

cudaError_t pack_planes_cuda(PackedView input, MutableBitPlanesView output, cudaStream_t stream) noexcept {
    if (detail::validate_output(input, output) != Status::ok) return cudaErrorInvalidValue;
    uint32_t n = word_count32(input.base_count);
    if (!n) return cudaSuccess;
    constexpr uint32_t threads = 256;
    pack_planes_kernel<<<(n + threads - 1u)/threads, threads, 0, stream>>>(input, output, n);
    return cudaGetLastError();
}

}  // namespace cellerator::sequence
