#pragma once
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace cellerator::experimental::moonshot {
cudaError_t launch_patch16(const __half* l, const __half* x, const __half* r,
                          float* y, std::uint32_t patches, cudaStream_t stream);
namespace patch16 {
// Capacities are element counts. Matrices are contiguous row-major [patches,16,16].
// Borrowed storage; the caller retains ownership until the stream completes.
struct Request {
    const __half* l{};
    const __half* x{};
    const __half* r{};
    float* y{};
    std::size_t l_capacity{}, x_capacity{}, r_capacity{}, y_capacity{};
    std::uint32_t patches{};
    int device{};
    int compute_major{7}, compute_minor{};
};
// Pure host validation trusts the supplied capacity and architecture metadata.
cudaError_t validate(const Request& request);
// Queries runtime device and pointer provenance; no allocation or synchronization.
cudaError_t launch_prepared(const Request& request, cudaStream_t stream);
} // namespace patch16
} // namespace cellerator::experimental::moonshot
