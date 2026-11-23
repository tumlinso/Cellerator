#pragma once

#include <Cellerator/ops/matrix/narrow_matmul.hh>
#include <cublasLt.h>

namespace cellerator::ops::matrix {

struct NarrowCudaWorkspace {
    float *decoded_a, *decoded_b;
    size_t decoded_a_elements, decoded_b_elements;
    void *library;
    size_t library_bytes;
};

// All buffers are device allocations. Keep the bound buffers, handle and
// prepared state live through stream completion. A prepared instance is used
// by one submitted stream at a time. Preparation uses the handle's device.
class NarrowMatmulCuda {
public:
    NarrowMatmulCuda() = default;
    ~NarrowMatmulCuda();
    NarrowMatmulCuda(const NarrowMatmulCuda &) = delete;
    NarrowMatmulCuda &operator=(const NarrowMatmulCuda &) = delete;

    NarrowStatus prepare(cublasLtHandle_t handle, const NarrowMatmul &p,
                         const NarrowCudaWorkspace &workspace) noexcept;
    NarrowStatus enqueue(cudaStream_t stream) const noexcept;

private:
    void clear() noexcept;
    NarrowStatus select(cudaDataType_t a, cudaDataType_t b, cublasComputeType_t compute) noexcept;

    cublasLtHandle_t handle_ = nullptr;
    cublasLtMatmulDesc_t operation_ = nullptr;
    cublasLtMatrixLayout_t a_ = nullptr, b_ = nullptr, output_ = nullptr;
    cublasLtMatmulAlgo_t algorithm_{};
    NarrowMatmul p_{};
    NarrowCudaWorkspace workspace_{};
    bool decode_ = false, ready_ = false;
};

// One device result and up to min(64, ceil(count/1024)) caller-owned partials.
// For an empty dot, a, b and partial may be null.
constexpr size_t int8_dot_cuda_partials(size_t count) noexcept {
    return count > int8_dot_max_count ? 0 : count > 64*1024 ? 64 : (count + 1023)/1024;
}
NarrowStatus int8_dot_cuda(const int8_t *a, const int8_t *b, size_t count,
                           int32_t *partial, size_t partial_count,
                           int32_t *out, cudaStream_t stream) noexcept;

}
