#pragma once

#include <stddef.h>
#include <stdint.h>
#include <cusparse.h>
#include <cuda_runtime_api.h>

#include <Cellerator/memory/view.hh>
#include <Cellerator/tensor/base/compressed.hh>

namespace cellerator::ops::matrix::cusparse {

using csr_view = tensor::base::CompressedView<int32_t, const void>;
// CUDA Toolkit 12.4 first shipped the API, in cuSPARSE 12.3.0.
inline constexpr bool spmv_preprocess_available = CUSPARSE_VERSION >= 12300;

// Nonempty, zero-based CSR with unique columns within each row. The caller
// validates the indices and provides disjoint device storage for inputs, output,
// and workspace. The handle stays on its device and stream in HOST pointer mode.
// FP32 is used for sparse and dense storage, accumulation, output and scalars.
// Exact FP16 sparse with FP32 dense operands is handled by the native kernels.
//
// Provision an empty plan, then bind workspace (including when size is zero).
// Enqueue requires successful provisioning and binding, and successful rebinding
// if addresses changed. Serialize host calls. Keep the handle, buffers and
// descriptors live through stream completion. Wait for prior work before
// changing descriptors, replacing workspace, or releasing the plan. Keep the
// preprocessing workspace at the same address and intact until release. The plan
// performs no waits.
class spmv_plan {
public:
    spmv_plan() = default;
    spmv_plan(const spmv_plan &) = delete;
    spmv_plan &operator=(const spmv_plan &) = delete;
    ~spmv_plan() { release(); }

    cusparseStatus_t provision(cusparseHandle_t, cusparseOperation_t, csr_view,
        cudaDataType_t sparse_type, memory::const_array_view<float>,
        memory::array_view<float>, cusparseSpMVAlg_t) noexcept;
    size_t workspace_bytes() const noexcept { return bytes_; }
    cusparseStatus_t bind_workspace(void *, size_t, bool preprocess = false) noexcept;
    cusparseStatus_t rebind(const void *values, const float *x, float *y) noexcept;
    cusparseStatus_t enqueue(float alpha = 1.0f, float beta = 0.0f) const noexcept;
    void release() noexcept;

private:
    cusparseHandle_t handle_ = nullptr;
    cusparseSpMatDescr_t a_ = nullptr;
    cusparseDnVecDescr_t x_ = nullptr, y_ = nullptr;
    cusparseOperation_t op_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpMVAlg_t alg_ = CUSPARSE_SPMV_ALG_DEFAULT;
    void *work_ = nullptr;
    size_t bytes_ = 0;
    bool preprocessed_ = false;
};

// SpMM has the same lifetime contract, with row-major dense operands and
// element leading dimensions that may exceed the number of columns.
class spmm_plan {
public:
    spmm_plan() = default;
    spmm_plan(const spmm_plan &) = delete;
    spmm_plan &operator=(const spmm_plan &) = delete;
    ~spmm_plan() { release(); }

    cusparseStatus_t provision(cusparseHandle_t, cusparseOperation_t, csr_view,
        cudaDataType_t sparse_type, memory::const_matrix_view<float>,
        memory::matrix_view<float>, cusparseSpMMAlg_t) noexcept;
    size_t workspace_bytes() const noexcept { return bytes_; }
    cusparseStatus_t bind_workspace(void *, size_t, bool preprocess = false) noexcept;
    cusparseStatus_t rebind(const void *values, const float *x, float *y) noexcept;
    cusparseStatus_t enqueue(float alpha = 1.0f, float beta = 0.0f) const noexcept;
    void release() noexcept;

private:
    cusparseHandle_t handle_ = nullptr;
    cusparseSpMatDescr_t a_ = nullptr;
    cusparseDnMatDescr_t x_ = nullptr, y_ = nullptr;
    cusparseOperation_t op_ = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpMMAlg_t alg_ = CUSPARSE_SPMM_ALG_DEFAULT;
    void *work_ = nullptr;
    size_t bytes_ = 0;
    bool preprocessed_ = false;
};

// FP32 samples of left * right^T in the supplied CSR order. The descriptor
// and workspace lifetime rules match spmm_plan.
class sddmm_plan {
public:
    sddmm_plan() = default;
    sddmm_plan(const sddmm_plan&) = delete;
    sddmm_plan& operator=(const sddmm_plan&) = delete;
    ~sddmm_plan() { release(); }

    cusparseStatus_t provision(cusparseHandle_t, csr_view,
        memory::const_matrix_view<float> left, memory::const_matrix_view<float> right) noexcept;
    size_t workspace_bytes() const noexcept { return bytes_; }
    cusparseStatus_t bind_workspace(void*, size_t) noexcept;
    cusparseStatus_t enqueue() const noexcept;
    void release() noexcept;

private:
    cusparseHandle_t handle_ = nullptr;
    cusparseSpMatDescr_t samples_ = nullptr;
    cusparseDnMatDescr_t left_ = nullptr, right_ = nullptr;
    void* work_ = nullptr;
    size_t bytes_ = 0;
};

cudaError_t refresh_csr_values(float* csr, const float* physical, const uint32_t* csr_to_physical,
    uint32_t count, cudaStream_t) noexcept;
cudaError_t scatter_csr_samples(float* physical, const float* csr, const uint32_t* csr_to_physical,
    uint32_t count, cudaStream_t) noexcept;
cudaError_t update_and_refresh_csr(float* physical, float* csr, const float* operand,
    const uint32_t* csr_to_physical, uint32_t count, bool gradient_step, float rate,
    bool operand_in_csr_order, cudaStream_t) noexcept;

} // namespace cellerator::ops::matrix::cusparse
