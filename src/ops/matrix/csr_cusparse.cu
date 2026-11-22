#include "csr_cusparse.cuh"

namespace cellerator::ops::matrix::cusparse {
namespace {

bool valid_csr(csr_view a, cusparseOperation_t op) noexcept {
    const auto &s = a.structure;
    return (op == CUSPARSE_OPERATION_NON_TRANSPOSE || op == CUSPARSE_OPERATION_TRANSPOSE)
        && s.axis == tensor::base::CompressedAxis::row && s.rows && s.cols && s.nnz
        && s.rows <= INT32_MAX && s.cols <= INT32_MAX && s.nnz <= INT32_MAX
        && s.major_ptr && s.minor_idx && a.values;
}

cusparseStatus_t create_csr(cusparseHandle_t handle, csr_view a,
    cusparseSpMatDescr_t *descriptor) noexcept {
    cusparsePointerMode_t mode;
    const auto status = cusparseGetPointerMode(handle, &mode);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    if (mode != CUSPARSE_POINTER_MODE_HOST) return CUSPARSE_STATUS_INVALID_VALUE;

    const auto &s = a.structure;
    return cusparseCreateCsr(descriptor, int64_t(s.rows), int64_t(s.cols), int64_t(s.nnz),
        const_cast<int32_t *>(s.major_ptr), const_cast<int32_t *>(s.minor_idx),
        const_cast<void *>(a.values), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
}

} // namespace

cusparseStatus_t spmv_plan::provision(cusparseHandle_t handle, cusparseOperation_t op,
    csr_view a, cudaDataType_t sparse_type, memory::const_array_view<float> x,
    memory::array_view<float> y, cusparseSpMVAlg_t alg) noexcept {
    if (sparse_type != CUDA_R_32F) return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (alg != CUSPARSE_SPMV_ALG_DEFAULT && alg != CUSPARSE_SPMV_CSR_ALG1
        && alg != CUSPARSE_SPMV_CSR_ALG2) return CUSPARSE_STATUS_NOT_SUPPORTED;
    const bool transpose = op == CUSPARSE_OPERATION_TRANSPOSE;
    const uint64_t m = transpose ? a.structure.cols : a.structure.rows;
    const uint64_t n = transpose ? a.structure.rows : a.structure.cols;
    if (handle_ || !handle || !valid_csr(a, op) || !x.data || !y.data
        || x.count != n || y.count != m) return CUSPARSE_STATUS_INVALID_VALUE;

    size_t bytes = 0;
    const float alpha = 1, beta = 0;
    auto status = create_csr(handle, a, &a_);
    if (status == CUSPARSE_STATUS_SUCCESS)
        status = cusparseCreateDnVec(&x_, int64_t(n), const_cast<float *>(x.data), CUDA_R_32F);
    if (status == CUSPARSE_STATUS_SUCCESS)
        status = cusparseCreateDnVec(&y_, int64_t(m), y.data, CUDA_R_32F);
    if (status == CUSPARSE_STATUS_SUCCESS)
        status = cusparseSpMV_bufferSize(handle, op, &alpha, a_, x_, &beta, y_,
            CUDA_R_32F, alg, &bytes);
    if (status != CUSPARSE_STATUS_SUCCESS) { release(); return status; }

    handle_ = handle; op_ = op; alg_ = alg; bytes_ = bytes;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t spmv_plan::bind_workspace(void *work, size_t bytes, bool preprocess) noexcept {
    if (!handle_ || bytes < bytes_ || (bytes_ && !work) || (preprocessed_ && work != work_))
        return CUSPARSE_STATUS_INVALID_VALUE;
    if (preprocess && !preprocessed_) {
#if CUSPARSE_VERSION >= 12300
        const float alpha = 1, beta = 0;
        const auto status = cusparseSpMV_preprocess(handle_, op_, &alpha, a_, x_, &beta, y_,
            CUDA_R_32F, alg_, work);
        if (status != CUSPARSE_STATUS_SUCCESS) return status;
        preprocessed_ = true;
#else
        return CUSPARSE_STATUS_NOT_SUPPORTED;
#endif
    }
    work_ = work;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t spmv_plan::rebind(const void *values, const float *x, float *y) noexcept {
    if (!handle_ || !values || !x || !y) return CUSPARSE_STATUS_INVALID_VALUE;
    auto status = cusparseSpMatSetValues(a_, const_cast<void *>(values));
    if (status == CUSPARSE_STATUS_SUCCESS) status = cusparseDnVecSetValues(x_, const_cast<float *>(x));
    if (status == CUSPARSE_STATUS_SUCCESS) status = cusparseDnVecSetValues(y_, y);
    return status;
}

cusparseStatus_t spmv_plan::enqueue(float alpha, float beta) const noexcept {
    return cusparseSpMV(handle_, op_, &alpha, a_, x_, &beta, y_, CUDA_R_32F, alg_, work_);
}

void spmv_plan::release() noexcept {
    if (y_) (void)cusparseDestroyDnVec(y_);
    if (x_) (void)cusparseDestroyDnVec(x_);
    if (a_) (void)cusparseDestroySpMat(a_);
    handle_ = nullptr; a_ = nullptr; x_ = y_ = nullptr;
    work_ = nullptr; bytes_ = 0; preprocessed_ = false;
}

cusparseStatus_t spmm_plan::provision(cusparseHandle_t handle, cusparseOperation_t op,
    csr_view a, cudaDataType_t sparse_type, memory::const_matrix_view<float> x,
    memory::matrix_view<float> y, cusparseSpMMAlg_t alg) noexcept {
    if (sparse_type != CUDA_R_32F) return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (alg != CUSPARSE_SPMM_ALG_DEFAULT && alg != CUSPARSE_SPMM_CSR_ALG1
        && alg != CUSPARSE_SPMM_CSR_ALG2 && alg != CUSPARSE_SPMM_CSR_ALG3)
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    const bool transpose = op == CUSPARSE_OPERATION_TRANSPOSE;
    if (transpose && alg == CUSPARSE_SPMM_CSR_ALG3) return CUSPARSE_STATUS_NOT_SUPPORTED;
    const uint64_t m = transpose ? a.structure.cols : a.structure.rows;
    const uint64_t n = transpose ? a.structure.rows : a.structure.cols;
    const size_t width = x.columns;
    if (handle_ || !handle || !valid_csr(a, op) || !x.data || !y.data || !width
        || x.rows != n || y.rows != m || y.columns != width
        || x.column_stride != 1 || y.column_stride != 1
        || x.row_stride < width || y.row_stride < width
        || x.row_stride > INT64_MAX || y.row_stride > INT64_MAX)
        return CUSPARSE_STATUS_INVALID_VALUE;

    size_t bytes = 0;
    const float alpha = 1, beta = 0;
    auto status = create_csr(handle, a, &a_);
    if (status == CUSPARSE_STATUS_SUCCESS)
        status = cusparseCreateDnMat(&x_, int64_t(n), int64_t(width), int64_t(x.row_stride),
            const_cast<float *>(x.data), CUDA_R_32F, CUSPARSE_ORDER_ROW);
    if (status == CUSPARSE_STATUS_SUCCESS)
        status = cusparseCreateDnMat(&y_, int64_t(m), int64_t(width), int64_t(y.row_stride),
            y.data, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    if (status == CUSPARSE_STATUS_SUCCESS)
        status = cusparseSpMM_bufferSize(handle, op, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, a_, x_, &beta, y_, CUDA_R_32F, alg, &bytes);
    if (status != CUSPARSE_STATUS_SUCCESS) { release(); return status; }

    handle_ = handle; op_ = op; alg_ = alg; bytes_ = bytes;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t spmm_plan::bind_workspace(void *work, size_t bytes, bool preprocess) noexcept {
    if (!handle_ || bytes < bytes_ || (bytes_ && !work) || (preprocessed_ && work != work_))
        return CUSPARSE_STATUS_INVALID_VALUE;
    if (preprocess && !preprocessed_) {
        if (alg_ != CUSPARSE_SPMM_CSR_ALG1 && alg_ != CUSPARSE_SPMM_CSR_ALG3)
            return CUSPARSE_STATUS_NOT_SUPPORTED;
        const float alpha = 1, beta = 0;
        const auto status = cusparseSpMM_preprocess(handle_, op_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, a_, x_, &beta, y_, CUDA_R_32F, alg_, work);
        if (status != CUSPARSE_STATUS_SUCCESS) return status;
        preprocessed_ = true;
    }
    work_ = work;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t spmm_plan::rebind(const void *values, const float *x, float *y) noexcept {
    if (!handle_ || !values || !x || !y) return CUSPARSE_STATUS_INVALID_VALUE;
    auto status = cusparseSpMatSetValues(a_, const_cast<void *>(values));
    if (status == CUSPARSE_STATUS_SUCCESS) status = cusparseDnMatSetValues(x_, const_cast<float *>(x));
    if (status == CUSPARSE_STATUS_SUCCESS) status = cusparseDnMatSetValues(y_, y);
    return status;
}

cusparseStatus_t spmm_plan::enqueue(float alpha, float beta) const noexcept {
    return cusparseSpMM(handle_, op_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, a_, x_, &beta, y_, CUDA_R_32F, alg_, work_);
}

void spmm_plan::release() noexcept {
    if (y_) (void)cusparseDestroyDnMat(y_);
    if (x_) (void)cusparseDestroyDnMat(x_);
    if (a_) (void)cusparseDestroySpMat(a_);
    handle_ = nullptr; a_ = nullptr; x_ = y_ = nullptr;
    work_ = nullptr; bytes_ = 0; preprocessed_ = false;
}

cusparseStatus_t sddmm_plan::provision(cusparseHandle_t handle, csr_view csr,
    memory::const_matrix_view<float> left, memory::const_matrix_view<float> right) noexcept {
    if (handle_ || !handle || !valid_csr(csr, CUSPARSE_OPERATION_NON_TRANSPOSE)
        || !left.data || !right.data || left.rows != csr.structure.rows || right.rows != csr.structure.cols
        || !left.columns || left.columns != right.columns || left.column_stride != 1 || right.column_stride != 1
        || left.row_stride < left.columns || right.row_stride < right.columns
        || left.row_stride > INT64_MAX || right.row_stride > INT64_MAX)
        return CUSPARSE_STATUS_INVALID_VALUE;
    auto status = create_csr(handle, csr, &samples_);
    if (status == CUSPARSE_STATUS_SUCCESS)
        status = cusparseCreateDnMat(&left_, left.rows, left.columns, left.row_stride,
            const_cast<float*>(left.data), CUDA_R_32F, CUSPARSE_ORDER_ROW);
    if (status == CUSPARSE_STATUS_SUCCESS)
        status = cusparseCreateDnMat(&right_, right.rows, right.columns, right.row_stride,
            const_cast<float*>(right.data), CUDA_R_32F, CUSPARSE_ORDER_ROW);
    const float one = 1, zero = 0;
    if (status == CUSPARSE_STATUS_SUCCESS)
        status = cusparseSDDMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, &one, left_, right_, &zero, samples_, CUDA_R_32F,
            CUSPARSE_SDDMM_ALG_DEFAULT, &bytes_);
    if (status == CUSPARSE_STATUS_SUCCESS) handle_ = handle;
    else release();
    return status;
}

cusparseStatus_t sddmm_plan::bind_workspace(void* work, size_t bytes) noexcept {
    if (!handle_ || bytes < bytes_ || (bytes_ && !work)) return CUSPARSE_STATUS_INVALID_VALUE;
    const float one = 1, zero = 0;
    auto status = cusparseSDDMM_preprocess(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE, &one, left_, right_, &zero, samples_, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, work);
    if (status == CUSPARSE_STATUS_SUCCESS) work_ = work;
    return status;
}

cusparseStatus_t sddmm_plan::enqueue() const noexcept {
    const float one = 1, zero = 0;
    return cusparseSDDMM(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
        &one, left_, right_, &zero, samples_, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, work_);
}

void sddmm_plan::release() noexcept {
    if (right_) (void)cusparseDestroyDnMat(right_);
    if (left_) (void)cusparseDestroyDnMat(left_);
    if (samples_) (void)cusparseDestroySpMat(samples_);
    right_ = left_ = nullptr;
    samples_ = nullptr;
    handle_ = nullptr;
    work_ = nullptr;
    bytes_ = 0;
}

namespace {
template<bool scatter>
__global__ void reorder_values(float* dst, const float* src, const uint32_t* map, uint32_t n) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if constexpr (scatter) dst[map[i]] = src[i];
    else dst[i] = src[map[i]];
}

template<bool csr_operand>
__global__ void update_values(float* physical, float* csr, const float* operand,
    const uint32_t* map, uint32_t n, bool gradient, float rate) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto p = map[i];
    const float v = operand[csr_operand ? i : p];
    const float next = gradient ? fmaf(-rate, v, physical[p]) : __fadd_rn(physical[p], v);
    physical[p] = next;
    csr[i] = next;
}
}

cudaError_t refresh_csr_values(float* csr, const float* physical, const uint32_t* map,
    uint32_t n, cudaStream_t stream) noexcept {
    if (!n) return cudaSuccess;
    reorder_values<false><<<(uint64_t(n)+255)/256, 256, 0, stream>>>(csr, physical, map, n);
    return cudaPeekAtLastError();
}

cudaError_t scatter_csr_samples(float* physical, const float* csr, const uint32_t* map,
    uint32_t n, cudaStream_t stream) noexcept {
    if (!n) return cudaSuccess;
    reorder_values<true><<<(uint64_t(n)+255)/256, 256, 0, stream>>>(physical, csr, map, n);
    return cudaPeekAtLastError();
}

cudaError_t update_and_refresh_csr(float* physical, float* csr, const float* operand,
    const uint32_t* map, uint32_t n, bool gradient, float rate, bool csr_operand, cudaStream_t stream) noexcept {
    if (!n) return cudaSuccess;
    const auto blocks = (uint64_t(n)+255)/256;
    if (csr_operand)
        update_values<true><<<blocks, 256, 0, stream>>>(physical, csr, operand, map, n, gradient, rate);
    else
        update_values<false><<<blocks, 256, 0, stream>>>(physical, csr, operand, map, n, gradient, rate);
    return cudaPeekAtLastError();
}

} // namespace cellerator::ops::matrix::cusparse
