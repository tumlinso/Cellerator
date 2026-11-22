#include "../../src/ops/matrix/csr_cusparse.cuh"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>

namespace csr = cellerator::ops::matrix::cusparse;
namespace mem = cellerator::memory;

namespace {

void require(bool ok, const char *message) {
    if (ok) return;
    std::cerr << "CSR cuSPARSE plans: " << message << '\n';
    std::abort();
}

void check(cudaError_t status) {
    require(status == cudaSuccess, cudaGetErrorString(status));
}

void check(cusparseStatus_t status) {
    require(status == CUSPARSE_STATUS_SUCCESS, cusparseGetErrorString(status));
}

template<class T>
struct device_array {
    T *data = nullptr;

    explicit device_array(size_t count) {
        if (count) check(cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)));
    }
    device_array(const device_array &) = delete;
    device_array &operator=(const device_array &) = delete;
    ~device_array() { if (data) check(cudaFree(data)); }
};

template<class T>
void upload(T *device, const std::vector<T> &host) {
    check(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
}

std::vector<float> dense(size_t rows, size_t width, size_t stride, float seed) {
    std::vector<float> result(rows * stride, 313.0f);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < width; ++j)
            result[i * stride + j] = seed + static_cast<float>(i * 3 + j) * 0.25f;
    return result;
}

void verify(const float *device, const std::vector<float> &old_y,
    const std::vector<float> &x, const std::vector<float> &values,
    const std::vector<int32_t> &offsets, const std::vector<int32_t> &cols,
    size_t width, size_t ldx, size_t ldy, bool transpose, float alpha, float beta) {
    std::vector<double> expected(old_y.begin(), old_y.end());
    for (size_t i = 0; i < old_y.size() / ldy; ++i)
        for (size_t j = 0; j < width; ++j)
            expected[i * ldy + j] *= double(beta);
    for (size_t i = 0; i + 1 < offsets.size(); ++i) {
        for (int32_t p = offsets[i]; p < offsets[i + 1]; ++p) {
            const size_t from = transpose ? i : size_t(cols[p]);
            const size_t to = transpose ? size_t(cols[p]) : i;
            for (size_t j = 0; j < width; ++j)
                expected[to * ldy + j] += double(alpha) * double(values[p]) * double(x[from * ldx + j]);
        }
    }

    std::vector<float> result(old_y.size());
    check(cudaMemcpy(result.data(), device, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < result.size(); ++i) {
        if (i % ldy >= width) require(result[i] == old_y[i], "padding changed");
        else require(std::abs(double(result[i]) - expected[i]) <= 2e-6 * (1 + std::abs(expected[i])),
            "output disagrees with FP64 reference");
    }
}

template<bool matrix>
void run(cusparseHandle_t handle, cudaStream_t stream, bool transpose) {
    const std::vector<int32_t> offsets{0, 2, 2, 4}, cols{0, 2, 1, 2};
    const std::vector<float> a{2, -1, 0.5f, 3}, b{-0.5f, 4, -2, 1};
    const size_t m = transpose ? 4 : 3, n = transpose ? 3 : 4;
    constexpr size_t width = matrix ? 5 : 1, ldx = matrix ? 8 : 1, ldy = matrix ? 9 : 1;
    const auto x = dense(n, width, ldx, 0.25f), x2 = dense(n, width, ldx, -2);
    const auto y = dense(m, width, ldy, 2.5f), y2 = dense(m, width, ldy, -1);
    device_array<int32_t> rows(offsets.size()), indices(cols.size());
    device_array<float> values(a.size()), values2(b.size()), dx(x.size()), dx2(x2.size()),
        dy(y.size()), dy2(y2.size());
    upload(rows.data, offsets); upload(indices.data, cols);
    upload(values.data, a); upload(dx.data, x); upload(dy.data, y);
    upload(values2.data, a); upload(dx2.data, x2); upload(dy2.data, y2);

    const csr::csr_view view{
        {3, 4, 4, cellerator::tensor::base::CompressedAxis::row, rows.data, indices.data}, values.data};
    const auto op = transpose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    std::conditional_t<matrix, csr::spmm_plan, csr::spmv_plan> plan;
    auto provision = [&](const csr::csr_view &a_view, cudaDataType_t type) {
        if constexpr (matrix) {
            return plan.provision(handle, op, a_view, type,
                mem::const_matrix_view<float>{dx.data, n, width, ldx},
                mem::matrix_view<float>{dy.data, m, width, ldy},
                transpose ? CUSPARSE_SPMM_CSR_ALG2 : CUSPARSE_SPMM_CSR_ALG1);
        } else {
            return plan.provision(handle, op, a_view, type,
                mem::const_array_view<float>{dx.data, n}, mem::array_view<float>{dy.data, m},
                CUSPARSE_SPMV_CSR_ALG1);
        }
    };

    auto empty = view;
    empty.structure.nnz = 0;
    require(provision(empty, CUDA_R_32F) == CUSPARSE_STATUS_INVALID_VALUE, "reject empty CSR");
    require(provision(view, CUDA_R_16F) == CUSPARSE_STATUS_NOT_SUPPORTED, "reject unsupported mix");
    check(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));
    require(provision(view, CUDA_R_32F) == CUSPARSE_STATUS_INVALID_VALUE, "host scalar mode");
    check(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

    check(provision(view, CUDA_R_32F));
    require(provision(view, CUDA_R_32F) == CUSPARSE_STATUS_INVALID_VALUE, "reject in-use provision");
    const size_t bytes = plan.workspace_bytes();
    device_array<unsigned char> workspace(bytes);
    if (bytes)
        require(plan.bind_workspace(workspace.data, bytes - 1) == CUSPARSE_STATUS_INVALID_VALUE,
            "reject short workspace");
    if constexpr (matrix) {
        if (transpose)
            require(plan.bind_workspace(workspace.data, bytes, true) == CUSPARSE_STATUS_NOT_SUPPORTED,
                "reject preprocessing for CSR ALG2");
    } else if constexpr (!csr::spmv_preprocess_available) {
        require(plan.bind_workspace(workspace.data, bytes, true) == CUSPARSE_STATUS_NOT_SUPPORTED,
            "reject SpMV preprocessing on older cuSPARSE");
    }
    const bool preprocess = !transpose && (matrix || csr::spmv_preprocess_available);
    check(plan.bind_workspace(workspace.data, bytes, preprocess));
    check(plan.enqueue(1.25f, -0.5f));
    check(cudaStreamSynchronize(stream));
    verify(dy.data, y, x, a, offsets, cols, width, ldx, ldy, transpose, 1.25f, -0.5f);

    if (preprocess && bytes) {
        device_array<unsigned char> replacement(bytes);
        require(plan.bind_workspace(replacement.data, bytes) == CUSPARSE_STATUS_INVALID_VALUE,
            "reject replacement of active preprocessing workspace");
    }
    check(plan.bind_workspace(workspace.data, bytes, preprocess));
    upload(values.data, b);
    check(plan.enqueue());
    check(cudaStreamSynchronize(stream));
    verify(dy.data, y, x, b, offsets, cols, width, ldx, ldy, transpose, 1, 0);

    require(plan.rebind(values2.data, nullptr, dy2.data) == CUSPARSE_STATUS_INVALID_VALUE,
        "reject null input binding");
    check(plan.rebind(values2.data, dx2.data, dy2.data));
    check(plan.enqueue(-2, 1));
    check(cudaStreamSynchronize(stream));
    verify(dy2.data, y2, x2, a, offsets, cols, width, ldx, ldy, transpose, -2, 1);
    verify(dy.data, y, x, b, offsets, cols, width, ldx, ldy, transpose, 1, 0);

    plan.release();
    require(plan.workspace_bytes() == 0, "release resets workspace");
    check(provision(view, CUDA_R_32F));
    require(plan.workspace_bytes() == bytes, "same shape reuses workspace sizing");
    check(plan.bind_workspace(workspace.data, bytes));
    check(plan.enqueue());
    check(cudaStreamSynchronize(stream));
    verify(dy.data, y, x, b, offsets, cols, width, ldx, ldy, transpose, 1, 0);
    plan.release();
}

} // namespace

int main() {
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cusparseHandle_t handle;
    check(cusparseCreate(&handle));
    check(cusparseSetStream(handle, stream));
    for (bool transpose : {false, true}) {
        run<false>(handle, stream, transpose);
        run<true>(handle, stream, transpose);
    }
    check(cusparseDestroy(handle));
    check(cudaStreamDestroy(stream));
}
