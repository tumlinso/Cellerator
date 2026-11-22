#pragma once

#include "../../../src/ops/matrix/csr_cusparse.cuh"
#include <algorithm>

namespace trace_sparse = cellerator::ops::matrix::cusparse;

void sparse_ok(cusparseStatus_t status) {
    require(status == CUSPARSE_STATUS_SUCCESS, cusparseGetErrorString(status));
}

struct trace_cusparse {
    cusparseHandle_t handle{};
    trace_sparse::spmm_plan forward, transpose;
    trace_sparse::sddmm_plan gradient;
    void* indices{};
    void* multiply_work{};
    void* gradient_work{};
    cudaStream_t stream{};

    ~trace_cusparse() {
        if (handle) cudaStreamSynchronize(stream);
        forward.release();
        transpose.release();
        gradient.release();
        if (handle) cusparseDestroy(handle);
        if (indices) cudaFree(indices);
        if (multiply_work) cudaFree(multiply_work);
        if (gradient_work) cudaFree(gradient_work);
    }

    void prepare(const std::vector<std::uint32_t>& offsets, const std::vector<std::uint32_t>& columns,
        std::size_t cols, std::size_t width, float* weights, float* dx, float* y, float* edge_gradient,
        float* input, float* cotangent, cudaStream_t stream) {
        this->stream = stream;
        const std::size_t rows = offsets.size() - 1, edges = columns.size();
        cuda_ok(cudaMalloc(&indices, (offsets.size() + edges) * sizeof(std::int32_t)));
        auto* row = static_cast<std::int32_t*>(indices);
        auto* col = row + offsets.size();
        cuda_ok(cudaMemcpyAsync(row, offsets.data(), offsets.size() * sizeof(*row), cudaMemcpyHostToDevice, stream));
        cuda_ok(cudaMemcpyAsync(col, columns.data(), edges * sizeof(*col), cudaMemcpyHostToDevice, stream));

        sparse_ok(cusparseCreate(&handle));
        sparse_ok(cusparseSetStream(handle, stream));
        const trace_sparse::csr_view csr{{rows, cols, edges,
            cellerator::tensor::base::CompressedAxis::row, row, col}, weights};
        sparse_ok(forward.provision(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, csr, CUDA_R_32F,
            {input, cols, width, width}, {y, rows, width, width}, CUSPARSE_SPMM_CSR_ALG2));
        sparse_ok(transpose.provision(handle, CUSPARSE_OPERATION_TRANSPOSE, csr, CUDA_R_32F,
            {cotangent, rows, width, width}, {dx, cols, width, width}, CUSPARSE_SPMM_CSR_ALG2));
        const auto bytes = std::max(forward.workspace_bytes(), transpose.workspace_bytes());
        if (bytes) cuda_ok(cudaMalloc(&multiply_work, bytes));
        sparse_ok(forward.bind_workspace(multiply_work, bytes));
        sparse_ok(transpose.bind_workspace(multiply_work, bytes));

        sparse_ok(gradient.provision(handle, {csr.structure, edge_gradient},
            {cotangent, rows, width, width}, {input, cols, width, width}));
        const auto gradient_bytes = gradient.workspace_bytes();
        if (gradient_bytes) cuda_ok(cudaMalloc(&gradient_work, gradient_bytes));
        sparse_ok(gradient.bind_workspace(gradient_work, gradient_bytes));
    }

    void edge_gradient() const {
        sparse_ok(gradient.enqueue());
    }
};

