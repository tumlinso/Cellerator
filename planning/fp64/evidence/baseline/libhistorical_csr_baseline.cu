#include <Cellerator/compute/candidate/sparse/project.hh>
#include <cuda_runtime.h>
#include <cstdint>

namespace cellerator::compute::sparse::project {
namespace {
constexpr int kSpmmColsThreads = 128;
__global__ void csr_spmm_fwd_f32_kernel_(
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *values,
    std::uint32_t rows,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld,
    const std::uint32_t *value_indices,
    float input_scale,
    float destination_scale) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x);
    const std::int64_t col = static_cast<std::int64_t>(threadIdx.x) +
        static_cast<std::int64_t>(blockIdx.y) * blockDim.x;
    if (row >= rows || col >= out_cols) return;

    float accum = 0.0f;
    for (std::uint32_t edge = major_ptr[row]; edge < major_ptr[row + 1u]; ++edge) {
        const std::uint32_t value = value_indices == nullptr ? edge : value_indices[edge];
        accum += values[value] * rhs[static_cast<std::int64_t>(minor_idx[edge]) * rhs_ld + col];
    }
    float* destination = out + static_cast<std::int64_t>(row) * out_ld + col;
    *destination = destination_scale == 0.0f
        ? input_scale * accum
        : fmaf(input_scale, accum, destination_scale * *destination);
}
} // namespace
void csr_spmm_fwd_f32_baseline(const runtime::execution_context& ctx,
    const std::uint32_t* major_ptr,const std::uint32_t* minor_idx,const float* values,
    std::uint32_t rows,std::uint32_t,const float* rhs,std::int64_t rhs_ld,
    std::int64_t out_cols,float* out,std::int64_t out_ld,const std::uint32_t* value_indices,
    float input_scale,float destination_scale) {
    runtime::cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(csr_spmm_f32)");
    if(!rows || !out_cols)return;
    const dim3 grid(rows,static_cast<unsigned int>((out_cols+kSpmmColsThreads-1)/kSpmmColsThreads),1u);
    csr_spmm_fwd_f32_kernel_<<<grid,kSpmmColsThreads,0,ctx.stream>>>(major_ptr,minor_idx,values,rows,rhs,rhs_ld,out_cols,out,out_ld,value_indices,input_scale,destination_scale);
    runtime::cuda_require(cudaGetLastError(),"csr_spmm_f32_kernel");
}
} // namespace cellerator::compute::sparse::project

extern "C" void fp64_old_csr_spmm_fwd_f32(
    const cellerator::runtime::execution_context* ctx,
    const std::uint32_t* major_ptr, const std::uint32_t* minor_idx,
    const float* values, std::uint32_t rows, std::uint32_t cols,
    const float* rhs, std::int64_t rhs_ld, std::int64_t out_cols,
    float* out, std::int64_t out_ld, const std::uint32_t* value_indices,
    float input_scale, float destination_scale) {
    cellerator::compute::sparse::project::csr_spmm_fwd_f32_baseline(
        *ctx, major_ptr, minor_idx, values, rows, cols, rhs, rhs_ld,
        out_cols, out, out_ld, value_indices, input_scale, destination_scale);
}
