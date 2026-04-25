#include "stateReduce.hh"

#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cellerator::models::state_reduce {

namespace {

using namespace nvcuda;
namespace cs = ::cellshard;
namespace css = ::cellshard::sparse;

constexpr int kScalarThreads = 256;

inline void require_model_(const StateReduceModel *model, const char *label) {
    if (model == nullptr) throw std::invalid_argument(std::string(label) + " requires a model");
}

inline void require_batch_(const StateReduceBatchView &batch, const char *label) {
    if (batch.rows == 0u) throw std::invalid_argument(std::string(label) + " requires batch rows > 0");
    if (batch.layout == StateReduceLayout::blocked_ell) {
        if (batch.blocked_ell.rows == 0u || batch.blocked_ell.cols == 0u) {
            throw std::invalid_argument(std::string(label) + " requires an initialized blocked-ELL batch");
        }
        return;
    }
    if (batch.sliced_ell.rows == 0u || batch.sliced_ell.cols == 0u) {
        throw std::invalid_argument(std::string(label) + " requires an initialized sliced-ELL batch");
    }
}

inline void require_runtime_mode_(const StateReduceModel *model, const char *label) {
    if (model->distributed.requested_slots > 1u) {
        throw std::runtime_error(
            std::string(label)
            + " does not yet implement multi-GPU NCCL synchronization; use requested_slots=1 for this runtime slice");
    }
}

template<typename T>
autograd::device_buffer<T> allocate_device_zeroed_(int device, std::size_t count, const char *label) {
    autograd::cuda_require(cudaSetDevice(device), "cudaSetDevice(allocate_device_zeroed)");
    autograd::device_buffer<T> out = autograd::allocate_device_buffer<T>(count);
    if (count != 0u) {
        autograd::cuda_require(cudaMemset(out.data, 0, count * sizeof(T)), label);
    }
    return out;
}

__global__ void add_bias_kernel_(float *dst, const float *bias, std::uint32_t rows, std::uint32_t cols) {
    const std::uint32_t linear = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    const std::uint32_t total = rows * cols;
    if (linear >= total) return;
    dst[linear] += bias[linear % cols];
}

__global__ void float_to_half_kernel_(const float *src, __half *dst, std::uint32_t count) {
    const std::uint32_t idx = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    if (idx >= count) return;
    dst[idx] = __float2half_rn(src[idx]);
}

__global__ void dense_matmul_fwd_kernel_(
    const float *lhs,
    const float *rhs,
    std::uint32_t rows,
    std::uint32_t inner,
    std::uint32_t cols,
    float *out) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.y) * static_cast<std::uint32_t>(blockDim.y)
        + static_cast<std::uint32_t>(threadIdx.y);
    const std::uint32_t col = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    if (row >= rows || col >= cols) return;

    float accum = 0.0f;
    for (std::uint32_t k = 0u; k < inner; ++k) {
        accum += lhs[static_cast<std::size_t>(row) * inner + k] * rhs[static_cast<std::size_t>(k) * cols + col];
    }
    out[static_cast<std::size_t>(row) * cols + col] = accum;
}

__global__ void dense_left_grad_kernel_(
    const float *grad_out,
    const float *rhs,
    std::uint32_t rows,
    std::uint32_t inner,
    std::uint32_t cols,
    float *grad_lhs) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.y) * static_cast<std::uint32_t>(blockDim.y)
        + static_cast<std::uint32_t>(threadIdx.y);
    const std::uint32_t col = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    if (row >= rows || col >= inner) return;

    float accum = 0.0f;
    for (std::uint32_t k = 0u; k < cols; ++k) {
        accum += grad_out[static_cast<std::size_t>(row) * cols + k] * rhs[static_cast<std::size_t>(col) * cols + k];
    }
    grad_lhs[static_cast<std::size_t>(row) * inner + col] = accum;
}

__global__ void dense_right_grad_kernel_(
    const float *lhs,
    const float *grad_out,
    std::uint32_t rows,
    std::uint32_t inner,
    std::uint32_t cols,
    float *grad_rhs) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.y) * static_cast<std::uint32_t>(blockDim.y)
        + static_cast<std::uint32_t>(threadIdx.y);
    const std::uint32_t col = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    if (row >= inner || col >= cols) return;

    float accum = 0.0f;
    for (std::uint32_t sample = 0u; sample < rows; ++sample) {
        accum += lhs[static_cast<std::size_t>(sample) * inner + row] * grad_out[static_cast<std::size_t>(sample) * cols + col];
    }
    grad_rhs[static_cast<std::size_t>(row) * cols + col] = accum;
}

__global__ void wmma_decode_kernel_(
    const __half *lhs,
    const __half *rhs,
    std::uint32_t rows,
    std::uint32_t inner,
    std::uint32_t cols,
    float *out) {
    const std::uint32_t tile_row = static_cast<std::uint32_t>(blockIdx.y) * 16u;
    const std::uint32_t tile_col = static_cast<std::uint32_t>(blockIdx.x) * 16u;
    if (tile_row >= rows || tile_col >= cols) return;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (std::uint32_t k = 0u; k < inner; k += 16u) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, lhs + static_cast<std::size_t>(tile_row) * inner + k, inner);
        wmma::load_matrix_sync(b_frag, rhs + static_cast<std::size_t>(k) * cols + tile_col, cols);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    wmma::store_matrix_sync(out + static_cast<std::size_t>(tile_row) * cols + tile_col, acc, cols, wmma::mem_row_major);
}

__global__ void sliced_ell_spmm_fwd_kernel_(
    csv::sliced_ell_view src,
    const float *rhs,
    std::uint32_t rhs_cols,
    float *out) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.y) * static_cast<std::uint32_t>(blockDim.y)
        + static_cast<std::uint32_t>(threadIdx.y);
    const std::uint32_t col = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    if (row >= src.rows || col >= rhs_cols) return;

    std::uint32_t slice = 0u, row_begin = 0u, width = 0u;
    std::size_t slot_base = 0u;
    if (src.slice_count != 0u) {
        if (src.slice_rows == 32u) {
            slice = row >> 5u;
            if (slice >= src.slice_count) slice = src.slice_count - 1u;
        } else if (src.slice_rows != 0u) {
            slice = row / src.slice_rows;
            if (slice >= src.slice_count) slice = src.slice_count - 1u;
        } else {
            while (slice + 1u < src.slice_count && row >= src.slice_row_offsets[slice + 1u]) ++slice;
        }
        row_begin = src.slice_row_offsets[slice];
        width = src.slice_widths[slice];
        slot_base = static_cast<std::size_t>(src.slice_slot_offsets[slice])
            + static_cast<std::size_t>(row - row_begin) * width;
    }

    float accum = 0.0f;
    for (std::uint32_t slot = 0u; slot < width; ++slot) {
        const std::uint32_t gene = src.col_idx[slot_base + slot];
        if (gene >= src.cols || gene == css::sliced_ell_invalid_col) continue;
        accum += __half2float(src.val[slot_base + slot]) * rhs[static_cast<std::size_t>(gene) * rhs_cols + col];
    }
    out[static_cast<std::size_t>(row) * rhs_cols + col] = accum;
}

__global__ void reconstruction_zero_baseline_kernel_(
    const float *reconstruction,
    std::uint32_t rows,
    std::uint32_t cols,
    float scale,
    float *loss_out,
    float *grad_out) {
    const std::uint32_t idx = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    const std::uint32_t total = rows * cols;
    if (idx >= total) return;

    const float value = reconstruction[idx];
    grad_out[idx] = 2.0f * scale * value;
    atomicAdd(loss_out, scale * value * value);
}

__global__ void reconstruction_positive_blocked_ell_kernel_(
    csv::blocked_ell_view src,
    const float *reconstruction,
    float positive_scale,
    float baseline_scale,
    float *loss_out,
    float *grad_out) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x);
    if (row >= src.rows) return;
    const std::uint32_t block_size = src.block_size;
    const std::uint32_t ell_width = block_size != 0u ? src.ell_cols / block_size : 0u;
    const std::uint32_t row_block = block_size != 0u ? row / block_size : 0u;

    for (std::uint32_t ell_col = static_cast<std::uint32_t>(threadIdx.x); ell_col < src.ell_cols; ell_col += static_cast<std::uint32_t>(blockDim.x)) {
        const std::uint32_t slot = block_size != 0u ? ell_col / block_size : 0u;
        const std::uint32_t lane = block_size != 0u ? ell_col % block_size : 0u;
        const std::uint32_t block_col = ell_width != 0u
            ? src.blockColIdx[static_cast<std::size_t>(row_block) * ell_width + slot]
            : css::blocked_ell_invalid_col;
        const std::uint32_t gene = block_col != css::blocked_ell_invalid_col
            ? block_col * block_size + lane
            : src.cols;
        if (gene >= src.cols) continue;

        const float observed = __half2float(src.val[static_cast<std::size_t>(row) * src.ell_cols + ell_col]);
        if (observed == 0.0f) continue;
        const std::size_t dense_idx = static_cast<std::size_t>(row) * src.cols + gene;
        const float predicted = reconstruction[dense_idx];
        const float correction = positive_scale * (predicted - observed) * (predicted - observed)
            - baseline_scale * predicted * predicted;
        atomicAdd(loss_out, correction);
        atomicAdd(grad_out + dense_idx, 2.0f * (positive_scale * (predicted - observed) - baseline_scale * predicted));
    }
}

__global__ void reconstruction_positive_sliced_ell_kernel_(
    csv::sliced_ell_view src,
    const float *reconstruction,
    float positive_scale,
    float baseline_scale,
    float *loss_out,
    float *grad_out) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x);
    if (row >= src.rows) return;

    std::uint32_t slice = 0u, row_begin = 0u, width = 0u;
    std::size_t slot_base = 0u;
    if (src.slice_count != 0u) {
        if (src.slice_rows == 32u) {
            slice = row >> 5u;
            if (slice >= src.slice_count) slice = src.slice_count - 1u;
        } else if (src.slice_rows != 0u) {
            slice = row / src.slice_rows;
            if (slice >= src.slice_count) slice = src.slice_count - 1u;
        } else {
            while (slice + 1u < src.slice_count && row >= src.slice_row_offsets[slice + 1u]) ++slice;
        }
        row_begin = src.slice_row_offsets[slice];
        width = src.slice_widths[slice];
        slot_base = static_cast<std::size_t>(src.slice_slot_offsets[slice])
            + static_cast<std::size_t>(row - row_begin) * width;
    }

    for (std::uint32_t slot = static_cast<std::uint32_t>(threadIdx.x); slot < width; slot += static_cast<std::uint32_t>(blockDim.x)) {
        const std::uint32_t gene = src.col_idx[slot_base + slot];
        if (gene >= src.cols || gene == css::sliced_ell_invalid_col) continue;
        const float observed = __half2float(src.val[slot_base + slot]);
        if (observed == 0.0f) continue;
        const std::size_t dense_idx = static_cast<std::size_t>(row) * src.cols + gene;
        const float predicted = reconstruction[dense_idx];
        const float correction = positive_scale * (predicted - observed) * (predicted - observed)
            - baseline_scale * predicted * predicted;
        atomicAdd(loss_out, correction);
        atomicAdd(grad_out + dense_idx, 2.0f * (positive_scale * (predicted - observed) - baseline_scale * predicted));
    }
}

__global__ void graph_loss_kernel_(
    const float *latent,
    std::uint32_t rows,
    std::uint32_t latent_dim,
    StateReduceGraphView graph,
    float scale,
    float *loss_out,
    float *grad_out) {
    const std::uint32_t edge = static_cast<std::uint32_t>(blockIdx.x);
    if (edge >= graph.edge_count) return;
    const std::uint32_t src = graph.src[edge];
    const std::uint32_t dst = graph.dst[edge];
    const float weight = graph.weight != nullptr ? graph.weight[edge] : 1.0f;
    if (src >= rows || dst >= rows) return;

    float local_loss = 0.0f;
    for (std::uint32_t dim = static_cast<std::uint32_t>(threadIdx.x); dim < latent_dim; dim += static_cast<std::uint32_t>(blockDim.x)) {
        const std::size_t src_idx = static_cast<std::size_t>(src) * latent_dim + dim;
        const std::size_t dst_idx = static_cast<std::size_t>(dst) * latent_dim + dim;
        const float diff = latent[src_idx] - latent[dst_idx];
        local_loss += weight * diff * diff * scale;
        atomicAdd(grad_out + src_idx, 2.0f * weight * diff * scale);
        atomicAdd(grad_out + dst_idx, -2.0f * weight * diff * scale);
    }
    atomicAdd(loss_out, local_loss);
}

__global__ void blocked_ell_encoder_grad_kernel_(
    csv::blocked_ell_view src,
    const float *grad_latent,
    std::uint32_t latent_dim,
    float *grad_weight) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t dim = static_cast<std::uint32_t>(threadIdx.x);
    if (row >= src.rows || dim >= latent_dim) return;
    const std::uint32_t block_size = src.block_size;
    const std::uint32_t ell_width = block_size != 0u ? src.ell_cols / block_size : 0u;
    const std::uint32_t row_block = block_size != 0u ? row / block_size : 0u;
    const float grad = grad_latent[static_cast<std::size_t>(row) * latent_dim + dim];

    for (std::uint32_t ell_col = 0u; ell_col < src.ell_cols; ++ell_col) {
        const std::uint32_t slot = block_size != 0u ? ell_col / block_size : 0u;
        const std::uint32_t lane = block_size != 0u ? ell_col % block_size : 0u;
        const std::uint32_t block_col = ell_width != 0u
            ? src.blockColIdx[static_cast<std::size_t>(row_block) * ell_width + slot]
            : css::blocked_ell_invalid_col;
        const std::uint32_t gene = block_col != css::blocked_ell_invalid_col
            ? block_col * block_size + lane
            : src.cols;
        if (gene >= src.cols) continue;
        const float value = __half2float(src.val[static_cast<std::size_t>(row) * src.ell_cols + ell_col]);
        if (value == 0.0f) continue;
        atomicAdd(grad_weight + static_cast<std::size_t>(gene) * latent_dim + dim, value * grad);
    }
}

__global__ void sliced_ell_encoder_grad_kernel_(
    csv::sliced_ell_view src,
    const float *grad_latent,
    std::uint32_t latent_dim,
    float *grad_weight) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t dim = static_cast<std::uint32_t>(threadIdx.x);
    if (row >= src.rows || dim >= latent_dim) return;
    const float grad = grad_latent[static_cast<std::size_t>(row) * latent_dim + dim];

    std::uint32_t slice = 0u, row_begin = 0u, width = 0u;
    std::size_t slot_base = 0u;
    if (src.slice_count != 0u) {
        if (src.slice_rows == 32u) {
            slice = row >> 5u;
            if (slice >= src.slice_count) slice = src.slice_count - 1u;
        } else if (src.slice_rows != 0u) {
            slice = row / src.slice_rows;
            if (slice >= src.slice_count) slice = src.slice_count - 1u;
        } else {
            while (slice + 1u < src.slice_count && row >= src.slice_row_offsets[slice + 1u]) ++slice;
        }
        row_begin = src.slice_row_offsets[slice];
        width = src.slice_widths[slice];
        slot_base = static_cast<std::size_t>(src.slice_slot_offsets[slice])
            + static_cast<std::size_t>(row - row_begin) * width;
    }

    for (std::uint32_t slot = 0u; slot < width; ++slot) {
        const std::uint32_t gene = src.col_idx[slot_base + slot];
        if (gene >= src.cols || gene == css::sliced_ell_invalid_col) continue;
        const float value = __half2float(src.val[slot_base + slot]);
        if (value == 0.0f) continue;
        atomicAdd(grad_weight + static_cast<std::size_t>(gene) * latent_dim + dim, value * grad);
    }
}

__global__ void row_sum_kernel_(const float *src, std::uint32_t rows, std::uint32_t cols, float *dst) {
    const std::uint32_t col = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    if (col >= cols) return;
    float accum = 0.0f;
    for (std::uint32_t row = 0u; row < rows; ++row) {
        accum += src[static_cast<std::size_t>(row) * cols + col];
    }
    dst[col] = accum;
}

__global__ void add_inplace_kernel_(float *dst, const float *src, std::uint32_t count) {
    const std::uint32_t idx = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    if (idx >= count) return;
    dst[idx] += src[idx];
}

__global__ void adamw_update_kernel_(
    float *param,
    float *m1,
    float *m2,
    const float *grad,
    std::uint32_t count,
    float learning_rate,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2) {
    const std::uint32_t idx = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    if (idx >= count) return;

    const float g = grad[idx] + weight_decay * param[idx];
    const float next_m1 = beta1 * m1[idx] + (1.0f - beta1) * g;
    const float next_m2 = beta2 * m2[idx] + (1.0f - beta2) * g * g;
    const float m1_hat = next_m1 / bias_correction1;
    const float m2_hat = next_m2 / bias_correction2;
    param[idx] -= learning_rate * m1_hat / (sqrtf(m2_hat) + eps);
    m1[idx] = next_m1;
    m2[idx] = next_m2;
}

inline void launch_bias_(const autograd::execution_context &ctx, float *dst, const float *bias, std::uint32_t rows, std::uint32_t cols) {
    if (rows == 0u || cols == 0u) return;
    const std::uint32_t total = rows * cols;
    const int blocks = static_cast<int>((total + kScalarThreads - 1u) / kScalarThreads);
    add_bias_kernel_<<<blocks, kScalarThreads, 0, ctx.stream>>>(dst, bias, rows, cols);
    autograd::cuda_require(cudaGetLastError(), "state_reduce add_bias_kernel");
}

inline void launch_float_to_half_(
    const autograd::execution_context &ctx,
    const float *src,
    __half *dst,
    std::uint32_t count) {
    if (count == 0u) return;
    const int blocks = static_cast<int>((count + kScalarThreads - 1u) / kScalarThreads);
    float_to_half_kernel_<<<blocks, kScalarThreads, 0, ctx.stream>>>(src, dst, count);
    autograd::cuda_require(cudaGetLastError(), "state_reduce float_to_half_kernel");
}

inline void launch_dense_fwd_(
    const autograd::execution_context &ctx,
    const float *lhs,
    const float *rhs,
    std::uint32_t rows,
    std::uint32_t inner,
    std::uint32_t cols,
    float *out) {
    if (rows == 0u || inner == 0u || cols == 0u) return;
    const dim3 block(16u, 16u);
    const dim3 grid((cols + block.x - 1u) / block.x, (rows + block.y - 1u) / block.y);
    dense_matmul_fwd_kernel_<<<grid, block, 0, ctx.stream>>>(lhs, rhs, rows, inner, cols, out);
    autograd::cuda_require(cudaGetLastError(), "state_reduce dense_matmul_fwd_kernel");
}

inline void launch_dense_left_grad_(
    const autograd::execution_context &ctx,
    const float *grad_out,
    const float *rhs,
    std::uint32_t rows,
    std::uint32_t inner,
    std::uint32_t cols,
    float *grad_lhs) {
    if (rows == 0u || inner == 0u || cols == 0u) return;
    const dim3 block(16u, 16u);
    const dim3 grid((inner + block.x - 1u) / block.x, (rows + block.y - 1u) / block.y);
    dense_left_grad_kernel_<<<grid, block, 0, ctx.stream>>>(grad_out, rhs, rows, inner, cols, grad_lhs);
    autograd::cuda_require(cudaGetLastError(), "state_reduce dense_left_grad_kernel");
}

inline void launch_dense_right_grad_(
    const autograd::execution_context &ctx,
    const float *lhs,
    const float *grad_out,
    std::uint32_t rows,
    std::uint32_t inner,
    std::uint32_t cols,
    float *grad_rhs) {
    if (rows == 0u || inner == 0u || cols == 0u) return;
    const dim3 block(16u, 16u);
    const dim3 grid((cols + block.x - 1u) / block.x, (inner + block.y - 1u) / block.y);
    dense_right_grad_kernel_<<<grid, block, 0, ctx.stream>>>(lhs, grad_out, rows, inner, cols, grad_rhs);
    autograd::cuda_require(cudaGetLastError(), "state_reduce dense_right_grad_kernel");
}

inline void launch_wmma_decode_(
    const autograd::execution_context &ctx,
    const __half *lhs,
    const __half *rhs,
    std::uint32_t rows,
    std::uint32_t inner,
    std::uint32_t cols,
    float *out) {
    if (rows == 0u || inner == 0u || cols == 0u) return;
    const dim3 block(32u, 1u, 1u);
    const dim3 grid((cols + 15u) / 16u, (rows + 15u) / 16u, 1u);
    wmma_decode_kernel_<<<grid, block, 0, ctx.stream>>>(lhs, rhs, rows, inner, cols, out);
    autograd::cuda_require(cudaGetLastError(), "state_reduce wmma_decode_kernel");
}

inline void encode_batch_(
    StateReduceModel *model,
    const StateReduceBatchView &batch,
    float *latent_out) {
    const std::uint32_t rows = batch.rows;
    const std::uint32_t latent_dim = model->config.latent_dim;
    if (batch.layout == StateReduceLayout::blocked_ell) {
        const csv::blocked_ell_view &view = batch.blocked_ell;
        if (model->config.backend == StateReduceBackend::cusparse_heavy) {
            autograd::base::blocked_ell_spmm_fwd_f16_f32_lib(
                model->ctx,
                &model->sparse_cache,
                view.blockColIdx,
                view.blockColIdx,
                view.val,
                view.rows,
                view.cols,
                view.block_size,
                view.ell_cols,
                model->encoder_weight.data,
                static_cast<std::int64_t>(latent_dim),
                static_cast<std::int64_t>(latent_dim),
                latent_out,
                static_cast<std::int64_t>(latent_dim));
        } else {
            autograd::base::blocked_ell_spmm_fwd_f16_f32(
                model->ctx,
                view.blockColIdx,
                view.val,
                view.rows,
                view.cols,
                view.block_size,
                view.ell_cols,
                model->encoder_weight.data,
                static_cast<std::int64_t>(latent_dim),
                static_cast<std::int64_t>(latent_dim),
                latent_out,
                static_cast<std::int64_t>(latent_dim));
        }
    } else {
        const dim3 block(16u, 16u);
        const dim3 grid((latent_dim + block.x - 1u) / block.x, (rows + block.y - 1u) / block.y);
        sliced_ell_spmm_fwd_kernel_<<<grid, block, 0, model->ctx.stream>>>(
            batch.sliced_ell,
            model->encoder_weight.data,
            latent_dim,
            latent_out);
        autograd::cuda_require(cudaGetLastError(), "state_reduce sliced_ell_spmm_fwd_kernel");
    }
    if (model->config.use_encoder_bias) launch_bias_(model->ctx, latent_out, model->encoder_bias.data, rows, latent_dim);
}

inline void refresh_half_mirrors_(StateReduceModel *model) {
    launch_float_to_half_(
        model->ctx,
        model->decoder_factor.data,
        model->decoder_factor_half.data,
        model->config.latent_dim * model->config.factor_dim);
    launch_float_to_half_(
        model->ctx,
        model->gene_dictionary.data,
        model->gene_dictionary_half.data,
        model->config.factor_dim * model->config.input_genes);
}

struct forward_buffers_ {
    autograd::device_buffer<float> latent;
    autograd::device_buffer<float> factors;
    autograd::device_buffer<float> reconstruction;
    autograd::device_buffer<__half> factors_half;
};

inline forward_buffers_ run_forward_(
    StateReduceModel *model,
    const StateReduceBatchView &batch) {
    forward_buffers_ out{
        allocate_device_zeroed_<float>(model->ctx.device, static_cast<std::size_t>(batch.rows) * model->config.latent_dim, "cudaMemset(state_reduce latent)"),
        allocate_device_zeroed_<float>(model->ctx.device, static_cast<std::size_t>(batch.rows) * model->config.factor_dim, "cudaMemset(state_reduce factors)"),
        allocate_device_zeroed_<float>(model->ctx.device, static_cast<std::size_t>(batch.rows) * model->config.input_genes, "cudaMemset(state_reduce reconstruction)"),
        allocate_device_zeroed_<__half>(model->ctx.device, static_cast<std::size_t>(batch.rows) * model->config.factor_dim, "cudaMemset(state_reduce factors_half)")
    };

    encode_batch_(model, batch, out.latent.data);
    launch_dense_fwd_(
        model->ctx,
        out.latent.data,
        model->decoder_factor.data,
        batch.rows,
        model->config.latent_dim,
        model->config.factor_dim,
        out.factors.data);

    if (model->config.backend == StateReduceBackend::wmma_fused
        && (batch.rows % 16u) == 0u
        && (model->config.factor_dim % 16u) == 0u
        && (model->config.input_genes % 16u) == 0u) {
        launch_float_to_half_(
            model->ctx,
            out.factors.data,
            out.factors_half.data,
            batch.rows * model->config.factor_dim);
        launch_wmma_decode_(
            model->ctx,
            out.factors_half.data,
            model->gene_dictionary_half.data,
            batch.rows,
            model->config.factor_dim,
            model->config.input_genes,
            out.reconstruction.data);
    } else {
        launch_dense_fwd_(
            model->ctx,
            out.factors.data,
            model->gene_dictionary.data,
            batch.rows,
            model->config.factor_dim,
            model->config.input_genes,
            out.reconstruction.data);
    }
    return out;
}

inline StateReduceTrainMetrics compute_loss_(
    StateReduceModel *model,
    const StateReduceBatchView &batch,
    const StateReduceLossConfig &loss_config,
    const forward_buffers_ &forward,
    float *grad_reconstruction,
    float *grad_graph) {
    StateReduceTrainMetrics metrics{};
    auto recon_loss = allocate_device_zeroed_<float>(model->ctx.device, 1u, "cudaMemset(state_reduce recon loss)");
    auto graph_loss = allocate_device_zeroed_<float>(model->ctx.device, 1u, "cudaMemset(state_reduce graph loss)");
    const float inv_total = 1.0f / static_cast<float>(static_cast<double>(batch.rows) * static_cast<double>(model->config.input_genes));
    const float baseline_scale = loss_config.reconstruction_weight * loss_config.zero_weight * inv_total;
    const float positive_scale = loss_config.reconstruction_weight * loss_config.nonzero_weight * inv_total;
    const int recon_blocks = static_cast<int>((batch.rows * model->config.input_genes + kScalarThreads - 1u) / kScalarThreads);
    reconstruction_zero_baseline_kernel_<<<recon_blocks, kScalarThreads, 0, model->ctx.stream>>>(
        forward.reconstruction.data,
        batch.rows,
        model->config.input_genes,
        baseline_scale,
        recon_loss.data,
        grad_reconstruction);
    autograd::cuda_require(cudaGetLastError(), "state_reduce reconstruction_zero_baseline_kernel");

    if (batch.layout == StateReduceLayout::blocked_ell) {
        reconstruction_positive_blocked_ell_kernel_<<<batch.rows, 128, 0, model->ctx.stream>>>(
            batch.blocked_ell,
            forward.reconstruction.data,
            positive_scale,
            baseline_scale,
            recon_loss.data,
            grad_reconstruction);
        autograd::cuda_require(cudaGetLastError(), "state_reduce reconstruction_positive_blocked_ell_kernel");
    } else {
        reconstruction_positive_sliced_ell_kernel_<<<batch.rows, 128, 0, model->ctx.stream>>>(
            batch.sliced_ell,
            forward.reconstruction.data,
            positive_scale,
            baseline_scale,
            recon_loss.data,
            grad_reconstruction);
        autograd::cuda_require(cudaGetLastError(), "state_reduce reconstruction_positive_sliced_ell_kernel");
    }

    if (batch.graph.edge_count != 0u && loss_config.graph_weight != 0.0f) {
        const float graph_scale = loss_config.graph_weight / static_cast<float>(batch.graph.edge_count);
        graph_loss_kernel_<<<batch.graph.edge_count, 128, 0, model->ctx.stream>>>(
            forward.latent.data,
            batch.rows,
            model->config.latent_dim,
            batch.graph,
            graph_scale,
            graph_loss.data,
            grad_graph);
        autograd::cuda_require(cudaGetLastError(), "state_reduce graph_loss_kernel");
    }

    autograd::download_device_buffer(recon_loss, &metrics.reconstruction, 1u);
    autograd::download_device_buffer(graph_loss, &metrics.graph, 1u);
    metrics.total = metrics.reconstruction + metrics.graph;
    return metrics;
}

inline void apply_optimizer_(
    StateReduceModel *model,
    const StateReduceOptimizerConfig &optimizer_config,
    const float *grad_encoder_weight,
    const float *grad_encoder_bias,
    const float *grad_decoder_factor,
    const float *grad_gene_dictionary) {
    model->step += 1u;
    const float bias_correction1 = 1.0f - std::pow(optimizer_config.beta1, static_cast<float>(model->step));
    const float bias_correction2 = 1.0f - std::pow(optimizer_config.beta2, static_cast<float>(model->step));

    const std::uint32_t encoder_weight_count = model->config.input_genes * model->config.latent_dim;
    const std::uint32_t encoder_bias_count = model->config.use_encoder_bias ? model->config.latent_dim : 0u;
    const std::uint32_t decoder_factor_count = model->config.latent_dim * model->config.factor_dim;
    const std::uint32_t gene_dictionary_count = model->config.factor_dim * model->config.input_genes;

    auto launch_update = [&](float *param, float *m1, float *m2, const float *grad, std::uint32_t count) {
        if (count == 0u) return;
        const int blocks = static_cast<int>((count + kScalarThreads - 1u) / kScalarThreads);
        adamw_update_kernel_<<<blocks, kScalarThreads, 0, model->ctx.stream>>>(
            param,
            m1,
            m2,
            grad,
            count,
            optimizer_config.learning_rate,
            optimizer_config.beta1,
            optimizer_config.beta2,
            optimizer_config.eps,
            optimizer_config.weight_decay,
            bias_correction1,
            bias_correction2);
        autograd::cuda_require(cudaGetLastError(), "state_reduce adamw_update_kernel");
    };

    launch_update(
        model->encoder_weight.data,
        model->encoder_weight_m1.data,
        model->encoder_weight_m2.data,
        grad_encoder_weight,
        encoder_weight_count);
    launch_update(
        model->encoder_bias.data,
        model->encoder_bias_m1.data,
        model->encoder_bias_m2.data,
        grad_encoder_bias,
        encoder_bias_count);
    launch_update(
        model->decoder_factor.data,
        model->decoder_factor_m1.data,
        model->decoder_factor_m2.data,
        grad_decoder_factor,
        decoder_factor_count);
    launch_update(
        model->gene_dictionary.data,
        model->gene_dictionary_m1.data,
        model->gene_dictionary_m2.data,
        grad_gene_dictionary,
        gene_dictionary_count);
    refresh_half_mirrors_(model);
}

inline void fill_host_weights_(std::vector<float> &values, float scale, float phase) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        const float x = static_cast<float>(i) * 0.173f + phase;
        values[i] = std::sin(x) * scale;
    }
}

} // namespace

void init(
    StateReduceModel *model,
    StateReduceModelConfig config,
    StateReduceDistributedConfig distributed) {
    require_model_(model, "state_reduce::init");
    if (config.input_genes == 0u) throw std::invalid_argument("StateReduceModelConfig.input_genes must be > 0");
    if (config.latent_dim == 0u) throw std::invalid_argument("StateReduceModelConfig.latent_dim must be > 0");
    if (config.factor_dim == 0u) throw std::invalid_argument("StateReduceModelConfig.factor_dim must be > 0");

    model->config = config;
    model->distributed = distributed;
    model->step = 0u;
    autograd::init(&model->ctx, config.device);
    autograd::init(&model->scratch);
    autograd::init(&model->sparse_cache);
    autograd::init(&model->fleet);
    if (distributed.requested_slots > 1u) {
        autograd::discover_fleet(&model->fleet);
        model->fleet_ready = true;
        if (model->fleet.local.device_count < distributed.requested_slots) {
            throw std::runtime_error("state_reduce requested more fleet slots than are visible");
        }
    }

    const std::uint32_t encoder_weight_count = config.input_genes * config.latent_dim;
    const std::uint32_t encoder_bias_count = config.use_encoder_bias ? config.latent_dim : 0u;
    const std::uint32_t decoder_factor_count = config.latent_dim * config.factor_dim;
    const std::uint32_t gene_dictionary_count = config.factor_dim * config.input_genes;

    autograd::cuda_require(cudaSetDevice(model->ctx.device), "cudaSetDevice(state_reduce init)");
    model->encoder_weight = autograd::allocate_device_buffer<float>(encoder_weight_count);
    model->encoder_bias = autograd::allocate_device_buffer<float>(encoder_bias_count);
    model->decoder_factor = autograd::allocate_device_buffer<float>(decoder_factor_count);
    model->gene_dictionary = autograd::allocate_device_buffer<float>(gene_dictionary_count);
    model->encoder_weight_m1 = allocate_device_zeroed_<float>(model->ctx.device, encoder_weight_count, "cudaMemset(state_reduce encoder_weight_m1)");
    model->encoder_weight_m2 = allocate_device_zeroed_<float>(model->ctx.device, encoder_weight_count, "cudaMemset(state_reduce encoder_weight_m2)");
    model->encoder_bias_m1 = allocate_device_zeroed_<float>(model->ctx.device, encoder_bias_count, "cudaMemset(state_reduce encoder_bias_m1)");
    model->encoder_bias_m2 = allocate_device_zeroed_<float>(model->ctx.device, encoder_bias_count, "cudaMemset(state_reduce encoder_bias_m2)");
    model->decoder_factor_m1 = allocate_device_zeroed_<float>(model->ctx.device, decoder_factor_count, "cudaMemset(state_reduce decoder_factor_m1)");
    model->decoder_factor_m2 = allocate_device_zeroed_<float>(model->ctx.device, decoder_factor_count, "cudaMemset(state_reduce decoder_factor_m2)");
    model->gene_dictionary_m1 = allocate_device_zeroed_<float>(model->ctx.device, gene_dictionary_count, "cudaMemset(state_reduce gene_dictionary_m1)");
    model->gene_dictionary_m2 = allocate_device_zeroed_<float>(model->ctx.device, gene_dictionary_count, "cudaMemset(state_reduce gene_dictionary_m2)");
    model->decoder_factor_half = autograd::allocate_device_buffer<__half>(decoder_factor_count);
    model->gene_dictionary_half = autograd::allocate_device_buffer<__half>(gene_dictionary_count);

    std::vector<float> encoder_weight_host(encoder_weight_count, 0.0f);
    std::vector<float> encoder_bias_host(encoder_bias_count, 0.0f);
    std::vector<float> decoder_factor_host(decoder_factor_count, 0.0f);
    std::vector<float> gene_dictionary_host(gene_dictionary_count, 0.0f);
    fill_host_weights_(encoder_weight_host, 0.05f, 0.13f);
    fill_host_weights_(decoder_factor_host, 0.05f, 0.29f);
    fill_host_weights_(gene_dictionary_host, 0.05f, 0.47f);

    autograd::upload_device_buffer(&model->encoder_weight, encoder_weight_host.data(), encoder_weight_host.size());
    if (encoder_bias_count != 0u) autograd::upload_device_buffer(&model->encoder_bias, encoder_bias_host.data(), encoder_bias_host.size());
    autograd::upload_device_buffer(&model->decoder_factor, decoder_factor_host.data(), decoder_factor_host.size());
    autograd::upload_device_buffer(&model->gene_dictionary, gene_dictionary_host.data(), gene_dictionary_host.size());
    refresh_half_mirrors_(model);
}

void clear(StateReduceModel *model) {
    require_model_(model, "state_reduce::clear");
    model->encoder_weight = autograd::device_buffer<float>();
    model->encoder_bias = autograd::device_buffer<float>();
    model->decoder_factor = autograd::device_buffer<float>();
    model->gene_dictionary = autograd::device_buffer<float>();
    model->encoder_weight_m1 = autograd::device_buffer<float>();
    model->encoder_weight_m2 = autograd::device_buffer<float>();
    model->encoder_bias_m1 = autograd::device_buffer<float>();
    model->encoder_bias_m2 = autograd::device_buffer<float>();
    model->decoder_factor_m1 = autograd::device_buffer<float>();
    model->decoder_factor_m2 = autograd::device_buffer<float>();
    model->gene_dictionary_m1 = autograd::device_buffer<float>();
    model->gene_dictionary_m2 = autograd::device_buffer<float>();
    model->decoder_factor_half = autograd::device_buffer<__half>();
    model->gene_dictionary_half = autograd::device_buffer<__half>();
    if (model->fleet_ready) autograd::clear(&model->fleet);
    autograd::clear(&model->sparse_cache);
    autograd::clear(&model->scratch);
    autograd::clear(&model->ctx);
    model->fleet_ready = false;
    model->step = 0u;
    model->config = StateReduceModelConfig();
    model->distributed = StateReduceDistributedConfig();
}

autograd::device_buffer<float> infer_embeddings(
    StateReduceModel *model,
    const StateReduceBatchView &batch) {
    require_model_(model, "state_reduce::infer_embeddings");
    require_batch_(batch, "state_reduce::infer_embeddings");
    require_runtime_mode_(model, "state_reduce::infer_embeddings");
    autograd::cuda_require(cudaSetDevice(model->ctx.device), "cudaSetDevice(state_reduce::infer_embeddings)");
    autograd::device_buffer<float> latent = allocate_device_zeroed_<float>(
        model->ctx.device,
        static_cast<std::size_t>(batch.rows) * model->config.latent_dim,
        "cudaMemset(state_reduce infer latent)");
    encode_batch_(model, batch, latent.data);
    return latent;
}

StateReduceTrainMetrics evaluate(
    StateReduceModel *model,
    const StateReduceBatchView &batch,
    const StateReduceLossConfig &loss_config) {
    require_model_(model, "state_reduce::evaluate");
    require_batch_(batch, "state_reduce::evaluate");
    require_runtime_mode_(model, "state_reduce::evaluate");
    autograd::cuda_require(cudaSetDevice(model->ctx.device), "cudaSetDevice(state_reduce::evaluate)");
    const forward_buffers_ forward = run_forward_(model, batch);
    auto grad_reconstruction = allocate_device_zeroed_<float>(
        model->ctx.device,
        static_cast<std::size_t>(batch.rows) * model->config.input_genes,
        "cudaMemset(state_reduce eval grad_reconstruction)");
    auto grad_graph = allocate_device_zeroed_<float>(
        model->ctx.device,
        static_cast<std::size_t>(batch.rows) * model->config.latent_dim,
        "cudaMemset(state_reduce eval grad_graph)");
    return compute_loss_(model, batch, loss_config, forward, grad_reconstruction.data, grad_graph.data);
}

StateReduceTrainMetrics train_step(
    StateReduceModel *model,
    const StateReduceBatchView &batch,
    const StateReduceLossConfig &loss_config,
    const StateReduceOptimizerConfig &optimizer_config) {
    require_model_(model, "state_reduce::train_step");
    require_batch_(batch, "state_reduce::train_step");
    require_runtime_mode_(model, "state_reduce::train_step");
    autograd::cuda_require(cudaSetDevice(model->ctx.device), "cudaSetDevice(state_reduce::train_step)");

    const forward_buffers_ forward = run_forward_(model, batch);
    auto grad_reconstruction = allocate_device_zeroed_<float>(
        model->ctx.device,
        static_cast<std::size_t>(batch.rows) * model->config.input_genes,
        "cudaMemset(state_reduce grad_reconstruction)");
    auto grad_graph = allocate_device_zeroed_<float>(
        model->ctx.device,
        static_cast<std::size_t>(batch.rows) * model->config.latent_dim,
        "cudaMemset(state_reduce grad_graph)");
    const StateReduceTrainMetrics metrics = compute_loss_(
        model,
        batch,
        loss_config,
        forward,
        grad_reconstruction.data,
        grad_graph.data);

    auto grad_factors = allocate_device_zeroed_<float>(
        model->ctx.device,
        static_cast<std::size_t>(batch.rows) * model->config.factor_dim,
        "cudaMemset(state_reduce grad_factors)");
    auto grad_gene_dictionary = allocate_device_zeroed_<float>(
        model->ctx.device,
        static_cast<std::size_t>(model->config.factor_dim) * model->config.input_genes,
        "cudaMemset(state_reduce grad_gene_dictionary)");
    auto grad_latent = allocate_device_zeroed_<float>(
        model->ctx.device,
        static_cast<std::size_t>(batch.rows) * model->config.latent_dim,
        "cudaMemset(state_reduce grad_latent)");
    auto grad_decoder_factor = allocate_device_zeroed_<float>(
        model->ctx.device,
        static_cast<std::size_t>(model->config.latent_dim) * model->config.factor_dim,
        "cudaMemset(state_reduce grad_decoder_factor)");
    auto grad_encoder_weight = allocate_device_zeroed_<float>(
        model->ctx.device,
        static_cast<std::size_t>(model->config.input_genes) * model->config.latent_dim,
        "cudaMemset(state_reduce grad_encoder_weight)");
    auto grad_encoder_bias = allocate_device_zeroed_<float>(
        model->ctx.device,
        model->config.use_encoder_bias ? model->config.latent_dim : 0u,
        "cudaMemset(state_reduce grad_encoder_bias)");

    launch_dense_left_grad_(
        model->ctx,
        grad_reconstruction.data,
        model->gene_dictionary.data,
        batch.rows,
        model->config.factor_dim,
        model->config.input_genes,
        grad_factors.data);
    launch_dense_right_grad_(
        model->ctx,
        forward.factors.data,
        grad_reconstruction.data,
        batch.rows,
        model->config.factor_dim,
        model->config.input_genes,
        grad_gene_dictionary.data);
    launch_dense_left_grad_(
        model->ctx,
        grad_factors.data,
        model->decoder_factor.data,
        batch.rows,
        model->config.latent_dim,
        model->config.factor_dim,
        grad_latent.data);
    launch_dense_right_grad_(
        model->ctx,
        forward.latent.data,
        grad_factors.data,
        batch.rows,
        model->config.latent_dim,
        model->config.factor_dim,
        grad_decoder_factor.data);

    const std::uint32_t latent_count = batch.rows * model->config.latent_dim;
    if (batch.graph.edge_count != 0u && loss_config.graph_weight != 0.0f) {
        const int add_blocks = static_cast<int>((latent_count + kScalarThreads - 1u) / kScalarThreads);
        add_inplace_kernel_<<<add_blocks, kScalarThreads, 0, model->ctx.stream>>>(
            grad_latent.data,
            grad_graph.data,
            latent_count);
        autograd::cuda_require(cudaGetLastError(), "state_reduce add_inplace_kernel graph");
    }

    if (batch.layout == StateReduceLayout::blocked_ell) {
        blocked_ell_encoder_grad_kernel_<<<batch.rows, model->config.latent_dim, 0, model->ctx.stream>>>(
            batch.blocked_ell,
            grad_latent.data,
            model->config.latent_dim,
            grad_encoder_weight.data);
        autograd::cuda_require(cudaGetLastError(), "state_reduce blocked_ell_encoder_grad_kernel");
    } else {
        sliced_ell_encoder_grad_kernel_<<<batch.rows, model->config.latent_dim, 0, model->ctx.stream>>>(
            batch.sliced_ell,
            grad_latent.data,
            model->config.latent_dim,
            grad_encoder_weight.data);
        autograd::cuda_require(cudaGetLastError(), "state_reduce sliced_ell_encoder_grad_kernel");
    }

    if (model->config.use_encoder_bias) {
        const int bias_blocks = static_cast<int>((model->config.latent_dim + kScalarThreads - 1u) / kScalarThreads);
        row_sum_kernel_<<<bias_blocks, kScalarThreads, 0, model->ctx.stream>>>(
            grad_latent.data,
            batch.rows,
            model->config.latent_dim,
            grad_encoder_bias.data);
        autograd::cuda_require(cudaGetLastError(), "state_reduce row_sum_kernel");
    }

    apply_optimizer_(
        model,
        optimizer_config,
        grad_encoder_weight.data,
        grad_encoder_bias.data,
        grad_decoder_factor.data,
        grad_gene_dictionary.data);
    return metrics;
}

} // namespace cellerator::models::state_reduce
