#include "dTT_model.hh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace cellerator::models::developmental_time_trajectory {

namespace {

namespace css = ::cellshard::sparse;
namespace autograd = dt::autograd;
namespace csv = dt::csv;

constexpr int kScalarThreads = 256;

inline void require_model_(const DevelopmentalTimeTrajectoryModel *model, const char *label) {
    if (model == nullptr) throw std::invalid_argument(std::string(label) + " requires a model");
}

inline void require_batch_(const DevelopmentalTimeTrajectoryBatchView &batch, const char *label, bool require_targets) {
    dt::DevelopmentalTimeBatchView inner = batch.features;
    if (batch.graph.edge_count != 0u && (batch.graph.src == nullptr || batch.graph.dst == nullptr)) {
        throw std::invalid_argument(std::string(label) + " requires graph src/dst");
    }
    if (require_targets && inner.target_time == nullptr) {
        throw std::invalid_argument(std::string(label) + " requires target_time");
    }
    if (inner.rows == 0u) throw std::invalid_argument(std::string(label) + " requires batch rows > 0");
    if (inner.layout == dt::DevelopmentalTimeLayout::blocked_ell) {
        if (inner.blocked_ell.rows == 0u || inner.blocked_ell.cols == 0u) {
            throw std::invalid_argument(std::string(label) + " requires an initialized blocked-ELL batch");
        }
    } else if (inner.sliced_ell.rows == 0u || inner.sliced_ell.cols == 0u) {
        throw std::invalid_argument(std::string(label) + " requires an initialized sliced-ELL batch");
    }
}

template<typename T>
autograd::device_buffer<T> allocate_device_zeroed_(int device, std::size_t count, const char *label) {
    autograd::cuda_require(cudaSetDevice(device), "cudaSetDevice(allocate_device_zeroed)");
    autograd::device_buffer<T> out = autograd::allocate_device_buffer<T>(count);
    if (count != 0u) autograd::cuda_require(cudaMemset(out.data, 0, count * sizeof(T)), label);
    return out;
}

__device__ __forceinline__ float sigmoid_(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void add_bias_kernel_(float *dst, const float *bias, std::uint32_t rows, std::uint32_t cols) {
    const std::uint32_t linear = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    const std::uint32_t total = rows * cols;
    if (linear >= total) return;
    dst[linear] += bias[linear % cols];
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

__global__ void silu_forward_kernel_(const float *src, float *dst, std::uint32_t count) {
    const std::uint32_t idx = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    if (idx >= count) return;
    const float x = src[idx];
    dst[idx] = x * sigmoid_(x);
}

__global__ void silu_backward_kernel_(const float *grad_out, const float *preact, float *grad_in, std::uint32_t count) {
    const std::uint32_t idx = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    if (idx >= count) return;
    const float x = preact[idx];
    const float sig = sigmoid_(x);
    grad_in[idx] = grad_out[idx] * (sig + x * sig * (1.0f - sig));
}

__global__ void huber_loss_kernel_(
    const float *predicted,
    const float *target,
    std::uint32_t rows,
    float delta,
    float scale,
    float *loss_out,
    float *grad_out) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    if (row >= rows) return;
    const float diff = predicted[row] - target[row];
    const float abs_diff = fabsf(diff);
    float loss = 0.0f, grad = 0.0f;
    if (abs_diff <= delta) {
        loss = 0.5f * diff * diff;
        grad = diff;
    } else {
        loss = delta * (abs_diff - 0.5f * delta);
        grad = delta * (diff >= 0.0f ? 1.0f : -1.0f);
    }
    atomicAdd(loss_out, loss * scale);
    grad_out[row] = grad * scale;
}

__global__ void graph_aggregate_kernel_(
    const float *hidden,
    std::uint32_t rows,
    std::uint32_t hidden_dim,
    DevelopmentalTimeGraphView graph,
    float *agg,
    float *norm) {
    const std::uint32_t edge = static_cast<std::uint32_t>(blockIdx.x);
    if (edge >= graph.edge_count) return;
    const std::uint32_t src = graph.src[edge], dst = graph.dst[edge];
    if (src >= rows || dst >= rows) return;
    const float weight = graph.weight != nullptr ? graph.weight[edge] : 1.0f;
    if (threadIdx.x == 0) atomicAdd(norm + dst, weight);
    for (std::uint32_t dim = static_cast<std::uint32_t>(threadIdx.x); dim < hidden_dim; dim += static_cast<std::uint32_t>(blockDim.x)) {
        atomicAdd(agg + static_cast<std::size_t>(dst) * hidden_dim + dim,
                  weight * hidden[static_cast<std::size_t>(src) * hidden_dim + dim]);
    }
}

__global__ void graph_finalize_kernel_(
    const float *hidden,
    const float *agg,
    const float *norm,
    std::uint32_t rows,
    std::uint32_t hidden_dim,
    float graph_mix,
    float *combined) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.y) * static_cast<std::uint32_t>(blockDim.y)
        + static_cast<std::uint32_t>(threadIdx.y);
    const std::uint32_t dim = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    if (row >= rows || dim >= hidden_dim) return;
    const std::size_t idx = static_cast<std::size_t>(row) * hidden_dim + dim;
    float mean = 0.0f;
    if (norm[row] > 0.0f) mean = agg[idx] / norm[row];
    combined[idx] = hidden[idx] + graph_mix * mean;
}

__global__ void copy_kernel_(const float *src, float *dst, std::uint32_t count) {
    const std::uint32_t idx = static_cast<std::uint32_t>(blockIdx.x) * static_cast<std::uint32_t>(blockDim.x)
        + static_cast<std::uint32_t>(threadIdx.x);
    if (idx >= count) return;
    dst[idx] = src[idx];
}

__global__ void graph_scatter_grad_kernel_(
    const float *grad_combined,
    const float *norm,
    std::uint32_t rows,
    std::uint32_t hidden_dim,
    DevelopmentalTimeGraphView graph,
    float graph_mix,
    float *grad_hidden_total) {
    const std::uint32_t edge = static_cast<std::uint32_t>(blockIdx.x);
    if (edge >= graph.edge_count) return;
    const std::uint32_t src = graph.src[edge], dst = graph.dst[edge];
    if (src >= rows || dst >= rows || norm[dst] <= 0.0f) return;
    const float weight = graph.weight != nullptr ? graph.weight[edge] : 1.0f;
    const float scale = graph_mix * weight / norm[dst];
    for (std::uint32_t dim = static_cast<std::uint32_t>(threadIdx.x); dim < hidden_dim; dim += static_cast<std::uint32_t>(blockDim.x)) {
        atomicAdd(
            grad_hidden_total + static_cast<std::size_t>(src) * hidden_dim + dim,
            scale * grad_combined[static_cast<std::size_t>(dst) * hidden_dim + dim]);
    }
}

__global__ void graph_smoothness_loss_kernel_(
    const float *predicted,
    std::uint32_t rows,
    DevelopmentalTimeGraphView graph,
    float scale,
    float *loss_out,
    float *grad_out) {
    const std::uint32_t edge = static_cast<std::uint32_t>(blockIdx.x);
    if (edge >= graph.edge_count) return;
    const std::uint32_t src = graph.src[edge], dst = graph.dst[edge];
    if (src >= rows || dst >= rows) return;
    const float weight = graph.weight != nullptr ? graph.weight[edge] : 1.0f;
    const float diff = predicted[src] - predicted[dst];
    atomicAdd(loss_out, scale * weight * diff * diff);
    atomicAdd(grad_out + src, 2.0f * scale * weight * diff);
    atomicAdd(grad_out + dst, -2.0f * scale * weight * diff);
}

__global__ void graph_order_loss_kernel_(
    const float *predicted,
    std::uint32_t rows,
    DevelopmentalTimeGraphView graph,
    float margin,
    float scale,
    float *loss_out,
    float *grad_out) {
    const std::uint32_t edge = static_cast<std::uint32_t>(blockIdx.x);
    if (edge >= graph.edge_count) return;
    const std::uint32_t src = graph.src[edge], dst = graph.dst[edge];
    if (src >= rows || dst >= rows) return;
    const float weight = graph.weight != nullptr ? graph.weight[edge] : 1.0f;
    const float gap = predicted[dst] - predicted[src];
    const float violation = margin - gap;
    if (violation <= 0.0f) return;
    atomicAdd(loss_out, scale * weight * violation * violation);
    atomicAdd(grad_out + src, 2.0f * scale * weight * violation);
    atomicAdd(grad_out + dst, -2.0f * scale * weight * violation);
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
        const std::uint32_t gene = block_col != css::blocked_ell_invalid_col ? block_col * block_size + lane : src.cols;
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
    if (rows == 0u || cols == 0u || bias == nullptr) return;
    const std::uint32_t total = rows * cols;
    const int blocks = static_cast<int>((total + kScalarThreads - 1u) / kScalarThreads);
    add_bias_kernel_<<<blocks, kScalarThreads, 0, ctx.stream>>>(dst, bias, rows, cols);
    autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory add_bias_kernel");
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
    autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory dense_matmul_fwd_kernel");
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
    autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory dense_left_grad_kernel");
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
    autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory dense_right_grad_kernel");
}

inline void launch_silu_forward_(const autograd::execution_context &ctx, const float *src, float *dst, std::uint32_t count) {
    if (count == 0u) return;
    const int blocks = static_cast<int>((count + kScalarThreads - 1u) / kScalarThreads);
    silu_forward_kernel_<<<blocks, kScalarThreads, 0, ctx.stream>>>(src, dst, count);
    autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory silu_forward_kernel");
}

inline void launch_silu_backward_(const autograd::execution_context &ctx, const float *grad_out, const float *preact, float *grad_in, std::uint32_t count) {
    if (count == 0u) return;
    const int blocks = static_cast<int>((count + kScalarThreads - 1u) / kScalarThreads);
    silu_backward_kernel_<<<blocks, kScalarThreads, 0, ctx.stream>>>(grad_out, preact, grad_in, count);
    autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory silu_backward_kernel");
}

inline void encode_batch_(DevelopmentalTimeTrajectoryModel *model, const dt::DevelopmentalTimeBatchView &batch, float *stem_out) {
    if (batch.layout == dt::DevelopmentalTimeLayout::blocked_ell) {
        const csv::blocked_ell_view &view = batch.blocked_ell;
        if (model->config.backend == dt::DevelopmentalTimeBackend::tensor_cusparse) {
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
                static_cast<std::int64_t>(model->config.stem_dim),
                static_cast<std::int64_t>(model->config.stem_dim),
                stem_out,
                static_cast<std::int64_t>(model->config.stem_dim));
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
                static_cast<std::int64_t>(model->config.stem_dim),
                static_cast<std::int64_t>(model->config.stem_dim),
                stem_out,
                static_cast<std::int64_t>(model->config.stem_dim));
        }
    } else {
        const dim3 block(16u, 16u);
        const dim3 grid((model->config.stem_dim + block.x - 1u) / block.x, (batch.rows + block.y - 1u) / block.y);
        sliced_ell_spmm_fwd_kernel_<<<grid, block, 0, model->ctx.stream>>>(
            batch.sliced_ell,
            model->encoder_weight.data,
            model->config.stem_dim,
            stem_out);
        autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory sliced_ell_spmm_fwd_kernel");
    }
    if (model->config.use_encoder_bias) launch_bias_(model->ctx, stem_out, model->encoder_bias.data, batch.rows, model->config.stem_dim);
}

struct forward_buffers_ {
    autograd::device_buffer<float> stem_linear;
    autograd::device_buffer<float> stem_activated;
    autograd::device_buffer<float> hidden_linear;
    autograd::device_buffer<float> hidden_activated;
    autograd::device_buffer<float> aggregated_hidden;
    autograd::device_buffer<float> graph_norm;
    autograd::device_buffer<float> combined_hidden;
    autograd::device_buffer<float> predicted;
};

inline forward_buffers_ run_forward_(
    DevelopmentalTimeTrajectoryModel *model,
    const DevelopmentalTimeTrajectoryBatchView &batch,
    float graph_mix) {
    forward_buffers_ out{
        allocate_device_zeroed_<float>(model->ctx.device, static_cast<std::size_t>(batch.features.rows) * model->config.stem_dim, "cudaMemset(dtt stem_linear)"),
        allocate_device_zeroed_<float>(model->ctx.device, static_cast<std::size_t>(batch.features.rows) * model->config.stem_dim, "cudaMemset(dtt stem_activated)"),
        allocate_device_zeroed_<float>(model->ctx.device, static_cast<std::size_t>(batch.features.rows) * model->config.hidden_dim, "cudaMemset(dtt hidden_linear)"),
        allocate_device_zeroed_<float>(model->ctx.device, static_cast<std::size_t>(batch.features.rows) * model->config.hidden_dim, "cudaMemset(dtt hidden_activated)"),
        allocate_device_zeroed_<float>(model->ctx.device, static_cast<std::size_t>(batch.features.rows) * model->config.hidden_dim, "cudaMemset(dtt aggregated_hidden)"),
        allocate_device_zeroed_<float>(model->ctx.device, batch.features.rows, "cudaMemset(dtt graph_norm)"),
        allocate_device_zeroed_<float>(model->ctx.device, static_cast<std::size_t>(batch.features.rows) * model->config.hidden_dim, "cudaMemset(dtt combined_hidden)"),
        allocate_device_zeroed_<float>(model->ctx.device, batch.features.rows, "cudaMemset(dtt predicted)")
    };

    encode_batch_(model, batch.features, out.stem_linear.data);
    launch_silu_forward_(model->ctx, out.stem_linear.data, out.stem_activated.data, batch.features.rows * model->config.stem_dim);
    launch_dense_fwd_(
        model->ctx,
        out.stem_activated.data,
        model->hidden_weight.data,
        batch.features.rows,
        model->config.stem_dim,
        model->config.hidden_dim,
        out.hidden_linear.data);
    if (model->config.use_hidden_bias) {
        launch_bias_(model->ctx, out.hidden_linear.data, model->hidden_bias.data, batch.features.rows, model->config.hidden_dim);
    }
    launch_silu_forward_(model->ctx, out.hidden_linear.data, out.hidden_activated.data, batch.features.rows * model->config.hidden_dim);

    if (batch.graph.edge_count != 0u && graph_mix != 0.0f) {
        graph_aggregate_kernel_<<<batch.graph.edge_count, 128, 0, model->ctx.stream>>>(
            out.hidden_activated.data,
            batch.features.rows,
            model->config.hidden_dim,
            batch.graph,
            out.aggregated_hidden.data,
            out.graph_norm.data);
        autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory graph_aggregate_kernel");
    }
    const dim3 block(16u, 16u);
    const dim3 grid((model->config.hidden_dim + block.x - 1u) / block.x, (batch.features.rows + block.y - 1u) / block.y);
    graph_finalize_kernel_<<<grid, block, 0, model->ctx.stream>>>(
        out.hidden_activated.data,
        out.aggregated_hidden.data,
        out.graph_norm.data,
        batch.features.rows,
        model->config.hidden_dim,
        graph_mix,
        out.combined_hidden.data);
    autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory graph_finalize_kernel");

    launch_dense_fwd_(
        model->ctx,
        out.combined_hidden.data,
        model->output_weight.data,
        batch.features.rows,
        model->config.hidden_dim,
        1u,
        out.predicted.data);
    if (model->config.use_output_bias) launch_bias_(model->ctx, out.predicted.data, model->output_bias.data, batch.features.rows, 1u);
    return out;
}

inline DevelopmentalTimeTrajectoryMetrics compute_loss_(
    DevelopmentalTimeTrajectoryModel *model,
    const DevelopmentalTimeTrajectoryBatchView &batch,
    const DevelopmentalTimeTrajectoryLossConfig &loss_config,
    const forward_buffers_ &forward,
    float *grad_predicted) {
    DevelopmentalTimeTrajectoryMetrics metrics{};
    auto regression = allocate_device_zeroed_<float>(model->ctx.device, 1u, "cudaMemset(dtt regression)");
    auto smoothness = allocate_device_zeroed_<float>(model->ctx.device, 1u, "cudaMemset(dtt smoothness)");
    auto order = allocate_device_zeroed_<float>(model->ctx.device, 1u, "cudaMemset(dtt order)");

    const int blocks = static_cast<int>((batch.features.rows + kScalarThreads - 1u) / kScalarThreads);
    huber_loss_kernel_<<<blocks, kScalarThreads, 0, model->ctx.stream>>>(
        forward.predicted.data,
        batch.features.target_time,
        batch.features.rows,
        loss_config.huber_delta,
        1.0f / static_cast<float>(batch.features.rows),
        regression.data,
        grad_predicted);
    autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory huber_loss_kernel");

    if (batch.graph.edge_count != 0u) {
        if (loss_config.smoothness_weight != 0.0f) {
            graph_smoothness_loss_kernel_<<<batch.graph.edge_count, 1, 0, model->ctx.stream>>>(
                forward.predicted.data,
                batch.features.rows,
                batch.graph,
                loss_config.smoothness_weight / static_cast<float>(batch.graph.edge_count),
                smoothness.data,
                grad_predicted);
            autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory graph_smoothness_loss_kernel");
        }
        if (loss_config.order_weight != 0.0f) {
            graph_order_loss_kernel_<<<batch.graph.edge_count, 1, 0, model->ctx.stream>>>(
                forward.predicted.data,
                batch.features.rows,
                batch.graph,
                loss_config.order_margin,
                loss_config.order_weight / static_cast<float>(batch.graph.edge_count),
                order.data,
                grad_predicted);
            autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory graph_order_loss_kernel");
        }
    }

    autograd::download_device_buffer(regression, &metrics.regression, 1u);
    autograd::download_device_buffer(smoothness, &metrics.smoothness, 1u);
    autograd::download_device_buffer(order, &metrics.order, 1u);
    metrics.total = metrics.regression + metrics.smoothness + metrics.order;
    return metrics;
}

inline void apply_optimizer_(
    DevelopmentalTimeTrajectoryModel *model,
    const DevelopmentalTimeTrajectoryOptimizerConfig &optimizer_config,
    const float *grad_encoder_weight,
    const float *grad_encoder_bias,
    const float *grad_hidden_weight,
    const float *grad_hidden_bias,
    const float *grad_output_weight,
    const float *grad_output_bias) {
    model->step += 1u;
    const float bias_correction1 = 1.0f - std::pow(optimizer_config.beta1, static_cast<float>(model->step));
    const float bias_correction2 = 1.0f - std::pow(optimizer_config.beta2, static_cast<float>(model->step));
    const std::uint32_t encoder_weight_count = model->config.input_genes * model->config.stem_dim;
    const std::uint32_t encoder_bias_count = model->config.use_encoder_bias ? model->config.stem_dim : 0u;
    const std::uint32_t hidden_weight_count = model->config.stem_dim * model->config.hidden_dim;
    const std::uint32_t hidden_bias_count = model->config.use_hidden_bias ? model->config.hidden_dim : 0u;
    const std::uint32_t output_weight_count = model->config.hidden_dim;
    const std::uint32_t output_bias_count = model->config.use_output_bias ? 1u : 0u;

    auto launch_update = [&](float *param, float *m1, float *m2, const float *grad, std::uint32_t count) {
        if (count == 0u) return;
        const int blocks = static_cast<int>((count + kScalarThreads - 1u) / kScalarThreads);
        adamw_update_kernel_<<<blocks, kScalarThreads, 0, model->ctx.stream>>>(
            param, m1, m2, grad, count,
            optimizer_config.learning_rate,
            optimizer_config.beta1,
            optimizer_config.beta2,
            optimizer_config.eps,
            optimizer_config.weight_decay,
            bias_correction1,
            bias_correction2);
        autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory adamw_update_kernel");
    };

    launch_update(model->encoder_weight.data, model->encoder_weight_m1.data, model->encoder_weight_m2.data, grad_encoder_weight, encoder_weight_count);
    launch_update(model->encoder_bias.data, model->encoder_bias_m1.data, model->encoder_bias_m2.data, grad_encoder_bias, encoder_bias_count);
    launch_update(model->hidden_weight.data, model->hidden_weight_m1.data, model->hidden_weight_m2.data, grad_hidden_weight, hidden_weight_count);
    launch_update(model->hidden_bias.data, model->hidden_bias_m1.data, model->hidden_bias_m2.data, grad_hidden_bias, hidden_bias_count);
    launch_update(model->output_weight.data, model->output_weight_m1.data, model->output_weight_m2.data, grad_output_weight, output_weight_count);
    launch_update(model->output_bias.data, model->output_bias_m1.data, model->output_bias_m2.data, grad_output_bias, output_bias_count);
}

} // namespace

void init(DevelopmentalTimeTrajectoryModel *model, DevelopmentalTimeTrajectoryModelConfig config) {
    require_model_(model, "developmental_time_trajectory::init");
    dt::DevelopmentalTimeModelConfig base{};
    base.input_genes = config.input_genes;
    base.stem_dim = config.stem_dim;
    base.hidden_dim = config.hidden_dim;
    base.use_encoder_bias = config.use_encoder_bias;
    base.use_hidden_bias = config.use_hidden_bias;
    base.use_output_bias = config.use_output_bias;
    base.device = config.device;
    base.backend = config.backend;
    dt::init(model, base);
}

void clear(DevelopmentalTimeTrajectoryModel *model) {
    require_model_(model, "developmental_time_trajectory::clear");
    dt::clear(model);
}

autograd::device_buffer<float> infer_time(
    DevelopmentalTimeTrajectoryModel *model,
    const DevelopmentalTimeTrajectoryBatchView &batch,
    float graph_mix) {
    require_model_(model, "developmental_time_trajectory::infer_time");
    require_batch_(batch, "developmental_time_trajectory::infer_time", false);
    autograd::cuda_require(cudaSetDevice(model->ctx.device), "cudaSetDevice(developmental_time_trajectory::infer_time)");
    forward_buffers_ forward = run_forward_(model, batch, graph_mix);
    return std::move(forward.predicted);
}

DevelopmentalTimeTrajectoryMetrics evaluate(
    DevelopmentalTimeTrajectoryModel *model,
    const DevelopmentalTimeTrajectoryBatchView &batch,
    const DevelopmentalTimeTrajectoryLossConfig &loss_config) {
    require_model_(model, "developmental_time_trajectory::evaluate");
    require_batch_(batch, "developmental_time_trajectory::evaluate", true);
    autograd::cuda_require(cudaSetDevice(model->ctx.device), "cudaSetDevice(developmental_time_trajectory::evaluate)");
    forward_buffers_ forward = run_forward_(model, batch, 0.5f);
    autograd::device_buffer<float> grad_predicted = allocate_device_zeroed_<float>(model->ctx.device, batch.features.rows, "cudaMemset(dtt eval grad_predicted)");
    return compute_loss_(model, batch, loss_config, forward, grad_predicted.data);
}

DevelopmentalTimeTrajectoryMetrics train_step(
    DevelopmentalTimeTrajectoryModel *model,
    const DevelopmentalTimeTrajectoryBatchView &batch,
    const DevelopmentalTimeTrajectoryLossConfig &loss_config,
    const DevelopmentalTimeTrajectoryOptimizerConfig &optimizer_config) {
    require_model_(model, "developmental_time_trajectory::train_step");
    require_batch_(batch, "developmental_time_trajectory::train_step", true);
    autograd::cuda_require(cudaSetDevice(model->ctx.device), "cudaSetDevice(developmental_time_trajectory::train_step)");
    forward_buffers_ forward = run_forward_(model, batch, 0.5f);

    autograd::device_buffer<float> grad_predicted = allocate_device_zeroed_<float>(model->ctx.device, batch.features.rows, "cudaMemset(dtt grad_predicted)");
    DevelopmentalTimeTrajectoryMetrics metrics = compute_loss_(model, batch, loss_config, forward, grad_predicted.data);

    autograd::device_buffer<float> grad_combined_hidden = allocate_device_zeroed_<float>(
        model->ctx.device, static_cast<std::size_t>(batch.features.rows) * model->config.hidden_dim, "cudaMemset(dtt grad_combined_hidden)");
    autograd::device_buffer<float> grad_hidden_total = allocate_device_zeroed_<float>(
        model->ctx.device, static_cast<std::size_t>(batch.features.rows) * model->config.hidden_dim, "cudaMemset(dtt grad_hidden_total)");
    autograd::device_buffer<float> grad_hidden_linear = allocate_device_zeroed_<float>(
        model->ctx.device, static_cast<std::size_t>(batch.features.rows) * model->config.hidden_dim, "cudaMemset(dtt grad_hidden_linear)");
    autograd::device_buffer<float> grad_output_weight = allocate_device_zeroed_<float>(
        model->ctx.device, model->config.hidden_dim, "cudaMemset(dtt grad_output_weight)");
    autograd::device_buffer<float> grad_output_bias = allocate_device_zeroed_<float>(
        model->ctx.device, model->config.use_output_bias ? 1u : 0u, "cudaMemset(dtt grad_output_bias)");
    autograd::device_buffer<float> grad_stem_activated = allocate_device_zeroed_<float>(
        model->ctx.device, static_cast<std::size_t>(batch.features.rows) * model->config.stem_dim, "cudaMemset(dtt grad_stem_activated)");
    autograd::device_buffer<float> grad_stem_linear = allocate_device_zeroed_<float>(
        model->ctx.device, static_cast<std::size_t>(batch.features.rows) * model->config.stem_dim, "cudaMemset(dtt grad_stem_linear)");
    autograd::device_buffer<float> grad_hidden_weight = allocate_device_zeroed_<float>(
        model->ctx.device, static_cast<std::size_t>(model->config.stem_dim) * model->config.hidden_dim, "cudaMemset(dtt grad_hidden_weight)");
    autograd::device_buffer<float> grad_hidden_bias = allocate_device_zeroed_<float>(
        model->ctx.device, model->config.use_hidden_bias ? model->config.hidden_dim : 0u, "cudaMemset(dtt grad_hidden_bias)");
    autograd::device_buffer<float> grad_encoder_weight = allocate_device_zeroed_<float>(
        model->ctx.device, static_cast<std::size_t>(model->config.input_genes) * model->config.stem_dim, "cudaMemset(dtt grad_encoder_weight)");
    autograd::device_buffer<float> grad_encoder_bias = allocate_device_zeroed_<float>(
        model->ctx.device, model->config.use_encoder_bias ? model->config.stem_dim : 0u, "cudaMemset(dtt grad_encoder_bias)");

    launch_dense_left_grad_(
        model->ctx,
        grad_predicted.data,
        model->output_weight.data,
        batch.features.rows,
        model->config.hidden_dim,
        1u,
        grad_combined_hidden.data);
    launch_dense_right_grad_(
        model->ctx,
        forward.combined_hidden.data,
        grad_predicted.data,
        batch.features.rows,
        model->config.hidden_dim,
        1u,
        grad_output_weight.data);
    if (model->config.use_output_bias) {
        row_sum_kernel_<<<1, 1, 0, model->ctx.stream>>>(grad_predicted.data, batch.features.rows, 1u, grad_output_bias.data);
        autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory row_sum output_bias");
    }

    {
        const std::uint32_t count = batch.features.rows * model->config.hidden_dim;
        const int blocks = static_cast<int>((count + kScalarThreads - 1u) / kScalarThreads);
        copy_kernel_<<<blocks, kScalarThreads, 0, model->ctx.stream>>>(grad_combined_hidden.data, grad_hidden_total.data, count);
        autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory copy_kernel");
    }
    if (batch.graph.edge_count != 0u) {
        graph_scatter_grad_kernel_<<<batch.graph.edge_count, 128, 0, model->ctx.stream>>>(
            grad_combined_hidden.data,
            forward.graph_norm.data,
            batch.features.rows,
            model->config.hidden_dim,
            batch.graph,
            0.5f,
            grad_hidden_total.data);
        autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory graph_scatter_grad_kernel");
    }

    launch_silu_backward_(
        model->ctx,
        grad_hidden_total.data,
        forward.hidden_linear.data,
        grad_hidden_linear.data,
        batch.features.rows * model->config.hidden_dim);
    launch_dense_left_grad_(
        model->ctx,
        grad_hidden_linear.data,
        model->hidden_weight.data,
        batch.features.rows,
        model->config.stem_dim,
        model->config.hidden_dim,
        grad_stem_activated.data);
    launch_dense_right_grad_(
        model->ctx,
        forward.stem_activated.data,
        grad_hidden_linear.data,
        batch.features.rows,
        model->config.stem_dim,
        model->config.hidden_dim,
        grad_hidden_weight.data);
    if (model->config.use_hidden_bias) {
        const int blocks = static_cast<int>((model->config.hidden_dim + kScalarThreads - 1u) / kScalarThreads);
        row_sum_kernel_<<<blocks, kScalarThreads, 0, model->ctx.stream>>>(
            grad_hidden_linear.data,
            batch.features.rows,
            model->config.hidden_dim,
            grad_hidden_bias.data);
        autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory row_sum hidden_bias");
    }

    launch_silu_backward_(
        model->ctx,
        grad_stem_activated.data,
        forward.stem_linear.data,
        grad_stem_linear.data,
        batch.features.rows * model->config.stem_dim);
    if (batch.features.layout == dt::DevelopmentalTimeLayout::blocked_ell) {
        blocked_ell_encoder_grad_kernel_<<<batch.features.rows, model->config.stem_dim, 0, model->ctx.stream>>>(
            batch.features.blocked_ell,
            grad_stem_linear.data,
            model->config.stem_dim,
            grad_encoder_weight.data);
        autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory blocked_ell_encoder_grad_kernel");
    } else {
        sliced_ell_encoder_grad_kernel_<<<batch.features.rows, model->config.stem_dim, 0, model->ctx.stream>>>(
            batch.features.sliced_ell,
            grad_stem_linear.data,
            model->config.stem_dim,
            grad_encoder_weight.data);
        autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory sliced_ell_encoder_grad_kernel");
    }
    if (model->config.use_encoder_bias) {
        const int blocks = static_cast<int>((model->config.stem_dim + kScalarThreads - 1u) / kScalarThreads);
        row_sum_kernel_<<<blocks, kScalarThreads, 0, model->ctx.stream>>>(
            grad_stem_linear.data,
            batch.features.rows,
            model->config.stem_dim,
            grad_encoder_bias.data);
        autograd::cuda_require(cudaGetLastError(), "developmental_time_trajectory row_sum encoder_bias");
    }

    apply_optimizer_(
        model,
        optimizer_config,
        grad_encoder_weight.data,
        grad_encoder_bias.data,
        grad_hidden_weight.data,
        grad_hidden_bias.data,
        grad_output_weight.data,
        grad_output_bias.data);
    return metrics;
}

} // namespace cellerator::models::developmental_time_trajectory
