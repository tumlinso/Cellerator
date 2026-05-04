#include <Cellerator/abi.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) throw std::runtime_error(message);
}

void require_status(cellerator_status status, const char *message) {
    if (status != CELLERATOR_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(message) + ": " + cellerator_status_string(status) + " " + cellerator_last_error());
    }
}

bool close_value(float lhs, float rhs, float tol = 1.0e-4f) {
    return std::fabs(lhs - rhs) <= tol;
}

float softplus_host(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return std::exp(value);
    return std::log1p(std::exp(value));
}

float sigmoid_host(float value) {
    if (value >= 0.0f) {
        const float z = std::exp(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(value);
    return z / (1.0f + z);
}

struct quantize_eval_host {
    float loss = 0.0f;
    float grad_offset = 0.0f;
    float grad_scale = 0.0f;
};

quantize_eval_host eval_quantize_host(float value, float scale, float offset, float max_code) {
    const float relaxed = (value - offset) / scale;
    const float clamped = std::fmin(std::fmax(relaxed, 0.0f), max_code);
    const float code = std::nearbyint(clamped);
    const float reconstruction = offset + code * scale;
    const float error = reconstruction - value;
    const bool active = relaxed > 0.0f && relaxed < max_code;
    return {
        error * error,
        2.0f * error * (1.0f - (active ? 1.0f : 0.0f)),
        2.0f * error * (code - (active ? relaxed : 0.0f))
    };
}

void compute_quantize_reference(
    const float *dense,
    std::uint32_t rows,
    std::uint32_t cols,
    const float *log_scale,
    const float *offset,
    const cellerator_feature_affine_reconstruction_config &config,
    float *reconstruction_loss,
    float *range_loss,
    float *grad_log_scale,
    float *grad_offset) {
    const float inv_elements = 1.0f / static_cast<float>(static_cast<double>(rows) * static_cast<double>(cols));
    const float inv_features = 1.0f / static_cast<float>(cols);
    const float max_code = static_cast<float>((1u << config.bits) - 1u);
    *reconstruction_loss = 0.0f;
    *range_loss = 0.0f;
    for (std::uint32_t col = 0; col < cols; ++col) {
        grad_log_scale[col] = 0.0f;
        grad_offset[col] = 0.0f;
    }

    for (std::uint32_t col = 0; col < cols; ++col) {
        const float scale = softplus_host(log_scale[col]) + config.scale_floor;
        const float scale_grad_factor = sigmoid_host(log_scale[col]);
        for (std::uint32_t row = 0; row < rows; ++row) {
            const float value = dense[static_cast<std::size_t>(row) * cols + col];
            const quantize_eval_host eval = eval_quantize_host(value, scale, offset[col], max_code);
            *reconstruction_loss += eval.loss * inv_elements;
            grad_offset[col] += eval.grad_offset * inv_elements * config.reconstruction_weight;
            grad_log_scale[col] += eval.grad_scale * inv_elements * config.reconstruction_weight * scale_grad_factor;
        }

        const float dynamic_range = scale * max_code;
        if (dynamic_range < config.min_dynamic_range) {
            const float diff = config.min_dynamic_range - dynamic_range;
            *range_loss += diff * diff * inv_features;
            grad_log_scale[col] += -2.0f * diff * max_code * inv_features * config.range_weight * scale_grad_factor;
        }
        *range_loss += offset[col] * offset[col] * inv_features;
        grad_offset[col] += 2.0f * offset[col] * inv_features * config.range_weight;
    }
}

template<typename T>
T *device_copy(const T *src, std::size_t count) {
    T *dst = nullptr;
    require_cuda(cudaMalloc(reinterpret_cast<void **>(&dst), count * sizeof(T)), "cudaMalloc failed");
    require_cuda(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D failed");
    return dst;
}

template<typename T>
T *device_alloc(std::size_t count) {
    T *dst = nullptr;
    require_cuda(cudaMalloc(reinterpret_cast<void **>(&dst), count * sizeof(T)), "cudaMalloc failed");
    require_cuda(cudaMemset(dst, 0, count * sizeof(T)), "cudaMemset failed");
    return dst;
}

template<typename T>
void download(const T *src, T *dst, std::size_t count) {
    require_cuda(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H failed");
}

cellerator_tensor_desc vector_desc(void *data, cellerator_dtype dtype, std::uint64_t count) {
    cellerator_tensor_desc desc{};
    desc.version = CELLERATOR_ABI_VERSION;
    desc.size = sizeof(desc);
    desc.data = data;
    desc.dtype = dtype;
    desc.rank = 1u;
    desc.shape[0] = count;
    desc.stride[0] = 1;
    return desc;
}

cellerator_tensor_desc matrix_desc(void *data, std::uint64_t rows, std::uint64_t cols, std::int64_t ld) {
    cellerator_tensor_desc desc{};
    desc.version = CELLERATOR_ABI_VERSION;
    desc.size = sizeof(desc);
    desc.data = data;
    desc.dtype = CELLERATOR_DTYPE_F32;
    desc.rank = 2u;
    desc.shape[0] = rows;
    desc.shape[1] = cols;
    desc.stride[0] = ld;
    desc.stride[1] = 1;
    return desc;
}

cellerator_tensor_desc scalar_desc(void *data) {
    cellerator_tensor_desc desc{};
    desc.version = CELLERATOR_ABI_VERSION;
    desc.size = sizeof(desc);
    desc.data = data;
    desc.dtype = CELLERATOR_DTYPE_F32;
    desc.rank = 1u;
    desc.shape[0] = 1u;
    desc.stride[0] = 1;
    return desc;
}

} // namespace

int main() {
    int device_count = 0;
    require_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount failed");
    require(device_count > 0, "abiRuntimeTest requires at least one visible CUDA device");

    cellerator_execution_config exec{};
    exec.version = CELLERATOR_ABI_VERSION;
    exec.size = sizeof(exec);
    exec.device = -1;
    cellerator_context *context = nullptr;
    require_status(cellerator_context_create(&exec, &context), "context create failed");

    const std::uint32_t major_ptr_host[] = { 0u, 2u, 4u, 5u };
    const std::uint32_t minor_idx_host[] = { 0u, 2u, 1u, 2u, 0u };
    const __half values_host[] = {
        __float2half(1.0f),
        __float2half(2.0f),
        __float2half(3.0f),
        __float2half(4.0f),
        __float2half(5.0f)
    };
    const float rhs_host[] = {
        1.0f, 2.0f,
        10.0f, 20.0f,
        100.0f, 200.0f
    };
    const float grad_out_host[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };
    const float row_scales_host[] = { 2.0f, 3.0f, 4.0f };
    const float ones_host[] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

    std::uint32_t *major_ptr = device_copy(major_ptr_host, 4u);
    std::uint32_t *minor_idx = device_copy(minor_idx_host, 5u);
    __half *values = device_copy(values_host, 5u);
    float *rhs = device_copy(rhs_host, 6u);
    float *grad_out = device_copy(grad_out_host, 6u);
    float *row_scales = device_copy(row_scales_host, 3u);
    float *ones = device_copy(ones_host, 5u);

    cellerator_sparse_layout_desc csr{};
    csr.version = CELLERATOR_ABI_VERSION;
    csr.size = sizeof(csr);
    csr.layout = CELLERATOR_LAYOUT_CSR;
    csr.rows = 3u;
    csr.cols = 3u;
    csr.nnz = 5u;
    csr.value_dtype = CELLERATOR_DTYPE_F16;
    csr.major_ptr = major_ptr;
    csr.minor_idx = minor_idx;
    csr.values = values;

    float *projection_out = device_alloc<float>(6u);
    auto rhs_desc = matrix_desc(rhs, 3u, 2u, 2);
    auto projection_desc = matrix_desc(projection_out, 3u, 2u, 2);
    cellerator_saved_state *projection_saved = nullptr;
    require_status(
        cellerator_sparse_projection_forward(context, &csr, &rhs_desc, &projection_desc, &projection_saved),
        "projection forward failed");
    require(projection_saved != nullptr, "projection forward did not return saved state");
    require_status(cellerator_context_synchronize(context), "context synchronize failed");
    float projection_host[6] = {};
    download(projection_out, projection_host, 6u);
    require(close_value(projection_host[0], 201.0f), "projection out 0 mismatch");
    require(close_value(projection_host[1], 402.0f), "projection out 1 mismatch");
    require(close_value(projection_host[2], 430.0f), "projection out 2 mismatch");
    require(close_value(projection_host[5], 10.0f), "projection out 5 mismatch");

    float *projection_grad_values = device_alloc<float>(5u);
    float *projection_grad_rhs = device_alloc<float>(6u);
    auto grad_out_desc = matrix_desc(grad_out, 3u, 2u, 2);
    auto grad_values_desc = vector_desc(projection_grad_values, CELLERATOR_DTYPE_F32, 5u);
    auto grad_rhs_desc = matrix_desc(projection_grad_rhs, 3u, 2u, 2);
    require_status(
        cellerator_sparse_projection_backward(context, &csr, &rhs_desc, &grad_out_desc, &grad_values_desc, &grad_rhs_desc),
        "projection backward failed");
    require_status(cellerator_context_synchronize(context), "context synchronize failed");
    float grad_values_host[5] = {};
    float grad_rhs_host[6] = {};
    download(projection_grad_values, grad_values_host, 5u);
    download(projection_grad_rhs, grad_rhs_host, 6u);
    require(close_value(grad_values_host[0], 5.0f), "projection grad value 0 mismatch");
    require(close_value(grad_values_host[1], 500.0f), "projection grad value 1 mismatch");
    require(close_value(grad_values_host[2], 110.0f), "projection grad value 2 mismatch");
    require(close_value(grad_values_host[3], 1100.0f), "projection grad value 3 mismatch");
    require(close_value(grad_values_host[4], 17.0f), "projection grad value 4 mismatch");
    require(close_value(grad_rhs_host[0], 26.0f), "projection grad rhs 0 mismatch");
    require(close_value(grad_rhs_host[1], 32.0f), "projection grad rhs 1 mismatch");
    require(close_value(grad_rhs_host[2], 9.0f), "projection grad rhs 2 mismatch");
    require(close_value(grad_rhs_host[3], 12.0f), "projection grad rhs 3 mismatch");
    require(close_value(grad_rhs_host[4], 14.0f), "projection grad rhs 4 mismatch");
    require(close_value(grad_rhs_host[5], 20.0f), "projection grad rhs 5 mismatch");

    const std::uint32_t block_col_idx_host[] = { 0u, 2u, 1u, 2u, 0u, 0xffffffffu };
    const __half blocked_values_host[] = {
        __float2half(1.0f),
        __float2half(2.0f),
        __float2half(3.0f),
        __float2half(4.0f),
        __float2half(5.0f),
        __float2half(0.0f)
    };
    std::uint32_t *block_col_idx = device_copy(block_col_idx_host, 6u);
    __half *blocked_values = device_copy(blocked_values_host, 6u);
    float *blocked_out = device_alloc<float>(6u);
    cellerator_sparse_layout_desc blocked{};
    blocked.version = CELLERATOR_ABI_VERSION;
    blocked.size = sizeof(blocked);
    blocked.layout = CELLERATOR_LAYOUT_BLOCKED_ELL;
    blocked.rows = 3u;
    blocked.cols = 3u;
    blocked.nnz = 5u;
    blocked.block_size = 1u;
    blocked.ell_cols = 2u;
    blocked.value_dtype = CELLERATOR_DTYPE_F16;
    blocked.block_col_idx = block_col_idx;
    blocked.values = blocked_values;
    auto blocked_out_desc = matrix_desc(blocked_out, 3u, 2u, 2);
    require_status(
        cellerator_sparse_projection_forward(context, &blocked, &rhs_desc, &blocked_out_desc, nullptr),
        "blocked projection forward failed");
    require_status(cellerator_context_synchronize(context), "context synchronize failed");
    float blocked_host[6] = {};
    download(blocked_out, blocked_host, 6u);
    require(close_value(blocked_host[0], projection_host[0]), "blocked projection out 0 mismatch");
    require(close_value(blocked_host[5], projection_host[5]), "blocked projection out 5 mismatch");

    float *row_scale_out = device_alloc<float>(5u);
    float *row_grad_values = device_alloc<float>(5u);
    float *row_grad_scales = device_alloc<float>(3u);
    auto row_scales_desc = vector_desc(row_scales, CELLERATOR_DTYPE_F32, 3u);
    auto row_scale_out_desc = vector_desc(row_scale_out, CELLERATOR_DTYPE_F32, 5u);
    auto ones_desc = vector_desc(ones, CELLERATOR_DTYPE_F32, 5u);
    auto row_grad_values_desc = vector_desc(row_grad_values, CELLERATOR_DTYPE_F32, 5u);
    auto row_grad_scales_desc = vector_desc(row_grad_scales, CELLERATOR_DTYPE_F32, 3u);
    cellerator_saved_state *row_scale_saved = nullptr;
    require_status(
        cellerator_csr_row_scale_forward(context, &csr, &row_scales_desc, &row_scale_out_desc, &row_scale_saved),
        "row-scale forward failed");
    require(row_scale_saved != nullptr, "row-scale forward did not return saved state");
    require_status(
        cellerator_csr_row_scale_backward(context, &csr, &ones_desc, &row_scales_desc, &row_grad_values_desc, &row_grad_scales_desc),
        "row-scale backward failed");
    require_status(cellerator_context_synchronize(context), "context synchronize failed");
    float row_scale_host[5] = {};
    float row_grad_values_host[5] = {};
    float row_grad_scales_host[3] = {};
    download(row_scale_out, row_scale_host, 5u);
    download(row_grad_values, row_grad_values_host, 5u);
    download(row_grad_scales, row_grad_scales_host, 3u);
    require(close_value(row_scale_host[0], 2.0f), "row-scale out 0 mismatch");
    require(close_value(row_scale_host[4], 20.0f), "row-scale out 4 mismatch");
    require(close_value(row_grad_values_host[0], 2.0f), "row-scale grad value 0 mismatch");
    require(close_value(row_grad_values_host[4], 4.0f), "row-scale grad value 4 mismatch");
    require(close_value(row_grad_scales_host[0], 3.0f), "row-scale grad scale 0 mismatch");
    require(close_value(row_grad_scales_host[1], 7.0f), "row-scale grad scale 1 mismatch");
    require(close_value(row_grad_scales_host[2], 5.0f), "row-scale grad scale 2 mismatch");

    const float log_scale_host[] = { -0.35f, 0.15f, 0.55f };
    const float offset_host[] = { 0.25f, 0.10f, 0.50f };
    const float dense_host[] = {
        1.0f, 0.0f, 2.0f,
        0.0f, 3.0f, 4.0f,
        5.0f, 0.0f, 0.0f
    };
    cellerator_feature_affine_reconstruction_config reconstruction_config{};
    reconstruction_config.version = CELLERATOR_ABI_VERSION;
    reconstruction_config.size = sizeof(reconstruction_config);
    reconstruction_config.bits = 2u;
    reconstruction_config.scale_floor = 1.0e-3f;
    reconstruction_config.reconstruction_weight = 1.0f;
    reconstruction_config.range_weight = 0.05f;
    reconstruction_config.min_dynamic_range = 1.25f;
    float ref_reconstruction = 0.0f;
    float ref_range = 0.0f;
    float ref_grad_log[3] = {};
    float ref_grad_offset[3] = {};
    compute_quantize_reference(
        dense_host,
        3u,
        3u,
        log_scale_host,
        offset_host,
        reconstruction_config,
        &ref_reconstruction,
        &ref_range,
        ref_grad_log,
        ref_grad_offset);

    float *log_scale = device_copy(log_scale_host, 3u);
    float *offset = device_copy(offset_host, 3u);
    float *reconstruction_loss = device_alloc<float>(1u);
    float *range_loss = device_alloc<float>(1u);
    float *grad_log = device_alloc<float>(3u);
    float *grad_offset = device_alloc<float>(3u);
    auto log_scale_desc = vector_desc(log_scale, CELLERATOR_DTYPE_F32, 3u);
    auto offset_desc = vector_desc(offset, CELLERATOR_DTYPE_F32, 3u);
    auto reconstruction_loss_desc = scalar_desc(reconstruction_loss);
    auto range_loss_desc = scalar_desc(range_loss);
    auto grad_log_desc = vector_desc(grad_log, CELLERATOR_DTYPE_F32, 3u);
    auto grad_offset_desc = vector_desc(grad_offset, CELLERATOR_DTYPE_F32, 3u);
    require_status(
        cellerator_sparse_reconstruction_loss_forward_backward(
            context,
            &csr,
            &log_scale_desc,
            &offset_desc,
            &reconstruction_config,
            &reconstruction_loss_desc,
            &range_loss_desc,
            &grad_log_desc,
            &grad_offset_desc),
        "reconstruction loss forward/backward failed");
    require_status(cellerator_context_synchronize(context), "context synchronize failed");
    float reconstruction_host = 0.0f;
    float range_host = 0.0f;
    float grad_log_host[3] = {};
    float grad_offset_host[3] = {};
    download(reconstruction_loss, &reconstruction_host, 1u);
    download(range_loss, &range_host, 1u);
    download(grad_log, grad_log_host, 3u);
    download(grad_offset, grad_offset_host, 3u);
    require(close_value(reconstruction_host, ref_reconstruction, 1.0e-4f), "reconstruction loss mismatch");
    require(close_value(range_host, ref_range, 1.0e-4f), "range loss mismatch");
    for (std::uint32_t i = 0; i < 3u; ++i) {
        require(close_value(grad_log_host[i], ref_grad_log[i], 1.0e-4f), "grad log-scale mismatch");
        require(close_value(grad_offset_host[i], ref_grad_offset[i], 1.0e-4f), "grad offset mismatch");
    }

    auto bad_out_desc = matrix_desc(projection_out, 2u, 2u, 2);
    const cellerator_status bad_status = cellerator_sparse_projection_forward(context, &csr, &rhs_desc, &bad_out_desc, nullptr);
    require(bad_status == CELLERATOR_STATUS_INVALID_SHAPE, "invalid projection shape should fail with invalid-shape status");

    cellerator_saved_state_destroy(row_scale_saved);
    cellerator_saved_state_destroy(projection_saved);
    cellerator_context_destroy(context);
    cudaFree(grad_offset);
    cudaFree(grad_log);
    cudaFree(range_loss);
    cudaFree(reconstruction_loss);
    cudaFree(offset);
    cudaFree(log_scale);
    cudaFree(row_grad_scales);
    cudaFree(row_grad_values);
    cudaFree(row_scale_out);
    cudaFree(blocked_out);
    cudaFree(blocked_values);
    cudaFree(block_col_idx);
    cudaFree(projection_grad_rhs);
    cudaFree(projection_grad_values);
    cudaFree(projection_out);
    cudaFree(ones);
    cudaFree(row_scales);
    cudaFree(grad_out);
    cudaFree(rhs);
    cudaFree(values);
    cudaFree(minor_idx);
    cudaFree(major_ptr);
    return 0;
}
