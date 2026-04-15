#include "../src/compute/autograd/autograd.hh"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace autograd = ::cellerator::compute::autograd;
namespace cs = ::cellshard;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
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
    const autograd::feature_affine_quantize_config &config,
    float *reconstruction_loss,
    float *range_loss,
    float *grad_log_scale,
    float *grad_offset) {
    const float inv_elements = rows == 0u
        ? 0.0f
        : 1.0f / static_cast<float>(static_cast<double>(rows) * static_cast<double>(cols));
    const float inv_features = cols == 0u ? 0.0f : 1.0f / static_cast<float>(cols);
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
autograd::device_buffer<T> allocate_on_device(int device, std::size_t count) {
    autograd::cuda_require(cudaSetDevice(device), "cudaSetDevice(allocate_on_device)");
    return autograd::allocate_device_buffer<T>(count);
}

template<typename T>
void upload_on_device(int device, autograd::device_buffer<T> *dst, const T *src, std::size_t count) {
    autograd::cuda_require(cudaSetDevice(device), "cudaSetDevice(upload_on_device)");
    autograd::upload_device_buffer(dst, src, count);
}

template<typename T>
void download_on_device(int device, const autograd::device_buffer<T> &src, T *dst, std::size_t count) {
    autograd::cuda_require(cudaSetDevice(device), "cudaSetDevice(download_on_device)");
    autograd::download_device_buffer(src, dst, count);
}

} // namespace

int main() {
    int device_count = 0;
    require(cudaGetDeviceCount(&device_count) == cudaSuccess, "cudaGetDeviceCount failed");
    require(device_count > 0, "computeAutogradRuntimeTest requires at least one visible CUDA device");

    autograd::execution_context ctx;
    autograd::init(&ctx);

    autograd::scratch_arena scratch;
    autograd::init(&scratch);

    const std::uint32_t major_ptr_host[] = { 0, 2, 4, 5 };
    const std::uint32_t minor_idx_host[] = { 0, 2, 1, 2, 0 };
    const __half values_host[] = {
        __float2half(1.0f),
        __float2half(2.0f),
        __float2half(3.0f),
        __float2half(4.0f),
        __float2half(5.0f)
    };
    const float row_scales_host[] = { 2.0f, 3.0f, 4.0f };
    const float ones_grad_host[] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    const float vector_host[] = { 1.0f, 10.0f, 100.0f };
    const float rhs_host[] = {
        1.0f, 2.0f,
        10.0f, 20.0f,
        100.0f, 200.0f
    };
    const float grad_out_spmm_host[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f
    };

    auto major_ptr = autograd::allocate_device_buffer<std::uint32_t>(4);
    auto minor_idx = autograd::allocate_device_buffer<std::uint32_t>(5);
    auto values = autograd::allocate_device_buffer<__half>(5);
    auto row_scales = autograd::allocate_device_buffer<float>(3);
    auto ones_grad = autograd::allocate_device_buffer<float>(5);
    auto vector = autograd::allocate_device_buffer<float>(3);
    auto rhs = autograd::allocate_device_buffer<float>(6);
    auto grad_out_spmm = autograd::allocate_device_buffer<float>(6);

    autograd::upload_device_buffer(&major_ptr, major_ptr_host, 4);
    autograd::upload_device_buffer(&minor_idx, minor_idx_host, 5);
    autograd::upload_device_buffer(&values, values_host, 5);
    autograd::upload_device_buffer(&row_scales, row_scales_host, 3);
    autograd::upload_device_buffer(&ones_grad, ones_grad_host, 5);
    autograd::upload_device_buffer(&vector, vector_host, 3);
    autograd::upload_device_buffer(&rhs, rhs_host, 6);
    autograd::upload_device_buffer(&grad_out_spmm, grad_out_spmm_host, 6);

    auto scaled_values = autograd::allocate_device_buffer<float>(5);
    autograd::base::csr_row_scale_fwd_f16_f32(
        ctx,
        major_ptr.data,
        values.data,
        row_scales.data,
        3,
        scaled_values.data);
    float scaled_host[5] = {};
    autograd::download_device_buffer(scaled_values, scaled_host, 5);
    require(close_value(scaled_host[0], 2.0f), "scaled value 0 mismatch");
    require(close_value(scaled_host[1], 4.0f), "scaled value 1 mismatch");
    require(close_value(scaled_host[2], 9.0f), "scaled value 2 mismatch");
    require(close_value(scaled_host[3], 12.0f), "scaled value 3 mismatch");
    require(close_value(scaled_host[4], 20.0f), "scaled value 4 mismatch");

    auto grad_values = autograd::allocate_device_buffer<float>(5);
    autograd::base::csr_row_scale_bwd_values_f16_f32(
        ctx,
        major_ptr.data,
        ones_grad.data,
        row_scales.data,
        3,
        grad_values.data);
    float grad_values_host[5] = {};
    autograd::download_device_buffer(grad_values, grad_values_host, 5);
    require(close_value(grad_values_host[0], 2.0f), "row-scale grad value 0 mismatch");
    require(close_value(grad_values_host[1], 2.0f), "row-scale grad value 1 mismatch");
    require(close_value(grad_values_host[2], 3.0f), "row-scale grad value 2 mismatch");
    require(close_value(grad_values_host[3], 3.0f), "row-scale grad value 3 mismatch");
    require(close_value(grad_values_host[4], 4.0f), "row-scale grad value 4 mismatch");

    auto grad_scales = autograd::allocate_device_buffer<float>(3);
    autograd::base::csr_row_scale_bwd_scales_f16_f32(
        ctx,
        major_ptr.data,
        values.data,
        ones_grad.data,
        3,
        grad_scales.data);
    float grad_scales_host[3] = {};
    autograd::download_device_buffer(grad_scales, grad_scales_host, 3);
    require(close_value(grad_scales_host[0], 3.0f), "row-scale grad scale 0 mismatch");
    require(close_value(grad_scales_host[1], 7.0f), "row-scale grad scale 1 mismatch");
    require(close_value(grad_scales_host[2], 5.0f), "row-scale grad scale 2 mismatch");

    auto scalar_sum = autograd::allocate_device_buffer<float>(1);
    autograd::base::sparse_value_sum_f16_f32(ctx, &scratch, values.data, 5, scalar_sum.data);
    float scalar_sum_host = 0.0f;
    autograd::download_device_buffer(scalar_sum, &scalar_sum_host, 1);
    require(close_value(scalar_sum_host, 15.0f), "sparse value sum mismatch");

    auto fill_grad = autograd::allocate_device_buffer<float>(5);
    autograd::base::sparse_value_sum_bwd_fill_f32(ctx, scalar_sum.data, 5, fill_grad.data);
    float fill_grad_host[5] = {};
    autograd::download_device_buffer(fill_grad, fill_grad_host, 5);
    for (float value : fill_grad_host) require(close_value(value, 15.0f), "sparse value fill mismatch");

    auto spmv_out = autograd::allocate_device_buffer<float>(3);
    autograd::base::csr_spmv_fwd_f16_f32(ctx, major_ptr.data, minor_idx.data, values.data, 3, vector.data, spmv_out.data);
    float spmv_out_host[3] = {};
    autograd::download_device_buffer(spmv_out, spmv_out_host, 3);
    require(close_value(spmv_out_host[0], 201.0f), "spmv out 0 mismatch");
    require(close_value(spmv_out_host[1], 430.0f), "spmv out 1 mismatch");
    require(close_value(spmv_out_host[2], 5.0f), "spmv out 2 mismatch");

    auto spmv_grad_values = autograd::allocate_device_buffer<float>(5);
    autograd::base::csr_spmv_bwd_values_f16_f32(
        ctx,
        major_ptr.data,
        minor_idx.data,
        spmv_out.data,
        vector.data,
        3,
        spmv_grad_values.data);
    float spmv_grad_values_host[5] = {};
    autograd::download_device_buffer(spmv_grad_values, spmv_grad_values_host, 5);
    require(close_value(spmv_grad_values_host[0], 201.0f), "spmv grad value 0 mismatch");
    require(close_value(spmv_grad_values_host[1], 20100.0f), "spmv grad value 1 mismatch");

    auto spmv_grad_vector = autograd::allocate_device_buffer<float>(3);
    autograd::base::csr_spmv_bwd_vector_f16_f32(
        ctx,
        major_ptr.data,
        minor_idx.data,
        values.data,
        spmv_out.data,
        3,
        3,
        spmv_grad_vector.data);
    float spmv_grad_vector_host[3] = {};
    autograd::download_device_buffer(spmv_grad_vector, spmv_grad_vector_host, 3);
    require(close_value(spmv_grad_vector_host[0], 226.0f), "spmv grad vector 0 mismatch");
    require(close_value(spmv_grad_vector_host[1], 1290.0f), "spmv grad vector 1 mismatch");
    require(close_value(spmv_grad_vector_host[2], 2122.0f), "spmv grad vector 2 mismatch");

    auto spmm_out = autograd::allocate_device_buffer<float>(6);
    autograd::base::csr_spmm_fwd_f16_f32(
        ctx,
        major_ptr.data,
        minor_idx.data,
        values.data,
        3,
        3,
        rhs.data,
        2,
        2,
        spmm_out.data,
        2);
    float spmm_out_host[6] = {};
    autograd::download_device_buffer(spmm_out, spmm_out_host, 6);
    require(close_value(spmm_out_host[0], 201.0f), "spmm out 0 mismatch");
    require(close_value(spmm_out_host[1], 402.0f), "spmm out 1 mismatch");
    require(close_value(spmm_out_host[2], 430.0f), "spmm out 2 mismatch");
    require(close_value(spmm_out_host[3], 860.0f), "spmm out 3 mismatch");
    require(close_value(spmm_out_host[4], 5.0f), "spmm out 4 mismatch");
    require(close_value(spmm_out_host[5], 10.0f), "spmm out 5 mismatch");

    cs::sparse::compressed blocked_src;
    cs::sparse::blocked_ell blocked_host;
    cs::sparse::init(&blocked_src, 3u, 3u, 5u, cs::sparse::compressed_by_row);
    cs::sparse::init(&blocked_host);
    blocked_src.majorPtr = const_cast<std::uint32_t *>(major_ptr_host);
    blocked_src.minorIdx = const_cast<std::uint32_t *>(minor_idx_host);
    blocked_src.val = const_cast<__half *>(values_host);
    require(cs::convert::blocked_ell_from_compressed(&blocked_src, 1u, &blocked_host) != 0, "blocked_ell_from_compressed runtime setup failed");
    auto blocked_idx = autograd::allocate_device_buffer<std::uint32_t>(static_cast<std::size_t>(cs::sparse::row_block_count(&blocked_host)) * cs::sparse::ell_width_blocks(&blocked_host));
    auto blocked_values = autograd::allocate_device_buffer<__half>(static_cast<std::size_t>(blocked_host.rows) * blocked_host.ell_cols);
    autograd::upload_device_buffer(&blocked_idx, blocked_host.blockColIdx, static_cast<std::size_t>(cs::sparse::row_block_count(&blocked_host)) * cs::sparse::ell_width_blocks(&blocked_host));
    autograd::upload_device_buffer(&blocked_values, blocked_host.val, static_cast<std::size_t>(blocked_host.rows) * blocked_host.ell_cols);

    auto blocked_spmm_ref = autograd::allocate_device_buffer<float>(6);
    autograd::base::blocked_ell_spmm_fwd_f16_f32(
        ctx,
        blocked_idx.data,
        blocked_values.data,
        blocked_host.rows,
        blocked_host.cols,
        blocked_host.block_size,
        blocked_host.ell_cols,
        rhs.data,
        2,
        2,
        blocked_spmm_ref.data,
        2);
    float blocked_spmm_ref_host[6] = {};
    autograd::download_device_buffer(blocked_spmm_ref, blocked_spmm_ref_host, 6);
    require(close_value(blocked_spmm_ref_host[0], 201.0f), "blocked ell ref out 0 mismatch");
    require(close_value(blocked_spmm_ref_host[3], 860.0f), "blocked ell ref out 3 mismatch");

    autograd::cusparse_cache blocked_cache;
    autograd::init(&blocked_cache);
    auto blocked_spmm_lib = autograd::allocate_device_buffer<float>(6);
    autograd::base::blocked_ell_spmm_fwd_f16_f32_lib(
        ctx,
        &blocked_cache,
        blocked_host.val,
        blocked_idx.data,
        blocked_values.data,
        blocked_host.rows,
        blocked_host.cols,
        blocked_host.block_size,
        blocked_host.ell_cols,
        rhs.data,
        2,
        2,
        blocked_spmm_lib.data,
        2);
    float blocked_spmm_lib_host[6] = {};
    autograd::download_device_buffer(blocked_spmm_lib, blocked_spmm_lib_host, 6);
    require(close_value(blocked_spmm_lib_host[0], 201.0f), "blocked ell lib out 0 mismatch");
    require(close_value(blocked_spmm_lib_host[5], 10.0f), "blocked ell lib out 5 mismatch");

    const float quantize_dense_host[] = {
        1.0f, 0.0f, 2.0f,
        0.0f, 3.0f, 4.0f,
        5.0f, 0.0f, 0.0f
    };
    const float quantize_log_scale_host[] = { -0.35f, 0.15f, 0.55f };
    const float quantize_offset_host[] = { 0.25f, 0.10f, 0.50f };
    autograd::feature_affine_quantize_config quantize_config;
    quantize_config.bits = 2u;
    quantize_config.scale_floor = 1.0e-3f;
    quantize_config.reconstruction_weight = 1.0f;
    quantize_config.range_weight = 0.05f;
    quantize_config.min_dynamic_range = 1.25f;

    auto quantize_log_scale = autograd::allocate_device_buffer<float>(3);
    auto quantize_offset = autograd::allocate_device_buffer<float>(3);
    auto quantize_reconstruction = autograd::allocate_device_buffer<float>(1);
    auto quantize_range = autograd::allocate_device_buffer<float>(1);
    auto quantize_grad_log = autograd::allocate_device_buffer<float>(3);
    auto quantize_grad_offset = autograd::allocate_device_buffer<float>(3);
    autograd::upload_device_buffer(&quantize_log_scale, quantize_log_scale_host, 3);
    autograd::upload_device_buffer(&quantize_offset, quantize_offset_host, 3);

    float quantize_ref_reconstruction = 0.0f;
    float quantize_ref_range = 0.0f;
    float quantize_ref_grad_log[3] = {};
    float quantize_ref_grad_offset[3] = {};
    compute_quantize_reference(
        quantize_dense_host,
        3u,
        3u,
        quantize_log_scale_host,
        quantize_offset_host,
        quantize_config,
        &quantize_ref_reconstruction,
        &quantize_ref_range,
        quantize_ref_grad_log,
        quantize_ref_grad_offset);

    autograd::base::csr_feature_affine_quantize_fwd_bwd_f16_f32(
        ctx,
        major_ptr.data,
        minor_idx.data,
        values.data,
        3u,
        3u,
        quantize_log_scale.data,
        quantize_offset.data,
        quantize_config,
        quantize_reconstruction.data,
        quantize_range.data,
        quantize_grad_log.data,
        quantize_grad_offset.data);

    float quantize_reconstruction_host = 0.0f;
    float quantize_range_host = 0.0f;
    float quantize_grad_log_host[3] = {};
    float quantize_grad_offset_host[3] = {};
    autograd::download_device_buffer(quantize_reconstruction, &quantize_reconstruction_host, 1);
    autograd::download_device_buffer(quantize_range, &quantize_range_host, 1);
    autograd::download_device_buffer(quantize_grad_log, quantize_grad_log_host, 3);
    autograd::download_device_buffer(quantize_grad_offset, quantize_grad_offset_host, 3);
    require(close_value(quantize_reconstruction_host, quantize_ref_reconstruction, 1.0e-4f), "csr quantize reconstruction loss mismatch");
    require(close_value(quantize_range_host, quantize_ref_range, 1.0e-4f), "csr quantize range loss mismatch");
    for (int i = 0; i < 3; ++i) {
        require(close_value(quantize_grad_log_host[i], quantize_ref_grad_log[i], 2.0e-4f), "csr quantize grad log mismatch");
        require(close_value(quantize_grad_offset_host[i], quantize_ref_grad_offset[i], 2.0e-4f), "csr quantize grad offset mismatch");
    }

    autograd::base::blocked_ell_feature_affine_quantize_fwd_bwd_f16_f32(
        ctx,
        blocked_idx.data,
        blocked_values.data,
        blocked_host.rows,
        blocked_host.cols,
        blocked_host.block_size,
        blocked_host.ell_cols,
        quantize_log_scale.data,
        quantize_offset.data,
        quantize_config,
        quantize_reconstruction.data,
        quantize_range.data,
        quantize_grad_log.data,
        quantize_grad_offset.data);
    autograd::download_device_buffer(quantize_reconstruction, &quantize_reconstruction_host, 1);
    autograd::download_device_buffer(quantize_range, &quantize_range_host, 1);
    autograd::download_device_buffer(quantize_grad_log, quantize_grad_log_host, 3);
    autograd::download_device_buffer(quantize_grad_offset, quantize_grad_offset_host, 3);
    require(close_value(quantize_reconstruction_host, quantize_ref_reconstruction, 1.0e-4f), "blocked ell quantize reconstruction loss mismatch");
    require(close_value(quantize_range_host, quantize_ref_range, 1.0e-4f), "blocked ell quantize range loss mismatch");
    for (int i = 0; i < 3; ++i) {
        require(close_value(quantize_grad_log_host[i], quantize_ref_grad_log[i], 2.0e-4f), "blocked ell quantize grad log mismatch");
        require(close_value(quantize_grad_offset_host[i], quantize_ref_grad_offset[i], 2.0e-4f), "blocked ell quantize grad offset mismatch");
    }

    autograd::clear(&blocked_cache);
    cs::sparse::clear(&blocked_host);

    auto spmm_grad_values = autograd::allocate_device_buffer<float>(5);
    autograd::base::csr_spmm_bwd_values_f16_f32(
        ctx,
        major_ptr.data,
        minor_idx.data,
        grad_out_spmm.data,
        rhs.data,
        3,
        2,
        2,
        spmm_grad_values.data);
    float spmm_grad_values_host[5] = {};
    autograd::download_device_buffer(spmm_grad_values, spmm_grad_values_host, 5);
    require(close_value(spmm_grad_values_host[0], 5.0f), "spmm grad value 0 mismatch");
    require(close_value(spmm_grad_values_host[1], 500.0f), "spmm grad value 1 mismatch");
    require(close_value(spmm_grad_values_host[2], 110.0f), "spmm grad value 2 mismatch");
    require(close_value(spmm_grad_values_host[3], 1100.0f), "spmm grad value 3 mismatch");
    require(close_value(spmm_grad_values_host[4], 17.0f), "spmm grad value 4 mismatch");

    auto spmm_grad_rhs = autograd::allocate_device_buffer<float>(6);
    autograd::base::csr_spmm_bwd_rhs_f16_f32(
        ctx,
        major_ptr.data,
        minor_idx.data,
        values.data,
        grad_out_spmm.data,
        3,
        3,
        2,
        2,
        spmm_grad_rhs.data,
        2);
    float spmm_grad_rhs_host[6] = {};
    autograd::download_device_buffer(spmm_grad_rhs, spmm_grad_rhs_host, 6);
    require(close_value(spmm_grad_rhs_host[0], 26.0f), "spmm grad rhs 0 mismatch");
    require(close_value(spmm_grad_rhs_host[1], 32.0f), "spmm grad rhs 1 mismatch");
    require(close_value(spmm_grad_rhs_host[2], 9.0f), "spmm grad rhs 2 mismatch");
    require(close_value(spmm_grad_rhs_host[3], 12.0f), "spmm grad rhs 3 mismatch");
    require(close_value(spmm_grad_rhs_host[4], 14.0f), "spmm grad rhs 4 mismatch");
    require(close_value(spmm_grad_rhs_host[5], 20.0f), "spmm grad rhs 5 mismatch");

    if (device_count >= 4) {
        autograd::fleet_context fleet;
        autograd::init(&fleet);
        autograd::discover_fleet(&fleet, true, cudaStreamNonBlocking, true);

        unsigned int pair_slots[2] = {};
        require(autograd::default_mode_pair_slots(fleet, 0u, pair_slots, 2u) == 2u, "pair slots unavailable");
        require(autograd::fleet_slot_available(fleet, pair_slots[0]), "pair leader slot 0 unavailable");
        require(autograd::fleet_slot_available(fleet, pair_slots[1]), "pair peer slot unavailable");

        const int pair_device0 = autograd::fleet_device_id(fleet, pair_slots[0]);
        const int pair_device1 = autograd::fleet_device_id(fleet, pair_slots[1]);

        const std::uint32_t pair_major0_host[] = { 0, 2, 4 };
        const std::uint32_t pair_minor0_host[] = { 0, 2, 1, 2 };
        const __half pair_values0_host[] = {
            __float2half(1.0f),
            __float2half(2.0f),
            __float2half(3.0f),
            __float2half(4.0f)
        };
        const std::uint32_t pair_major1_host[] = { 0, 1 };
        const std::uint32_t pair_minor1_host[] = { 0 };
        const __half pair_values1_host[] = { __float2half(5.0f) };

        auto pair_major0 = allocate_on_device<std::uint32_t>(pair_device0, 3);
        auto pair_minor0 = allocate_on_device<std::uint32_t>(pair_device0, 4);
        auto pair_values0 = allocate_on_device<__half>(pair_device0, 4);
        auto pair_vector0 = allocate_on_device<float>(pair_device0, 3);
        auto pair_out0 = allocate_on_device<float>(pair_device0, 2);

        auto pair_major1 = allocate_on_device<std::uint32_t>(pair_device1, 2);
        auto pair_minor1 = allocate_on_device<std::uint32_t>(pair_device1, 1);
        auto pair_values1 = allocate_on_device<__half>(pair_device1, 1);
        auto pair_vector1 = allocate_on_device<float>(pair_device1, 3);
        auto pair_out1 = allocate_on_device<float>(pair_device1, 1);

        upload_on_device(pair_device0, &pair_major0, pair_major0_host, 3);
        upload_on_device(pair_device0, &pair_minor0, pair_minor0_host, 4);
        upload_on_device(pair_device0, &pair_values0, pair_values0_host, 4);
        upload_on_device(pair_device0, &pair_vector0, vector_host, 3);

        upload_on_device(pair_device1, &pair_major1, pair_major1_host, 2);
        upload_on_device(pair_device1, &pair_minor1, pair_minor1_host, 1);
        upload_on_device(pair_device1, &pair_values1, pair_values1_host, 1);
        upload_on_device(pair_device1, &pair_vector1, vector_host, 3);

        const std::uint32_t *pair_major_ptrs[] = { pair_major0.data, pair_major1.data };
        const std::uint32_t *pair_minor_ptrs[] = { pair_minor0.data, pair_minor1.data };
        const __half *pair_value_ptrs[] = { pair_values0.data, pair_values1.data };
        const float *pair_vector_ptrs[] = { pair_vector0.data, pair_vector1.data };
        const std::uint32_t pair_rows[] = { 2u, 1u };
        float *pair_out_ptrs[] = { pair_out0.data, pair_out1.data };

        autograd::dist::launch_csr_spmv_fwd_f16_f32(
            &fleet,
            pair_slots,
            2,
            pair_major_ptrs,
            pair_minor_ptrs,
            pair_value_ptrs,
            pair_rows,
            pair_vector_ptrs,
            pair_out_ptrs);
        autograd::synchronize_slots(fleet, pair_slots, 2);

        float pair_out0_host[2] = {};
        float pair_out1_host[1] = {};
        download_on_device(pair_device0, pair_out0, pair_out0_host, 2);
        download_on_device(pair_device1, pair_out1, pair_out1_host, 1);
        require(close_value(pair_out0_host[0], 201.0f), "pair-local row-shard output 0 mismatch");
        require(close_value(pair_out0_host[1], 430.0f), "pair-local row-shard output 1 mismatch");
        require(close_value(pair_out1_host[0], 5.0f), "pair-local row-shard output 2 mismatch");

        unsigned int fleet_slots[4] = {};
        require(autograd::default_mode_fleet_slots(fleet, fleet_slots, 4) == 4u, "default fleet slots unavailable");
        for (unsigned int i = 0; i < 4; ++i) {
            require(autograd::fleet_slot_available(fleet, fleet_slots[i]), "fleet slot unavailable");
        }

        const int fleet_device0 = autograd::fleet_device_id(fleet, fleet_slots[0]);
        const int fleet_device1 = autograd::fleet_device_id(fleet, fleet_slots[1]);
        const int fleet_device2 = autograd::fleet_device_id(fleet, fleet_slots[2]);
        const int fleet_device3 = autograd::fleet_device_id(fleet, fleet_slots[3]);

        const std::uint32_t feature_major_slot0_host[] = { 0, 1, 1 };
        const std::uint32_t feature_minor_slot0_host[] = { 0 };
        const __half feature_values_slot0_host[] = { __float2half(1.0f) };
        const float feature_vector_slot0_host[] = { 10.0f };

        const std::uint32_t feature_major_slot1_host[] = { 0, 0, 1 };
        const std::uint32_t feature_minor_slot1_host[] = { 0 };
        const __half feature_values_slot1_host[] = { __float2half(3.0f) };
        const float feature_vector_slot1_host[] = { 20.0f };

        const std::uint32_t feature_major_slot2_host[] = { 0, 1, 1 };
        const std::uint32_t feature_minor_slot2_host[] = { 0 };
        const __half feature_values_slot2_host[] = { __float2half(2.0f) };
        const float feature_vector_slot2_host[] = { 30.0f };

        const std::uint32_t feature_major_slot3_host[] = { 0, 0, 1 };
        const std::uint32_t feature_minor_slot3_host[] = { 0 };
        const __half feature_values_slot3_host[] = { __float2half(4.0f) };
        const float feature_vector_slot3_host[] = { 40.0f };

        auto feature_major0 = allocate_on_device<std::uint32_t>(fleet_device0, 3);
        auto feature_minor0 = allocate_on_device<std::uint32_t>(fleet_device0, 1);
        auto feature_values0 = allocate_on_device<__half>(fleet_device0, 1);
        auto feature_vector0 = allocate_on_device<float>(fleet_device0, 1);
        auto feature_out0 = allocate_on_device<float>(fleet_device0, 2);
        auto feature_sum0 = allocate_on_device<float>(fleet_device0, 2);

        auto feature_major1 = allocate_on_device<std::uint32_t>(fleet_device1, 3);
        auto feature_minor1 = allocate_on_device<std::uint32_t>(fleet_device1, 1);
        auto feature_values1 = allocate_on_device<__half>(fleet_device1, 1);
        auto feature_vector1 = allocate_on_device<float>(fleet_device1, 1);
        auto feature_out1 = allocate_on_device<float>(fleet_device1, 2);

        auto feature_major2 = allocate_on_device<std::uint32_t>(fleet_device2, 3);
        auto feature_minor2 = allocate_on_device<std::uint32_t>(fleet_device2, 1);
        auto feature_values2 = allocate_on_device<__half>(fleet_device2, 1);
        auto feature_vector2 = allocate_on_device<float>(fleet_device2, 1);
        auto feature_out2 = allocate_on_device<float>(fleet_device2, 2);

        auto feature_major3 = allocate_on_device<std::uint32_t>(fleet_device3, 3);
        auto feature_minor3 = allocate_on_device<std::uint32_t>(fleet_device3, 1);
        auto feature_values3 = allocate_on_device<__half>(fleet_device3, 1);
        auto feature_vector3 = allocate_on_device<float>(fleet_device3, 1);
        auto feature_out3 = allocate_on_device<float>(fleet_device3, 2);

        upload_on_device(fleet_device0, &feature_major0, feature_major_slot0_host, 3);
        upload_on_device(fleet_device0, &feature_minor0, feature_minor_slot0_host, 1);
        upload_on_device(fleet_device0, &feature_values0, feature_values_slot0_host, 1);
        upload_on_device(fleet_device0, &feature_vector0, feature_vector_slot0_host, 1);

        upload_on_device(fleet_device1, &feature_major1, feature_major_slot1_host, 3);
        upload_on_device(fleet_device1, &feature_minor1, feature_minor_slot1_host, 1);
        upload_on_device(fleet_device1, &feature_values1, feature_values_slot1_host, 1);
        upload_on_device(fleet_device1, &feature_vector1, feature_vector_slot1_host, 1);

        upload_on_device(fleet_device2, &feature_major2, feature_major_slot2_host, 3);
        upload_on_device(fleet_device2, &feature_minor2, feature_minor_slot2_host, 1);
        upload_on_device(fleet_device2, &feature_values2, feature_values_slot2_host, 1);
        upload_on_device(fleet_device2, &feature_vector2, feature_vector_slot2_host, 1);

        upload_on_device(fleet_device3, &feature_major3, feature_major_slot3_host, 3);
        upload_on_device(fleet_device3, &feature_minor3, feature_minor_slot3_host, 1);
        upload_on_device(fleet_device3, &feature_values3, feature_values_slot3_host, 1);
        upload_on_device(fleet_device3, &feature_vector3, feature_vector_slot3_host, 1);

        const std::uint32_t *feature_major_ptrs[] = {
            feature_major0.data,
            feature_major1.data,
            feature_major2.data,
            feature_major3.data
        };
        const std::uint32_t *feature_minor_ptrs[] = {
            feature_minor0.data,
            feature_minor1.data,
            feature_minor2.data,
            feature_minor3.data
        };
        const __half *feature_value_ptrs[] = {
            feature_values0.data,
            feature_values1.data,
            feature_values2.data,
            feature_values3.data
        };
        const float *feature_vector_ptrs[] = {
            feature_vector0.data,
            feature_vector1.data,
            feature_vector2.data,
            feature_vector3.data
        };
        const std::uint32_t feature_rows[] = { 2u, 2u, 2u, 2u };
        float *feature_out_ptrs[] = {
            feature_out0.data,
            feature_out1.data,
            feature_out2.data,
            feature_out3.data
        };

        autograd::dist::launch_csr_spmv_fwd_f16_f32(
            &fleet,
            fleet_slots,
            4,
            feature_major_ptrs,
            feature_minor_ptrs,
            feature_value_ptrs,
            feature_rows,
            feature_vector_ptrs,
            feature_out_ptrs);
        autograd::dist::reduce_sum_to_leader_f32(
            &fleet,
            fleet_slots,
            4,
            const_cast<const float *const *>(feature_out_ptrs),
            2,
            feature_sum0.data);

        const unsigned int leader_slot_only[] = { fleet_slots[0] };
        autograd::synchronize_slots(fleet, leader_slot_only, 1);

        float feature_sum_host[2] = {};
        download_on_device(fleet_device0, feature_sum0, feature_sum_host, 2);
        require(close_value(feature_sum_host[0], 70.0f), "4-gpu feature-shard leader sum row 0 mismatch");
        require(close_value(feature_sum_host[1], 220.0f), "4-gpu feature-shard leader sum row 1 mismatch");

        autograd::clear(&fleet);
    }

    autograd::clear(&scratch);
    autograd::clear(&ctx);
    return 0;
}
