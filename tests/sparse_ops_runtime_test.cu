#include "../src/ops/matrix/csr_multiply.cuh"
#include <Cellerator/runtime/multi_gpu/fleet.cuh>
#include <Cellerator/compute/operators/sparse/ops.hh>
#include <Cellerator/compute/candidate/sparse/project.hh>
#include <Cellerator/ops/quantized/api.cuh>
#include <Cellerator/memory/cuda/shared_buffer.hh>
#include <Cellerator/tensor/base/device_views.cuh>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace runtime = ::cellerator::runtime;
namespace memory_cuda = ::cellerator::memory::cuda;
namespace sparse_ops = ::cellerator::compute::sparse::ops;
namespace matrix_ops = ::cellerator::ops::matrix;
namespace sparse_project = ::cellerator::compute::sparse::project;
namespace base = ::cellerator::tensor::base;
namespace msq = ::cellerator::tensor::quantized;
namespace mso = ::cellerator::ops::quantized;

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
    const sparse_ops::feature_affine_quantize_config &config,
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
memory_cuda::SharedDeviceBuffer<T> allocate_on_device(int device, std::size_t count) {
    return memory_cuda::SharedDeviceBuffer<T>::allocate(count, device);
}

template<typename T>
void upload_on_device(int device, memory_cuda::SharedDeviceBuffer<T> *dst, const T *src, std::size_t count) {
    auto stream = memory_cuda::Stream(device);
    dst->upload_async(src, count, stream);
    stream.synchronize();
}

template<typename T>
void download_on_device(int device, const memory_cuda::SharedDeviceBuffer<T> &src, T *dst, std::size_t count) {
    auto stream = memory_cuda::Stream(device);
    src.download_async(dst, count, stream);
    stream.synchronize();
}

template<typename T>
void upload_and_wait(memory_cuda::SharedDeviceBuffer<T> &dst, const T *src,
                     std::size_t count, const memory_cuda::Stream &stream) {
    dst.upload_async(src, count, stream);
    stream.synchronize();
}

template<typename T>
void download_and_wait(const memory_cuda::SharedDeviceBuffer<T> &src, T *dst,
                       std::size_t count, const memory_cuda::Stream &stream) {
    src.download_async(dst, count, stream);
    stream.synchronize();
}

void check_csr_multiply_edges(const runtime::execution_context& ctx,
                              const memory_cuda::Stream& stream) {
    const uint32_t row[] = {0, 3, 3}, col[] = {0, 0, 2};
    const __half w[] = {__float2half(1), __float2half(2), __float2half(4)};
    const float x[] = {5, 50, 99, 6, 60, 99, 7, 70, 99};
    const float initial[] = {-1, -1, -1, -1, -1, -1, -1, -1};
    auto drow = allocate_on_device<uint32_t>(ctx.device, 3);
    auto dcol = allocate_on_device<uint32_t>(ctx.device, 3);
    auto dw = allocate_on_device<__half>(ctx.device, 3);
    auto dx = allocate_on_device<float>(ctx.device, 9);
    auto dy = allocate_on_device<float>(ctx.device, 8);
    drow.upload_async(row, 3, stream);
    dcol.upload_async(col, 3, stream);
    dw.upload_async(w, 3, stream);
    dx.upload_async(x, 9, stream);
    dy.upload_async(initial, 8, stream);

    matrix_ops::Csr<const __half> a{{2, 3, 3, base::CompressedAxis::row, drow.data(), dcol.data()}, dw.data()};
    float actual[8] = {};
    runtime::cuda_require(matrix_ops::csr_spmv(a, dx.data(), dy.data(), ctx.stream), "csr_spmv");
    download_and_wait(dy, actual, 2, stream);
    require(actual[0] == 411 && actual[1] == 0, "SpMV duplicate and empty row");

    runtime::cuda_require(matrix_ops::csr_spmm(a, dx.data(), 3, 2, dy.data(), 4, ctx.stream), "csr_spmm");
    download_and_wait(dy, actual, 8, stream);
    const float expected[] = {43, 430, -1, -1, 0, 0, -1, -1};
    for (int i = 0; i != 8; ++i) require(actual[i] == expected[i], "SpMM duplicate, empty row, and padded stride");

    dy.upload_async(initial, 8, stream);
    runtime::cuda_require(matrix_ops::csr_spmm_transpose(a, dx.data(), 3, 2, dy.data(), 3, ctx.stream),
                          "csr_spmm_transpose with padded output");
    download_and_wait(dy, actual, 8, stream);
    const float expected_transpose[] = {15, 150, -1, 0, 0, -1, 20, 200};
    for (int i = 0; i != 8; ++i) require(actual[i] == expected_transpose[i], "transpose respects logical columns");

    matrix_ops::Csr<const __half> empty{{0, 3, 0, base::CompressedAxis::row, nullptr, nullptr}, nullptr};
    runtime::cuda_require(matrix_ops::csr_spmv_transpose(empty, nullptr, dy.data(), ctx.stream), "csr_spmv_transpose");
    download_and_wait(dy, actual, 3, stream);
    for (int i = 0; i != 3; ++i) require(actual[i] == 0, "empty transpose overwrites destination");
}

} // namespace

int main() {
    int device_count = 0;
    require(cudaGetDeviceCount(&device_count) == cudaSuccess, "cudaGetDeviceCount failed");
    require(device_count > 0, "sparseOpsRuntimeTest requires at least one visible CUDA device");

    runtime::execution_context ctx;
    runtime::init(&ctx);
    auto transfer_stream = memory_cuda::Stream::borrow(ctx.stream, ctx.device);
    check_csr_multiply_edges(ctx, transfer_stream);

    runtime::scratch_arena scratch;
    runtime::init(&scratch);

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

    auto major_ptr = memory_cuda::SharedDeviceBuffer<std::uint32_t>::allocate(4, ctx.device);
    auto minor_idx = memory_cuda::SharedDeviceBuffer<std::uint32_t>::allocate(5, ctx.device);
    auto values = memory_cuda::SharedDeviceBuffer<__half>::allocate(5, ctx.device);
    auto row_scales = memory_cuda::SharedDeviceBuffer<float>::allocate(3, ctx.device);
    auto ones_grad = memory_cuda::SharedDeviceBuffer<float>::allocate(5, ctx.device);
    auto vector = memory_cuda::SharedDeviceBuffer<float>::allocate(3, ctx.device);
    auto rhs = memory_cuda::SharedDeviceBuffer<float>::allocate(6, ctx.device);
    auto grad_out_spmm = memory_cuda::SharedDeviceBuffer<float>::allocate(6, ctx.device);

    upload_and_wait(major_ptr, major_ptr_host, 4, transfer_stream);
    upload_and_wait(minor_idx, minor_idx_host, 5, transfer_stream);
    upload_and_wait(values, values_host, 5, transfer_stream);
    upload_and_wait(row_scales, row_scales_host, 3, transfer_stream);
    upload_and_wait(ones_grad, ones_grad_host, 5, transfer_stream);
    upload_and_wait(vector, vector_host, 3, transfer_stream);
    upload_and_wait(rhs, rhs_host, 6, transfer_stream);
    upload_and_wait(grad_out_spmm, grad_out_spmm_host, 6, transfer_stream);

    auto scaled_values = memory_cuda::SharedDeviceBuffer<float>::allocate(5, ctx.device);
    sparse_ops::base::csr_row_scale_fwd_f16_f32(
        ctx,
        major_ptr.data(),
        values.data(),
        row_scales.data(),
        3,
        scaled_values.data());
    float scaled_host[5] = {};
    download_and_wait(scaled_values, scaled_host, 5, transfer_stream);
    require(close_value(scaled_host[0], 2.0f), "scaled value 0 mismatch");
    require(close_value(scaled_host[1], 4.0f), "scaled value 1 mismatch");
    require(close_value(scaled_host[2], 9.0f), "scaled value 2 mismatch");
    require(close_value(scaled_host[3], 12.0f), "scaled value 3 mismatch");
    require(close_value(scaled_host[4], 20.0f), "scaled value 4 mismatch");

    auto grad_values = memory_cuda::SharedDeviceBuffer<float>::allocate(5, ctx.device);
    sparse_ops::base::csr_row_scale_bwd_values_f16_f32(
        ctx,
        major_ptr.data(),
        ones_grad.data(),
        row_scales.data(),
        3,
        grad_values.data());
    float grad_values_host[5] = {};
    download_and_wait(grad_values, grad_values_host, 5, transfer_stream);
    require(close_value(grad_values_host[0], 2.0f), "row-scale grad value 0 mismatch");
    require(close_value(grad_values_host[1], 2.0f), "row-scale grad value 1 mismatch");
    require(close_value(grad_values_host[2], 3.0f), "row-scale grad value 2 mismatch");
    require(close_value(grad_values_host[3], 3.0f), "row-scale grad value 3 mismatch");
    require(close_value(grad_values_host[4], 4.0f), "row-scale grad value 4 mismatch");

    auto grad_scales = memory_cuda::SharedDeviceBuffer<float>::allocate(3, ctx.device);
    sparse_ops::base::csr_row_scale_bwd_scales_f16_f32(
        ctx,
        major_ptr.data(),
        values.data(),
        ones_grad.data(),
        3,
        grad_scales.data());
    float grad_scales_host[3] = {};
    download_and_wait(grad_scales, grad_scales_host, 3, transfer_stream);
    require(close_value(grad_scales_host[0], 3.0f), "row-scale grad scale 0 mismatch");
    require(close_value(grad_scales_host[1], 7.0f), "row-scale grad scale 1 mismatch");
    require(close_value(grad_scales_host[2], 5.0f), "row-scale grad scale 2 mismatch");

    auto scalar_sum = memory_cuda::SharedDeviceBuffer<float>::allocate(1, ctx.device);
    sparse_ops::base::sparse_value_sum_f16_f32(ctx, &scratch, values.data(), 5, scalar_sum.data());
    float scalar_sum_host = 0.0f;
    download_and_wait(scalar_sum, &scalar_sum_host, 1, transfer_stream);
    require(close_value(scalar_sum_host, 15.0f), "sparse value sum mismatch");

    auto fill_grad = memory_cuda::SharedDeviceBuffer<float>::allocate(5, ctx.device);
    sparse_ops::base::sparse_value_sum_bwd_fill_f32(ctx, scalar_sum.data(), 5, fill_grad.data());
    float fill_grad_host[5] = {};
    download_and_wait(fill_grad, fill_grad_host, 5, transfer_stream);
    for (float value : fill_grad_host) require(close_value(value, 15.0f), "sparse value fill mismatch");

    auto spmv_out = memory_cuda::SharedDeviceBuffer<float>::allocate(3, ctx.device);
    matrix_ops::Csr<const __half> csr{{3, 3, 5, base::CompressedAxis::row, major_ptr.data(), minor_idx.data()},
        values.data()};
    runtime::cuda_require(matrix_ops::csr_spmv(csr, vector.data(), spmv_out.data(), ctx.stream), "csr_spmv");
    float spmv_out_host[3] = {};
    download_and_wait(spmv_out, spmv_out_host, 3, transfer_stream);
    require(close_value(spmv_out_host[0], 201.0f), "spmv out 0 mismatch");
    require(close_value(spmv_out_host[1], 430.0f), "spmv out 1 mismatch");
    require(close_value(spmv_out_host[2], 5.0f), "spmv out 2 mismatch");

    auto spmv_grad_values = memory_cuda::SharedDeviceBuffer<float>::allocate(5, ctx.device);
    sparse_ops::base::csr_spmv_bwd_values_f16_f32(
        ctx,
        major_ptr.data(),
        minor_idx.data(),
        spmv_out.data(),
        vector.data(),
        3,
        spmv_grad_values.data());
    float spmv_grad_values_host[5] = {};
    download_and_wait(spmv_grad_values, spmv_grad_values_host, 5, transfer_stream);
    require(close_value(spmv_grad_values_host[0], 201.0f), "spmv grad value 0 mismatch");
    require(close_value(spmv_grad_values_host[1], 20100.0f), "spmv grad value 1 mismatch");

    auto spmv_grad_vector = memory_cuda::SharedDeviceBuffer<float>::allocate(3, ctx.device);
    runtime::cuda_require(matrix_ops::csr_spmv_transpose(
        csr, spmv_out.data(), spmv_grad_vector.data(), ctx.stream), "csr_spmv_transpose");
    float spmv_grad_vector_host[3] = {};
    download_and_wait(spmv_grad_vector, spmv_grad_vector_host, 3, transfer_stream);
    require(close_value(spmv_grad_vector_host[0], 226.0f), "spmv grad vector 0 mismatch");
    require(close_value(spmv_grad_vector_host[1], 1290.0f), "spmv grad vector 1 mismatch");
    require(close_value(spmv_grad_vector_host[2], 2122.0f), "spmv grad vector 2 mismatch");

    auto spmm_out = memory_cuda::SharedDeviceBuffer<float>::allocate(6, ctx.device);
    runtime::cuda_require(matrix_ops::csr_spmm(
        csr, rhs.data(), 2, 2, spmm_out.data(), 2, ctx.stream), "csr_spmm");
    float spmm_out_host[6] = {};
    download_and_wait(spmm_out, spmm_out_host, 6, transfer_stream);
    require(close_value(spmm_out_host[0], 201.0f), "spmm out 0 mismatch");
    require(close_value(spmm_out_host[1], 402.0f), "spmm out 1 mismatch");
    require(close_value(spmm_out_host[2], 430.0f), "spmm out 2 mismatch");
    require(close_value(spmm_out_host[3], 860.0f), "spmm out 3 mismatch");
    require(close_value(spmm_out_host[4], 5.0f), "spmm out 4 mismatch");
    require(close_value(spmm_out_host[5], 10.0f), "spmm out 5 mismatch");

    constexpr std::size_t blocked_slots = 6u;
    std::uint32_t block_col_idx_host[blocked_slots] = { 0u, 2u, 1u, 2u, 0u, base::blocked_ell_invalid_col };
    __half blocked_values_host[blocked_slots] = {
        __float2half(1.0f),
        __float2half(2.0f),
        __float2half(3.0f),
        __float2half(4.0f),
        __float2half(5.0f),
        __float2half(0.0f)
    };
    const base::device::BlockedEllView blocked_host{
        3u, 3u, 5u, 1u, 2u, block_col_idx_host, blocked_values_host};
    auto blocked_idx = memory_cuda::SharedDeviceBuffer<std::uint32_t>::allocate(blocked_slots, ctx.device);
    auto blocked_values = memory_cuda::SharedDeviceBuffer<__half>::allocate(blocked_slots, ctx.device);
    upload_and_wait(blocked_idx, blocked_host.blockColIdx, blocked_slots, transfer_stream);
    upload_and_wait(blocked_values, blocked_host.val, blocked_slots, transfer_stream);

    auto blocked_spmm_ref = memory_cuda::SharedDeviceBuffer<float>::allocate(6, ctx.device);
    sparse_project::blocked_ell_spmm_fwd_f16_f32(
        ctx,
        blocked_idx.data(),
        blocked_values.data(),
        blocked_host.rows,
        blocked_host.cols,
        blocked_host.block_size,
        blocked_host.ell_cols,
        rhs.data(),
        2,
        2,
        blocked_spmm_ref.data(),
        2);
    float blocked_spmm_ref_host[6] = {};
    download_and_wait(blocked_spmm_ref, blocked_spmm_ref_host, 6, transfer_stream);
    require(close_value(blocked_spmm_ref_host[0], 201.0f), "blocked ell ref out 0 mismatch");
    require(close_value(blocked_spmm_ref_host[3], 860.0f), "blocked ell ref out 3 mismatch");

    runtime::cusparse_cache blocked_cache;
    runtime::init(&blocked_cache);
    auto blocked_spmm_lib = memory_cuda::SharedDeviceBuffer<float>::allocate(6, ctx.device);
    sparse_project::blocked_ell_spmm_fwd_f16_f32_lib(
        ctx,
        &blocked_cache,
        blocked_host.val,
        blocked_idx.data(),
        blocked_values.data(),
        blocked_host.rows,
        blocked_host.cols,
        blocked_host.block_size,
        blocked_host.ell_cols,
        rhs.data(),
        2,
        2,
        blocked_spmm_lib.data(),
        2);
    float blocked_spmm_lib_host[6] = {};
    download_and_wait(blocked_spmm_lib, blocked_spmm_lib_host, 6, transfer_stream);
    require(close_value(blocked_spmm_lib_host[0], 201.0f), "blocked ell lib out 0 mismatch");
    require(close_value(blocked_spmm_lib_host[5], 10.0f), "blocked ell lib out 5 mismatch");

    std::vector<float> blocked_slot_values(static_cast<std::size_t>(blocked_host.rows) * blocked_host.ell_cols, 0.0f);
    for (std::size_t i = 0; i < blocked_slot_values.size(); ++i) {
        blocked_slot_values[i] = __half2float(blocked_host.val[i]);
    }

    const float per_gene_scales_host[] = { 1.0f, 1.0f, 1.0f };
    std::vector<unsigned char> quantized_packed_values(
        static_cast<std::size_t>(blocked_host.rows)
            * static_cast<std::size_t>(msq::blocked_ell::row_bytes<8>(blocked_host.ell_cols)),
        0u);
    auto quantized_host = msq::make_blocked_ell_view<8>(
        static_cast<int>(blocked_host.rows),
        static_cast<int>(blocked_host.cols),
        static_cast<int>(blocked_host.nnz),
        static_cast<int>(blocked_host.block_size),
        static_cast<int>(blocked_host.ell_cols),
        msq::blocked_ell::row_bytes<8>(blocked_host.ell_cols),
        blocked_host.blockColIdx,
        quantized_packed_values.data(),
        msq::make_per_gene_affine(per_gene_scales_host));
    require(msq::blocked_ell::pack_row_major_values(&quantized_host, quantized_packed_values.data(), blocked_slot_values.data()) == 0,
            "quantized blocked ell host pack failed");

    auto quantized_blocked_values = memory_cuda::SharedDeviceBuffer<std::uint8_t>::allocate(quantized_packed_values.size(), ctx.device);
    auto per_gene_scales = memory_cuda::SharedDeviceBuffer<float>::allocate(3, ctx.device);
    upload_and_wait(quantized_blocked_values, quantized_packed_values.data(), quantized_packed_values.size(), transfer_stream);
    upload_and_wait(per_gene_scales, per_gene_scales_host, 3, transfer_stream);
    ::cellerator::tensor::quantized::RuntimeBlockedEllView quantized_view{};
    quantized_view.rows = blocked_host.rows;
    quantized_view.cols = blocked_host.cols;
    quantized_view.nnz = blocked_host.nnz;
    quantized_view.block_size = blocked_host.block_size;
    quantized_view.ell_cols = blocked_host.ell_cols;
    quantized_view.row_stride_bytes = static_cast<std::uint32_t>(msq::blocked_ell::row_bytes<8>(blocked_host.ell_cols));
    quantized_view.bits = 8u;
    quantized_view.decode_policy = msq::blocked_ell::decode_policy_per_gene_affine;
    quantized_view.block_col_idx = blocked_idx.data();
    quantized_view.packed_values = quantized_blocked_values.data();
    quantized_view.column_scales = per_gene_scales.data();

    auto quantized_spmm_out = memory_cuda::SharedDeviceBuffer<float>::allocate(6, ctx.device);
    sparse_project::quantized_blocked_ell_spmm_fwd_f32(
        ctx,
        quantized_view,
        rhs.data(),
        2,
        2,
        quantized_spmm_out.data(),
        2);
    float quantized_spmm_host[6] = {};
    download_and_wait(quantized_spmm_out, quantized_spmm_host, 6, transfer_stream);
    require(close_value(quantized_spmm_host[0], blocked_spmm_ref_host[0]), "quantized blocked ell per_gene out 0 mismatch");
    require(close_value(quantized_spmm_host[5], blocked_spmm_ref_host[5]), "quantized blocked ell per_gene out 5 mismatch");

    const float row_offsets_host[] = { 0.0f, 3.0f, 5.0f };
    const float column_scales_host[] = { 1.0f, 1.0f, 1.0f };
    std::vector<unsigned char> quantized_row_offset_packed(
        static_cast<std::size_t>(blocked_host.rows)
            * static_cast<std::size_t>(msq::blocked_ell::row_bytes<4>(blocked_host.ell_cols)),
        0u);
    auto quantized_row_offset_host = msq::make_blocked_ell_view<4>(
        static_cast<int>(blocked_host.rows),
        static_cast<int>(blocked_host.cols),
        static_cast<int>(blocked_host.nnz),
        static_cast<int>(blocked_host.block_size),
        static_cast<int>(blocked_host.ell_cols),
        msq::blocked_ell::row_bytes<4>(blocked_host.ell_cols),
        blocked_host.blockColIdx,
        quantized_row_offset_packed.data(),
        msq::make_column_scale_row_offset(column_scales_host, row_offsets_host));
    require(msq::blocked_ell::pack_row_major_values(&quantized_row_offset_host, quantized_row_offset_packed.data(), blocked_slot_values.data()) == 0,
            "quantized blocked ell row-offset host pack failed");

    auto quantized_row_offset_values = memory_cuda::SharedDeviceBuffer<std::uint8_t>::allocate(quantized_row_offset_packed.size(), ctx.device);
    auto row_offsets = memory_cuda::SharedDeviceBuffer<float>::allocate(3, ctx.device);
    auto column_scales = memory_cuda::SharedDeviceBuffer<float>::allocate(3, ctx.device);
    upload_and_wait(quantized_row_offset_values, quantized_row_offset_packed.data(), quantized_row_offset_packed.size(), transfer_stream);
    upload_and_wait(row_offsets, row_offsets_host, 3, transfer_stream);
    upload_and_wait(column_scales, column_scales_host, 3, transfer_stream);

    ::cellerator::tensor::quantized::RuntimeBlockedEllView quantized_row_offset_view{};
    quantized_row_offset_view.rows = blocked_host.rows;
    quantized_row_offset_view.cols = blocked_host.cols;
    quantized_row_offset_view.nnz = blocked_host.nnz;
    quantized_row_offset_view.block_size = blocked_host.block_size;
    quantized_row_offset_view.ell_cols = blocked_host.ell_cols;
    quantized_row_offset_view.row_stride_bytes = static_cast<std::uint32_t>(msq::blocked_ell::row_bytes<4>(blocked_host.ell_cols));
    quantized_row_offset_view.bits = 4u;
    quantized_row_offset_view.decode_policy = msq::blocked_ell::decode_policy_column_scale_row_offset;
    quantized_row_offset_view.block_col_idx = blocked_idx.data();
    quantized_row_offset_view.packed_values = quantized_row_offset_values.data();
    quantized_row_offset_view.column_scales = column_scales.data();
    quantized_row_offset_view.row_offsets = row_offsets.data();

    auto quantized_row_offset_out = memory_cuda::SharedDeviceBuffer<float>::allocate(6, ctx.device);
    sparse_project::quantized_blocked_ell_spmm_fwd_f32(
        ctx,
        quantized_row_offset_view,
        rhs.data(),
        2,
        2,
        quantized_row_offset_out.data(),
        2);
    float quantized_row_offset_host_out[6] = {};
    download_and_wait(quantized_row_offset_out, quantized_row_offset_host_out, 6, transfer_stream);
    require(close_value(quantized_row_offset_host_out[1], blocked_spmm_ref_host[1]), "quantized blocked ell row-offset out 1 mismatch");
    require(close_value(quantized_row_offset_host_out[4], blocked_spmm_ref_host[4]), "quantized blocked ell row-offset out 4 mismatch");


    const float quantize_dense_host[] = {
        1.0f, 0.0f, 2.0f,
        0.0f, 3.0f, 4.0f,
        5.0f, 0.0f, 0.0f
    };
    const float quantize_log_scale_host[] = { -0.35f, 0.15f, 0.55f };
    const float quantize_offset_host[] = { 0.25f, 0.10f, 0.50f };
    sparse_ops::feature_affine_quantize_config quantize_config;
    quantize_config.bits = 2u;
    quantize_config.scale_floor = 1.0e-3f;
    quantize_config.reconstruction_weight = 1.0f;
    quantize_config.range_weight = 0.05f;
    quantize_config.min_dynamic_range = 1.25f;

    auto quantize_log_scale = memory_cuda::SharedDeviceBuffer<float>::allocate(3, ctx.device);
    auto quantize_offset = memory_cuda::SharedDeviceBuffer<float>::allocate(3, ctx.device);
    auto quantize_reconstruction = memory_cuda::SharedDeviceBuffer<float>::allocate(1, ctx.device);
    auto quantize_range = memory_cuda::SharedDeviceBuffer<float>::allocate(1, ctx.device);
    auto quantize_grad_log = memory_cuda::SharedDeviceBuffer<float>::allocate(3, ctx.device);
    auto quantize_grad_offset = memory_cuda::SharedDeviceBuffer<float>::allocate(3, ctx.device);
    upload_and_wait(quantize_log_scale, quantize_log_scale_host, 3, transfer_stream);
    upload_and_wait(quantize_offset, quantize_offset_host, 3, transfer_stream);

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

    sparse_ops::base::csr_feature_affine_quantize_fwd_bwd_f16_f32(
        ctx,
        major_ptr.data(),
        minor_idx.data(),
        values.data(),
        3u,
        3u,
        quantize_log_scale.data(),
        quantize_offset.data(),
        quantize_config,
        quantize_reconstruction.data(),
        quantize_range.data(),
        quantize_grad_log.data(),
        quantize_grad_offset.data());

    float quantize_reconstruction_host = 0.0f;
    float quantize_range_host = 0.0f;
    float quantize_grad_log_host[3] = {};
    float quantize_grad_offset_host[3] = {};
    download_and_wait(quantize_reconstruction, &quantize_reconstruction_host, 1, transfer_stream);
    download_and_wait(quantize_range, &quantize_range_host, 1, transfer_stream);
    download_and_wait(quantize_grad_log, quantize_grad_log_host, 3, transfer_stream);
    download_and_wait(quantize_grad_offset, quantize_grad_offset_host, 3, transfer_stream);
    require(close_value(quantize_reconstruction_host, quantize_ref_reconstruction, 1.0e-4f), "csr quantize reconstruction loss mismatch");
    require(close_value(quantize_range_host, quantize_ref_range, 1.0e-4f), "csr quantize range loss mismatch");
    for (int i = 0; i < 3; ++i) {
        require(close_value(quantize_grad_log_host[i], quantize_ref_grad_log[i], 2.0e-4f), "csr quantize grad log mismatch");
        require(close_value(quantize_grad_offset_host[i], quantize_ref_grad_offset[i], 2.0e-4f), "csr quantize grad offset mismatch");
    }

    sparse_ops::base::blocked_ell_feature_affine_quantize_fwd_bwd_f16_f32(
        ctx,
        blocked_idx.data(),
        blocked_values.data(),
        blocked_host.rows,
        blocked_host.cols,
        blocked_host.block_size,
        blocked_host.ell_cols,
        quantize_log_scale.data(),
        quantize_offset.data(),
        quantize_config,
        quantize_reconstruction.data(),
        quantize_range.data(),
        quantize_grad_log.data(),
        quantize_grad_offset.data());
    download_and_wait(quantize_reconstruction, &quantize_reconstruction_host, 1, transfer_stream);
    download_and_wait(quantize_range, &quantize_range_host, 1, transfer_stream);
    download_and_wait(quantize_grad_log, quantize_grad_log_host, 3, transfer_stream);
    download_and_wait(quantize_grad_offset, quantize_grad_offset_host, 3, transfer_stream);
    require(close_value(quantize_reconstruction_host, quantize_ref_reconstruction, 1.0e-4f), "blocked ell quantize reconstruction loss mismatch");
    require(close_value(quantize_range_host, quantize_ref_range, 1.0e-4f), "blocked ell quantize range loss mismatch");
    for (int i = 0; i < 3; ++i) {
        require(close_value(quantize_grad_log_host[i], quantize_ref_grad_log[i], 2.0e-4f), "blocked ell quantize grad log mismatch");
        require(close_value(quantize_grad_offset_host[i], quantize_ref_grad_offset[i], 2.0e-4f), "blocked ell quantize grad offset mismatch");
    }

    runtime::clear(&blocked_cache);

    auto spmm_grad_values = memory_cuda::SharedDeviceBuffer<float>::allocate(5, ctx.device);
    sparse_ops::base::csr_spmm_bwd_values_f16_f32(
        ctx,
        major_ptr.data(),
        minor_idx.data(),
        grad_out_spmm.data(),
        rhs.data(),
        3,
        2,
        2,
        spmm_grad_values.data());
    float spmm_grad_values_host[5] = {};
    download_and_wait(spmm_grad_values, spmm_grad_values_host, 5, transfer_stream);
    require(close_value(spmm_grad_values_host[0], 5.0f), "spmm grad value 0 mismatch");
    require(close_value(spmm_grad_values_host[1], 500.0f), "spmm grad value 1 mismatch");
    require(close_value(spmm_grad_values_host[2], 110.0f), "spmm grad value 2 mismatch");
    require(close_value(spmm_grad_values_host[3], 1100.0f), "spmm grad value 3 mismatch");
    require(close_value(spmm_grad_values_host[4], 17.0f), "spmm grad value 4 mismatch");

    auto spmm_grad_rhs = memory_cuda::SharedDeviceBuffer<float>::allocate(6, ctx.device);
    runtime::cuda_require(matrix_ops::csr_spmm_transpose(
        csr, grad_out_spmm.data(), 2, 2, spmm_grad_rhs.data(), 2, ctx.stream), "csr_spmm_transpose");
    float spmm_grad_rhs_host[6] = {};
    download_and_wait(spmm_grad_rhs, spmm_grad_rhs_host, 6, transfer_stream);
    require(close_value(spmm_grad_rhs_host[0], 26.0f), "spmm grad rhs 0 mismatch");
    require(close_value(spmm_grad_rhs_host[1], 32.0f), "spmm grad rhs 1 mismatch");
    require(close_value(spmm_grad_rhs_host[2], 9.0f), "spmm grad rhs 2 mismatch");
    require(close_value(spmm_grad_rhs_host[3], 12.0f), "spmm grad rhs 3 mismatch");
    require(close_value(spmm_grad_rhs_host[4], 14.0f), "spmm grad rhs 4 mismatch");
    require(close_value(spmm_grad_rhs_host[5], 20.0f), "spmm grad rhs 5 mismatch");

    if (device_count >= 2) {
        runtime::fleet_context fleet;
        runtime::init(&fleet);
        runtime::discover_fleet(&fleet, true, cudaStreamNonBlocking, true);

        unsigned int pair_slots[2] = {};
        require(runtime::default_mode_pair_slots(fleet, 0u, pair_slots, 2u) == 2u, "pair slots unavailable");
        require(runtime::fleet_slot_available(fleet, pair_slots[0]), "pair leader slot 0 unavailable");
        require(runtime::fleet_slot_available(fleet, pair_slots[1]), "pair peer slot unavailable");

        const int pair_device0 = runtime::fleet_device_id(fleet, pair_slots[0]);
        const int pair_device1 = runtime::fleet_device_id(fleet, pair_slots[1]);

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

        matrix_ops::Csr<const __half> csr[] = {
            {{2, 3, 4, base::CompressedAxis::row, pair_major0.data(), pair_minor0.data()}, pair_values0.data()},
            {{1, 3, 1, base::CompressedAxis::row, pair_major1.data(), pair_minor1.data()}, pair_values1.data()},
        };
        const float *pair_vector_ptrs[] = { pair_vector0.data(), pair_vector1.data() };
        float *pair_out_ptrs[] = { pair_out0.data(), pair_out1.data() };

        sparse_ops::dist::launch_csr_spmv_fwd_f16_f32(
            &fleet,
            pair_slots,
            2,
            csr,
            pair_vector_ptrs,
            pair_out_ptrs);
        runtime::synchronize_slots(fleet, pair_slots, 2);

        float pair_out0_host[2] = {};
        float pair_out1_host[1] = {};
        download_on_device(pair_device0, pair_out0, pair_out0_host, 2);
        download_on_device(pair_device1, pair_out1, pair_out1_host, 1);
        require(close_value(pair_out0_host[0], 201.0f), "pair-local row-shard output 0 mismatch");
        require(close_value(pair_out0_host[1], 430.0f), "pair-local row-shard output 1 mismatch");
        require(close_value(pair_out1_host[0], 5.0f), "pair-local row-shard output 2 mismatch");

        {
            const float subset_partial0_host[] = { 1.0f, 2.0f };
            const float subset_partial1_host[] = { 3.0f, 4.0f };
            auto subset_partial0 = allocate_on_device<float>(pair_device0, 2);
            auto subset_partial1 = allocate_on_device<float>(pair_device1, 2);
            auto subset_sum0 = allocate_on_device<float>(pair_device0, 2);
            upload_on_device(pair_device0, &subset_partial0, subset_partial0_host, 2);
            upload_on_device(pair_device1, &subset_partial1, subset_partial1_host, 2);
            const float *subset_partial_ptrs[] = { subset_partial0.data(), subset_partial1.data() };
            sparse_ops::dist::reduce_sum_to_leader_f32(
                &fleet,
                pair_slots,
                2,
                subset_partial_ptrs,
                2,
                subset_sum0.data());
            const unsigned int pair_leader_slot[] = { pair_slots[0] };
            runtime::synchronize_slots(fleet, pair_leader_slot, 1);
            float subset_sum_host[2] = {};
            download_on_device(pair_device0, subset_sum0, subset_sum_host, 2);
            require(close_value(subset_sum_host[0], 4.0f), "2-gpu subset leader sum row 0 mismatch");
            require(close_value(subset_sum_host[1], 6.0f), "2-gpu subset leader sum row 1 mismatch");
        }

        if (device_count >= 4) {
            unsigned int fleet_slots[4] = {};
            require(runtime::default_mode_fleet_slots(fleet, fleet_slots, 4) == 4u, "default fleet slots unavailable");
            for (unsigned int i = 0; i < 4; ++i) {
                require(runtime::fleet_slot_available(fleet, fleet_slots[i]), "fleet slot unavailable");
            }

            const int fleet_device0 = runtime::fleet_device_id(fleet, fleet_slots[0]);
            const int fleet_device1 = runtime::fleet_device_id(fleet, fleet_slots[1]);
            const int fleet_device2 = runtime::fleet_device_id(fleet, fleet_slots[2]);
            const int fleet_device3 = runtime::fleet_device_id(fleet, fleet_slots[3]);

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

            matrix_ops::Csr<const __half> csr[] = {
                {{2, 1, 1, base::CompressedAxis::row, feature_major0.data(), feature_minor0.data()}, feature_values0.data()},
                {{2, 1, 1, base::CompressedAxis::row, feature_major1.data(), feature_minor1.data()}, feature_values1.data()},
                {{2, 1, 1, base::CompressedAxis::row, feature_major2.data(), feature_minor2.data()}, feature_values2.data()},
                {{2, 1, 1, base::CompressedAxis::row, feature_major3.data(), feature_minor3.data()}, feature_values3.data()},
            };
            const float *feature_vector_ptrs[] = {
                feature_vector0.data(),
                feature_vector1.data(),
                feature_vector2.data(),
                feature_vector3.data()
            };
            float *feature_out_ptrs[] = {
                feature_out0.data(),
                feature_out1.data(),
                feature_out2.data(),
                feature_out3.data()
            };

            sparse_ops::dist::launch_csr_spmv_fwd_f16_f32(
                &fleet,
                fleet_slots,
                4,
                csr,
                feature_vector_ptrs,
                feature_out_ptrs);
            sparse_ops::dist::reduce_sum_to_leader_f32(
                &fleet,
                fleet_slots,
                4,
                const_cast<const float *const *>(feature_out_ptrs),
                2,
                feature_sum0.data());

            const unsigned int leader_slot_only[] = { fleet_slots[0] };
            runtime::synchronize_slots(fleet, leader_slot_only, 1);

            float feature_sum_host[2] = {};
            download_on_device(fleet_device0, feature_sum0, feature_sum_host, 2);
            require(close_value(feature_sum_host[0], 70.0f), "4-gpu feature-shard leader sum row 0 mismatch");
            require(close_value(feature_sum_host[1], 220.0f), "4-gpu feature-shard leader sum row 1 mismatch");
        }

        runtime::clear(&fleet);
    }

    runtime::clear(&scratch);
    runtime::clear(&ctx);
    return 0;
}
