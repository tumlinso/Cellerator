#include <Cellerator/modeling/cuda/grouped_linear.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace cellerator::modeling::cuda {
namespace {

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
}

std::size_t matrix_bytes(std::int64_t rows, std::int64_t columns, DType dtype) {
    if (rows < 0 || columns < 0 ||
        (columns != 0 && static_cast<std::uint64_t>(rows) >
                             std::numeric_limits<std::size_t>::max() /
                                 static_cast<std::uint64_t>(columns)))
        throw std::invalid_argument("grouped linear matrix shape overflows");
    return checked_bytes(static_cast<std::size_t>(rows) *
                             static_cast<std::size_t>(columns),
                         item_size(dtype));
}

bool overlaps(ConstDeviceView a, std::size_t a_bytes,
              ConstDeviceView b, std::size_t b_bytes) noexcept {
    if (!a_bytes || !b_bytes) return false;
    const auto ap = reinterpret_cast<std::uintptr_t>(a.data);
    const auto bp = reinterpret_cast<std::uintptr_t>(b.data);
    return ap <= bp ? bp - ap < a_bytes : ap - bp < b_bytes;
}

void check_disjoint(ConstDeviceView a, std::size_t a_bytes,
                    ConstDeviceView b, std::size_t b_bytes,
                    const char* description) {
    if (overlaps(a, a_bytes, b, b_bytes))
        throw std::invalid_argument(std::string(description) + " buffers overlap");
}

void check_aligned(ConstDeviceView view, DType dtype, const char* description) {
    if (view.data && reinterpret_cast<std::uintptr_t>(view.data) % item_size(dtype))
        throw std::invalid_argument(std::string(description) + " is misaligned");
}

ConstDeviceView as_const(DeviceView view) noexcept {
    return {view.data, view.bytes, view.device};
}

cudaDataType_t data_type(DType dtype) {
    return dtype == DType::float16 ? CUDA_R_16F : CUDA_R_32F;
}

void gemm(cublasHandle_t handle, DType dtype,
          cublasOperation_t trans_a, cublasOperation_t trans_b,
          int m, int n, int k, const void* a, int lda,
          const void* b, int ldb, void* c, int ldc, float beta) {
    const float alpha = 1.0f;
    check_cublas(cublasGemmEx(handle, trans_a, trans_b, m, n, k,
                              &alpha, a, data_type(dtype), lda,
                              b, data_type(dtype), ldb,
                              &beta, c, data_type(dtype), ldc,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                 "grouped linear GEMM");
}

}  // namespace

GroupedLinear::GroupedLinear(std::int64_t input_features,
                             std::int64_t output_features,
                             std::vector<std::int64_t> row_offsets,
                             std::vector<ConstDeviceView> expert_weights,
                             DType dtype, int device)
    : input_features_(input_features), output_features_(output_features),
      row_offsets_(std::move(row_offsets)), weights_(std::move(expert_weights)),
      dtype_(dtype), device_(device) {
    if (input_features_ <= 0 || output_features_ <= 0 || device_ < 0 ||
        row_offsets_.size() != weights_.size() + 1 || row_offsets_.empty() ||
        row_offsets_.front() != 0 ||
        !std::is_sorted(row_offsets_.begin(), row_offsets_.end()))
        throw std::invalid_argument("invalid grouped linear dimensions or row offsets");
    if (input_features_ > std::numeric_limits<int>::max() ||
        output_features_ > std::numeric_limits<int>::max())
        throw std::invalid_argument("cuBLAS dimensions exceed its integer interface");
    (void)matrix_bytes(row_offsets_.back(), input_features_, dtype_);
    (void)matrix_bytes(row_offsets_.back(), output_features_, dtype_);
    for (std::size_t e = 0; e + 1 < row_offsets_.size(); ++e)
        if (row_offsets_[e + 1] - row_offsets_[e] > std::numeric_limits<int>::max())
            throw std::invalid_argument("expert segment exceeds cuBLAS integer dimensions");
    const auto weight_bytes = matrix_bytes(output_features_, input_features_, dtype_);
    for (const auto& weight : weights_) {
        require_view(weight, weight_bytes, device_, "grouped linear expert weight");
        const auto alignment = item_size(dtype_);
        if (reinterpret_cast<std::uintptr_t>(weight.data) % alignment)
            throw std::invalid_argument("grouped linear weight is misaligned");
    }

    DeviceGuard guard(device_);
    check_cublas(cublasCreate(&handle_), "create grouped linear cuBLAS handle");
    const auto mode = dtype_ == DType::float16 ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    const auto status = cublasSetMathMode(handle_, mode);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle_);
        handle_ = nullptr;
        check_cublas(status, "set grouped linear math mode");
    }
}

GroupedLinear::~GroupedLinear() noexcept {
    if (handle_ != nullptr) {
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        const bool changed = have_previous && previous != device_ &&
                             cudaSetDevice(device_) == cudaSuccess;
        (void)cublasDestroy(handle_);
        if (changed) (void)cudaSetDevice(previous);
    }
}

void GroupedLinear::set_stream(const Stream& stream) const {
    if (stream.device() != device_)
        throw std::invalid_argument("grouped linear/stream device mismatch");
    DeviceGuard guard(device_);
    check_cublas(cublasSetStream(handle_, stream.get()), "bind grouped linear stream");
}

void GroupedLinear::forward(ConstDeviceView input, DeviceView output,
                            const Stream& stream) const {
    const auto total_rows = row_offsets_.back();
    const auto input_bytes = matrix_bytes(total_rows, input_features_, dtype_);
    const auto output_bytes = matrix_bytes(total_rows, output_features_, dtype_);
    require_view(input, input_bytes, device_, "grouped linear input");
    require_view(output, output_bytes, device_, "grouped linear output");
    check_aligned(input, dtype_, "grouped linear input");
    check_aligned(output, dtype_, "grouped linear output");
    check_disjoint(input, input_bytes, output, output_bytes, "grouped linear input/output");
    for (const auto& weight : weights_)
        check_disjoint(output, output_bytes, weight, matrix_bytes(output_features_, input_features_, dtype_),
                       "grouped linear output/weight");
    set_stream(stream);
    DeviceGuard guard(device_);
    const auto type_size = item_size(dtype_);
    for (std::size_t e = 0; e < weights_.size(); ++e) {
        const auto rows = row_offsets_[e + 1] - row_offsets_[e];
        if (rows == 0) continue;
        const auto input_offset = static_cast<std::size_t>(row_offsets_[e]) * input_features_ * type_size;
        const auto output_offset = static_cast<std::size_t>(row_offsets_[e]) * output_features_ * type_size;
        // Row-major Y = X W^T maps to column-major Y^T = W X^T.
        gemm(handle_, dtype_, CUBLAS_OP_T, CUBLAS_OP_N,
             static_cast<int>(output_features_), static_cast<int>(rows),
             static_cast<int>(input_features_), weights_[e].data,
             static_cast<int>(input_features_),
             static_cast<const char*>(input.data) + input_offset,
             static_cast<int>(input_features_),
             static_cast<char*>(output.data) + output_offset,
             static_cast<int>(output_features_), 0.0f);
    }
}

void GroupedLinear::backward(ConstDeviceView input,
                             ConstDeviceView output_gradient,
                             DeviceView input_gradient,
                             const std::vector<DeviceView>& weight_gradients,
                             GradientMode input_gradient_mode,
                             GradientMode weight_gradient_mode,
                             const Stream& stream) const {
    if ((input_gradient_mode != GradientMode::overwrite &&
         input_gradient_mode != GradientMode::accumulate) ||
        (weight_gradient_mode != GradientMode::overwrite &&
         weight_gradient_mode != GradientMode::accumulate))
        throw std::invalid_argument("invalid grouped linear gradient mode");
    if (weight_gradients.size() != weights_.size())
        throw std::invalid_argument("one weight-gradient view is required per expert");
    const auto total_rows = row_offsets_.back();
    const auto input_bytes = matrix_bytes(total_rows, input_features_, dtype_);
    const auto output_bytes = matrix_bytes(total_rows, output_features_, dtype_);
    const auto weight_bytes = matrix_bytes(output_features_, input_features_, dtype_);
    require_view(input, input_bytes, device_, "grouped linear backward input");
    require_view(output_gradient, output_bytes, device_, "grouped linear output gradient");
    require_view(input_gradient, input_bytes, device_, "grouped linear input gradient");
    check_aligned(input, dtype_, "grouped linear backward input");
    check_aligned(output_gradient, dtype_, "grouped linear output gradient");
    check_aligned(input_gradient, dtype_, "grouped linear input gradient");
    check_disjoint(input, input_bytes, input_gradient, input_bytes, "input/input-gradient");
    check_disjoint(output_gradient, output_bytes, input_gradient, input_bytes,
                   "output-gradient/input-gradient");
    for (const auto& weight : weights_)
        check_disjoint(input_gradient, input_bytes, weight, weight_bytes,
                       "input-gradient/weight");
    for (const auto& gradient : weight_gradients) {
        require_view(gradient, weight_bytes, device_, "grouped linear weight gradient");
        check_aligned(gradient, dtype_, "grouped linear weight gradient");
        check_disjoint(input, input_bytes, gradient, weight_bytes, "input/weight-gradient");
        check_disjoint(output_gradient, output_bytes, gradient, weight_bytes,
                       "output-gradient/weight-gradient");
        check_disjoint(input_gradient, input_bytes, gradient, weight_bytes,
                       "input-gradient/weight-gradient");
        for (const auto& weight : weights_)
            check_disjoint(gradient, weight_bytes, weight, weight_bytes,
                           "weight/weight-gradient");
    }
    for (std::size_t i = 0; i < weight_gradients.size(); ++i)
        for (std::size_t j = i + 1; j < weight_gradients.size(); ++j)
            check_disjoint(as_const(weight_gradients[i]), weight_bytes,
                           as_const(weight_gradients[j]), weight_bytes,
                           "distinct weight-gradient");
    set_stream(stream);
    DeviceGuard guard(device_);
    const auto type_size = item_size(dtype_);
    const float beta_x = input_gradient_mode == GradientMode::accumulate ? 1.0f : 0.0f;
    const float beta_w = weight_gradient_mode == GradientMode::accumulate ? 1.0f : 0.0f;
    for (std::size_t e = 0; e < weights_.size(); ++e) {
        const auto rows = row_offsets_[e + 1] - row_offsets_[e];
        if (rows == 0) {
            if (weight_gradient_mode == GradientMode::overwrite) {
                const auto error = cudaMemsetAsync(weight_gradients[e].data, 0, weight_bytes, stream.get());
                check(error, "zero empty-expert weight gradient");
            }
            continue;
        }
        const auto input_offset = static_cast<std::size_t>(row_offsets_[e]) * input_features_ * type_size;
        const auto output_offset = static_cast<std::size_t>(row_offsets_[e]) * output_features_ * type_size;
        // Row-major dX = dY W maps to column-major dX^T = W^T dY^T.
        gemm(handle_, dtype_, CUBLAS_OP_N, CUBLAS_OP_N,
             static_cast<int>(input_features_), static_cast<int>(rows),
             static_cast<int>(output_features_), weights_[e].data,
             static_cast<int>(input_features_),
             static_cast<const char*>(output_gradient.data) + output_offset,
             static_cast<int>(output_features_),
             static_cast<char*>(input_gradient.data) + input_offset,
             static_cast<int>(input_features_), beta_x);
        // Row-major dW = dY^T X maps to column-major dW^T = X^T dY.
        gemm(handle_, dtype_, CUBLAS_OP_N, CUBLAS_OP_T,
             static_cast<int>(input_features_), static_cast<int>(output_features_),
             static_cast<int>(rows),
             static_cast<const char*>(input.data) + input_offset,
             static_cast<int>(input_features_),
             static_cast<const char*>(output_gradient.data) + output_offset,
             static_cast<int>(output_features_), weight_gradients[e].data,
             static_cast<int>(input_features_), beta_w);
    }
}

}  // namespace cellerator::modeling::cuda
