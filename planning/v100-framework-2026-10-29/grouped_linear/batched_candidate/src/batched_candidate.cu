#include "batched_candidate.hh"

#include <algorithm>
#include <limits>
#include <string>

namespace cellerator::planning::grouped_linear {
namespace {
using cellerator::modeling::cuda::DeviceGuard;
using cellerator::modeling::cuda::DeviceView;
using cellerator::modeling::cuda::DType;
using cellerator::modeling::cuda::Stream;
using cellerator::modeling::cuda::check;
using cellerator::modeling::cuda::checked_bytes;
using cellerator::modeling::cuda::item_size;
using cellerator::modeling::cuda::require_view;

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
}

std::size_t matrix_bytes(std::int64_t rows, std::int64_t columns, DType dtype) {
    if (rows < 0 || columns < 0)
        throw std::invalid_argument("negative grouped-linear candidate shape");
    return checked_bytes(checked_bytes(static_cast<std::size_t>(rows),
                                       static_cast<std::size_t>(columns)),
                         item_size(dtype));
}

bool overlaps(const void* a, std::size_t a_bytes,
              const void* b, std::size_t b_bytes) noexcept {
    if (!a_bytes || !b_bytes) return false;
    const auto ap = reinterpret_cast<std::uintptr_t>(a);
    const auto bp = reinterpret_cast<std::uintptr_t>(b);
    return ap <= bp ? bp - ap < a_bytes : ap - bp < b_bytes;
}

void check_disjoint(const void* a, std::size_t a_bytes,
                    const void* b, std::size_t b_bytes,
                    const char* description) {
    if (overlaps(a, a_bytes, b, b_bytes))
        throw std::invalid_argument(std::string(description) + " buffers overlap");
}

void check_alignment(const void* data, DType dtype, const char* description) {
    if (data && reinterpret_cast<std::uintptr_t>(data) % item_size(dtype))
        throw std::invalid_argument(std::string(description) + " is misaligned");
}

__global__ void fill_activation_output_pointers(
    const char* input, char* output, std::size_t input_stride_bytes,
    std::size_t output_stride_bytes, int count,
    const void** input_pointers, void** output_pointers) {
    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert < count) {
        input_pointers[expert] = input + static_cast<std::size_t>(expert) * input_stride_bytes;
        output_pointers[expert] = output + static_cast<std::size_t>(expert) * output_stride_bytes;
    }
}

cudaDataType_t data_type(DType dtype) {
    return dtype == DType::float16 ? CUDA_R_16F : CUDA_R_32F;
}

}  // namespace

bool supports_equal_row_batch(const std::vector<std::int64_t>& row_offsets,
                              std::int64_t minimum_experts) noexcept {
    if (minimum_experts < 1 || row_offsets.size() <= 1 ||
        row_offsets.size() - 1 < static_cast<std::size_t>(minimum_experts) ||
        row_offsets.front() != 0 || !std::is_sorted(row_offsets.begin(), row_offsets.end()))
        return false;
    const auto rows = row_offsets[1] - row_offsets[0];
    if (rows <= 0) return false;
    for (std::size_t e = 1; e + 1 < row_offsets.size(); ++e)
        if (row_offsets[e + 1] - row_offsets[e] != rows) return false;
    return true;
}

EqualRowBatchedForward::EqualRowBatchedForward(
    std::int64_t input_features, std::int64_t output_features,
    std::vector<std::int64_t> row_offsets,
    std::vector<cellerator::modeling::cuda::ConstDeviceView> weights,
    DType dtype, int device)
    : input_features_(input_features), output_features_(output_features),
      row_offsets_(std::move(row_offsets)), weights_(std::move(weights)),
      dtype_(dtype), device_(device) {
    if (input_features_ <= 0 || output_features_ <= 0 || device_ < 0 ||
        row_offsets_.empty() || row_offsets_.size() != weights_.size() + 1 ||
        row_offsets_.front() != 0 ||
        !std::is_sorted(row_offsets_.begin(), row_offsets_.end()))
        throw std::invalid_argument("invalid equal-row grouped-linear candidate metadata");
    if (input_features_ > std::numeric_limits<int>::max() ||
        output_features_ > std::numeric_limits<int>::max())
        throw std::invalid_argument("candidate dimensions exceed cuBLAS integer interface");
    if (weights_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("expert count exceeds cuBLAS batch-count interface");
    total_rows_ = row_offsets_.back();
    (void)matrix_bytes(total_rows_, input_features_, dtype_);
    (void)matrix_bytes(total_rows_, output_features_, dtype_);
    rows_per_expert_ = weights_.empty() ? 0 : row_offsets_[1] - row_offsets_[0];
    if (rows_per_expert_ > std::numeric_limits<int>::max())
        throw std::invalid_argument("expert segment exceeds cuBLAS integer dimensions");
    eligible_ = supports_equal_row_batch(row_offsets_);
    if (!eligible_) return;

    const auto weight_bytes = matrix_bytes(output_features_, input_features_, dtype_);
    std::vector<const void*> host_weights;
    host_weights.reserve(weights_.size());
    for (const auto& weight : weights_) {
        require_view(weight, weight_bytes, device_, "candidate expert weight");
        check_alignment(weight.data, dtype_, "candidate expert weight");
        host_weights.push_back(weight.data);
    }

    const auto count = static_cast<std::size_t>(weights_.size());
    device_weights_ = cellerator::modeling::cuda::DeviceBuffer(
        checked_bytes(count, sizeof(void*)), device_);
    device_inputs_ = cellerator::modeling::cuda::DeviceBuffer(
        checked_bytes(count, sizeof(void*)), device_);
    device_outputs_ = cellerator::modeling::cuda::DeviceBuffer(
        checked_bytes(count, sizeof(void*)), device_);

    DeviceGuard guard(device_);
    check(cudaMemcpy(const_cast<void*>(device_weights_.data()), host_weights.data(),
                     checked_bytes(count, sizeof(void*)), cudaMemcpyHostToDevice),
          "prepare equal-row weight pointers");
    try {
        check_cublas(cublasCreate(&handle_), "create equal-row batched cuBLAS handle");
        const auto mode = dtype_ == DType::float16
            ? CUBLAS_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
        check_cublas(cublasSetMathMode(handle_, mode), "set equal-row batched math mode");
        check_cublas(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST),
                     "set equal-row batched scalar mode");
    } catch (...) {
        if (handle_ != nullptr) (void)cublasDestroy(handle_);
        handle_ = nullptr;
        throw;
    }
}

EqualRowBatchedForward::~EqualRowBatchedForward() noexcept {
    if (handle_ != nullptr) {
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        const bool changed = have_previous && previous != device_ &&
                             cudaSetDevice(device_) == cudaSuccess;
        (void)cublasDestroy(handle_);
        if (changed) (void)cudaSetDevice(previous);
    }
}

void EqualRowBatchedForward::set_stream(const Stream& stream) const {
    if (stream.device() != device_)
        throw std::invalid_argument("candidate/stream device mismatch");
    DeviceGuard guard(device_);
    check_cublas(cublasSetStream(handle_, stream.get()), "bind equal-row batched stream");
}

void EqualRowBatchedForward::forward(
    cellerator::modeling::cuda::ConstDeviceView input, DeviceView output,
    const Stream& stream) const {
    if (!eligible_)
        throw std::logic_error("equal-row batched candidate is not eligible; use fallback");
    const auto input_bytes = matrix_bytes(total_rows_, input_features_, dtype_);
    const auto output_bytes = matrix_bytes(total_rows_, output_features_, dtype_);
    const auto weight_bytes = matrix_bytes(output_features_, input_features_, dtype_);
    require_view(input, input_bytes, device_, "candidate input");
    require_view(output, output_bytes, device_, "candidate output");
    check_alignment(input.data, dtype_, "candidate input");
    check_alignment(output.data, dtype_, "candidate output");
    check_disjoint(input.data, input_bytes, output.data, output_bytes,
                   "candidate input/output");
    for (const auto& weight : weights_)
        check_disjoint(output.data, output_bytes, weight.data, weight_bytes,
                       "candidate output/weight");
    set_stream(stream);
    DeviceGuard guard(device_);

    const auto type_size = item_size(dtype_);
    const auto input_stride = checked_bytes(
        checked_bytes(static_cast<std::size_t>(rows_per_expert_),
                      static_cast<std::size_t>(input_features_)), type_size);
    const auto output_stride = checked_bytes(
        checked_bytes(static_cast<std::size_t>(rows_per_expert_),
                      static_cast<std::size_t>(output_features_)), type_size);
    const auto count = static_cast<int>(weights_.size());
    constexpr int threads = 128;
    const auto blocks = (count + threads - 1) / threads;
    fill_activation_output_pointers<<<blocks, threads, 0, stream.get()>>>(
        static_cast<const char*>(input.data), static_cast<char*>(output.data),
        input_stride, output_stride, count,
        reinterpret_cast<const void**>(device_inputs_.data()),
        reinterpret_cast<void**>(device_outputs_.data()));
    check(cudaPeekAtLastError(), "fill equal-row batched pointers");

    const auto type = data_type(dtype_);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cublas(cublasGemmBatchedEx(
        handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(output_features_), static_cast<int>(rows_per_expert_),
        static_cast<int>(input_features_), &alpha,
        reinterpret_cast<const void* const*>(device_weights_.data()), type,
        static_cast<int>(input_features_),
        reinterpret_cast<const void* const*>(device_inputs_.data()), type,
        static_cast<int>(input_features_), &beta,
        reinterpret_cast<void* const*>(device_outputs_.data()), type,
        static_cast<int>(output_features_), count, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP), "equal-row cublasGemmBatchedEx");
}

}  // namespace cellerator::planning::grouped_linear
