#pragma once

#include <Cellerator/modeling/cuda/runtime.hh>

#include <cublas_v2.h>

#include <cstdint>
#include <vector>

namespace cellerator::planning::grouped_linear {

// The initial crossover is deliberately conservative and evidence-bounded.
constexpr std::int64_t kMinimumBatchedExperts = 8;

bool supports_equal_row_batch(const std::vector<std::int64_t>& row_offsets,
                              std::int64_t minimum_experts = kMinimumBatchedExperts) noexcept;

// Isolated forward-only probe. It borrows weight allocations and owns only its
// handle and device pointer arrays. Use one executor per concurrent stream, as
// with the production GroupedLinear contract.
class EqualRowBatchedForward final {
public:
    EqualRowBatchedForward(std::int64_t input_features,
                           std::int64_t output_features,
                           std::vector<std::int64_t> row_offsets,
                           std::vector<cellerator::modeling::cuda::ConstDeviceView> weights,
                           cellerator::modeling::cuda::DType dtype,
                           int device);
    ~EqualRowBatchedForward() noexcept;

    EqualRowBatchedForward(const EqualRowBatchedForward&) = delete;
    EqualRowBatchedForward& operator=(const EqualRowBatchedForward&) = delete;
    EqualRowBatchedForward(EqualRowBatchedForward&&) = delete;
    EqualRowBatchedForward& operator=(EqualRowBatchedForward&&) = delete;

    bool eligible() const noexcept { return eligible_; }
    void forward(cellerator::modeling::cuda::ConstDeviceView input,
                 cellerator::modeling::cuda::DeviceView output,
                 const cellerator::modeling::cuda::Stream& stream) const;

private:
    void set_stream(const cellerator::modeling::cuda::Stream& stream) const;

    std::int64_t input_features_ = 0;
    std::int64_t output_features_ = 0;
    std::int64_t rows_per_expert_ = 0;
    std::int64_t total_rows_ = 0;
    std::vector<std::int64_t> row_offsets_;
    std::vector<cellerator::modeling::cuda::ConstDeviceView> weights_;
    cellerator::modeling::cuda::DType dtype_;
    int device_ = 0;
    bool eligible_ = false;
    cellerator::modeling::cuda::DeviceBuffer device_weights_;
    mutable cellerator::modeling::cuda::DeviceBuffer device_inputs_;
    mutable cellerator::modeling::cuda::DeviceBuffer device_outputs_;
    mutable cublasHandle_t handle_ = nullptr;
};

}  // namespace cellerator::planning::grouped_linear
