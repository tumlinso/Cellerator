#pragma once

#include <Cellerator/modeling/cuda/runtime.hh>

#include <cublas_v2.h>

#include <cstdint>
#include <vector>

namespace cellerator::modeling::cuda {

enum class GradientMode { overwrite, accumulate };

// Expert-major linear transforms over a ragged row partition. Each expert e
// owns a row-major [output_features, input_features] weight matrix. Activations
// and results are packed row-major by expert, with rows [offsets[e], offsets[e+1]).
// The executor stores only borrowed weight views and host offsets; callers own
// all model parameters, activations, outputs, and gradients.
//
// Calls are asynchronous on the supplied stream. A handle is owned per
// executor, and is rebound to that stream for each call. One executor must not
// be called concurrently from multiple host threads; use one executor per
// concurrent submission stream. FP16 uses FP32 accumulation and enables
// cuBLAS Tensor Op math (cuBLAS chooses supported shapes); FP32 uses FP32 math.
// One constructor-selected DType applies to activations, outputs, and all
// gradients; FP32 accumulation is internal to FP16 GEMMs.
class GroupedLinear final {
public:
    GroupedLinear(std::int64_t input_features,
                  std::int64_t output_features,
                  std::vector<std::int64_t> row_offsets,
                  std::vector<ConstDeviceView> expert_weights,
                  DType dtype,
                  int device);
    ~GroupedLinear() noexcept;
    GroupedLinear(const GroupedLinear&) = delete;
    GroupedLinear& operator=(const GroupedLinear&) = delete;
    GroupedLinear(GroupedLinear&&) = delete;
    GroupedLinear& operator=(GroupedLinear&&) = delete;

    std::int64_t expert_count() const noexcept {
        return static_cast<std::int64_t>(weights_.size());
    }
    std::int64_t row_count() const noexcept { return row_offsets_.back(); }
    std::int64_t input_features() const noexcept { return input_features_; }
    std::int64_t output_features() const noexcept { return output_features_; }
    DType dtype() const noexcept { return dtype_; }
    int device() const noexcept { return device_; }

    // y[e] = x[e] * W[e]^T. Output is fully overwritten. Empty expert
    // segments are skipped and do not evaluate their parameter matrices.
    void forward(ConstDeviceView input, DeviceView output,
                 const Stream& stream) const;

    // Reverse mode for forward():
    //   dX[e] = dY[e] * W[e]
    //   dW[e] = dY[e]^T * X[e]
    // dX and dW each have an explicit write/accumulate policy. Gradient
    // buffers are per expert for dW, in the same row-major shape as W[e].
    void backward(ConstDeviceView input, ConstDeviceView output_gradient,
                  DeviceView input_gradient,
                  const std::vector<DeviceView>& weight_gradients,
                  GradientMode input_gradient_mode,
                  GradientMode weight_gradient_mode,
                  const Stream& stream) const;

private:
    void set_stream(const Stream& stream) const;
    std::int64_t input_features_;
    std::int64_t output_features_;
    std::vector<std::int64_t> row_offsets_;
    std::vector<ConstDeviceView> weights_;
    DType dtype_;
    int device_;
    mutable cublasHandle_t handle_ = nullptr;
};

}  // namespace cellerator::modeling::cuda
