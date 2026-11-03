#pragma once

#include <Cellerator/memory/cuda/resources.hh>
#include <Cellerator/execution/conditional/dispatch.hh>
#include <Cellerator/modeling/cuda/dispatch.hh>
#include <vector>

namespace cellerator::modeling::cuda {

// Immutable device projection. Constructor validates all host indexes and
// synchronizes its upload stream before returning. It preserves supplied
// origin_dispatch order, which defines floating-point summation order.
class DispatchPlan final {
public:
    // Converts host double weights to the CUDA FP32 weight contract. Finite
    // rounding and underflow to zero are accepted; overflow/nonfinite results
    // are rejected. This is not a float64-equivalent numerical projection.
    explicit DispatchPlan(
        const cellerator::execution::conditional::DispatchPlan& host,
        const ::cellerator::memory::cuda::Stream& stream);
    DispatchPlan(std::int64_t origins, std::vector<std::int64_t> selection_origin,
                 std::vector<float> weights, std::vector<std::int64_t> origin_offsets,
                 std::vector<std::int64_t> origin_dispatch,
                 std::vector<std::int64_t> expert_offsets,
                 const ::cellerator::memory::cuda::Stream& stream);
    DispatchPlanView view() const noexcept;
    std::int64_t origin_count() const noexcept { return origins_; }
    std::int64_t selection_count() const noexcept { return selections_; }
    int device() const noexcept { return device_; }
    const std::vector<std::int64_t>& expert_offsets() const noexcept { return expert_offsets_; }
    void gather(::cellerator::memory::cuda::ConstDeviceView input,
                ::cellerator::memory::cuda::DeviceView output,
                std::int64_t features, ::cellerator::memory::cuda::DType dtype,
                const ::cellerator::memory::cuda::Stream& stream) const;
    void combine(::cellerator::memory::cuda::ConstDeviceView input,
                 ::cellerator::memory::cuda::DeviceView output,
                 std::int64_t features, ::cellerator::memory::cuda::DType dtype,
                 const ::cellerator::memory::cuda::Stream& stream) const;
    void gather_backward(::cellerator::memory::cuda::ConstDeviceView gradient,
                         ::cellerator::memory::cuda::DeviceView input_gradient,
                         std::int64_t features,
                         ::cellerator::memory::cuda::DType dtype,
                         const ::cellerator::memory::cuda::Stream& stream) const;
    void combine_backward(
        ::cellerator::memory::cuda::ConstDeviceView gradient,
        ::cellerator::memory::cuda::ConstDeviceView expert_output,
        ::cellerator::memory::cuda::DeviceView expert_gradient,
        ::cellerator::memory::cuda::DeviceView weight_gradient,
        std::int64_t features, ::cellerator::memory::cuda::DType dtype,
        const ::cellerator::memory::cuda::Stream& stream) const;
private:
    std::int64_t origins_ = 0, selections_ = 0;
    int device_ = 0;
    std::vector<std::int64_t> expert_offsets_;
    ::cellerator::memory::cuda::DeviceBuffer selection_origin_, weights_,
        origin_offsets_, origin_dispatch_;
};
} // namespace cellerator::modeling::cuda
