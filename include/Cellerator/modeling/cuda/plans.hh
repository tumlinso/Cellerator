#pragma once

#include <Cellerator/modeling/cuda/runtime.hh>
#include <Cellerator/execution/conditional/dispatch.hh>
#include <Cellerator/modeling/cuda/dispatch.hh>
#include <Cellerator/modeling/cuda/quantization.hh>
#include <memory>
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
    explicit DispatchPlan(const cellerator::execution::conditional::DispatchPlan& host, const Stream& stream);
    DispatchPlan(std::int64_t origins, std::vector<std::int64_t> selection_origin,
                 std::vector<float> weights, std::vector<std::int64_t> origin_offsets,
                 std::vector<std::int64_t> origin_dispatch,
                 std::vector<std::int64_t> expert_offsets, const Stream& stream);
    DispatchPlanView view() const noexcept;
    std::int64_t origin_count() const noexcept { return origins_; }
    std::int64_t selection_count() const noexcept { return selections_; }
    int device() const noexcept { return device_; }
    const std::vector<std::int64_t>& expert_offsets() const noexcept { return expert_offsets_; }
    void gather(ConstDeviceView input, DeviceView output, std::int64_t features,
                DType dtype, const Stream& stream) const;
    void combine(ConstDeviceView input, DeviceView output, std::int64_t features,
                 DType dtype, const Stream& stream) const;
    void gather_backward(ConstDeviceView gradient, DeviceView input_gradient,
                         std::int64_t features, DType dtype, const Stream& stream) const;
    void combine_backward(ConstDeviceView gradient, ConstDeviceView expert_output,
                          DeviceView expert_gradient, DeviceView weight_gradient,
                          std::int64_t features, DType dtype, const Stream& stream) const;
private:
    std::int64_t origins_ = 0, selections_ = 0;
    int device_ = 0;
    std::vector<std::int64_t> expert_offsets_;
    DeviceBuffer selection_origin_, weights_, origin_offsets_, origin_dispatch_;
};

// Levels are finite, strictly increasing per feature, with 1..2^bits entries.
// profile identity belongs to the caller; feature_map explicitly maps physical
// columns to profile columns. Precision conversion to float is explicit.
class QuantizationProfile final {
public:
    QuantizationProfile(std::vector<std::uint32_t> offsets,
                        std::vector<float> levels,
                        std::vector<std::uint32_t> feature_map,
                        std::uint32_t bits, const Stream& stream);
    QuantizationProfileView view() const noexcept;
    int device() const noexcept { return device_; }
    std::uint32_t feature_count() const noexcept { return features_; }
    std::uint32_t bits() const noexcept { return bits_; }
private:
    int device_ = 0;
    std::uint32_t profiles_ = 0, features_ = 0, levels_count_ = 0, bits_ = 0;
    DeviceBuffer offsets_, levels_, feature_map_;
};

// Mutable encoded state owns packed codes, exact FP32 overrides and validation
// status. It retains its immutable profile. All launches use the caller stream;
// cross-stream use requires explicit events or synchronization.
class QuantizedState final {
public:
    QuantizedState(std::shared_ptr<const QuantizationProfile> profile,
                   std::uint32_t rows, std::uint32_t override_capacity = 0);
    void encode(ConstDeviceView input, DType dtype, ConstDeviceView exact_mask,
                const Stream& stream);
    void decode(DeviceView output, DType dtype, const Stream& stream) const;
    std::uint32_t status(const Stream& stream) const; // synchronizing check
    std::uint32_t override_count(const Stream& stream) const;
    std::uint32_t rows() const noexcept { return rows_; }
    std::uint32_t features() const noexcept { return profile_->feature_count(); }
    std::size_t packed_bytes() const noexcept { return packed_.bytes(); }
    std::uint32_t override_capacity() const noexcept { return capacity_; }
    int device() const noexcept { return profile_->device(); }
    ConstDeviceView packed_view() const noexcept { return packed_.view(); }
private:
    OverrideView overrides() noexcept;
    ConstOverrideView overrides() const noexcept;
    std::shared_ptr<const QuantizationProfile> profile_;
    std::uint32_t rows_ = 0, capacity_ = 0;
    DeviceBuffer packed_, indices_, values_, count_;
    mutable DeviceBuffer status_;
    bool encoded_ = false;
};
} // namespace cellerator::modeling::cuda
