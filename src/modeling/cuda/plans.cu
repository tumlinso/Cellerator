#include <Cellerator/modeling/cuda/plans.hh>
#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::modeling::cuda {
namespace {
template<class T> DeviceBuffer upload(const std::vector<T>& values, const Stream& stream) {
    DeviceBuffer result(checked_bytes(values.size(), sizeof(T)), stream.device());
    result.upload_async(values.data(), result.bytes(), stream);
    // Host vectors may be destroyed immediately after this helper returns.
    stream.synchronize();
    return result;
}
std::size_t elements(std::int64_t rows, std::int64_t features) {
    if (rows < 0 || features < 0) throw std::invalid_argument("negative shape");
    return checked_bytes(static_cast<std::size_t>(rows), static_cast<std::size_t>(features));
}
void check_stream(int device, const Stream& stream) {
    if (device != stream.device()) throw std::invalid_argument("plan/stream device mismatch");
}
void aligned(ConstDeviceView view, std::size_t alignment) {
    if (reinterpret_cast<std::uintptr_t>(view.data) % alignment)
        throw std::invalid_argument("device view is not aligned for its numeric type");
}
void disjoint(ConstDeviceView input, DeviceView output, std::size_t input_bytes, std::size_t output_bytes) {
    if (!input_bytes || !output_bytes) return;
    const auto a = reinterpret_cast<std::uintptr_t>(input.data), b = reinterpret_cast<std::uintptr_t>(output.data);
    if ((a <= b && b - a < input_bytes) || (b < a && a - b < output_bytes))
        throw std::invalid_argument("input and output storage overlap");
}
}
DispatchPlan::DispatchPlan(const cellerator::execution::conditional::DispatchPlan& host, const Stream& stream)
    : DispatchPlan(static_cast<std::int64_t>(host.origin_count()),
                   std::vector<std::int64_t>(host.dispatch_origin_slots().begin(), host.dispatch_origin_slots().end()),
                   std::vector<float>(host.dispatch_weights().begin(), host.dispatch_weights().end()),
                   std::vector<std::int64_t>(host.origin_offsets().begin(), host.origin_offsets().end()),
                   std::vector<std::int64_t>(host.origin_dispatch_indices().begin(), host.origin_dispatch_indices().end()),
                   std::vector<std::int64_t>(host.expert_offsets().begin(), host.expert_offsets().end()), stream) {}
DispatchPlan::DispatchPlan(std::int64_t origins, std::vector<std::int64_t> selection_origin,
                          std::vector<float> weights, std::vector<std::int64_t> origin_offsets,
                          std::vector<std::int64_t> origin_dispatch,
                          std::vector<std::int64_t> expert_offsets, const Stream& stream)
    : origins_(origins), device_(stream.device()), expert_offsets_(std::move(expert_offsets)) {
    if (origins < 0 || selection_origin.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::invalid_argument("invalid dispatch shape");
    selections_ = static_cast<std::int64_t>(selection_origin.size());
    if (weights.empty()) weights.assign(selection_origin.size(), 1.0f);
    if (weights.size() != selection_origin.size() || origin_dispatch.size() != selection_origin.size()
        || origin_offsets.size() != static_cast<std::size_t>(origins) + 1
        || origin_offsets.front() != 0 || origin_offsets.back() != selections_
        || !std::is_sorted(origin_offsets.begin(), origin_offsets.end()))
        throw std::invalid_argument("invalid dispatch array dimensions or offsets");
    if (expert_offsets_.empty()) expert_offsets_ = {0, selections_};
    if (expert_offsets_.front() != 0 || expert_offsets_.back() != selections_
        || !std::is_sorted(expert_offsets_.begin(), expert_offsets_.end()))
        throw std::invalid_argument("invalid expert offsets");
    std::vector<bool> seen(selection_origin.size(), false);
    for (std::int64_t r = 0; r < origins_; ++r) {
        for (auto j = origin_offsets[r]; j < origin_offsets[r + 1]; ++j) {
            auto d = origin_dispatch[j];
            if (d < 0 || d >= selections_ || seen[d] || selection_origin[d] != r)
                throw std::invalid_argument("reverse dispatch must be a matching permutation");
            seen[d] = true;
        }
    }
    for (auto weight : weights) if (!std::isfinite(weight)) throw std::invalid_argument("dispatch weights must be finite");
    selection_origin_ = upload(selection_origin, stream);
    weights_ = upload(weights, stream);
    origin_offsets_ = upload(origin_offsets, stream);
    origin_dispatch_ = upload(origin_dispatch, stream);
}
DispatchPlanView DispatchPlan::view() const noexcept {
    return {static_cast<const std::int64_t*>(selection_origin_.data()),
            static_cast<const std::int64_t*>(origin_offsets_.data()),
            static_cast<const std::int64_t*>(origin_dispatch_.data()),
            static_cast<const float*>(weights_.data()), selections_, origins_};
}
void DispatchPlan::gather(ConstDeviceView input, DeviceView output, std::int64_t features, DType dtype, const Stream& stream) const {
    check_stream(device_, stream); const auto ni = elements(origins_, features), no = elements(selections_, features), unit = item_size(dtype);
    aligned(input, unit); aligned(output, unit);
    require_view(input, checked_bytes(ni, unit), device_, "gather input"); require_view(output, checked_bytes(no, unit), device_, "gather output");
    disjoint(input, output, checked_bytes(ni, unit), checked_bytes(no, unit)); DeviceGuard guard(device_);
    check(dtype == DType::float16 ? gather_fp16(static_cast<const __half*>(input.data), ni, static_cast<__half*>(output.data), no, view(), features, stream.get())
                                : gather_fp32(static_cast<const float*>(input.data), ni, static_cast<float*>(output.data), no, view(), features, stream.get()), "dispatch gather");
}
void DispatchPlan::combine(ConstDeviceView input, DeviceView output, std::int64_t features, DType dtype, const Stream& stream) const {
    check_stream(device_, stream); const auto ni = elements(selections_, features), no = elements(origins_, features), unit = item_size(dtype);
    aligned(input, unit); aligned(output, unit);
    require_view(input, checked_bytes(ni, unit), device_, "combine input"); require_view(output, checked_bytes(no, unit), device_, "combine output");
    disjoint(input, output, checked_bytes(ni, unit), checked_bytes(no, unit)); DeviceGuard guard(device_);
    check(dtype == DType::float16 ? combine_fp16(static_cast<const __half*>(input.data), ni, static_cast<__half*>(output.data), no, view(), features, stream.get())
                                : combine_fp32(static_cast<const float*>(input.data), ni, static_cast<float*>(output.data), no, view(), features, stream.get()), "dispatch combine");
}
void DispatchPlan::gather_backward(ConstDeviceView gradient, DeviceView input_gradient, std::int64_t features, DType dtype, const Stream& stream) const {
    check_stream(device_, stream); const auto ni = elements(selections_, features), no = elements(origins_, features), unit = item_size(dtype);
    aligned(gradient, unit); aligned(input_gradient, unit);
    require_view(gradient, checked_bytes(ni, unit), device_, "gather gradient"); require_view(input_gradient, checked_bytes(no, unit), device_, "input gradient");
    disjoint(gradient, input_gradient, checked_bytes(ni, unit), checked_bytes(no, unit)); DeviceGuard guard(device_);
    check(dtype == DType::float16 ? gather_backward_fp16(static_cast<const __half*>(gradient.data), ni, static_cast<__half*>(input_gradient.data), no, view(), features, stream.get())
                                : gather_backward_fp32(static_cast<const float*>(gradient.data), ni, static_cast<float*>(input_gradient.data), no, view(), features, stream.get()), "dispatch gather backward");
}
void DispatchPlan::combine_backward(ConstDeviceView gradient, ConstDeviceView expert_output, DeviceView expert_gradient, DeviceView weight_gradient, std::int64_t features, DType dtype, const Stream& stream) const {
    check_stream(device_, stream); const auto ng = elements(origins_, features), ne = elements(selections_, features), unit = item_size(dtype);
    aligned(gradient, unit); aligned(expert_output, unit); aligned(expert_gradient, unit); aligned(weight_gradient, sizeof(float));
    require_view(gradient, checked_bytes(ng, unit), device_, "combine gradient"); require_view(expert_output, checked_bytes(ne, unit), device_, "expert output");
    require_view(expert_gradient, checked_bytes(ne, unit), device_, "expert gradient"); require_view(weight_gradient, checked_bytes(selections_, sizeof(float)), device_, "weight gradient");
    disjoint(gradient, expert_gradient, checked_bytes(ng, unit), checked_bytes(ne, unit));
    disjoint(expert_output, expert_gradient, checked_bytes(ne, unit), checked_bytes(ne, unit));
    disjoint(expert_output, weight_gradient, checked_bytes(ne, unit), checked_bytes(selections_, sizeof(float)));
    disjoint(gradient, weight_gradient, checked_bytes(ng, unit), checked_bytes(selections_, sizeof(float)));
    disjoint(expert_gradient, weight_gradient, checked_bytes(ne, unit), checked_bytes(selections_, sizeof(float)));
    DeviceGuard guard(device_);
    check(dtype == DType::float16 ? combine_backward_fp16(static_cast<const __half*>(gradient.data), ng, static_cast<const __half*>(expert_output.data), ne, static_cast<__half*>(expert_gradient.data), ne, static_cast<float*>(weight_gradient.data), selections_, view(), features, stream.get())
                                : combine_backward_fp32(static_cast<const float*>(gradient.data), ng, static_cast<const float*>(expert_output.data), ne, static_cast<float*>(expert_gradient.data), ne, static_cast<float*>(weight_gradient.data), selections_, view(), features, stream.get()), "dispatch combine backward");
}
QuantizationProfile::QuantizationProfile(std::vector<std::uint32_t> offsets, std::vector<float> levels,
                                       std::vector<std::uint32_t> feature_map, std::uint32_t bits, const Stream& stream)
    : device_(stream.device()), bits_(bits) {
    if (bits != 1 && bits != 2 && bits != 4 && bits != 8) throw std::invalid_argument("code bits must be 1, 2, 4 or 8");
    if (offsets.size() < 2 || feature_map.empty() || offsets.front() != 0 || offsets.back() != levels.size()
        || offsets.size() - 1 > std::numeric_limits<std::uint32_t>::max()
        || feature_map.size() > std::numeric_limits<std::uint32_t>::max()
        || levels.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("invalid codebook dimensions");
    profiles_ = static_cast<std::uint32_t>(offsets.size() - 1);
    features_ = static_cast<std::uint32_t>(feature_map.size()); levels_count_ = static_cast<std::uint32_t>(levels.size());
    for (std::size_t p = 0; p < profiles_; ++p) {
        if (offsets[p + 1] <= offsets[p] || offsets[p + 1] > levels.size() || offsets[p + 1] - offsets[p] > (1u << bits))
            throw std::invalid_argument("codebook has invalid number of levels");
        for (auto j = offsets[p]; j < offsets[p + 1]; ++j)
            if (!std::isfinite(levels[j]) || (j > offsets[p] && levels[j] <= levels[j - 1]))
                throw std::invalid_argument("codebook levels must be finite and strictly increasing");
    }
    for (auto p : feature_map) if (p >= profiles_) throw std::invalid_argument("invalid feature map");
    offsets_ = upload(offsets, stream); levels_ = upload(levels, stream); feature_map_ = upload(feature_map, stream);
}
QuantizationProfileView QuantizationProfile::view() const noexcept {
    return {static_cast<const std::uint32_t*>(offsets_.data()), static_cast<const float*>(levels_.data()), static_cast<const std::uint32_t*>(feature_map_.data()), profiles_, features_, levels_count_, bits_};
}
QuantizedState::QuantizedState(std::shared_ptr<const QuantizationProfile> profile, std::uint32_t rows, std::uint32_t override_capacity)
    : profile_(std::move(profile)), rows_(rows), capacity_(override_capacity) {
    if (!profile_) throw std::invalid_argument("null quantization profile");
    const auto n = checked_bytes(rows_, features());
    if (n > std::numeric_limits<std::uint32_t>::max()) throw std::length_error("quantized state exceeds 32-bit logical index capacity");
    if (capacity_ > n) throw std::invalid_argument("override capacity exceeds state size");
    const auto words = (checked_bytes(n, profile_->bits()) + 31) / 32;
    packed_ = DeviceBuffer(checked_bytes(words, 4), device());
    indices_ = DeviceBuffer(checked_bytes(capacity_, 4), device()); values_ = DeviceBuffer(checked_bytes(capacity_, 4), device());
    count_ = DeviceBuffer(4, device()); status_ = DeviceBuffer(4, device());
}
OverrideView QuantizedState::overrides() noexcept { return {static_cast<std::uint32_t*>(indices_.data()), static_cast<float*>(values_.data()), static_cast<std::uint32_t*>(count_.data()), capacity_}; }
ConstOverrideView QuantizedState::overrides() const noexcept { if (!capacity_) return {}; return {static_cast<const std::uint32_t*>(indices_.data()), static_cast<const float*>(values_.data()), static_cast<const std::uint32_t*>(count_.data()), capacity_}; }
void QuantizedState::encode(ConstDeviceView input, DType dtype, ConstDeviceView exact_mask, const Stream& stream) {
    check_stream(device(), stream); const auto n = checked_bytes(rows_, features());
    aligned(input, item_size(dtype));
    require_view(input, checked_bytes(n, item_size(dtype)), device(), "quantization input");
    if (exact_mask.data || exact_mask.bytes) {
        require_view(exact_mask, n, device(), "exact mask");
        if (capacity_ < n) throw std::invalid_argument("exact-mask encoding requires full state override capacity");
    }
    DeviceGuard guard(device());
    check(reset_status({static_cast<std::uint32_t*>(status_.data())}, stream.get()), "reset quantization status");
    check(reset_override_count(overrides(), stream.get()), "reset override count");
    const PackedStateView packed{static_cast<std::uint32_t*>(packed_.data()), packed_.bytes() / 4};
    const DeviceStatus status{static_cast<std::uint32_t*>(status_.data())};
    check(dtype == DType::float16 ? encode_state_fp16(static_cast<const __half*>(input.data), rows_, profile_->view(), packed, static_cast<const std::uint8_t*>(exact_mask.data), overrides(), status, stream.get())
                                : encode_state_fp32(static_cast<const float*>(input.data), rows_, profile_->view(), packed, static_cast<const std::uint8_t*>(exact_mask.data), overrides(), status, stream.get()), "encode quantized state");
    encoded_ = true;
}
void QuantizedState::decode(DeviceView output, DType dtype, const Stream& stream) const {
    if (!encoded_) throw std::logic_error("state must be encoded before decoding");
    check_stream(device(), stream); aligned(output, item_size(dtype)); require_view(output, checked_bytes(checked_bytes(rows_, features()), item_size(dtype)), device(), "decode output");
    DeviceGuard guard(device());
    const ConstPackedStateView packed{static_cast<const std::uint32_t*>(packed_.data()), packed_.bytes() / 4};
    const DeviceStatus status{static_cast<std::uint32_t*>(status_.data())};
    check(dtype == DType::float16 ? decode_state_fp16(packed, rows_, profile_->view(), overrides(), static_cast<__half*>(output.data), status, stream.get())
                                : decode_state_fp32(packed, rows_, profile_->view(), overrides(), static_cast<float*>(output.data), status, stream.get()), "decode quantized state");
}
std::uint32_t QuantizedState::status(const Stream& stream) const {
    if (!encoded_) throw std::logic_error("state has not been encoded");
    std::uint32_t result = 0; status_.download_async(&result, 4, stream); stream.synchronize(); return result;
}
std::uint32_t QuantizedState::override_count(const Stream& stream) const {
    if (!encoded_) throw std::logic_error("state has not been encoded");
    std::uint32_t result = 0; count_.download_async(&result, 4, stream); stream.synchronize(); return result;
}
} // namespace cellerator::modeling::cuda
