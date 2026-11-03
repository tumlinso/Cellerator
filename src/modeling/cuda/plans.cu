#include <Cellerator/modeling/cuda/plans.hh>
#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::modeling::cuda {
namespace memory_cuda = ::cellerator::memory::cuda;
namespace {
template<class T> memory_cuda::DeviceBuffer upload(const std::vector<T>& values, const memory_cuda::Stream& stream) {
    memory_cuda::DeviceBuffer result(memory_cuda::checked_bytes(values.size(), sizeof(T)), stream.device());
    result.upload_async(values.data(), result.bytes(), stream);
    // Host vectors may be destroyed immediately after this helper returns.
    stream.synchronize();
    return result;
}
std::size_t elements(std::int64_t rows, std::int64_t features) {
    if (rows < 0 || features < 0) throw std::invalid_argument("negative shape");
    return memory_cuda::checked_bytes(static_cast<std::size_t>(rows), static_cast<std::size_t>(features));
}
void check_stream(int device, const memory_cuda::Stream& stream) {
    if (device != stream.device()) throw std::invalid_argument("plan/stream device mismatch");
}
void aligned(memory_cuda::ConstDeviceView view, std::size_t alignment) {
    if (reinterpret_cast<std::uintptr_t>(view.data) % alignment)
        throw std::invalid_argument("device view is not aligned for its numeric type");
}
void disjoint(memory_cuda::ConstDeviceView input, memory_cuda::DeviceView output, std::size_t input_bytes, std::size_t output_bytes) {
    if (!input_bytes || !output_bytes) return;
    const auto a = reinterpret_cast<std::uintptr_t>(input.data), b = reinterpret_cast<std::uintptr_t>(output.data);
    if ((a <= b && b - a < input_bytes) || (b < a && a - b < output_bytes))
        throw std::invalid_argument("input and output storage overlap");
}
}
DispatchPlan::DispatchPlan(const cellerator::execution::conditional::DispatchPlan& host, const memory_cuda::Stream& stream)
    : DispatchPlan(static_cast<std::int64_t>(host.origin_count()),
                   std::vector<std::int64_t>(host.dispatch_origin_slots().begin(), host.dispatch_origin_slots().end()),
                   std::vector<float>(host.dispatch_weights().begin(), host.dispatch_weights().end()),
                   std::vector<std::int64_t>(host.origin_offsets().begin(), host.origin_offsets().end()),
                   std::vector<std::int64_t>(host.origin_dispatch_indices().begin(), host.origin_dispatch_indices().end()),
                   std::vector<std::int64_t>(host.expert_offsets().begin(), host.expert_offsets().end()), stream) {}
DispatchPlan::DispatchPlan(std::int64_t origins, std::vector<std::int64_t> selection_origin,
                          std::vector<float> weights, std::vector<std::int64_t> origin_offsets,
                          std::vector<std::int64_t> origin_dispatch,
                          std::vector<std::int64_t> expert_offsets, const memory_cuda::Stream& stream)
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
void DispatchPlan::gather(memory_cuda::ConstDeviceView input, memory_cuda::DeviceView output, std::int64_t features, memory_cuda::DType dtype, const memory_cuda::Stream& stream) const {
    check_stream(device_, stream); const auto ni = elements(origins_, features), no = elements(selections_, features), unit = memory_cuda::item_size(dtype);
    aligned(input, unit); aligned(output, unit);
    memory_cuda::require_view(input, memory_cuda::checked_bytes(ni, unit), device_, "gather input"); memory_cuda::require_view(output, memory_cuda::checked_bytes(no, unit), device_, "gather output");
    disjoint(input, output, memory_cuda::checked_bytes(ni, unit), memory_cuda::checked_bytes(no, unit)); memory_cuda::DeviceGuard guard(device_);
    memory_cuda::check(dtype == memory_cuda::DType::float16 ? gather_fp16(static_cast<const __half*>(input.data), ni, static_cast<__half*>(output.data), no, view(), features, stream.get())
                                : gather_fp32(static_cast<const float*>(input.data), ni, static_cast<float*>(output.data), no, view(), features, stream.get()), "dispatch gather");
}
void DispatchPlan::combine(memory_cuda::ConstDeviceView input, memory_cuda::DeviceView output, std::int64_t features, memory_cuda::DType dtype, const memory_cuda::Stream& stream) const {
    check_stream(device_, stream); const auto ni = elements(selections_, features), no = elements(origins_, features), unit = memory_cuda::item_size(dtype);
    aligned(input, unit); aligned(output, unit);
    memory_cuda::require_view(input, memory_cuda::checked_bytes(ni, unit), device_, "combine input"); memory_cuda::require_view(output, memory_cuda::checked_bytes(no, unit), device_, "combine output");
    disjoint(input, output, memory_cuda::checked_bytes(ni, unit), memory_cuda::checked_bytes(no, unit)); memory_cuda::DeviceGuard guard(device_);
    memory_cuda::check(dtype == memory_cuda::DType::float16 ? combine_fp16(static_cast<const __half*>(input.data), ni, static_cast<__half*>(output.data), no, view(), features, stream.get())
                                : combine_fp32(static_cast<const float*>(input.data), ni, static_cast<float*>(output.data), no, view(), features, stream.get()), "dispatch combine");
}
void DispatchPlan::gather_backward(memory_cuda::ConstDeviceView gradient, memory_cuda::DeviceView input_gradient, std::int64_t features, memory_cuda::DType dtype, const memory_cuda::Stream& stream) const {
    check_stream(device_, stream); const auto ni = elements(selections_, features), no = elements(origins_, features), unit = memory_cuda::item_size(dtype);
    aligned(gradient, unit); aligned(input_gradient, unit);
    memory_cuda::require_view(gradient, memory_cuda::checked_bytes(ni, unit), device_, "gather gradient"); memory_cuda::require_view(input_gradient, memory_cuda::checked_bytes(no, unit), device_, "input gradient");
    disjoint(gradient, input_gradient, memory_cuda::checked_bytes(ni, unit), memory_cuda::checked_bytes(no, unit)); memory_cuda::DeviceGuard guard(device_);
    memory_cuda::check(dtype == memory_cuda::DType::float16 ? gather_backward_fp16(static_cast<const __half*>(gradient.data), ni, static_cast<__half*>(input_gradient.data), no, view(), features, stream.get())
                                : gather_backward_fp32(static_cast<const float*>(gradient.data), ni, static_cast<float*>(input_gradient.data), no, view(), features, stream.get()), "dispatch gather backward");
}
void DispatchPlan::combine_backward(memory_cuda::ConstDeviceView gradient, memory_cuda::ConstDeviceView expert_output, memory_cuda::DeviceView expert_gradient, memory_cuda::DeviceView weight_gradient, std::int64_t features, memory_cuda::DType dtype, const memory_cuda::Stream& stream) const {
    check_stream(device_, stream); const auto ng = elements(origins_, features), ne = elements(selections_, features), unit = memory_cuda::item_size(dtype);
    aligned(gradient, unit); aligned(expert_output, unit); aligned(expert_gradient, unit); aligned(weight_gradient, sizeof(float));
    memory_cuda::require_view(gradient, memory_cuda::checked_bytes(ng, unit), device_, "combine gradient"); memory_cuda::require_view(expert_output, memory_cuda::checked_bytes(ne, unit), device_, "expert output");
    memory_cuda::require_view(expert_gradient, memory_cuda::checked_bytes(ne, unit), device_, "expert gradient"); memory_cuda::require_view(weight_gradient, memory_cuda::checked_bytes(selections_, sizeof(float)), device_, "weight gradient");
    disjoint(gradient, expert_gradient, memory_cuda::checked_bytes(ng, unit), memory_cuda::checked_bytes(ne, unit));
    disjoint(expert_output, expert_gradient, memory_cuda::checked_bytes(ne, unit), memory_cuda::checked_bytes(ne, unit));
    disjoint(expert_output, weight_gradient, memory_cuda::checked_bytes(ne, unit), memory_cuda::checked_bytes(selections_, sizeof(float)));
    disjoint(gradient, weight_gradient, memory_cuda::checked_bytes(ng, unit), memory_cuda::checked_bytes(selections_, sizeof(float)));
    disjoint(expert_gradient, weight_gradient, memory_cuda::checked_bytes(ne, unit), memory_cuda::checked_bytes(selections_, sizeof(float)));
    memory_cuda::DeviceGuard guard(device_);
    memory_cuda::check(dtype == memory_cuda::DType::float16 ? combine_backward_fp16(static_cast<const __half*>(gradient.data), ng, static_cast<const __half*>(expert_output.data), ne, static_cast<__half*>(expert_gradient.data), ne, static_cast<float*>(weight_gradient.data), selections_, view(), features, stream.get())
                                : combine_backward_fp32(static_cast<const float*>(gradient.data), ng, static_cast<const float*>(expert_output.data), ne, static_cast<float*>(expert_gradient.data), ne, static_cast<float*>(weight_gradient.data), selections_, view(), features, stream.get()), "dispatch combine backward");
}
} // namespace cellerator::modeling::cuda
