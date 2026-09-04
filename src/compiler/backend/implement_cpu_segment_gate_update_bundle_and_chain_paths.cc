#include <Cellerator/compiler/backend/implement_cpu_segment_gate_update_bundle_and_chain_paths_v1.hh>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::compiler::backend::v1 {

cpu_fallback_status_v1 run_cpu_segment_v1(
    const cpu_segment_request_v1& request) noexcept {
    if (request.offsets == nullptr || request.input == nullptr
        || request.output == nullptr || request.segment_count == 0)
        return cpu_fallback_status_v1::invalid_argument;
    for (std::uint64_t segment = 0; segment < request.segment_count; ++segment) {
        const auto begin = request.offsets[segment];
        const auto end = request.offsets[segment + 1];
        if (begin > end) return cpu_fallback_status_v1::invalid_offsets;
        if (request.kind == cpu_segment_kind_v1::sum) {
            float sum = 0;
            for (auto i = begin; i < end; ++i) sum += request.input[i];
            request.output[segment] = sum;
        } else if (request.kind == cpu_segment_kind_v1::maximum) {
            float maximum = -std::numeric_limits<float>::infinity();
            for (auto i = begin; i < end; ++i)
                maximum = std::max(maximum, request.input[i]);
            request.output[segment] = maximum;
        } else {
            if (begin == end) continue;
            float maximum = request.input[begin];
            for (auto i = begin + 1; i < end; ++i)
                maximum = std::max(maximum, request.input[i]);
            float denominator = 0;
            for (auto i = begin; i < end; ++i)
                denominator += std::exp(request.input[i] - maximum);
            for (auto i = begin; i < end; ++i)
                request.output[i] = std::exp(request.input[i] - maximum) / denominator;
        }
    }
    return cpu_fallback_status_v1::success;
}

cpu_fallback_status_v1 run_cpu_gate_v1(
    const cpu_gate_request_v1& request) noexcept {
    if (request.input == nullptr || request.gate == nullptr
        || request.output == nullptr || request.count == 0)
        return cpu_fallback_status_v1::invalid_argument;
    if (request.kind == cpu_gate_kind_v1::predicate) {
        const auto* gate = static_cast<const std::uint8_t*>(request.gate);
        for (std::uint64_t i = 0; i < request.count; ++i)
            request.output[i] = gate[i] == 0 ? 0.0F : request.input[i];
    } else {
        const auto* gate = static_cast<const float*>(request.gate);
        for (std::uint64_t i = 0; i < request.count; ++i)
            request.output[i] = gate[i] * request.input[i];
    }
    return cpu_fallback_status_v1::success;
}

cpu_fallback_status_v1 run_cpu_sparse_update_v1(
    const cpu_sparse_update_request_v1& request) noexcept {
    if (request.values == nullptr || request.indices == nullptr
        || request.updates == nullptr || request.value_count == 0)
        return cpu_fallback_status_v1::invalid_argument;
    for (std::uint64_t update = 0; update < request.update_count; ++update) {
        if (request.indices[update] >= request.value_count)
            return cpu_fallback_status_v1::index_out_of_range;
        request.values[request.indices[update]] += request.updates[update];
    }
    return cpu_fallback_status_v1::success;
}

cpu_fallback_status_v1 run_cpu_bundle_v1(
    const cpu_bundle_request_v1& request) noexcept {
    if (request.members == nullptr || request.member_count == 0
        || request.element_count == 0 || request.output == nullptr)
        return cpu_fallback_status_v1::invalid_argument;
    std::fill_n(request.output, request.element_count, 0.0F);
    for (std::uint32_t member = 0; member < request.member_count; ++member) {
        if (request.members[member] == nullptr)
            return cpu_fallback_status_v1::invalid_argument;
        for (std::uint64_t i = 0; i < request.element_count; ++i)
            request.output[i] += request.members[member][i];
    }
    return cpu_fallback_status_v1::success;
}

cpu_fallback_status_v1 run_cpu_chain_v1(
    const cpu_chain_request_v1& request) noexcept {
    if (request.input == nullptr || request.output == nullptr
        || request.element_count == 0 || request.stages == nullptr
        || request.stage_count == 0)
        return cpu_fallback_status_v1::invalid_argument;
    for (std::uint64_t i = 0; i < request.element_count; ++i) {
        float value = request.input[i];
        for (std::uint32_t stage = 0; stage < request.stage_count; ++stage)
            value = value * request.stages[stage].scale
                + request.stages[stage].bias;
        request.output[i] = value;
    }
    return cpu_fallback_status_v1::success;
}

}  // namespace cellerator::compiler::backend::v1
