#include <Cellerator/compiler/ir/semantic/implement_contraction_segment_and_normalization_operatio_v1.hh>

#include <algorithm>
#include <cmath>
#include <limits>

namespace Cellerator::compiler::ir::semantic {
namespace {

bool is_normalization(aggregate_operation_ir_v1 operation) noexcept {
    return operation >= aggregate_operation_ir_v1::normalize_softmax &&
        operation <= aggregate_operation_ir_v1::normalize_rms;
}

}  // namespace

aggregate_operation_status_ir_v1 validate_aggregate_operation_ir_v1(
    const aggregate_operation_definition_ir_v1& operation) noexcept {
    if (!operation.identity.valid()) return aggregate_operation_status_ir_v1::invalid_identity;
    if (operation.operation == aggregate_operation_ir_v1::support_contraction) {
        if (!operation.support_identity.valid())
            return aggregate_operation_status_ir_v1::invalid_support;
    } else if (!operation.segment_identity.valid()) {
        return aggregate_operation_status_ir_v1::invalid_segments;
    }
    if (operation.operation == aggregate_operation_ir_v1::segment_sum &&
        operation.neutral_element != 0.0)
        return aggregate_operation_status_ir_v1::invalid_neutral_element;
    if (operation.operation == aggregate_operation_ir_v1::segment_maximum &&
        operation.neutral_element != -std::numeric_limits<double>::infinity())
        return aggregate_operation_status_ir_v1::invalid_neutral_element;
    constexpr std::uint32_t effects = aggregate_writes_output_v1 |
        aggregate_advances_generation_v1;
    if ((operation.output_effects & effects) != effects)
        return aggregate_operation_status_ir_v1::invalid_effects;
    return aggregate_operation_status_ir_v1::success;
}

aggregate_operation_status_ir_v1 interpret_support_contraction_ir_v1(
    const aggregate_operation_definition_ir_v1& operation,
    const std::vector<double>& left,
    const std::vector<double>& right,
    const std::vector<std::uint8_t>& active_support,
    double* result) noexcept {
    const auto status = validate_aggregate_operation_ir_v1(operation);
    if (status != aggregate_operation_status_ir_v1::success) return status;
    if (operation.operation != aggregate_operation_ir_v1::support_contraction ||
        result == nullptr || left.size() != right.size() || left.size() != active_support.size())
        return aggregate_operation_status_ir_v1::invalid_input;
    double sum = operation.neutral_element;
    for (std::size_t index = 0; index < left.size(); ++index)
        if (active_support[index] != 0) sum += left[index] * right[index];
    *result = sum;
    return aggregate_operation_status_ir_v1::success;
}

aggregate_operation_status_ir_v1 interpret_segment_operation_ir_v1(
    const aggregate_operation_definition_ir_v1& operation,
    const std::vector<double>& values,
    const std::vector<std::uint64_t>& offsets,
    std::vector<double>* result) noexcept {
    const auto status = validate_aggregate_operation_ir_v1(operation);
    if (status != aggregate_operation_status_ir_v1::success) return status;
    if (operation.operation == aggregate_operation_ir_v1::support_contraction ||
        result == nullptr || offsets.empty() || offsets.front() != 0 ||
        offsets.back() != values.size() ||
        !std::is_sorted(offsets.begin(), offsets.end()))
        return aggregate_operation_status_ir_v1::invalid_input;
    result->clear();
    if (is_normalization(operation.operation)) result->resize(values.size());
    else result->reserve(offsets.size() - 1);

    for (std::size_t segment = 0; segment + 1 < offsets.size(); ++segment) {
        const auto begin = static_cast<std::size_t>(offsets[segment]);
        const auto end = static_cast<std::size_t>(offsets[segment + 1]);
        if (operation.operation == aggregate_operation_ir_v1::segment_sum) {
            double value = operation.neutral_element;
            for (auto index = begin; index < end; ++index) value += values[index];
            result->push_back(value);
        } else if (operation.operation == aggregate_operation_ir_v1::segment_maximum) {
            double value = operation.neutral_element;
            for (auto index = begin; index < end; ++index) value = std::max(value, values[index]);
            result->push_back(value);
        } else if (operation.operation == aggregate_operation_ir_v1::segment_mean ||
                   operation.operation == aggregate_operation_ir_v1::segment_variance) {
            double mean = 0.0;
            for (auto index = begin; index < end; ++index) mean += values[index];
            if (end != begin) mean /= static_cast<double>(end - begin);
            if (operation.operation == aggregate_operation_ir_v1::segment_mean) {
                result->push_back(mean);
            } else {
                double variance = 0.0;
                for (auto index = begin; index < end; ++index) {
                    const auto delta = values[index] - mean;
                    variance += delta * delta;
                }
                result->push_back(end == begin ? 0.0 : variance / static_cast<double>(end - begin));
            }
        } else {
            double denominator = 0.0;
            double maximum = -std::numeric_limits<double>::infinity();
            if (operation.operation == aggregate_operation_ir_v1::normalize_softmax)
                for (auto index = begin; index < end; ++index)
                    maximum = std::max(maximum, values[index]);
            for (auto index = begin; index < end; ++index) {
                if (operation.operation == aggregate_operation_ir_v1::normalize_softmax)
                    denominator += std::exp(values[index] - maximum);
                else if (operation.operation == aggregate_operation_ir_v1::normalize_l1)
                    denominator += std::abs(values[index]);
                else
                    denominator += values[index] * values[index];
            }
            if (operation.operation == aggregate_operation_ir_v1::normalize_l2)
                denominator = std::sqrt(denominator);
            if (operation.operation == aggregate_operation_ir_v1::normalize_rms)
                denominator = end == begin ? 0.0 :
                    std::sqrt(denominator / static_cast<double>(end - begin));
            for (auto index = begin; index < end; ++index) {
                const auto numerator = operation.operation == aggregate_operation_ir_v1::normalize_softmax
                    ? std::exp(values[index] - maximum) : values[index];
                (*result)[index] = denominator == 0.0 ? 0.0 : numerator / denominator;
            }
        }
    }
    return aggregate_operation_status_ir_v1::success;
}

cellerator::compute::operation::v2::segment_operation
lower_segment_operation_ir_v1(aggregate_operation_ir_v1 operation) noexcept {
    using result = cellerator::compute::operation::v2::segment_operation;
    switch (operation) {
    case aggregate_operation_ir_v1::segment_sum: return result::sum;
    case aggregate_operation_ir_v1::segment_maximum: return result::maximum;
    case aggregate_operation_ir_v1::normalize_softmax: return result::softmax;
    case aggregate_operation_ir_v1::normalize_l1: return result::l1_normalize;
    case aggregate_operation_ir_v1::normalize_l2: return result::l2_normalize;
    case aggregate_operation_ir_v1::normalize_rms: return result::rms_normalize;
    default: return result::none;
    }
}

}  // namespace Cellerator::compiler::ir::semantic
