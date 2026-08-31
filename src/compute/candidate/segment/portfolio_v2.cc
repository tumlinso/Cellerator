#include <Cellerator/compute/candidate/segment/portfolio_v2.hh>

#include <cstdint>
#include <limits>

namespace cellerator::compute::segment {
namespace {

constexpr std::uint32_t reduction_count = 6u;
constexpr std::uint32_t normalization_count = 6u;
constexpr std::uint32_t mechanism_count = 3u;
constexpr std::uint32_t order_count = 3u;
constexpr std::uint32_t total_candidates =
    reduction_count * mechanism_count * order_count
    + normalization_count * 2u * mechanism_count * order_count;

segment_result_v2 error(segment_status_v2 code, const char *message) noexcept {
    return {code, message};
}

std::uint64_t hash_byte(std::uint64_t hash, std::uint8_t value) noexcept {
    return (hash ^ value) * 1099511628211ULL;
}

segment_candidate_descriptor_v2 descriptor(
    segment_operation_v2 operation,
    segment_direction_v2 direction,
    segment_reduce_kind_v2 reduction,
    segment_normalize_kind_v2 normalization,
    segment_mechanism_v2 mechanism,
    segment_storage_order_v2 order) noexcept {
    segment_candidate_descriptor_v2 result{};
    result.operation = operation;
    result.direction = direction;
    result.reduction = reduction;
    result.normalization = normalization;
    result.mechanism = mechanism;
    result.storage_order = order;
    result.candidate_identity = segment_candidate_identity_v2(operation,
        direction, reduction, normalization, mechanism, order);
    result.stage_identity = result.candidate_identity ^ 0x73746167652d7632ULL;
    result.threads_per_cta = mechanism == segment_mechanism_v2::warp_per_output
        ? 32u : (mechanism == segment_mechanism_v2::cta_per_output
            ? 256u : 512u);
    result.warps_per_cta = result.threads_per_cta / 32u;
    return result;
}

bool multiply_fits(std::uint64_t left, std::uint64_t right,
    std::uint64_t &result) noexcept {
    if (left != 0u
        && right > std::numeric_limits<std::uint64_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

} // namespace

std::uint64_t segment_candidate_identity_v2(
    segment_operation_v2 operation,
    segment_direction_v2 direction,
    segment_reduce_kind_v2 reduction,
    segment_normalize_kind_v2 normalization,
    segment_mechanism_v2 mechanism,
    segment_storage_order_v2 storage_order) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    hash = hash_byte(hash, 2u);
    hash = hash_byte(hash, static_cast<std::uint8_t>(operation));
    hash = hash_byte(hash, static_cast<std::uint8_t>(direction));
    hash = hash_byte(hash, static_cast<std::uint8_t>(reduction));
    hash = hash_byte(hash, static_cast<std::uint8_t>(normalization));
    hash = hash_byte(hash, static_cast<std::uint8_t>(mechanism));
    return hash_byte(hash, static_cast<std::uint8_t>(storage_order));
}

std::uint32_t segment_candidate_count_v2() noexcept {
    return total_candidates;
}

segment_result_v2 enumerate_segment_candidates_v2(
    segment_candidate_buffer_v2 &buffer) noexcept {
    buffer.count = total_candidates;
    if (buffer.capacity < total_candidates || buffer.data == nullptr)
        return error(segment_status_v2::insufficient_workspace,
            "segment candidate buffer is too small");
    std::uint32_t index = 0u;
    for (std::uint8_t reduction = 1u; reduction <= reduction_count;
         ++reduction)
        for (std::uint8_t mechanism = 1u; mechanism <= mechanism_count;
             ++mechanism)
            for (std::uint8_t order = 1u; order <= order_count; ++order)
                buffer.data[index++] = descriptor(segment_operation_v2::reduce,
                    segment_direction_v2::forward,
                    static_cast<segment_reduce_kind_v2>(reduction),
                    segment_normalize_kind_v2::softmax,
                    static_cast<segment_mechanism_v2>(mechanism),
                    static_cast<segment_storage_order_v2>(order));
    for (std::uint8_t normalization = 1u;
         normalization <= normalization_count; ++normalization)
        for (std::uint8_t direction = 1u; direction <= 2u; ++direction)
            for (std::uint8_t mechanism = 1u; mechanism <= mechanism_count;
                 ++mechanism)
                for (std::uint8_t order = 1u; order <= order_count; ++order)
                    buffer.data[index++] = descriptor(
                        segment_operation_v2::normalize,
                        static_cast<segment_direction_v2>(direction),
                        segment_reduce_kind_v2::sum,
                        static_cast<segment_normalize_kind_v2>(normalization),
                        static_cast<segment_mechanism_v2>(mechanism),
                        static_cast<segment_storage_order_v2>(order));
    return {};
}

segment_result_v2 validate_segment_candidate_catalog_v2(
    const segment_candidate_descriptor_v2 *candidates,
    std::uint32_t count) noexcept {
    if (candidates == nullptr || count != total_candidates)
        return error(segment_status_v2::invalid_argument,
            "segment catalog does not contain the complete portfolio");
    // Validate against the deterministic enumeration rather than performing a
    // quadratic duplicate scan.
    std::uint32_t index = 0u;
    for (std::uint8_t reduction = 1u; reduction <= reduction_count;
         ++reduction)
        for (std::uint8_t mechanism = 1u; mechanism <= mechanism_count;
             ++mechanism)
            for (std::uint8_t order = 1u; order <= order_count; ++order) {
                const auto expected = descriptor(segment_operation_v2::reduce,
                    segment_direction_v2::forward,
                    static_cast<segment_reduce_kind_v2>(reduction),
                    segment_normalize_kind_v2::softmax,
                    static_cast<segment_mechanism_v2>(mechanism),
                    static_cast<segment_storage_order_v2>(order));
                const auto &actual = candidates[index++];
                if (actual.candidate_identity != expected.candidate_identity
                    || actual.stage_identity != expected.stage_identity
                    || actual.operation != expected.operation
                    || actual.direction != expected.direction
                    || actual.reduction != expected.reduction
                    || actual.mechanism != expected.mechanism
                    || actual.storage_order != expected.storage_order
                    || !actual.graph_capture_compatible
                    || !actual.requires_measurement
                    || actual.production_promoted)
                    return error(segment_status_v2::invalid_identity,
                        "segment reduction catalog identity is invalid");
            }
    for (std::uint8_t normalization = 1u;
         normalization <= normalization_count; ++normalization)
        for (std::uint8_t direction = 1u; direction <= 2u; ++direction)
            for (std::uint8_t mechanism = 1u; mechanism <= mechanism_count;
                 ++mechanism)
                for (std::uint8_t order = 1u; order <= order_count; ++order) {
                    const auto expected = descriptor(
                        segment_operation_v2::normalize,
                        static_cast<segment_direction_v2>(direction),
                        segment_reduce_kind_v2::sum,
                        static_cast<segment_normalize_kind_v2>(normalization),
                        static_cast<segment_mechanism_v2>(mechanism),
                        static_cast<segment_storage_order_v2>(order));
                    const auto &actual = candidates[index++];
                    if (actual.candidate_identity
                            != expected.candidate_identity
                        || actual.stage_identity != expected.stage_identity
                        || actual.operation != expected.operation
                        || actual.direction != expected.direction
                        || actual.normalization != expected.normalization
                        || actual.mechanism != expected.mechanism
                        || actual.storage_order != expected.storage_order
                        || !actual.graph_capture_compatible
                        || !actual.requires_measurement
                        || actual.production_promoted)
                        return error(segment_status_v2::invalid_identity,
                            "segment normalization catalog identity is invalid");
                }
    return {};
}

segment_result_v2 build_segment_prepared_manifest_v2(
    const segment_plan_v2 &plan,
    std::uint64_t physical_slot_count,
    segment_prepared_manifest_v2 &manifest) noexcept {
    manifest = {};
    const segment_result_v2 valid = validate_segment_plan_v2(plan);
    if (!valid) return valid;
    if (physical_slot_count < plan.local_value_count)
        return error(segment_status_v2::invalid_shape,
            "segment physical slots cannot prune logical values");
    std::uint64_t interactions = 0u;
    std::uint64_t input_bytes = 0u;
    if (!multiply_fits(plan.local_value_count, plan.dense_width, interactions)
        || !multiply_fits(interactions, sizeof(float), input_bytes))
        return error(segment_status_v2::invalid_shape,
            "segment profiling counters overflow");
    const std::uint64_t output_rows = plan.operation
            == segment_operation_v2::reduce
            || plan.normalization == segment_normalize_kind_v2::log_sum_exp
        ? plan.local_segment_count : plan.local_value_count;
    std::uint64_t output_elements = 0u;
    std::uint64_t output_bytes = 0u;
    if (!multiply_fits(output_rows, plan.dense_width, output_elements)
        || !multiply_fits(output_elements, sizeof(float), output_bytes)
        || (plan.operation == segment_operation_v2::reduce
            && plan.reduction
                == segment_reduce_kind_v2::first_second_moments
            && !multiply_fits(output_bytes, 2u, output_bytes)))
        return error(segment_status_v2::invalid_shape,
            "segment profiling output counters overflow");
    manifest.candidate_identity = segment_candidate_identity_v2(plan.operation,
        plan.direction, plan.reduction, plan.normalization,
        plan.mechanism, plan.storage_order);
    manifest.operation_identity = plan.operation_identity;
    manifest.stage_identity = plan.stage_identity;
    manifest.partition_identity = plan.partition_identity;
    manifest.operation = plan.operation;
    manifest.direction = plan.direction;
    manifest.mechanism = plan.mechanism;
    manifest.storage_order = plan.storage_order;
    manifest.logical_values = plan.local_value_count;
    manifest.physical_slots = physical_slot_count;
    manifest.physical_holes = physical_slot_count - plan.local_value_count;
    manifest.useful_interactions = interactions;
    manifest.input_bytes = input_bytes;
    manifest.output_bytes = output_bytes;
    manifest.threads_per_cta =
        plan.mechanism == segment_mechanism_v2::warp_per_output ? 32u
        : (plan.mechanism == segment_mechanism_v2::cta_per_output
            ? 256u : 512u);
    manifest.warps_per_cta = manifest.threads_per_cta / 32u;
    manifest.requires_measurement = plan.requires_measurement;
    manifest.production_promoted = false;
    return {};
}

} // namespace cellerator::compute::segment
