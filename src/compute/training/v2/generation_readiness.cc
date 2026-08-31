#include <Cellerator/compute/training/v2/generation_readiness.hh>

#include <cstdint>
#include <limits>

namespace cellerator::compute::training_v2 {
namespace {

using execution::training_v2::training_status_v2;

training_result_v2 error(training_status_v2 code, const char *message) noexcept {
    return {code, message};
}

bool add_fits(std::uint64_t &total, std::uint64_t value) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - total) return false;
    total += value;
    return true;
}

} // namespace

training_result_v2 validate_generation_readiness_v2(
    const generation_publication_v2 &publication,
    generation_publication_receipt_v2 &receipt) noexcept {
    receipt = {};
    if (!valid_handle(publication.structure) || publication.epoch.value == 0u
        || publication.expected_current.value == 0u
        || publication.pending.value != publication.expected_current.value + 1u
        || publication.pending.value == 0u
        || !valid_axis_identity(publication.source_axis)
        || !valid_axis_identity(publication.destination_axis)
        || publication.persistent_order
            != training_order_mode_v2::persistent_physical
        || publication.component_count == 0u || publication.components == nullptr)
        return error(training_status_v2::invalid_generation,
            "generation publication envelope is invalid");
    std::uint64_t previous_identity = 0u;
    std::uint64_t occupied = 0u;
    std::uint64_t required_count = 0u;
    for (std::uint64_t index = 0u; index < publication.component_count;
         ++index) {
        const generation_component_readiness_v2 &component =
            publication.components[index];
        if (component.component_identity == 0u
            || component.component_identity <= previous_identity)
            return error(training_status_v2::invalid_generation,
                "generation components are not uniquely sorted");
        previous_identity = component.component_identity;
        if (!component.required) continue;
        if (component.state != generation_component_state_v2::ready
            || component.generation.value != publication.pending.value
            || component.completion_token == 0u)
            return error(training_status_v2::stale_generation,
                "required generation component is not ready");
        if (!add_fits(occupied, component.occupied_slot_count))
            return error(training_status_v2::invalid_generation,
                "generation occupied-slot census overflows");
        ++required_count;
    }
    if (required_count == 0u)
        return error(training_status_v2::invalid_generation,
            "generation publication has no required component");
    receipt.published = publication.pending;
    receipt.required_component_count = required_count;
    receipt.occupied_slot_count = occupied;
    receipt.retained_order = publication.persistent_order;
    receipt.canonicalized = publication.canonicalization_requested;
    return {};
}

training_result_v2 publish_ready_generation_v2(
    const generation_publication_v2 &publication,
    value_generation &caller_generation,
    generation_publication_receipt_v2 &receipt) noexcept {
    if (caller_generation.value != publication.expected_current.value)
        return error(training_status_v2::stale_generation,
            "caller generation changed before publication");
    generation_publication_receipt_v2 candidate{};
    const training_result_v2 result =
        validate_generation_readiness_v2(publication, candidate);
    if (!result) return result;
    caller_generation = publication.pending;
    receipt = candidate;
    return {};
}

} // namespace cellerator::compute::training_v2
