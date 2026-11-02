#include <Cellerator/execution/dependency/distributed.hh>

namespace cellerator::execution::dependency {
namespace {
using Code = DistributedBindingCode;
bool power_of_two(std::size_t n) noexcept { return n != 0 && (n & (n - 1)) == 0; }
}

DistributedBindingStatus validate_extent(
    const ExtentRequirement& expected, const DeviceExtentBinding& actual) noexcept {
    if (!structure::valid_identity(expected.domain)
        || !structure::valid_identity(expected.logical_order)
        || !structure::valid_identity(expected.physical_order)
        || !structure::valid_identity(expected.structure))
        return {Code::invalid_identity};
    if (!expected.structure_epoch || !expected.value_generation
        || !expected.placement_revision)
        return {Code::invalid_revision};
    if (!power_of_two(expected.alignment)) return {Code::invalid_alignment};
    if (actual.bytes && !actual.data) return {Code::invalid_extent};
    if (actual.bytes < expected.minimum_bytes) return {Code::insufficient_capacity};
    if (actual.data && reinterpret_cast<std::uintptr_t>(actual.data) % expected.alignment)
        return {Code::invalid_alignment};
    if (!structure::same_identity(expected.domain, actual.domain)) return {Code::wrong_domain};
    if (!structure::same_identity(expected.logical_order, actual.logical_order)
        || !structure::same_identity(expected.physical_order, actual.physical_order))
        return {Code::wrong_order};
    if (!structure::same_identity(expected.structure, actual.structure))
        return {Code::wrong_structure};
    if (expected.structure_epoch != actual.structure_epoch) return {Code::stale_structure};
    if (expected.value_generation != actual.value_generation) return {Code::stale_values};
    if (expected.placement_revision != actual.placement_revision) return {Code::stale_placement};
    if (expected.device < 0 || actual.where.kind != memory::domain::device
        || actual.where.device_ordinal != expected.device
        || actual.ready_device != expected.device)
        return {Code::wrong_device};
    if (actual.ready_generation != actual.value_generation) return {Code::stale_readiness};
    return {};
}

DistributedBindingStatus validate_extents(
    std::span<const ExtentRequirement> expected,
    std::span<const DeviceExtentBinding> actual) noexcept {
    if (expected.size() != actual.size()) return {Code::count_mismatch};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        auto result = validate_extent(expected[index], actual[index]);
        if (!result) { result.index = index; return result; }
    }
    return {};
}

DistributedBindingStatus validate_transfer_requirements(
    const ops::mechanism::AccessContract& access,
    std::span<const TransferRequirement> requirements) noexcept {
    for (std::size_t index = 0; index < requirements.size(); ++index) {
        const auto& required = requirements[index];
        if (required.source_device < 0 || required.destination_device < 0
            || required.source_device == required.destination_device)
            return {Code::wrong_device, index};
        bool found = false;
        for (const auto& phase : access.phases) {
            if (phase.phase != required.phase) continue;
            for (const auto& read : phase.reads) {
                if (geometry::dependency::same_source(read.source, required.source.source)
                    && geometry::dependency::same_selection(read.selection, required.source.selection))
                    found = true;
            }
        }
        if (!found) return {Code::invalid_identity, index};
    }
    return {};
}
} // namespace cellerator::execution::dependency
