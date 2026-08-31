#include <Cellerator/compute/architecture/target_cover/strategy_registry.hh>

namespace cellerator::compute::architecture::target_cover {
namespace {
bool ordered(stable_identity left, stable_identity right) noexcept {
    return left.high < right.high || (left.high == right.high && left.low < right.low);
}
}

status validate_strategy_registry(const strategy_registry &registry) noexcept {
    if (registry.strategies == nullptr || registry.strategy_count == 0) {
        return {geometry::compiler::v2::workload_status_code::invalid_argument, 0};
    }
    for (std::uint64_t index = 0; index < registry.strategy_count; ++index) {
        const strategy &entry = registry.strategies[index];
        if (!geometry::compiler::v2::valid_identity(entry.identity)
            || !geometry::compiler::v2::valid_identity(entry.provider_identity)
            || entry.name == nullptr || entry.query_workspace == nullptr
            || entry.solve == nullptr || !entry.deterministic
            || (index != 0 && !ordered(registry.strategies[index - 1].identity,
                entry.identity))) {
            return {geometry::compiler::v2::workload_status_code::invalid_argument, index};
        }
    }
    return {};
}
}  // namespace cellerator::compute::architecture::target_cover
