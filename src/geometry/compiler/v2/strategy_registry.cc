#include <Cellerator/geometry/compiler/v2/strategy_registry.hh>

namespace cellerator::geometry::compiler::v2 {
namespace {
bool ordered(stable_identity left, stable_identity right) noexcept {
    return left.high < right.high || (left.high == right.high && left.low < right.low);
}
}

workload_status validate_semantic_strategy_registry(
    const semantic_strategy_registry &registry) noexcept {
    if (registry.strategies == nullptr || registry.strategy_count == 0) {
        return {workload_status_code::invalid_argument, 0};
    }
    for (std::uint64_t index = 0; index < registry.strategy_count; ++index) {
        const semantic_strategy &strategy = registry.strategies[index];
        if (!valid_identity(strategy.identity) || strategy.name == nullptr
            || strategy.query_workspace == nullptr || strategy.solve == nullptr
            || !strategy.deterministic
            || (index != 0 && !ordered(registry.strategies[index - 1].identity,
                strategy.identity))) {
            return {workload_status_code::invalid_argument, index};
        }
    }
    return {};
}
}  // namespace cellerator::geometry::compiler::v2
