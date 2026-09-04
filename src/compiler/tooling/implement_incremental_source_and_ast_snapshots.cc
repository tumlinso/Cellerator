#include <Cellerator/compiler/tooling/implement_incremental_source_and_ast_snapshots_v1.hh>

#include <algorithm>
#include <unordered_set>

namespace Cellerator::compiler::tooling {

incremental_snapshot_update_v1 update_incremental_snapshot_v1(
    const std::vector<snapshot_region_v1> &previous,
    std::vector<snapshot_region_v1> current,
    std::uint64_t generation) {
    incremental_snapshot_update_v1 result;
    std::unordered_set<std::uint64_t> invalidated;
    for (auto &region : current) {
        const auto prior = std::find_if(previous.begin(), previous.end(), [&](const auto &candidate) {
            return candidate.stable_locator == region.stable_locator &&
                   candidate.artifact == region.artifact;
        });
        const bool direct_match = prior != previous.end() &&
                                  prior->content_hash == region.content_hash &&
                                  prior->dependency_hash == region.dependency_hash;
        const bool dependency_changed = std::any_of(
            region.dependencies.begin(), region.dependencies.end(),
            [&](std::uint64_t dependency) { return invalidated.count(dependency) != 0; });
        if (direct_match && !dependency_changed) {
            region.payload = prior->payload;
            region.generation = prior->generation;
            ++result.metrics.reused;
        } else {
            if (dependency_changed) ++result.metrics.dependency_invalidated;
            region.generation = generation;
            invalidated.insert(region.stable_locator);
            ++result.metrics.rebuilt;
        }
    }
    result.regions = std::move(current);
    return result;
}

} // namespace Cellerator::compiler::tooling
