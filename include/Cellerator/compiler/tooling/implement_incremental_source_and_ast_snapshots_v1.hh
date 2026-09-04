#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::tooling {

enum class snapshot_artifact_v1 : std::uint8_t {
    source_tokens = 1,
    include_state,
    shadow_cpp,
    cpp_ast_bridge,
    cellerator_ast,
    semantic_ir,
};

struct snapshot_region_v1 {
    std::uint64_t stable_locator = 0;
    snapshot_artifact_v1 artifact = snapshot_artifact_v1::source_tokens;
    std::uint64_t content_hash = 0;
    std::uint64_t dependency_hash = 0;
    std::vector<std::uint64_t> dependencies;
    std::string payload;
    std::uint64_t generation = 0;
};

struct incremental_snapshot_metrics_v1 {
    std::uint64_t reused = 0;
    std::uint64_t rebuilt = 0;
    std::uint64_t dependency_invalidated = 0;
    [[nodiscard]] double reuse_fraction() const noexcept {
        const auto total = reused + rebuilt;
        return total == 0 ? 1.0 : static_cast<double>(reused) / static_cast<double>(total);
    }
};

struct incremental_snapshot_update_v1 {
    std::vector<snapshot_region_v1> regions;
    incremental_snapshot_metrics_v1 metrics;
};

[[nodiscard]] incremental_snapshot_update_v1 update_incremental_snapshot_v1(
    const std::vector<snapshot_region_v1> &previous,
    std::vector<snapshot_region_v1> current,
    std::uint64_t generation);

} // namespace Cellerator::compiler::tooling
