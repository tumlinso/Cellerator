#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace Cellerator::compiler::ast {

enum class incremental_identity_level_v1 : std::uint8_t { file = 1, field, subtree };

struct incremental_ast_identity_v1 {
    std::uint64_t identity = 0;
    incremental_identity_level_v1 level = incremental_identity_level_v1::subtree;
    std::uint64_t stable_locator = 0;
    std::uint64_t content_hash = 0;
    std::uint64_t dependency_hash = 0;
    bool macro_dependent = false;
    bool template_dependent = false;
    std::uint64_t macro_environment_hash = 0;
    std::uint64_t template_environment_hash = 0;
};

struct incremental_reuse_metrics_v1 {
    std::size_t total = 0;
    std::size_t reused = 0;
    std::size_t invalidated = 0;
    std::size_t created = 0;
    [[nodiscard]] double reused_fraction() const noexcept {
        return total ? static_cast<double>(reused) / static_cast<double>(total) : 1.0;
    }
};

struct incremental_reuse_result_v1 {
    std::vector<incremental_ast_identity_v1> identities;
    incremental_reuse_metrics_v1 metrics;
};

// New records carry identity zero. Exact fingerprint matches reuse the prior
// identity. Macro/template-dependent records also require their corresponding
// nonzero environment hashes to match, otherwise they invalidate conservatively.
[[nodiscard]] std::optional<incremental_reuse_result_v1>
reuse_incremental_ast_identities_v1(
    std::vector<incremental_ast_identity_v1> previous,
    std::vector<incremental_ast_identity_v1> current,
    std::string* error = nullptr);

} // namespace Cellerator::compiler::ast
