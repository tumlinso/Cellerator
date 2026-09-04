#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace cellerator::compiler::ir {

struct stable_id { std::uint64_t low{}; std::uint64_t high{}; };
struct local_id { std::uint32_t slot{}; std::uint32_t generation{}; };
struct identity_layers {
    std::optional<stable_id> semantic;
    std::optional<stable_id> artifact;
    local_id local;
    std::optional<stable_id> provenance;
};

bool same(stable_id lhs, stable_id rhs) noexcept;
identity_layers clone_identity(const identity_layers &source, local_id clone_local) noexcept;
identity_layers fuse_identities(
    const std::vector<identity_layers> &sources, local_id fused_local) noexcept;
std::vector<identity_layers> split_identity(
    const identity_layers &source, const std::vector<local_id> &parts);
bool override_semantic_identity(identity_layers &target, stable_id asserted,
    bool explicit_authority) noexcept;
identity_layers strip_for_hot_lowering(
    const identity_layers &source, bool needs_semantic_identity) noexcept;

} // namespace cellerator::compiler::ir
