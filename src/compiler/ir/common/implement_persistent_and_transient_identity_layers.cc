#include <Cellerator/compiler/ir/common/implement_persistent_and_transient_identity_layers_v1.hh>

namespace cellerator::compiler::ir {

bool same(stable_id lhs, stable_id rhs) noexcept {
    return lhs.low == rhs.low && lhs.high == rhs.high;
}

identity_layers clone_identity(const identity_layers &source, local_id clone_local) noexcept {
    auto result = source;
    result.local = clone_local;
    result.artifact.reset();
    return result;
}

identity_layers fuse_identities(
    const std::vector<identity_layers> &sources, local_id fused_local) noexcept {
    identity_layers result{};
    result.local = fused_local;
    if (sources.empty())
        return result;
    result.semantic = sources.front().semantic;
    result.provenance = sources.front().provenance;
    for (const auto &source : sources) {
        if (result.semantic && (!source.semantic || !same(*result.semantic, *source.semantic)))
            result.semantic.reset();
        if (result.provenance && (!source.provenance || !same(*result.provenance, *source.provenance)))
            result.provenance.reset();
    }
    return result;
}

std::vector<identity_layers> split_identity(
    const identity_layers &source, const std::vector<local_id> &parts) {
    std::vector<identity_layers> result;
    result.reserve(parts.size());
    for (const auto part : parts)
        result.push_back(clone_identity(source, part));
    return result;
}

bool override_semantic_identity(identity_layers &target, stable_id asserted,
    bool explicit_authority) noexcept {
    if (!explicit_authority && target.semantic && !same(*target.semantic, asserted))
        return false;
    target.semantic = asserted;
    return true;
}

identity_layers strip_for_hot_lowering(
    const identity_layers &source, bool needs_semantic_identity) noexcept {
    identity_layers result{};
    result.local = source.local;
    if (needs_semantic_identity)
        result.semantic = source.semantic;
    return result;
}

} // namespace cellerator::compiler::ir
