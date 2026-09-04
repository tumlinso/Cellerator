#include <Cellerator/compiler/ir/common/implement_persistent_and_transient_identity_layers_v1.hh>

#include <cassert>

using namespace cellerator::compiler::ir;

int main() {
    identity_layers original{{stable_id{1u, 2u}}, {stable_id{3u, 4u}},
        {7u, 1u}, {stable_id{5u, 6u}}};
    const auto clone = clone_identity(original, {8u, 1u});
    assert(clone.semantic && same(*clone.semantic, *original.semantic));
    assert(!clone.artifact && clone.local.slot == 8u);
    const auto parts = split_identity(original, {{9u, 1u}, {10u, 1u}});
    assert(parts.size() == 2u && parts[1].local.slot == 10u);
    const auto fused = fuse_identities({original, clone}, {11u, 1u});
    assert(fused.semantic && fused.provenance && !fused.artifact);

    auto asserted = original;
    assert(!override_semantic_identity(asserted, {99u, 1u}, false));
    assert(override_semantic_identity(asserted, {99u, 1u}, true));
    const auto hot = strip_for_hot_lowering(asserted, false);
    assert(!hot.semantic && !hot.artifact && !hot.provenance);
    assert(hot.local.slot == asserted.local.slot);
}
