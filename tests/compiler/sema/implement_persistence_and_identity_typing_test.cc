#include <Cellerator/compiler/sema/implement_persistence_and_identity_typing_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::sema::v1;
    const auto declared = make_identity_binding({1, 2}, identity_origin::declared_persistent,
                                                 {3, 1}, 7);
    const auto reused = make_identity_binding({1, 2}, identity_origin::cloned, {8, 1}, 7);
    assert(identity_cache_entry_reusable(declared, reused));
    auto stale = reused;
    stale.semantic_generation = 8;
    assert(!identity_cache_entry_reusable(declared, stale));

    const auto forced = make_identity_binding({9, 10}, identity_origin::user_forced,
                                               {4, 1}, 1);
    assert(identity_is_persistable(forced));
    assert(forced.unsafe_assertion_warning);
    const auto ephemeral = make_identity_binding({}, identity_origin::ephemeral, {5, 1}, 1);
    assert(!identity_is_persistable(ephemeral));
}
