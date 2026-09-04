#include <Cellerator/compiler/sema/implement_persistence_and_identity_typing_v1.hh>

namespace cellerator::compiler::sema::v1 {

identity_binding make_identity_binding(semantic_identity identity,
                                       identity_origin origin,
                                       compiler_identity_handle handle,
                                       std::uint64_t generation) noexcept {
    return {identity, origin, handle, generation,
            origin == identity_origin::user_forced};
}

bool identity_is_persistable(const identity_binding &binding) noexcept {
    if (binding.persistent.low == 0 && binding.persistent.high == 0)
        return false;
    return binding.origin == identity_origin::declared_persistent
        || binding.origin == identity_origin::user_forced
        || binding.origin == identity_origin::cloned;
}

bool identity_cache_entry_reusable(const identity_binding &cached,
                                   const identity_binding &requested) noexcept {
    return identity_is_persistable(cached) && identity_is_persistable(requested)
        && cached.persistent.low == requested.persistent.low
        && cached.persistent.high == requested.persistent.high
        && cached.semantic_generation == requested.semantic_generation;
}

}  // namespace cellerator::compiler::sema::v1
