#include <Cellerator/execution/identity_registry.hh>

#include <cassert>

namespace execution = cellerator::execution;

int main() {
    execution::identity_registry registry{};
    const execution::domain_id first{10u, 20u};
    const execution::domain_id second{11u, 20u};
    execution::domain_handle first_handle{}, repeated_handle{}, second_handle{};
    assert(execution::intern_identity(&registry, first, &first_handle)
        == execution::identity_registry_status::ok);
    assert(execution::intern_identity(&registry, first, &repeated_handle)
        == execution::identity_registry_status::ok);
    assert(execution::same_handle(first_handle, repeated_handle));
    assert(execution::intern_identity(&registry, second, &second_handle)
        == execution::identity_registry_status::ok);
    assert(!execution::same_handle(first_handle, second_handle));

    execution::domain_id resolved{};
    assert(execution::resolve_identity(registry, first_handle, &resolved)
        == execution::identity_registry_status::ok);
    assert(execution::same_identity(first, resolved));

    const execution::order_handle wrong_kind{
        first_handle.slot, first_handle.generation};
    execution::order_id wrong_resolved{};
    assert(execution::resolve_identity(registry, wrong_kind, &wrong_resolved)
        == execution::identity_registry_status::identity_kind_mismatch);

    assert(execution::release_identity(&registry, first_handle)
        == execution::identity_registry_status::ok);
    assert(execution::resolve_identity(registry, first_handle, &resolved)
        == execution::identity_registry_status::stale_handle);
    execution::domain_handle replacement{};
    assert(execution::intern_identity(&registry, {12u, 20u}, &replacement)
        == execution::identity_registry_status::ok);
    assert(replacement.slot == first_handle.slot
        && replacement.generation != first_handle.generation);

    execution::clear_identity_registry(&registry);
    assert(execution::resolve_identity(registry, replacement, &resolved)
        == execution::identity_registry_status::stale_handle);
    execution::domain_handle after_clear{};
    assert(execution::intern_identity(&registry, first, &after_clear)
        == execution::identity_registry_status::ok);
    assert(after_clear.generation != first_handle.generation);
    assert(execution::resolve_identity(registry, first_handle, &resolved)
        == execution::identity_registry_status::stale_handle);
    return 0;
}
