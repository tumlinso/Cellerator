#include <Cellerator/execution/identity_registry.hh>

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace execution = cellerator::execution;
namespace structure = cellerator::structure;

static_assert(!std::is_same<structure::entity_id,
    structure::occurrence_id>::value,
    "entity and occurrence identities must remain distinct types");
static_assert(sizeof(structure::entity_id) == sizeof(std::uint64_t),
    "domain-local entity identities must remain 64-bit values");
static_assert(sizeof(structure::occurrence_id) == sizeof(std::uint64_t),
    "domain-local occurrence identities must remain 64-bit values");
static_assert(sizeof(execution::axis_identity) == 32u,
    "runtime axis handle ABI must remain unchanged");
static_assert(sizeof(execution::persistent_axis_identity) == 72u,
    "field-serialized persistent axis record layout must remain unchanged");

int main() {
    const structure::entity_id entity{7u};
    const structure::occurrence_id occurrence{7u};
    assert(structure::valid_identity(entity));
    assert(structure::valid_identity(occurrence));
    assert(structure::same_identity(entity, structure::entity_id{7u}));
    assert(structure::same_identity(occurrence,
        structure::occurrence_id{7u}));

    const structure::LogicalAxis empty_axis{
        structure::domain_id{10u, 20u}, structure::order_id{11u, 21u}, 0u};
    assert(structure::valid_logical_axis(empty_axis));
    assert(!structure::valid_epoch(structure::structure_epoch{0u}));
    assert(structure::valid_epoch(structure::structure_epoch{1u}));

    execution::identity_registry registry{};
    const structure::domain_id first{10u, 20u};
    const structure::domain_id second{11u, 20u};
    execution::domain_handle first_handle{}, repeated_handle{}, second_handle{};
    assert(execution::intern_identity(&registry, first, &first_handle)
        == execution::identity_registry_status::ok);
    assert(execution::intern_identity(&registry, first, &repeated_handle)
        == execution::identity_registry_status::ok);
    assert(execution::same_handle(first_handle, repeated_handle));
    assert(execution::intern_identity(&registry, second, &second_handle)
        == execution::identity_registry_status::ok);
    assert(!execution::same_handle(first_handle, second_handle));

    structure::domain_id resolved{};
    assert(execution::resolve_identity(registry, first_handle, &resolved)
        == execution::identity_registry_status::ok);
    assert(structure::same_identity(first, resolved));

    const execution::order_handle wrong_kind{
        first_handle.slot, first_handle.generation};
    structure::order_id wrong_resolved{};
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
