#pragma once
#include <Cellerator/compiler/sema/relation_update_spine_bridge.hh>
#include <array>

namespace Cellerator::compiler::ir::realization {
namespace relation_update = cellerator::compute::relation;
enum class relation_entry_point : std::uint8_t {
    enqueue_forward, enqueue_transpose, enqueue_edge_gradient,
    enqueue_value_update_and_publish, begin_value_read,
};
struct relation_binding_action {
    relation_entry_point entry = relation_entry_point::enqueue_forward;
    std::uint32_t effect_index = 0;
    std::uint32_t publication_index = 0;
    cellerator::execution::value_generation expected{}, next{};
};
// Cold, value-owned recipe. No compiler execution state, streams, pointers,
// candidate identity or private numerical implementation. Observations follow
// the indicated publication effect and require the caller to return its lease.
struct lowered_relation_update {
    relation_update::relation_calculus_descriptor semantic{};
    relation_update::relation_effect_sequence effects{};
    std::uint32_t observation_after = 0; // mathematical effect-index bitset
    std::uint32_t count = 0;
    std::array<relation_binding_action, relation_update::max_relation_effects> actions{};
};
relation_update::status lower_relation_update(
    const relation_update::relation_calculus_descriptor&,
    const relation_update::relation_effect_sequence&, std::uint32_t observation_after,
    lowered_relation_update*) noexcept;
relation_update::status lower_relation_update(
    const sema::relation_update_source_result&, lowered_relation_update*) noexcept;
relation_update::status validate_lowered_relation_update(const lowered_relation_update&) noexcept;
bool equivalent(const lowered_relation_update&, const lowered_relation_update&) noexcept;

} // namespace Cellerator::compiler::ir::realization
