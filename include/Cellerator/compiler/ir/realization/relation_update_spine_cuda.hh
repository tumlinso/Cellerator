#pragma once
#include <Cellerator/compiler/ir/realization/relation_update_spine.hh>
#include <Cellerator/compute/operation/relation_update.hh>
namespace Cellerator::compiler::ir::realization {
// Caller-owned launch facts, borrowed through actual core completion. Alpha is
// a launch coefficient, never folded into immutable semantics. Native users can
// call these same core entrypoints directly with the canonical descriptors.
struct relation_action_bindings {
    relation_update::device_state_view input{}, cotangent{};
    relation_update::device_result_view output{}, adjoint{};
    relation_update::edge_plane_view gradient{}, delta{};
    relation_update::operand_version input_version{}, cotangent_version{};
    relation_update::gradient_stamp* stamp = nullptr;
    relation_update::value_read_lease* lease = nullptr;
    float alpha = 0;
    cudaStream_t owner = nullptr, consumer = nullptr;
};
// Stateless dispatch of one already ordered recipe action. The embedding caller
// follows the recipe in order, checks every status, and calls end_value_read
// after its observation. This adapter neither schedules nor owns a runtime.
inline relation_update::status enqueue_relation_action(
    relation_update::prepared_relation_pair& pair, const lowered_relation_update& recipe,
    std::uint32_t action_index, const relation_action_bindings& b) noexcept {
    using namespace relation_update;
    if (auto s = validate_lowered_relation_update(recipe); !s) return s;
    if (action_index >= recipe.count) return {status_code::invalid_argument, "recipe action index out of range"};
    const auto& a = recipe.actions[action_index];
    switch (a.entry) {
    case relation_entry_point::enqueue_forward:
        return enqueue(pair, recipe.semantic.forward, b.input, b.output, a.expected, b.owner);
    case relation_entry_point::enqueue_transpose:
        return enqueue(pair, recipe.semantic.transpose, b.cotangent, b.adjoint, a.expected, b.owner);
    case relation_entry_point::enqueue_edge_gradient:
        return enqueue_edge_gradient(pair, recipe.semantic, b.input, b.cotangent,
            b.input_version, b.cotangent_version, a.expected, b.gradient, b.stamp, b.owner);
    case relation_entry_point::enqueue_value_update_and_publish: {
        value_update_request request{};
        request.kind = recipe.semantic.update;
        request.operand = request.kind == value_update_kind::gradient_step ? b.gradient : b.delta;
        request.expected = a.expected; request.next = a.next; request.alpha = b.alpha;
        if (request.kind == value_update_kind::gradient_step) {
            if (!b.stamp) return {status_code::invalid_argument, "gradient step requires producer stamp"};
            request.gradient = *b.stamp;
        }
        return enqueue_value_update(pair, request, b.owner);
    }
    case relation_entry_point::begin_value_read:
        return begin_value_read(pair, a.expected, b.consumer, b.lease);
    }
    return {status_code::unsupported_semantics, "unknown recipe entry point"};
}
} // namespace Cellerator::compiler::ir::realization
