#include <Cellerator/compiler/ir/semantic/implement_gradient_and_publication_operations_v1.hh>

namespace Cellerator::compiler::ir::semantic {
namespace {
using code = gradient_publication_status_ir_v1;
using kind = gradient_publication_operation_ir_v1;
using namespace cellerator::compute::relation;
bool same_axis(const axis_descriptor& a, const axis_descriptor& b) noexcept {
    // Reuse fieldwise canonical comparison, including all persistent axis tags.
    operation_descriptor x{}, y{};
    x.topology.source = a; y.topology.source = b;
    return equivalent(x, y);
}
}
code lower_gradient_publication_program_ir_v1(
    const gradient_publication_program_ir_v1& p, relation_effect_sequence* output) noexcept {
    if (!output || !p.program_identity) return code::invalid_identity;
    *output = {};
    if (!validate(p.calculus)) return code::invalid_numerical_policy;
    if (!p.prepared_generation) return code::invalid_generation;
    if (p.stages.empty() || p.stages.size() > max_relation_effects) return code::invalid_stage;
    relation_effect_sequence effects{};
    effects.initial_generation.value = p.prepared_generation;
    std::array<std::uint32_t, max_relation_effects> ancestors{}, core_dependencies{};
    std::uint64_t current = p.prepared_generation;
    std::uint32_t last_publication = 0;
    bool pending_update = false;
    bool forward = false, transpose = false, gradient = false, update = false, publication = false;
    for (std::uint32_t i = 0; i < p.stages.size(); ++i) {
        const auto& s = p.stages[i];
        if (!s.identity || (s.dependencies & ~((1u << i) - 1u))) return code::invalid_dependency;
        for (std::uint32_t j = 0; j < i; ++j) {
            if (s.identity == p.stages[j].identity) return code::invalid_identity;
            if (s.dependencies & (1u << j)) {
                ancestors[i] |= ancestors[j] | (1u << j);
                core_dependencies[i] |= core_dependencies[j];
            }
        }
        if (s.kind == kind::observe_generation) {
            if (pending_update || !last_publication || !(ancestors[i] & last_publication) ||
                s.consumed_generation != current || s.published_generation)
                return code::invalid_generation;
            continue;
        }
        relation_effect e{};
        e.identity = s.identity;
        e.dependencies = core_dependencies[i];
        e.reads.value = s.consumed_generation;
        e.writes.value = s.published_generation;
        const auto& topology = p.calculus.forward.topology;
        switch (s.kind) {
        case kind::forward:
        case kind::transpose: {
            const auto& op = s.kind == kind::forward ? p.calculus.forward : p.calculus.transpose;
            if (!same_axis(s.input_axis, input_axis(op)) || !same_axis(s.output_axis, result_axis(op)))
                return code::invalid_axis;
            e.kind = s.kind == kind::forward ? relation_effect_kind::forward : relation_effect_kind::transpose;
            forward |= s.kind == kind::forward; transpose |= s.kind == kind::transpose;
            break;
        }
        case kind::value_gradient:
            if (!same_axis(s.input_axis, topology.source) || !same_axis(s.output_axis, topology.destination))
                return code::invalid_axis;
            if (s.gradient_order.low != topology.logical_edge_order.low ||
                s.gradient_order.high != topology.logical_edge_order.high) return code::invalid_order;
            e.kind = relation_effect_kind::edge_gradient; gradient = true;
            break;
        case kind::delta_add:
        case kind::gradient_step:
            if ((s.kind == kind::delta_add) != (p.calculus.update == value_update_kind::delta_add))
                return code::invalid_stage;
            if (s.gradient_order.low != topology.logical_edge_order.low ||
                s.gradient_order.high != topology.logical_edge_order.high) return code::invalid_order;
            e.kind = relation_effect_kind::value_update; update = true; pending_update = true;
            break;
        case kind::publish_generation:
            e.kind = relation_effect_kind::publication; publication = true; pending_update = false;
            current = s.consumed_generation; last_publication = 1u << i;
            break;
        default: return code::invalid_stage;
        }
        effects.stages[effects.count] = e;
        core_dependencies[i] |= 1u << effects.count++;
    }
    if (!forward || !transpose || !gradient || !update || !publication)
        return code::incomplete_gradient_closure;
    if (!validate(effects)) return code::invalid_dependency;
    *output = effects;
    return code::success;
}
code validate_gradient_publication_program_ir_v1(const gradient_publication_program_ir_v1& p) noexcept {
    relation_effect_sequence effects{};
    return lower_gradient_publication_program_ir_v1(p, &effects);
}
} // namespace Cellerator::compiler::ir::semantic
