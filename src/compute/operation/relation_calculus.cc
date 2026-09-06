#include <Cellerator/compute/operation/relation_calculus.hh>

namespace cellerator::compute::relation {
namespace {
bool same_gradient(const scalar_gradient_policy& a, const scalar_gradient_policy& b) noexcept {
    return a.input_storage == b.input_storage && a.accumulation == b.accumulation
        && a.output_storage == b.output_storage && a.output == b.output
        && a.channels_per_edge == b.channels_per_edge && a.permit_fma == b.permit_fma
        && a.permit_reassociation == b.permit_reassociation && a.nonfinite == b.nonfinite;
}
bool supported_arithmetic(const arithmetic_policy& a) noexcept {
    using execution::numeric_type;
    return a.relation_storage == numeric_type::f16 && a.input_storage == numeric_type::f32
        && a.multiply == numeric_type::f32 && a.accumulation == numeric_type::f32
        && a.output_storage == numeric_type::f32 && a.permit_fma
        && a.permit_reassociation && a.nonfinite == nonfinite_policy::propagate;
}
} // namespace
status validate(const relation_calculus_descriptor& c) noexcept {
    if (auto s = validate(c.forward); !s) return s;
    if (auto s = validate(c.transpose); !s) return s;
    if (c.forward.direction != orientation::forward || c.transpose.direction != orientation::transpose)
        return {status_code::unsupported_semantics, "calculus requires forward then adjoint orientations"};
    // Normalize only fields explicitly independent between the two operations.
    auto adjoint = c.transpose;
    adjoint.direction = c.forward.direction;
    adjoint.update = c.forward.update;
    if (!equivalent(c.forward, adjoint))
        return {status_code::unsupported_semantics, "adjoint must preserve exact topology, axes, width and arithmetic"};
    if (c.forward.dense_width != 1 && c.forward.dense_width != 16)
        return {status_code::unsupported_width, "bounded calculus supports N1 and N16"};
    if (c.forward.input_output_aliasing_legal)
        return {status_code::unsupported_semantics, "bounded calculus forbids input/output aliasing"};
    if (!supported_arithmetic(c.forward.arithmetic))
        return {status_code::unsupported_numeric_policy, "bounded apply requires f16 weights and permitted f32 fused reductions"};
    if (c.gradient != gradient_arithmetic::full_f32 && c.gradient != gradient_arithmetic::round_operands_f16_rne)
        return {status_code::unsupported_numeric_policy, "unknown gradient operand profile"};
    if (c.update != value_update_kind::delta_add && c.update != value_update_kind::gradient_step)
        return {status_code::unsupported_semantics, "unknown value update"};
    const auto& g = c.scalar_gradient;
    if (g.channels_per_edge != 1 || g.output != output_update::overwrite)
        return {status_code::unsupported_semantics, "edge VJP is scalar overwrite without normalization"};
    using execution::numeric_type;
    if (g.input_storage != numeric_type::f32 || g.accumulation != numeric_type::f32
        || g.output_storage != numeric_type::f32 || !g.permit_fma
        || !g.permit_reassociation || g.nonfinite != nonfinite_policy::propagate)
        return {status_code::unsupported_numeric_policy, "gradient requires explicit f32 fused reduction permissions"};
    return {};
}
bool equivalent(const relation_calculus_descriptor& a, const relation_calculus_descriptor& b) noexcept {
    return equivalent(a.forward, b.forward) && equivalent(a.transpose, b.transpose)
        && a.gradient == b.gradient && a.update == b.update
        && same_gradient(a.scalar_gradient, b.scalar_gradient);
}
status validate(const relation_effect_sequence& sequence) noexcept {
    if (sequence.count == 0 || sequence.count > max_relation_effects || sequence.initial_generation.value == 0)
        return {status_code::invalid_argument, "effect witness needs stages and an initial published generation"};
    std::array<std::uint32_t, max_relation_effects> ancestors{};
    std::uint64_t current = sequence.initial_generation.value;
    std::uint64_t pending = 0;
    std::uint32_t readers = 0, writer = 0, publication = 0;
    for (std::uint32_t i = 0; i < sequence.count; ++i) {
        const auto& stage = sequence.stages[i];
        const std::uint32_t earlier = (1u << i) - 1u;
        if (stage.identity == 0 || (stage.dependencies & ~earlier) != 0)
            return {status_code::invalid_argument, "effect has zero identity or self/future/out-of-range dependency"};
        for (std::uint32_t j = 0; j < i; ++j) {
            if (stage.identity == sequence.stages[j].identity)
                return {status_code::invalid_identity, "duplicate effect stage identity"};
            if ((stage.dependencies & (1u << j)) != 0)
                ancestors[i] |= ancestors[j] | (1u << j);
        }
        const auto depends = [&](std::uint32_t mask) { return (ancestors[i] & mask) == mask; };
        switch (stage.kind) {
        case relation_effect_kind::forward:
        case relation_effect_kind::transpose:
        case relation_effect_kind::edge_gradient:
            if (pending != 0 || stage.reads.value != current || stage.writes.value != 0 || !depends(publication))
                return {status_code::stale_generation, "read must follow publication of current generation"};
            readers |= 1u << i;
            break;
        case relation_effect_kind::value_update:
            if (pending != 0 || stage.reads.value != current || stage.writes.value <= current
                || !depends(readers | publication))
                return {status_code::invalid_state, "update must follow all old readers and strictly advance generation"};
            pending = stage.writes.value;
            writer = 1u << i;
            break;
        case relation_effect_kind::publication:
            if (pending == 0 || stage.reads.value != pending || stage.writes.value != 0 || !depends(writer))
                return {status_code::invalid_state, "publication must consume exact pending writer generation"};
            current = pending;
            pending = 0;
            readers = 0;
            publication = 1u << i;
            break;
        default:
            return {status_code::unsupported_semantics, "unknown effect kind"};
        }
    }
    if (pending != 0)
        return {status_code::invalid_state, "effect witness ends with an unpublished mutation"};
    return {};
}
} // namespace cellerator::compute::relation
