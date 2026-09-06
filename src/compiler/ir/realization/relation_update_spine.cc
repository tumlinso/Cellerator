#include <Cellerator/compiler/ir/realization/relation_update_spine.hh>

namespace Cellerator::compiler::ir::realization {
namespace {
using namespace relation_update;
bool same_effects(const relation_effect_sequence& a, const relation_effect_sequence& b) noexcept {
    if (a.initial_generation.value != b.initial_generation.value || a.count != b.count || a.count > max_relation_effects) return false;
    for (std::uint32_t i=0; i<a.count; ++i) {
        const auto& x=a.stages[i]; const auto& y=b.stages[i];
        if (x.identity!=y.identity || x.kind!=y.kind || x.dependencies!=y.dependencies ||
            x.reads.value!=y.reads.value || x.writes.value!=y.writes.value) return false;
    }
    return true;
}
bool same_actions(const lowered_relation_update& a, const lowered_relation_update& b) noexcept {
    if (a.count!=b.count || a.count>max_relation_effects || a.observation_after!=b.observation_after) return false;
    for (std::uint32_t i=0; i<a.count; ++i) {
        const auto& x=a.actions[i]; const auto& y=b.actions[i];
        if (x.entry!=y.entry || x.effect_index!=y.effect_index || x.publication_index!=y.publication_index ||
            x.expected.value!=y.expected.value || x.next.value!=y.next.value) return false;
    }
    return true;
}
}
relation_update::status lower_relation_update(const relation_calculus_descriptor& semantic,
    const relation_effect_sequence& effects, std::uint32_t observations, lowered_relation_update* out) noexcept {
    if (!out) return {status_code::invalid_argument,"null recipe output"};
    *out={};
    if (auto s=validate(semantic); !s) return s;
    if (auto s=validate(effects); !s) return s;
    if (observations & ~((1u<<effects.count)-1u)) return {status_code::invalid_argument,"observation outside effect sequence"};
    lowered_relation_update result{};
    result.semantic=semantic; result.effects=effects; result.observation_after=observations;
    for (std::uint32_t i=0;i<effects.count;++i) {
        const auto& e=effects.stages[i];
        if (result.count==max_relation_effects) return {status_code::insufficient_capacity,"too many recipe actions"};
        relation_binding_action a{}; a.effect_index=i; a.expected=e.reads;
        switch(e.kind) {
        case relation_effect_kind::forward: a.entry=relation_entry_point::enqueue_forward; break;
        case relation_effect_kind::transpose: a.entry=relation_entry_point::enqueue_transpose; break;
        case relation_effect_kind::edge_gradient: a.entry=relation_entry_point::enqueue_edge_gradient; break;
        case relation_effect_kind::value_update:
            if (i+1>=effects.count || effects.stages[i+1].kind!=relation_effect_kind::publication)
                return {status_code::unsupported_semantics,"core update requires adjacent publication"};
            a.entry=relation_entry_point::enqueue_value_update_and_publish;
            a.next=e.writes; a.publication_index=i+1;
            break;
        case relation_effect_kind::publication:
            if (!i || effects.stages[i-1].kind!=relation_effect_kind::value_update)
                return {status_code::invalid_state,"unpaired publication"};
            if (observations & (1u<<i)) {
                a.entry=relation_entry_point::begin_value_read;
                result.actions[result.count++]=a;
            }
            continue;
        default: return {status_code::unsupported_semantics,"unknown effect"};
        }
        if (observations & (1u<<i)) return {status_code::invalid_state,"observation requires publication"};
        result.actions[result.count++]=a;
    }
    *out=result; return {};
}
relation_update::status lower_relation_update(const sema::relation_update_source_result& source,
    lowered_relation_update* out) noexcept {
    if (!out) return {status_code::invalid_argument,"null recipe output"};
    *out={};
    if (!source.accepted()) return {status_code::invalid_argument,"source was not successfully lowered"};
    relation_effect_sequence effects{};
    if (semantic::lower_gradient_publication_program_ir_v1(source.program,&effects)!=semantic::gradient_publication_status_ir_v1::success ||
        !relation_update::equivalent(source.semantic,source.program.calculus) || !same_effects(source.effects,effects))
        return {status_code::unsupported_semantics,"source descriptor or effect witness was altered"};
    std::uint32_t observations=0, effect_count=0;
    for (const auto& stage:source.program.stages) {
        if (stage.kind==semantic::gradient_publication_operation_ir_v1::observe_generation) {
            // The bounded core exposes one lease. Repeated observations after
            // intervening operations are not silently hoisted or merged.
            if (!effect_count || effects.stages[effect_count-1].kind!=relation_effect_kind::publication ||
                (observations & (1u<<(effect_count-1))))
                return {status_code::unsupported_semantics,"observation must immediately follow its publication once"};
            observations|=1u<<(effect_count-1);
        } else ++effect_count;
    }
    return lower_relation_update(source.semantic,effects,observations,out);
}
relation_update::status validate_lowered_relation_update(const lowered_relation_update& recipe) noexcept {
    lowered_relation_update expected{};
    if (auto s=lower_relation_update(recipe.semantic,recipe.effects,recipe.observation_after,&expected); !s) return s;
    return same_actions(recipe,expected) ? status{} : status{status_code::invalid_state,"recipe does not match canonical effects"};
}
bool equivalent(const lowered_relation_update& a,const lowered_relation_update& b) noexcept {
    return relation_update::equivalent(a.semantic,b.semantic) && same_effects(a.effects,b.effects) && same_actions(a,b);
}
} // namespace Cellerator::compiler::ir::realization
