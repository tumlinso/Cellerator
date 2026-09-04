#include <Cellerator/compiler/sema/implement_output_update_effect_semantics_v1.hh>

namespace cellerator::compiler::sema::v1 {

output_effect_semantics resolve_output_effect(output_effect effect,
                                              bool alias_requested) noexcept {
    using update = compute::operation::v2::destination_update;
    switch (effect) {
    case output_effect::assign:
        return {effect, update::overwrite, alias_requested, false, false};
    case output_effect::add:
    case output_effect::subtract:
    case output_effect::multiply:
    case output_effect::maximum:
    case output_effect::shared_destination_accumulate:
        return {effect, update::accumulate, alias_requested, false,
                effect != output_effect::shared_destination_accumulate};
    case output_effect::partial_output:
        return {effect, update::partial_write, false, false, false};
    case output_effect::canonicalize:
        return {effect, update::overwrite, false, true, false};
    case output_effect::epilogue:
        return {effect, update::affine_accumulate, alias_requested, false, true};
    }
    return {};
}

bool agrees_with_output_contract(
    const output_effect_semantics &semantics,
    const compute::operation::v2::output_contract &contract) noexcept {
    return semantics.runtime_update == contract.update
        && semantics.input_output_aliasing_legal == contract.input_output_aliasing_legal
        && semantics.requires_order_transform == contract.explicit_order_transform;
}

}  // namespace cellerator::compiler::sema::v1
