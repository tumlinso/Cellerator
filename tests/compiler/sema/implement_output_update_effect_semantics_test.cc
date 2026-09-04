#include <Cellerator/compiler/sema/implement_output_update_effect_semantics_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::sema::v1;
    using namespace cellerator::compute::operation::v2;
    const auto add = resolve_output_effect(output_effect::add, true);
    output_contract runtime{};
    runtime.update = destination_update::accumulate;
    runtime.input_output_aliasing_legal = true;
    assert(agrees_with_output_contract(add, runtime));

    const auto partial = resolve_output_effect(output_effect::partial_output, true);
    assert(partial.runtime_update == destination_update::partial_write);
    assert(!partial.input_output_aliasing_legal);
    const auto canonical = resolve_output_effect(output_effect::canonicalize, false);
    assert(canonical.requires_order_transform);
    const auto epilogue = resolve_output_effect(output_effect::epilogue, false);
    assert(epilogue.requires_epilogue && epilogue.runtime_update == destination_update::affine_accumulate);
}
