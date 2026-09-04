#pragma once

#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>

namespace cellerator::compiler::sema::v1 {

enum class output_effect : std::uint8_t {
    assign = 1,
    add,
    subtract,
    multiply,
    maximum,
    shared_destination_accumulate,
    partial_output,
    canonicalize,
    epilogue
};

struct output_effect_semantics {
    output_effect effect = output_effect::assign;
    compute::operation::v2::destination_update runtime_update =
        compute::operation::v2::destination_update::overwrite;
    bool input_output_aliasing_legal = false;
    bool requires_order_transform = false;
    bool requires_epilogue = false;
};

output_effect_semantics resolve_output_effect(output_effect effect,
                                              bool alias_requested) noexcept;
bool agrees_with_output_contract(
    const output_effect_semantics &semantics,
    const compute::operation::v2::output_contract &contract) noexcept;

}  // namespace cellerator::compiler::sema::v1
