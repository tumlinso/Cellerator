#include <Cellerator/compute/operation/native_foundation_contract.hh>
#include <cassert>
using namespace cellerator;
namespace nf = compute::operation::nf1;

execution::persistent_axis_identity axis() {
    return {{execution::biological_abi_version, execution::serialized_record_kind::persistent_axis_identity,
             sizeof(execution::persistent_axis_identity)}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};
}
compute::operation::v2::numerical_policy numeric() {
    auto f = execution::numeric_type::f32;
    return {f, f, f, f, f, f};
}
int main() {
    auto a = axis();
    nf::operand_signature inputs[] = {{{1, 0}, {&a, 1}, 33, execution::numeric_type::f32},
                                     {{2, 0}, {&a, 1}, 33, execution::numeric_type::f32},
                                     {{3, 0}, {&a, 1}, 33, execution::numeric_type::f32}};
    nf::output_signature outputs[] = {{inputs[0], {100, 0}}, {inputs[1], {100, 0}}};
    nf::operation_contract op; op.definition = {8, 0}; op.arguments = inputs; op.outputs = outputs; op.numeric = numeric();
    assert(nf::validate_operation(op) == nf::status::success);
    auto foreign = a; foreign.domain.low = 99;
    auto bound = inputs[0]; bound.axes = {&foreign, 1};
    assert(nf::match_operand(inputs[0], bound) == nf::status::axis_mismatch);
    foreign = a; foreign.order.low = 99;
    assert(nf::match_operand(inputs[0], bound) == nf::status::axis_mismatch);
    outputs[1].operand.role = outputs[0].operand.role;
    assert(nf::validate_operation(op) == nf::status::duplicate_output);
    outputs[1].operand.role = {2, 0}; op.capabilities |= 128u;
    assert(nf::validate_operation(op) == nf::status::unsupported_capability);
    op.capabilities = nf::forward; outputs[0].effect.requires_initialized_destination = true;
    assert(nf::validate_operation(op) == nf::status::invalid_effect);
}
