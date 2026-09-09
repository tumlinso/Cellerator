#include "fixture.hh"
int main() {
    auto a = axis(); nf::operand_signature input{{1, 0}, {&a, 1}, 33, execution::numeric_type::f32};
    nf::output_signature output{input, {9, 0}};
    nf::operation_contract op; op.definition={1,0}; op.arguments={&input,1}; op.outputs={&output,1}; op.numeric=numeric();
    nf::derivative_request request;
    request.primal.instance={{{1,0},{2,0},{1},{3,0}},{{4,0},{1}},{{5,0},{1}}};
    request.primal.branches={{6,0},{1}}; request.direction_domain=input; request.response_domain=input;
    auto live=request.primal;
    assert(nf::validate_derivative(op,request,live,input,input)==nf::status::unsupported_derivative);
    op.capabilities |= nf::jvp;
    assert(nf::validate_derivative(op,request,live,input,input)==nf::status::success);
    live.instance.parameters.generation.value++;
    assert(nf::validate_derivative(op,request,live,input,input)==nf::status::stale_generation);
    live=request.primal; live.instance.state.instance.low++;
    assert(nf::validate_derivative(op,request,live,input,input)==nf::status::stale_generation);
    live=request.primal; live.branches.generation.value++;
    assert(nf::validate_derivative(op,request,live,input,input)==nf::status::stale_generation);
    live=request.primal; op.definition.low=99;
    assert(nf::validate_derivative(op,request,live,input,input)==nf::status::stale_generation);
    op.definition.low=1; request.smooth_at_primal=false;
    assert(nf::validate_derivative(op,request,live,input,input)==nf::status::unsupported_derivative);
    request.smooth_at_primal=true; request.convention=nf::derivative_convention::through_rounding;
    assert(nf::validate_derivative(op,request,live,input,input)==nf::status::unsupported_derivative);
    request.convention=nf::derivative_convention::mathematical_at_stored_values; request.direction_scale=0;
    assert(nf::validate_derivative(op,request,live,input,input)==nf::status::invalid_contract);
}
