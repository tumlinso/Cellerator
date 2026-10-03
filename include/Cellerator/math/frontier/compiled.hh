#pragma once
#include <Cellerator/math/frontier/operators.hh>
namespace cellerator::math::frontier {
namespace nf=matrix::nf;
namespace pg=execution::program;
// Typed payload behind launch_binding_v2::input. Every pointer is borrowed
// through synchronous launch. Responses use the native owner's generation guard.
struct polynomial_binding {
    polynomial_primal primal{};
    polynomial_result* forward=nullptr;
    const polynomial_tape* tape=nullptr;
    const Matrix *dX=nullptr,*dL=nullptr,*dR=nullptr,*dM=nullptr,*cotangent=nullptr;
    Matrix* tangent=nullptr;
    polynomial_adjoints* adjoints=nullptr;
};
struct polynomial_block { square_axes axes{};nf::compiled_block block{}; };
template<nf::capability Action> inline pg::program_status polynomial_launch(const void* owner,
        const pg::launch_binding_v2& launch,void* stream) noexcept {
    if(!owner || !launch.input || launch.output || launch.values || stream)
        return pg::program_status::invalid_argument;
    const auto& block=*static_cast<const polynomial_block*>(owner);
    const auto& b=*static_cast<const polynomial_binding*>(launch.input);
    status result=status::invalid_binding;
    if constexpr(Action==nf::forward) {
        if(b.forward)result=polynomial_forward(block.axes,b.primal,*b.forward);
    } else {
        if(!b.tape || !nf::same_axis(b.tape->axes.rows.identity,block.axes.rows.identity)
            || !nf::same_axis(b.tape->axes.columns.identity,block.axes.columns.identity)
            || b.tape->axes.rows.extent!=block.axes.rows.extent
            || b.tape->axes.columns.extent!=block.axes.columns.extent)
            return pg::program_status::invalid_argument;
        if constexpr(Action==nf::jvp) {
            if(b.dX&&b.dL&&b.dR&&b.dM&&b.tangent)
                result=polynomial_jvp(*b.tape,*b.dX,*b.dL,*b.dR,*b.dM,*b.tangent);
        } else if(b.cotangent&&b.adjoints)result=polynomial_vjp(*b.tape,*b.cotangent,*b.adjoints);
    }
    return result==status::success?pg::program_status::success:pg::program_status::invalid_argument;
}
// Caller owns contract spans through preparation and launch. Only finite FP64
// nearest-even propagate policies and four square operand roles are supported.
// Native frontier kernels enforce finite inputs and borrow all tape dependencies.
inline nf::status make_polynomial_block(square_axes axes,const nf::operation_contract& contract,
        polynomial_block& out) noexcept {
    auto checked=nf::validate_operation(contract);if(checked!=nf::status::success)return checked;
    if(!matrix::valid_axis(axes.rows)||!matrix::valid_axis(axes.columns)
        || !axes.rows.extent || axes.rows.extent!=axes.columns.extent
        || axes.rows.extent>UINT64_MAX/axes.rows.extent)return nf::status::axis_mismatch;
    const auto& p=contract.numeric;using execution::numeric_type;
    if(contract.capabilities!=(nf::forward|nf::jvp|nf::vjp)
        ||p.relation_storage!=numeric_type::f64||p.state_storage!=numeric_type::f64
        ||p.multiply!=numeric_type::f64||p.accumulation!=numeric_type::f64
        ||p.output_storage!=numeric_type::f64||p.scalar!=numeric_type::f64
        ||p.rounding!=compute::operation::v2::rounding_policy::nearest_even
        ||p.saturation!=compute::operation::v2::saturation_policy::none
        ||p.nan!=compute::operation::v2::nan_policy::propagate
        ||p.infinity!=compute::operation::v2::infinity_policy::propagate)
        return nf::status::unsupported_capability;
    auto matches=[&](const nf::operand_signature& a){return a.storage==numeric_type::f64
        &&a.element_count==axes.rows.extent*axes.columns.extent&&a.axes.size()==2
        &&nf::same_axis(a.axes[0],axes.rows.identity)&&nf::same_axis(a.axes[1],axes.columns.identity);};
    if(contract.arguments.size()!=4||contract.outputs.size()!=1)return nf::status::invalid_contract;
    for(const auto& arg:contract.arguments)if(!matches(arg))return nf::status::axis_mismatch;
    const auto& output=contract.outputs[0];
    if(!matches(output.operand))return nf::status::axis_mismatch;
    if(output.effect.update!=execution::output_update_kind::overwrite||output.effect.input_output_aliasing_legal)
        return nf::status::invalid_effect;
    polynomial_block next;next.axes=axes;next.block.contract=contract;
    next.block.effects={true,true,true,false}; // Native matrices allocate; no allocation-free memoization claim.
    next.block.forward_launch=polynomial_launch<nf::forward>;
    next.block.jvp_launch=polynomial_launch<nf::jvp>;
    next.block.vjp_launch=polynomial_launch<nf::vjp>;
    checked=nf::validate_compiled_block(next.block);if(checked==nf::status::success)out=next;
    return checked;
}
} // namespace cellerator::math::frontier
