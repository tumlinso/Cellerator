#include <Cellerator/compute/operation/differential/local_arithmetic.hh>
#include <array>
#include <cfenv>
#include <limits>

namespace cellerator::compute::differential {
namespace {
using status = numeric::local_status;
using op = numeric::local_operation;
namespace pg = execution::program;
template<class T> bool valid_span(std::span<T> span, std::size_t count) noexcept {
    return span.size() == count && (!count || span.data());
}
template<class A, class B> bool overlap(std::span<A> a, std::span<B> b) noexcept {
    if (a.empty() || b.empty()) return false;
    const auto x = reinterpret_cast<std::uintptr_t>(a.data());
    const auto y = reinterpret_cast<std::uintptr_t>(b.data());
    const auto max = std::numeric_limits<std::uintptr_t>::max();
    if (a.size() > (max-x)/sizeof(A) || b.size() > (max-y)/sizeof(B)) return true;
    return x < y + b.size_bytes() && y < x + a.size_bytes();
}
template<class T> status validate(op operation, const local_binding<T>& b, bool reverse) noexcept {
    const auto arity = numeric::local_arity(operation);
    if (!arity) return status::unsupported_operation;
    if (std::fegetround() != FE_TONEAREST) return status::unsupported_policy;
    const auto count = b.left.size();
    if (!valid_span(b.left,count) || !valid_span(b.right,arity == 2 ? count : 0)) return status::invalid_binding;
    if (reverse) {
        if (!valid_span(b.cotangent,count) || !valid_span(b.left_adjoint,count) ||
            !valid_span(b.right_adjoint,arity == 2 ? count : 0) ||
            overlap(b.left_adjoint,b.right_adjoint)) return status::invalid_binding;
    } else if (!valid_span(b.left_direction,count) ||
        !valid_span(b.right_direction,arity == 2 ? count : 0) ||
        !valid_span(b.output,count)) return status::invalid_binding;
    const std::array<std::span<const T>,5> reads{b.left,b.right,b.left_direction,b.right_direction,b.cotangent};
    for (auto read : reads) {
        if (reverse ? (overlap(read,b.left_adjoint) || overlap(read,b.right_adjoint)) : overlap(read,b.output))
            return status::invalid_binding;
    }
    return status::success;
}
template<class T> void partials(op operation, T a, T b, T& da, T& db) noexcept {
    switch (operation) {
    case op::add: da=T{1}; db=T{1}; break;
    case op::multiply: da=b; db=a; break;
    case op::tanh: {
        T primal{};
        numeric::detail::local_value_nearest(operation,a,T{},primal);
        da=T{1}-primal*primal; db=T{}; break;
    }
    }
}
template<class T> status action(op operation, const local_binding<T>& b, bool reverse) noexcept {
    const auto checked=validate(operation,b,reverse);
    if (checked!=status::success) return checked;
    const bool binary=numeric::local_arity(operation)==2;
    for (std::size_t i=0; i<b.left.size(); ++i) {
        T left{},right{};
        partials(operation,b.left[i],binary?b.right[i]:T{},left,right);
        if (reverse) {
            b.left_adjoint[i]=left*b.cotangent[i];
            if (binary) b.right_adjoint[i]=right*b.cotangent[i];
        } else {
            b.output[i]=left*b.left_direction[i];
            if (binary) b.output[i]+=right*b.right_direction[i];
        }
    }
    return status::success;
}
template<class T> pg::program_status execute(const local_block& block,
        const pg::launch_binding_v2& launch, nf1::capability capability) noexcept {
    const auto& b=*static_cast<const local_binding<T>*>(launch.input);
    const auto count=block.block.contract.outputs[0].operand.element_count;
    if (b.left.size()!=count) return pg::program_status::invalid_argument;
    status result;
    if (capability==nf1::forward)
        result=numeric::local_forward(block.operation,b.left,b.right,b.output);
    else result=action(block.operation,b,capability==nf1::vjp);
    return result==status::success ? pg::program_status::success : pg::program_status::invalid_argument;
}
template<nf1::capability Capability> pg::program_status callback(const void* state,
        const pg::launch_binding_v2& binding, void* stream) noexcept {
    if (!state || !binding.input || binding.output || binding.values || stream)
        return pg::program_status::invalid_argument;
    const auto& block=*static_cast<const local_block*>(state);
    if (!(block.block.contract.capabilities & Capability)) return pg::program_status::invalid_argument;
    if (block.block.contract.numeric.state_storage==execution::numeric_type::f32)
        return execute<float>(block,binding,Capability);
    return execute<double>(block,binding,Capability);
}
}
status local_jvp(op operation,const local_binding<float>& binding) noexcept { return action(operation,binding,false); }
status local_jvp(op operation,const local_binding<double>& binding) noexcept { return action(operation,binding,false); }
status local_vjp(op operation,const local_binding<float>& binding) noexcept { return action(operation,binding,true); }
status local_vjp(op operation,const local_binding<double>& binding) noexcept { return action(operation,binding,true); }
nf1::status make_local_block(op operation,const nf1::operation_contract& contract,local_block& output) noexcept {
    auto checked=nf1::validate_operation(contract);
    if (checked!=nf1::status::success) return checked;
    const auto arity=numeric::local_arity(operation);
    if (!arity || contract.arguments.size()!=arity || contract.outputs.size()!=1 ||
        (contract.capabilities & nf1::second_direction)) return nf1::status::unsupported_capability;
    const auto type=contract.numeric.state_storage;
    const auto& n=contract.numeric;
    namespace v2=operation::v2;
    if ((type!=execution::numeric_type::f32 && type!=execution::numeric_type::f64) ||
        n.relation_storage!=type || n.multiply!=type || n.accumulation!=type ||
        n.output_storage!=type || n.scalar!=type || n.rounding!=v2::rounding_policy::nearest_even ||
        n.saturation!=v2::saturation_policy::none || n.nan!=v2::nan_policy::propagate ||
        n.infinity!=v2::infinity_policy::propagate) return nf1::status::unsupported_capability;
    const auto& result=contract.outputs[0];
    if (result.operand.storage!=type || result.effect.update!=execution::output_update_kind::overwrite ||
        result.effect.input_output_aliasing_legal) return nf1::status::invalid_effect;
    for (const auto& input:contract.arguments) {
        if (input.storage!=type || input.element_count!=result.operand.element_count ||
            input.axes.size()!=result.operand.axes.size()) return nf1::status::axis_mismatch;
        for (std::size_t i=0;i<input.axes.size();++i)
            if (!nf1::same_axis(input.axes[i],result.operand.axes[i])) return nf1::status::axis_mismatch;
    }
    local_block candidate{};candidate.operation=operation;candidate.block.contract=contract;
    candidate.block.forward_launch=callback<nf1::forward>;
    if (contract.capabilities & nf1::jvp) candidate.block.jvp_launch=callback<nf1::jvp>;
    if (contract.capabilities & nf1::vjp) candidate.block.vjp_launch=callback<nf1::vjp>;
    checked=nf1::validate_compiled_block(candidate.block);
    if (checked==nf1::status::success) output=candidate;
    return checked;
}
} // namespace cellerator::compute::differential
