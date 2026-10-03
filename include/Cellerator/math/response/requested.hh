#pragma once
#include <Cellerator/execution/prepared_host_sweep.hh>
#include <Cellerator/compute/operation/relation_semantics.hh>
#include <algorithm>
#include <vector>
namespace cellerator::math::response {
namespace ph=execution::prepared_host;
namespace nf=ph::nf;
namespace df=ph::df;
namespace nn=ph::nn;
namespace ex=execution;
namespace rel=compute::relation;
enum class status { success,invalid_demand,invalid_binding,invalid_direction,stale_primal,
                    unsupported_capability,unsupported_policy,provider_failure };
enum class argument { state,parameters };
struct demand {
    std::vector<std::uint64_t> outputs,state,parameters;
};
struct trace { std::uint64_t primal_coordinates=0,requested_outputs=0,response_coordinates=0,native_callbacks=0; };
struct descriptor {
    nf::prepared_identity prepared{};
    rel::axis_descriptor state{},parameters{};
    // One canonical coefficient slot for each state coordinate; repeated slots
    // retain all contributions. This is cold metadata, not a parameter master.
    std::span<const std::uint64_t> parameter_slots;
};
template<class T> struct binding {
    std::span<const T> state,parameters;
    std::span<T> expanded_parameters,intermediate,output;
    ex::persistent_axis_identity state_axis{},parameter_axis{};
    const nf::instance_binding* instance=nullptr;
};
template<class T> struct saved_primal { ph::saved_primal native;binding<T> primal; };
template<class T> struct direction {
    argument role=argument::state;
    nf::identity identity{}; // Caller-defined direction identity, separate from primal owner.
    nf::generation_stamp point{};
    std::span<const T> values;
};
template<class T> struct jvp_result { std::vector<T> outputs;trace work; };
template<class T> struct vjp_result { std::vector<T> state,parameters;trace work; };
// Explicit custom extension seam: bind an actual owner's declared response callback
// into the existing runner. No registration, arbitrary differentiation or fallback.
nf::status bind_custom_rule(const nf::compiled_block&,nf::capability,const void* owner,
    std::uint64_t stage_id,std::uint64_t candidate_id,std::uint32_t binding_index,
    ex::program::prepared_stage_v2&) noexcept;

template<class T> class requested_scaled_tanh {
    descriptor descriptor_{};
    std::vector<std::uint64_t> slots_;
    demand demand_;
    ph::scaled_tanh<T> native_;
    std::vector<std::uint64_t> reverse_coordinates_;
    status preparation_=status::invalid_demand;
    static std::size_t position(const std::vector<std::uint64_t>& values,std::uint64_t coordinate){
        auto it=std::find(values.begin(),values.end(),coordinate);return it==values.end()?values.size():it-values.begin();
    }
    static bool indices(const std::vector<std::uint64_t>& values,std::uint64_t count){
        for(std::size_t i=0;i<values.size();++i){if(values[i]>=count)return false;for(std::size_t j=0;j<i;++j)if(values[j]==values[i])return false;}return true;
    }
    template<class A,class B> static bool overlaps(std::span<A> a,std::span<B> b){
        if(a.empty()||b.empty())return false;
        auto x=reinterpret_cast<std::uintptr_t>(a.data()),y=reinterpret_cast<std::uintptr_t>(b.data());
        auto limit=std::numeric_limits<std::uintptr_t>::max();
        if(a.size()>(limit-x)/sizeof(A)||b.size()>(limit-y)/sizeof(B))return true;
        return x<y+b.size_bytes()&&y<x+a.size_bytes();
    }
    bool output_alias(std::span<T> output,const saved_primal<T>& tape) const {
        const auto& p=tape.primal;
        return overlaps(output,p.state)||overlaps(output,p.parameters)||overlaps(output,p.expanded_parameters)
            ||overlaps(output,p.intermediate)||overlaps(output,p.output);
    }
    static ph::sweep_binding<T> native_binding(const binding<T>& b){return {b.state,b.expanded_parameters,b.intermediate,b.output,b.state_axis,b.instance};}
    status current(const saved_primal<T>& tape) const {
        if(preparation_!=status::success)return preparation_;
        if(std::fegetround()!=FE_TONEAREST)return status::unsupported_policy;
        if(!native_.primal_is_current(tape.native,native_binding(tape.primal)))return status::stale_primal;
        if(tape.primal.parameters.size()!=descriptor_.parameters.extent
            ||!nf::same_axis(tape.primal.parameter_axis,descriptor_.parameters.identity))return status::invalid_binding;
        return status::success;
    }
    bool valid_direction(const direction<T>* value,argument role,const saved_primal<T>& tape) const {
        if(!value)return true;
        auto expected=role==argument::state?tape.native.instance.state:tape.native.instance.parameters;
        auto size=role==argument::state?descriptor_.state.extent:descriptor_.parameters.extent;
        return value->role==role&&compute::operation::v2::valid_stable_id(value->identity)
            &&nf::same_stamp(value->point,expected)&&value->values.size()==size&&(!size||value->values.data());
    }
public:
    requested_scaled_tanh(descriptor description,demand requested)
      :descriptor_(description),slots_(description.parameter_slots.begin(),description.parameter_slots.end()),demand_(std::move(requested)),
       native_(description.prepared,description.state.identity,description.state.extent) {
        descriptor_.parameter_slots=slots_;
        if(native_.preparation_status()!=nf::status::success
           ||ex::validate_persistent_axis_identity(description.parameters.identity)!=ex::biological_validation_code::ok
           ||slots_.size()!=description.state.extent||!indices(demand_.outputs,description.state.extent)
           ||!indices(demand_.state,description.state.extent)||!indices(demand_.parameters,description.parameters.extent))return;
        for(auto slot:slots_)if(slot>=description.parameters.extent)return;
        // Canonical traversal fixes coefficient-gradient accumulation order.
        for(std::size_t i=0;i<slots_.size();++i)
            if(position(demand_.outputs,i)!=demand_.outputs.size()
               &&(position(demand_.state,i)!=demand_.state.size()||position(demand_.parameters,slots_[i])!=demand_.parameters.size()))reverse_coordinates_.push_back(i);
        preparation_=status::success;
    }
    requested_scaled_tanh(const requested_scaled_tanh&)=delete;
    requested_scaled_tanh& operator=(const requested_scaled_tanh&)=delete;
    status preparation_status() const noexcept{return preparation_;}
    const ex::program::prepared_program_v2& native_program() const noexcept{return native_.native_program();}
    const demand& requested() const noexcept{return demand_;}
    status forward(const binding<T>& b,saved_primal<T>& tape,void* stream=nullptr) noexcept {
        if(preparation_!=status::success)return preparation_;
        const auto n=descriptor_.state.extent,k=descriptor_.parameters.extent;
        if(stream||native_.forward_attempts()==std::numeric_limits<std::uint64_t>::max()||!b.instance||nf::validate_instance(descriptor_.prepared,*b.instance)!=nf::status::success
            ||!nf::valid_stamp(b.instance->parameters)||std::fegetround()!=FE_TONEAREST
            ||ex::validate_persistent_axis_identity(b.state_axis)!=ex::biological_validation_code::ok
            ||ex::validate_persistent_axis_identity(b.parameter_axis)!=ex::biological_validation_code::ok
            ||!nf::same_axis(b.state_axis,descriptor_.state.identity)||!nf::same_axis(b.parameter_axis,descriptor_.parameters.identity)
            ||b.state.size()!=n||b.parameters.size()!=k||b.expanded_parameters.size()!=n||b.intermediate.size()!=n||b.output.size()!=n
            ||(n&&(!b.state.data()||!b.expanded_parameters.data()||!b.intermediate.data()||!b.output.data()))||(k&&!b.parameters.data())
            ||overlaps(b.expanded_parameters,b.state)||overlaps(b.expanded_parameters,b.parameters)||overlaps(b.expanded_parameters,b.intermediate)
            ||overlaps(b.expanded_parameters,b.output)||overlaps(b.intermediate,b.state)||overlaps(b.intermediate,b.parameters)
            ||overlaps(b.output,b.state)||overlaps(b.output,b.parameters)||overlaps(b.output,b.intermediate))return status::invalid_binding;
        for(std::size_t i=0;i<n;++i)b.expanded_parameters[i]=b.parameters[slots_[i]];
        saved_primal<T> next;next.primal=b;
        if(native_.forward(native_binding(b),&next.native)!=ex::program::program_status::success)return status::provider_failure;
        tape=next;return status::success;
    }
    status jvp(const saved_primal<T>& tape,const direction<T>* state_direction,const direction<T>* parameter_direction,jvp_result<T>& out) const noexcept {
        auto s=current(tape);if(s!=status::success)return s;
        if((!state_direction&&!parameter_direction)||!valid_direction(state_direction,argument::state,tape)
            ||!valid_direction(parameter_direction,argument::parameters,tape))return status::invalid_direction;
        if(output_alias(std::span<T>(out.outputs),tape)
           ||(state_direction&&overlaps(std::span<T>(out.outputs),state_direction->values))
           ||(parameter_direction&&overlaps(std::span<T>(out.outputs),parameter_direction->values)))return status::invalid_binding;
        try {
            jvp_result<T> next;next.outputs.resize(demand_.outputs.size());
            next.work={descriptor_.state.extent,demand_.outputs.size(),demand_.outputs.size(),2*demand_.outputs.size()};
            for(std::size_t q=0;q<demand_.outputs.size();++q){auto i=demand_.outputs[q];
                T dx=state_direction?state_direction->values[i]:T{},dp=parameter_direction?parameter_direction->values[slots_[i]]:T{},dz{},dy{};
                df::local_binding<T> multiply{};multiply.left=tape.primal.state.subspan(i,1);multiply.right=tape.primal.expanded_parameters.subspan(i,1);
                multiply.left_direction={&dx,1};multiply.right_direction={&dp,1};multiply.output={&dz,1};
                if(df::local_jvp(nn::local_operation::multiply,multiply)!=nn::local_status::success)return status::provider_failure;
                df::local_binding<T> tanh{};tanh.left=tape.primal.intermediate.subspan(i,1);tanh.left_direction={&dz,1};tanh.output={&dy,1};
                if(df::local_jvp(nn::local_operation::tanh,tanh)!=nn::local_status::success)return status::provider_failure;
                next.outputs[q]=dy;
            }
            out=std::move(next);return status::success;
        }catch(...){return status::provider_failure;}
    }
    status vjp(const saved_primal<T>& tape,std::span<const T> cotangent,vjp_result<T>& out) const noexcept {
        auto s=current(tape);if(s!=status::success)return s;
        if(cotangent.size()!=demand_.outputs.size()||(!cotangent.empty()&&!cotangent.data()))return status::invalid_binding;
        if(output_alias(std::span<T>(out.state),tape)||output_alias(std::span<T>(out.parameters),tape)
           ||overlaps(std::span<T>(out.state),cotangent)||overlaps(std::span<T>(out.parameters),cotangent))return status::invalid_binding;
        try {
            vjp_result<T> next;next.state.resize(demand_.state.size());next.parameters.resize(demand_.parameters.size());
            next.work={descriptor_.state.extent,demand_.outputs.size(),reverse_coordinates_.size(),2*reverse_coordinates_.size()};
            for(auto i:reverse_coordinates_){T dz{},dx{},dp{};auto q=position(demand_.outputs,i);
                df::local_binding<T> tanh{};tanh.left=tape.primal.intermediate.subspan(i,1);tanh.cotangent=cotangent.subspan(q,1);tanh.left_adjoint={&dz,1};
                if(df::local_vjp(nn::local_operation::tanh,tanh)!=nn::local_status::success)return status::provider_failure;
                df::local_binding<T> multiply{};multiply.left=tape.primal.state.subspan(i,1);multiply.right=tape.primal.expanded_parameters.subspan(i,1);
                multiply.cotangent={&dz,1};multiply.left_adjoint={&dx,1};multiply.right_adjoint={&dp,1};
                if(df::local_vjp(nn::local_operation::multiply,multiply)!=nn::local_status::success)return status::provider_failure;
                auto state_slot=position(demand_.state,i),parameter_slot=position(demand_.parameters,slots_[i]);
                if(state_slot<next.state.size())next.state[state_slot]+=dx;
                if(parameter_slot<next.parameters.size())next.parameters[parameter_slot]+=dp;
            }
            out=std::move(next);return status::success;
        }catch(...){return status::provider_failure;}
    }
};
} // namespace cellerator::math::response
