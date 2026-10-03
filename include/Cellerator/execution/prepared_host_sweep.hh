#pragma once
#include <Cellerator/compute/operation/differential/local_arithmetic.hh>
#include <array>
#include <cfenv>
#include <limits>
#include <type_traits>

namespace cellerator::execution::prepared_host {
namespace nf = compute::operation::nf1;
namespace df = compute::differential;
namespace nn = compute::native_numeric;
namespace pg = execution::program;

// Fixed borrowed-buffer host composition: output[i] = tanh(state[i]*parameters[i]).
// Coordinates and physical output order are preserved; no conversion or allocation.
template<class T> struct sweep_binding {
    std::span<const T> state, parameters;
    std::span<T> intermediate, output;
    persistent_axis_identity axis{};
    const nf::instance_binding* instance = nullptr;
};
struct saved_primal {
    const void* program = nullptr;
    std::uint64_t run = 0;
    nf::instance_binding instance{};
    const nf::instance_binding* instance_owner = nullptr;
    const void* state = nullptr;
    const void* parameters = nullptr;
    const void* intermediate = nullptr;
    const void* output = nullptr;
};

template<class T> class scaled_tanh {
    static_assert(std::is_same_v<T,float> || std::is_same_v<T,double>);
    struct stage_owner {
        std::array<nf::operand_signature,2> inputs{};
        nf::output_signature output{};
        df::local_block block{};
    };
    nf::prepared_identity identity_{};
    persistent_axis_identity axis_{};
    std::size_t extent_{};
    std::array<stage_owner,2> owners_{};
    std::array<pg::prepared_stage_v2,2> stages_{};
    std::array<std::uint64_t,1> dependencies_{0};
    pg::prepared_program_v2 graph_{};
    nf::status preparation_ = nf::status::invalid_contract;
    std::uint64_t run_ = 0;

    template<class A,class B> static bool overlap(std::span<A> a,std::span<B> b) noexcept {
        if (a.empty() || b.empty()) return false;
        auto x=reinterpret_cast<std::uintptr_t>(a.data()), y=reinterpret_cast<std::uintptr_t>(b.data());
        auto m=std::numeric_limits<std::uintptr_t>::max();
        if (a.size()>(m-x)/sizeof(A) || b.size()>(m-y)/sizeof(B)) return true;
        return x<y+b.size_bytes() && y<x+a.size_bytes();
    }
    template<class A> bool capacity(std::span<A> a) const noexcept {
        return a.size()==extent_ && (!extent_ || a.data());
    }
public:
    scaled_tanh(nf::prepared_identity identity,persistent_axis_identity axis,std::size_t extent)
      : identity_(identity),axis_(axis),extent_(extent) {
        const auto type=std::is_same_v<T,float>?numeric_type::f32:numeric_type::f64;
        for (std::size_t i=0;i<2;++i) {
            auto& owner=owners_[i];
            owner.inputs={nf::operand_signature{{1,1},{&axis_,1},extent,type},
                          nf::operand_signature{{2,1},{&axis_,1},extent,type}};
            owner.output.operand={{3,1},{&axis_,1},extent,type};
            owner.output.assembly_owner={4,1};
            nf::operation_contract contract{};
            contract.definition=identity.definition;
            contract.arguments={owner.inputs.data(),i==0?2u:1u};
            contract.outputs={&owner.output,1};
            contract.numeric={type,type,type,type,type,type};
            preparation_=df::make_local_block(i==0?nn::local_operation::multiply:nn::local_operation::tanh,
                                              contract,owner.block);
            if (preparation_!=nf::status::success) return;
            preparation_=nf::bind_compiled_stage(owner.block.block,nf::forward,&owner.block,i+1,i+1,
                                                 static_cast<std::uint32_t>(i),stages_[i]);
            if (preparation_!=nf::status::success) return;
        }
        stages_[1].dependency_count=1;
        graph_={2,0,stages_.data(),2,dependencies_.data(),1};
    }
    scaled_tanh(const scaled_tanh&)=delete;
    scaled_tanh& operator=(const scaled_tanh&)=delete;
    scaled_tanh(scaled_tanh&&)=delete; // Native contracts borrow stable owner addresses.
    scaled_tanh& operator=(scaled_tanh&&)=delete;
    nf::status preparation_status() const noexcept { return preparation_; }
    const pg::prepared_program_v2& native_program() const noexcept { return graph_; }
    std::uint64_t forward_attempts() const noexcept { return run_; }
    std::size_t required_elements() const noexcept { return extent_; }
    pg::program_status forward(const sweep_binding<T>& b,saved_primal* saved=nullptr,
                               void* stream=nullptr) noexcept {
        // Entire fixed composition admits before either native callback writes.
        if (preparation_!=nf::status::success || stream || !b.instance
            || nf::validate_instance(identity_,*b.instance)!=nf::status::success
            || !nf::valid_stamp(b.instance->parameters)
            || validate_persistent_axis_identity(b.axis)!=biological_validation_code::ok
            || !nf::same_axis(axis_,b.axis)
            || std::fegetround()!=FE_TONEAREST || run_==std::numeric_limits<std::uint64_t>::max()
            || !capacity(b.state) || !capacity(b.parameters) || !capacity(b.intermediate) || !capacity(b.output)
            || overlap(b.intermediate,b.state) || overlap(b.intermediate,b.parameters)
            || overlap(b.output,b.state) || overlap(b.output,b.parameters) || overlap(b.output,b.intermediate))
            return pg::program_status::invalid_argument;
        // Every accepted forward attempt invalidates previous borrowed saved primals.
        ++run_;
        std::array<df::local_binding<T>,2> numeric{};
        numeric[0].left=b.state; numeric[0].right=b.parameters; numeric[0].output=b.intermediate;
        numeric[1].left=b.intermediate; numeric[1].output=b.output;
        std::array<pg::launch_binding_v2,2> bindings{};
        bindings[0].input=&numeric[0]; bindings[1].input=&numeric[1];
        auto result=pg::execute_prepared_program_v2(graph_,bindings.data(),2,nullptr);
        if (result==pg::program_status::success && saved)
            *saved={this,run_,*b.instance,b.instance,b.state.data(),b.parameters.data(),
                    b.intermediate.data(),b.output.data()};
        return result;
    }
    bool primal_is_current(const saved_primal& saved,const sweep_binding<T>& binding) const noexcept {
        if (!binding.instance || binding.instance!=saved.instance_owner
            || binding.state.data()!=saved.state || binding.parameters.data()!=saved.parameters
            || binding.intermediate.data()!=saved.intermediate || binding.output.data()!=saved.output
            || !capacity(binding.state) || !capacity(binding.parameters)
            || !capacity(binding.intermediate) || !capacity(binding.output)
            || validate_persistent_axis_identity(binding.axis)!=biological_validation_code::ok
            || !nf::same_axis(axis_,binding.axis)) return false;
        const auto& live=*binding.instance;
        return saved.program==this && saved.run!=0 && saved.run==run_
            && nf::validate_instance(identity_,live)==nf::status::success
            && nf::same_preparation(saved.instance.prepared,live.prepared)
            && nf::same_stamp(saved.instance.state,live.state)
            && nf::same_stamp(saved.instance.parameters,live.parameters)
            && compute::operation::v2::same_stable_id(saved.instance.parameter_tie_group,live.parameter_tie_group);
    }
};
} // namespace cellerator::execution::prepared_host
