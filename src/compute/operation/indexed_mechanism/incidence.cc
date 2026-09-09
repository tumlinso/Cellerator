#include <Cellerator/compute/operation/indexed_mechanism/incidence.hh>
#include <limits>
namespace cellerator::compute::operation::indexed {
namespace {
bool axes_valid(std::span<const indexed_axis> axes) {
    for(std::size_t i=0;i<axes.size();++i) {
        if(execution::validate_persistent_axis_identity(axes[i].identity)!=execution::biological_validation_code::ok)return false;
        for(std::size_t j=0;j<i;++j)if(nf1::same_axis(axes[i].identity,axes[j].identity))return false;
    }return true;
}
bool overlaps(std::span<const double> a,std::span<double> b) {
    if(a.empty()||b.empty())return false;
    auto x=reinterpret_cast<std::uintptr_t>(a.data()),y=reinterpret_cast<std::uintptr_t>(b.data());
    const auto max=std::numeric_limits<std::uintptr_t>::max();
    if(a.size()>(max-x)/sizeof(double)||b.size()>(max-y)/sizeof(double))return true;
    return x<y+b.size_bytes()&&y<x+a.size_bytes();
}
}
incidence_status argument_incidence::prepare(std::span<const indexed_axis> inputs,
        std::span<const indexed_axis> outputs,std::span<const mechanism_incidence> mechanisms) noexcept {
    if(!axes_valid(inputs)||!axes_valid(outputs))return incidence_status::invalid_axis;
    try {
        argument_incidence next;next.inputs_.assign(inputs.begin(),inputs.end());next.outputs_.assign(outputs.begin(),outputs.end());
        next.mechanisms_.reserve(mechanisms.size());
        for(const auto& mechanism:mechanisms) {
            if(!v2::valid_stable_id(mechanism.instance)||!v2::valid_stable_id(mechanism.evaluator))return incidence_status::invalid_identity;
            if(mechanism.outputs.empty())return incidence_status::invalid_binding;
            for(const auto& previous:next.mechanisms_)if(v2::same_stable_id(mechanism.instance,previous.instance))return incidence_status::invalid_identity;
            prepared_incidence prepared;prepared.instance=mechanism.instance;prepared.evaluator=mechanism.evaluator;
            prepared.arguments.resize(mechanism.arguments.size());prepared.outputs.resize(mechanism.outputs.size());
            std::vector<bool> seen(mechanism.arguments.size(),false);
            for(const auto& arg:mechanism.arguments) {
                if(arg.slot>=seen.size()||seen[arg.slot])return incidence_status::invalid_slot;seen[arg.slot]=true;
                if(!v2::valid_stable_id(arg.role))return incidence_status::invalid_identity;
                if(arg.axis>=inputs.size()||arg.index>=inputs[arg.axis].extent)return incidence_status::invalid_index;
                prepared.arguments[arg.slot]=arg;
            }
            seen.assign(mechanism.outputs.size(),false);
            for(const auto& out:mechanism.outputs) {
                if(out.slot>=seen.size()||seen[out.slot])return incidence_status::invalid_slot;seen[out.slot]=true;
                if(!v2::valid_stable_id(out.role)||!v2::valid_stable_id(out.assembly_owner))return incidence_status::invalid_identity;
                if(out.axis>=outputs.size()||out.index>=outputs[out.axis].extent)return incidence_status::invalid_index;
                if(!execution::valid_output_effect_contract(out.effect))return incidence_status::invalid_effect;
                prepared.outputs[out.slot]=out;
            }
            next.mechanisms_.push_back(std::move(prepared));
        }
        // Shared destinations require an explicit common assembly owner and sum
        // effect. Affine/partial-write duplication has no defined ordering here.
        for(std::size_t i=0;i<next.mechanisms_.size();++i)for(std::size_t j=0;j<next.mechanisms_[i].outputs.size();++j) {
            const auto& a=next.mechanisms_[i].outputs[j];
            for(std::size_t k=0;k<=i;++k)for(std::size_t l=0;l<next.mechanisms_[k].outputs.size();++l) {
                if(k==i&&l>=j)break;
                const auto& b=next.mechanisms_[k].outputs[l];
                if(a.axis==b.axis&&a.index==b.index &&
                    (!v2::same_stable_id(a.assembly_owner,b.assembly_owner)||
                     a.effect.update!=execution::output_update_kind::accumulate||b.effect.update!=execution::output_update_kind::accumulate))
                    return incidence_status::duplicate_writer;
            }
        }
        *this=std::move(next);return incidence_status::success;
    }catch(...){return incidence_status::allocation_failure;}
}
incidence_status argument_incidence::gather_f64(std::uint64_t instance,
        std::span<const host_axis_f64> bindings,std::span<double> arguments) const noexcept {
    if(instance>=mechanisms_.size()||bindings.size()!=inputs_.size())return incidence_status::invalid_binding;
    const auto& mechanism=mechanisms_[instance];
    if(arguments.size()!=mechanism.arguments.size()||(!arguments.empty()&&!arguments.data()))return incidence_status::invalid_binding;
    for(std::size_t i=0;i<bindings.size();++i) {
        if(execution::validate_persistent_axis_identity(bindings[i].identity)!=execution::biological_validation_code::ok ||
            !nf1::same_axis(bindings[i].identity,inputs_[i].identity)||bindings[i].values.size()!=inputs_[i].extent||
            (!bindings[i].values.empty()&&!bindings[i].values.data())||overlaps(bindings[i].values,arguments))return incidence_status::invalid_binding;
    }
    for(const auto& arg:mechanism.arguments)arguments[arg.slot]=bindings[arg.axis].values[arg.index];
    return incidence_status::success;
}
} // namespace cellerator::compute::operation::indexed
