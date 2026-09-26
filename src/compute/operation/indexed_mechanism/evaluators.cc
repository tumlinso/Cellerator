#include <Cellerator/compute/operation/indexed_mechanism/evaluators.hh>
#include <limits>
namespace cellerator::compute::operation::indexed {
namespace {
bool signature_matches(const nf1::operand_signature& signature,const argument_index& arg,
        const argument_incidence& incidence) {
    return v2::same_stable_id(signature.role,arg.role) && signature.element_count==1 &&
        signature.axes.size()==1 && nf1::same_axis(signature.axes[0],incidence.inputs()[arg.axis].identity);
}
bool matches(const nf1::operation_contract& contract,const prepared_incidence& mechanism,
        const argument_incidence& incidence) {
    if(!v2::same_stable_id(contract.definition,mechanism.evaluator)||
        contract.arguments.size()!=mechanism.arguments.size()||contract.outputs.size()!=mechanism.outputs.size())return false;
    for(std::size_t i=0;i<contract.arguments.size();++i)
        if(!signature_matches(contract.arguments[i],mechanism.arguments[i],incidence))return false;
    for(std::size_t i=0;i<contract.outputs.size();++i) {
        const auto& expected=contract.outputs[i];const auto& actual=mechanism.outputs[i];
        if(!v2::same_stable_id(expected.operand.role,actual.role)||expected.operand.element_count!=1||
            expected.operand.axes.size()!=1||!nf1::same_axis(expected.operand.axes[0],incidence.outputs()[actual.axis].identity)||
            !v2::same_stable_id(expected.assembly_owner,actual.assembly_owner)||
            expected.effect.update!=actual.effect.update||
            expected.effect.requires_initialized_destination!=actual.effect.requires_initialized_destination||
            expected.effect.input_output_aliasing_legal!=actual.effect.input_output_aliasing_legal||
            expected.effect.input_scale_binding_id!=actual.effect.input_scale_binding_id||
            expected.effect.destination_scale_binding_id!=actual.effect.destination_scale_binding_id)return false;
    }
    return true;
}
}
evaluator_status evaluator_catalogue::add(const evaluator_registration& entry) noexcept {
    if(!entry.candidate_id||nf1::validate_compiled_block(entry.block)!=nf1::status::success ||
        !entry.block.effects.dependencies_known||!entry.block.effects.writes_only_declared_outputs||
        !entry.block.effects.allocation_free_launch)return evaluator_status::invalid_contract;
    for(const auto& old:entries_)if(v2::same_stable_id(old.block.contract.definition,entry.block.contract.definition)&&
        old.block.contract.arguments.size()==entry.block.contract.arguments.size()&&
        old.block.contract.outputs.size()==entry.block.contract.outputs.size())return evaluator_status::duplicate_registration;
    try {entries_.push_back(entry);return evaluator_status::success;}catch(...){return evaluator_status::allocation_failure;}
}
evaluator_status grouped_evaluators::prepare(const argument_incidence& incidence,
        const evaluator_catalogue& catalogue,nf1::capability action) noexcept {
    try {
        grouped_evaluators next;std::vector<std::size_t> selected;
        std::size_t previous=std::numeric_limits<std::size_t>::max();
        for(std::size_t i=0;i<incidence.mechanisms().size();++i) {
            const auto& mechanism=incidence.mechanisms()[i];
            auto found=catalogue.entries().size();bool identity_found=false;
            for(std::size_t j=0;j<catalogue.entries().size();++j) {
                const auto& contract=catalogue.entries()[j].block.contract;
                identity_found|=v2::same_stable_id(contract.definition,mechanism.evaluator);
                if(matches(contract,mechanism,incidence)){found=j;break;}
            }
            if(found==catalogue.entries().size())return identity_found?evaluator_status::incompatible_incidence:evaluator_status::missing_evaluator;
            if(found!=previous) {
                next.groups_.push_back({&incidence,catalogue.entries()[found].provider_state,{}});
                selected.push_back(found);previous=found;
            }
            next.groups_.back().mechanism_indices.push_back(i);
        }
        if(next.groups_.size()>std::numeric_limits<std::uint32_t>::max())return evaluator_status::invalid_contract;
        next.stages_.resize(next.groups_.size());
        next.dependencies_.resize(next.groups_.empty()?0:next.groups_.size()-1);
        for(std::size_t i=0;i<next.groups_.size();++i) {
            const auto& entry=catalogue.entries()[selected[i]];
            const auto bound=nf1::bind_compiled_stage(entry.block,action,&next.groups_[i],i+1,entry.candidate_id,
                static_cast<std::uint32_t>(i),next.stages_[i]);
            if(bound!=nf1::status::success)return evaluator_status::unsupported_action;
            if(i){next.dependencies_[i-1]=i-1;next.stages_[i].first_dependency=i-1;next.stages_[i].dependency_count=1;}
        }
        *this=std::move(next);return evaluator_status::success;
    }catch(...){return evaluator_status::allocation_failure;}
}
execution::program::prepared_program_v2 grouped_evaluators::program() const noexcept {
    return {2,0,stages_.data(),stages_.size(),dependencies_.data(),dependencies_.size()};
}
} // namespace cellerator::compute::operation::indexed
