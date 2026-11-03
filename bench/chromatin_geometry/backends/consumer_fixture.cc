#include "consumer_fixture.hh"
#include "../reference/reference.hh"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>

namespace cgs {
ProblemView ConsumerHostFixture::problem() const {
    ProblemView p{};
    p.topology={groups.data(),groups.size(),arguments.data(),arguments.size(),outputs.data(),outputs.size(),coordinates,conditions,parameters};
    p.batch=batch;p.dtype=dtype;p.x=dtype==DType::fp16?static_cast<const void*>(x16.data()):x32.data();
    p.theta=theta.data();p.weights=dtype==DType::fp16?static_cast<const void*>(weights16.data()):weights32.data();
    p.weight_offsets=weight_offsets.data();p.group_weight_offsets=group_weight_offsets.data();
    p.weight_count=weights32.size();p.group_weight_count=group_weight_offsets.empty()?0:group_weight_offsets.back();
    p.neighborhoods={neighborhood_group_offsets.data(),neighborhood_groups.data(),neighborhood_feature_offsets.data(),
        neighborhood_features.data(),feature_group_masks.data(),argument_local_indices.data(),
        neighborhood_group_offsets.empty()?0:neighborhood_group_offsets.size()-1,neighborhood_groups.size(),
        neighborhood_features.size(),max_groups,max_features,max_group_weights};
    return p;
}
GateView ConsumerHostFixture::gates() const {
    GateView g{};g.profile=profile;g.probabilities=probabilities.data();g.admitted=admitted.data();
    g.primal_words=primal.data();g.derivative_words=derivative.data();g.words_per_condition=(batch+31)/32;g.revisions=revisions;return g;
}

ConsumerHostFixture make_consumer_fixture(const WorkloadFixture& workload,BodyKind body,DType dtype,GateProfile profile) {
    ConsumerHostFixture f;f.body=body;f.dtype=dtype;f.profile=profile;
    f.batch=workload.spec.cells;f.coordinates=workload.spec.features;
    f.conditions=workload.edge_roots.empty()?workload.programs.size():workload.edge_roots.size();
    f.parameters=(workload.spec.groups+1)/2; // adjacent groups explicitly share numerical parameters
    if(!f.conditions && workload.spec.groups)throw std::invalid_argument("consumer fixture has no conditions");
    f.theta.resize(f.parameters);for(std::size_t j=0;j<f.theta.size();++j)f.theta[j]=0.25f+float(j%7)*0.03125f;
    f.x32.resize(f.batch*f.coordinates);f.x16.resize(f.x32.size());f.dy.resize(f.x32.size());
    for(std::size_t j=0;j<f.x32.size();++j){
        f.x32[j]=0.75f+float(workload_mix(j+workload.spec.seed)%17)*0.015625f;
        if(j%97==0)f.x32[j]=0.f;
        f.x16[j]=reference::float_to_half_bits(f.x32[j]);
        f.dy[j]=float(int(workload_mix(j+77)%17)-8)*0.03125f;
    }
    f.group_weight_offsets.push_back(0);
    std::vector<std::pair<std::size_t,std::size_t>> parameter_shapes(f.parameters,{0,0});
    for(std::uint64_t gid=0;gid<workload.spec.groups;++gid){
        const auto ab=workload.group_feature_offsets[gid],ae=workload.group_feature_offsets[gid+1];
        const auto ob=workload.group_output_offsets[gid],oe=workload.group_output_offsets[gid+1];
        const auto I=body==BodyKind::grouped_linear?std::min<std::uint64_t>(8,ae-ab):ae-ab;
        const auto O=body==BodyKind::grouped_linear?std::min<std::uint64_t>(4,oe-ob):oe-ob;
        cellerator::ops::mechanism::GatedProductGroup g{f.arguments.size(),I,f.outputs.size(),O,gid/2,workload.group_program[gid],gid};
        f.groups.push_back(g);
        for(std::uint64_t a=0;a<I;++a){const auto coordinate=workload.group_features[ab+a];
            if(coordinate>=f.coordinates)throw std::invalid_argument("fixture participant coordinate outside axis");f.arguments.push_back(coordinate);}
        for(std::uint64_t o=0;o<O;++o){const auto coordinate=workload.group_outputs[ob+o];
            if(coordinate>=f.coordinates)throw std::invalid_argument("fixture output coordinate outside axis");f.outputs.push_back({coordinate,1.f});}
        f.group_weight_offsets.push_back(f.group_weight_offsets.back()+I*O);
        const auto shape=std::make_pair(static_cast<std::size_t>(I),static_cast<std::size_t>(O));
        if(gid%2 && parameter_shapes[gid/2]!=shape)throw std::invalid_argument("shared linear parameter shape differs");
        parameter_shapes[gid/2]=shape;
    }
    for(const auto shape:parameter_shapes){
        f.weight_offsets.push_back(f.weights32.size());
        for(std::size_t j=0;j<shape.first*shape.second;++j){
            const float w=float(int(workload_mix(j+f.weights32.size()+99)%13)-6)*0.03125f;
            f.weights32.push_back(w);f.weights16.push_back(reference::float_to_half_bits(w));
        }
    }
    f.admitted=workload.predicates_before;
    const auto words=(f.batch+31)/32;
    f.primal.assign(f.conditions*words,0);f.derivative.assign(f.conditions*words,0);
    f.probabilities.resize(f.batch*f.conditions);
    for(std::uint64_t b=0;b<f.batch;++b)for(std::uint64_t c=0;c<f.conditions;++c){
        const bool active=f.admitted[b*f.conditions+c]!=0;
        f.probabilities[b*f.conditions+c]=active?0.75f:0.25f;
        if(active)f.primal[c*words+b/32]|=1u<<(b%32);
        if(active||profile==GateProfile::ste)f.derivative[c*words+b/32]|=1u<<(b%32);
    }
    f.argument_local_indices.resize(f.arguments.size());
    f.neighborhood_group_offsets.push_back(0);f.neighborhood_feature_offsets.push_back(0);
    for(std::size_t begin=0;begin<f.groups.size();begin+=8){
        const auto end=std::min(begin+8,f.groups.size());
        std::map<std::uint64_t,std::uint32_t> masks;
        std::uint64_t weights=0;
        for(auto gid=begin;gid<end;++gid){
            f.neighborhood_groups.push_back(gid);const auto& g=f.groups[gid];weights+=g.argument_count*g.output_count;
            for(std::uint64_t a=0;a<g.argument_count;++a)masks[f.arguments[g.argument_begin+a]]|=1u<<(gid-begin);
        }
        std::map<std::uint64_t,std::uint32_t> local;
        for(const auto& item:masks){local[item.first]=local.size();f.neighborhood_features.push_back(item.first);f.feature_group_masks.push_back(item.second);}
        for(auto gid=begin;gid<end;++gid){const auto& g=f.groups[gid];
            for(std::uint64_t a=0;a<g.argument_count;++a)f.argument_local_indices[g.argument_begin+a]=local.at(f.arguments[g.argument_begin+a]);}
        f.neighborhood_group_offsets.push_back(f.neighborhood_groups.size());
        f.neighborhood_feature_offsets.push_back(f.neighborhood_features.size());
        f.max_groups=std::max<std::uint64_t>(f.max_groups,end-begin);
        f.max_features=std::max<std::uint64_t>(f.max_features,masks.size());f.max_group_weights=std::max(f.max_group_weights,weights);
    }
    f.revisions.structure_epoch=1;f.revisions.mapping_revision=1;f.revisions.accessibility_generation=1;
    const float temperature=1.f,threshold=0.5f;
    std::memcpy(&f.revisions.temperature_bits,&temperature,4);std::memcpy(&f.revisions.threshold_bits,&threshold,4);
    return f;
}
} // namespace cgs
