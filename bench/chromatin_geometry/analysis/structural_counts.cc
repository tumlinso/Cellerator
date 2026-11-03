// CPU-only structural accounting. This program never initializes CUDA,
// allocates device memory, launches a kernel, or reports GPU performance.
#include "backends/consumer_fixture.hh"
#include "data/replay.hh"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {
struct Cohorts {
    std::uint64_t count=0,chunks=0,tiles=0,active_warps=0,max_cohort=0;
    std::uint64_t max_chunks_sum=0,staged_forward_x=0,staged_backward_x=0;
    std::uint64_t staged_forward_weights=0,staged_backward_weights=0;
};
Cohorts describe(const cgs::ConsumerHostFixture& h,bool ste,unsigned tile){
    Cohorts s;
    const auto p=h.problem();
    for(std::size_t nh=0;nh<p.neighborhoods.count;++nh){
        const auto gb=h.neighborhood_group_offsets[nh],ge=h.neighborhood_group_offsets[nh+1];
        const auto fb=h.neighborhood_feature_offsets[nh],fe=h.neighborhood_feature_offsets[nh+1];
        const auto all=ge-gb==32?UINT32_MAX:(std::uint32_t(1)<<(ge-gb))-1;
        for(std::uint64_t base=0;base<h.batch;base+=tile){
            ++s.tiles;
            std::map<std::pair<std::uint32_t,std::uint32_t>,std::uint64_t> frequencies;
            for(auto cell=base;cell<std::min(h.batch,base+tile);++cell){
                std::uint32_t mask=0;
                for(auto k=gb;k<ge;++k){const auto& g=h.groups[h.neighborhood_groups[k]];
                    if(h.admitted[cell*h.conditions+g.condition_index])mask|=1u<<(k-gb);}
                const auto derivative=ste?all:0u;
                ++frequencies[{mask,derivative}];
                for(auto f=fb;f<fe;++f){
                    const auto membership=h.feature_group_masks[f];
                    s.staged_forward_x+=(membership&mask)!=0;
                    s.staged_backward_x+=(membership&(mask|derivative))!=0;
                }
            }
            // The implementation assigns a cohort to start_index % 4, rather
            // than to its ordinal. Count that actual static work assignment.
            std::uint64_t starts=0,warp_chunks[4]{};
            for(const auto& item:frequencies){const auto cells=item.second,chunks=(cells+31)/32;
                ++s.count;s.chunks+=chunks;s.max_cohort=std::max(s.max_cohort,cells);
                warp_chunks[starts%4]+=chunks;starts+=cells;
            }
            s.active_warps+=std::count_if(warp_chunks,warp_chunks+4,[](auto n){return n!=0;});
            s.max_chunks_sum+=*std::max_element(warp_chunks,warp_chunks+4);
        }
        // Both shared variants reuse their parameter stage across TWO cell
        // tiles. Buffer count changes pipelining, not that reuse boundary.
        for(std::uint64_t base=0;base<h.batch;base+=2*tile){
            std::uint32_t primal=0;
            for(auto cell=base;cell<std::min(h.batch,base+2*tile);++cell)
                for(auto k=gb;k<ge;++k)if(h.admitted[cell*h.conditions+h.groups[h.neighborhood_groups[k]].condition_index])primal|=1u<<(k-gb);
            for(auto k=gb;k<ge;++k){const auto& g=h.groups[h.neighborhood_groups[k]];
                const auto values=h.body==cgs::BodyKind::ordered_product?1:g.argument_count*g.output_count;
                if(primal&(1u<<(k-gb)))s.staged_forward_weights+=values;
                if(ste||(primal&(1u<<(k-gb))))s.staged_backward_weights+=values;
            }
        }
    }
    return s;
}
void emit(const cgs::WorkloadFixture& w,const std::string& input,bool after,cgs::BodyKind body,bool ste){
    auto host=cgs::make_consumer_fixture(w,body,cgs::DType::fp16,ste?cgs::GateProfile::ste:cgs::GateProfile::supplied);
    if(after)host.admitted=w.predicates_after;
    const auto p=host.problem();
    std::uint64_t primal_groups=0,forward_x=0,backward_x=0,warp_backward_x=0,weight_uses=0,unique_x=0;
    std::set<std::uint64_t> edge_ids;
    for(const auto& g:host.groups)edge_ids.insert(g.condition_index);
    for(std::uint64_t cell=0;cell<host.batch;++cell){
        std::vector<unsigned char> seen(host.coordinates,0);
        for(const auto& g:host.groups){const bool active=host.admitted[cell*host.conditions+g.condition_index];
            const auto I=g.argument_count,O=g.output_count;primal_groups+=active;
            if(active){
                forward_x+=body==cgs::BodyKind::ordered_product?I:I*O;
                weight_uses+=body==cgs::BodyKind::ordered_product?1:I*O;
                for(std::uint64_t a=0;a<I;++a)seen[host.arguments[g.argument_begin+a]]=1;
            }
            if(body==cgs::BodyKind::ordered_product){
                const auto scalar=I*(ste||active)+(active?I*(I-1):0);
                backward_x+=scalar;warp_backward_x+=I<=32?I*(ste||active):scalar;
            }else{const auto values=I*O*(ste?1:0)+(active?I*O:0);backward_x+=values;warp_backward_x+=values;} // F for D, plus separate P dW
        }
        unique_x+=std::count(seen.begin(),seen.end(),1);
    }
    std::vector<Cohorts> stats;
    for(const auto tile:{32u,64u,128u})stats.push_back(describe(host,ste,tile));
    const auto source_nodes=w.programs.front().nodes.size();
    const auto physical_nodes=w.spec.role==cgs::WorkloadRole::complete_consumer?w.demanded_nodes.size():source_nodes;
    const auto params=body==cgs::BodyKind::ordered_product?host.max_groups:host.max_group_weights;
    const auto floats128_1=128*host.max_features+params,floats128_2=256*host.max_features+params;
    std::cout<<w.spec.id<<','<<input<<','<<(after?"after":"before")<<','
        <<(w.spec.role==cgs::WorkloadRole::complete_consumer?"complete":"producer_only")<<','
        <<(body==cgs::BodyKind::ordered_product?"product":"linear")<<','<<(ste?"all_D_given_condition_P":"boolean")<<','
        <<host.batch<<','<<host.groups.size()<<','<<host.conditions<<','<<edge_ids.size()<<','
        <<source_nodes<<','<<physical_nodes<<','<<w.demanded_roots.size()<<','<<w.predicate_features<<','
        <<w.changed_input_features.size()<<','<<w.root_decision_changes<<','<<primal_groups<<','
        <<(ste?host.batch*host.groups.size():0)<<','<<forward_x<<','<<warp_backward_x<<','<<backward_x<<','<<unique_x<<','<<weight_uses<<','
        <<host.max_features<<','<<floats128_1*4<<','<<(floats128_1>12000)<<','<<floats128_2*4<<','<<(floats128_2>12000);
    for(const auto& s:stats){
        const double useful_lanes=s.chunks?double(host.batch*p.neighborhoods.count)/(32*s.chunks):0;
        const double warp_presence=s.tiles?double(s.active_warps)/(4*s.tiles):0;
        const double balance=s.max_chunks_sum?double(s.chunks)/(4*s.max_chunks_sum):0;
        std::cout<<','<<s.count<<','<<s.max_cohort<<','<<useful_lanes<<','<<warp_presence<<','<<balance<<','
            <<s.staged_forward_x<<','<<s.staged_backward_x<<','<<s.staged_forward_weights<<','<<s.staged_backward_weights;
    }
    std::cout<<'\n';
}
}
int main(int argc,char** argv){
    try{
        std::cout<<std::setprecision(9);
        std::cout<<"case,input,state,role,body,gate_envelope,cells,groups,condition_edges,numerically_demanded_edges,source_nodes,evaluated_nodes,demanded_unique_roots,predicate_features,changed_leaf_inputs,changed_root_decisions,primal_group_occurrences,gate_derivative_group_occurrences,direct_forward_x_value_uses,warp_control_backward_x_value_uses,cohort_scalar_backward_x_value_uses,unique_primal_x_per_cell_sum,forward_parameter_value_uses,max_neighborhood_features,stage128_one_bytes,stage128_one_rejected,stage128_two_bytes,stage128_two_rejected";
        for(const auto tile:{32,64,128})std::cout<<",cohorts"<<tile<<",max_cohort"<<tile<<",dispatch_lane_fill"<<tile<<",dispatch_warp_presence"<<tile<<",dispatch_chunk_balance"<<tile<<",staged_forward_x"<<tile<<",staged_backward_x"<<tile<<",staged_forward_parameters"<<tile<<",staged_backward_parameters"<<tile;
        std::cout<<'\n';
        for(const auto& spec:cgs::workload_manifest()){
            auto workload=cgs::materialize_workload(spec);
            const auto write=[&](const std::string& input){
                for(const auto body:{cgs::BodyKind::ordered_product,cgs::BodyKind::grouped_linear})
                    for(const auto ste:{false,true})for(const auto after:{false,true})emit(workload,input,after,body,ste);
            };
            write("synthetic");
            if(argc==2){cgs::apply_atac_replay(workload,argv[1]);write("10x_prefix_replay");}
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
