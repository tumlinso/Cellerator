#include "conditions.hh"
#include <Cellerator/compute/gating/cuda_worksets.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cgs {
namespace {
namespace oc=cellerator::ops::condition;
namespace cg=cellerator::compute::gating;
void check(cudaError_t e,const char* op){if(e!=cudaSuccess)throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(e));}
struct Allocation{void* data=nullptr;std::size_t bytes=0;explicit Allocation(std::size_t n):bytes(n){if(n)check(cudaMalloc(&data,n),"allocate condition fixture");}~Allocation(){if(data)cudaFree(data);}};

__device__ bool scalar_node(oc::NodeV1 node,const std::uint8_t* x,const std::uint8_t* state){
    if(node.opcode==oc::OpcodeV1::predicate)return x[node.predicate_index]==node.expected_value;
    if(node.opcode==oc::OpcodeV1::logical_not)return !state[node.first_child];
    bool value=node.opcode==oc::OpcodeV1::all_of;
    for(unsigned j=0;j<node.child_count;++j){
        if(node.opcode==oc::OpcodeV1::all_of)value=value&&state[node.first_child+j];
        else value=value||state[node.first_child+j];
    }return value;
}
template<bool Update>
__global__ void root_mapped_bytes(oc::ProgramViewV1 p,std::uint64_t cells,
    const std::uint64_t* level_offsets,const std::uint32_t* level_nodes,std::uint64_t levels,
    const std::uint8_t* before,const std::uint8_t* current,std::uint8_t* retained,
    const std::uint32_t* roots,std::uint8_t* admitted){
    const auto cell=static_cast<std::uint64_t>(blockIdx.x);if(cell>=cells)return;
    extern __shared__ std::uint8_t shared[];
    auto* values=shared;auto* changed=shared+p.node_count;
    auto* old_values=retained+cell*p.node_count;
    if(Update){for(std::uint64_t n=threadIdx.x;n<p.node_count;n+=blockDim.x)values[n]=old_values[n];__syncthreads();}
    const auto* old_input=before+cell*p.predicate_count;const auto* input=current+cell*p.predicate_count;
    for(std::uint64_t level=0;level<levels;++level){
        for(auto at=level_offsets[level]+threadIdx.x;at<level_offsets[level+1];at+=blockDim.x){
            const auto id=level_nodes[at];const auto node=p.nodes[id];bool need=true;
            if(Update){
                need=node.opcode==oc::OpcodeV1::predicate?old_input[node.predicate_index]!=input[node.predicate_index]:false;
                if(node.opcode!=oc::OpcodeV1::predicate)for(unsigned j=0;j<node.child_count;++j)need=need||changed[node.first_child+j];
                changed[id]=need;
            }
            if(need)values[id]=scalar_node(node,input,values);
        }
        __syncthreads();
    }
    if(admitted)for(std::uint64_t c=threadIdx.x;c<p.edge_count;c+=blockDim.x)admitted[cell*p.edge_count+c]=values[roots[c]];
    for(std::uint64_t n=threadIdx.x;n<p.node_count;n+=blockDim.x)old_values[n]=values[n];
}
__global__ void expose_state_planes(oc::ProgramViewV1 p,std::uint64_t cells,const std::uint32_t* roots,
    const std::uint8_t* state,bool ste,std::uint32_t* primal,std::uint32_t* derivative){
    const auto words=(cells+31)/32,work=static_cast<std::uint64_t>(blockIdx.x)*4+threadIdx.x/32;
    if(work>=p.edge_count*words)return;
    const auto c=work/words,w=work%words;const auto lane=threadIdx.x%32;const auto cell=w*32+lane;
    const bool active=cell<cells&&state[cell*p.node_count+roots[c]];
    const auto pv=__ballot_sync(0xffffffffu,active),dv=__ballot_sync(0xffffffffu,cell<cells&&(active||ste));
    if(!lane){primal[c*words+w]=pv;derivative[c*words+w]=dv;}
}
__global__ void scalar_dependency_update(oc::ProgramViewV1 p,std::uint64_t cells,
    const std::uint8_t* before,const std::uint8_t* after,std::uint8_t* values,
    std::uint8_t* dirty,const std::uint32_t* roots,std::uint64_t* bitmaps){
    const auto cell=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(cell>=cells)return;
    auto* state=values+cell*p.node_count;auto* changed=dirty+cell*p.node_count;
    const auto* oldx=before+cell*p.predicate_count;const auto* newx=after+cell*p.predicate_count;
    for(std::uint64_t n=0;n<p.node_count;++n){const auto node=p.nodes[n];bool need=false;
        if(node.opcode==oc::OpcodeV1::predicate)need=oldx[node.predicate_index]!=newx[node.predicate_index];
        else for(unsigned j=0;j<node.child_count;++j)need=need||changed[node.first_child+j];
        changed[n]=need;if(need)state[n]=scalar_node(node,newx,state);
    }
    const auto stride=(p.edge_count+63)/64;
    for(std::uint64_t w=0;w<stride;++w){std::uint64_t value=0;
        for(unsigned j=0;j<64 && w*64+j<p.edge_count;++j)if(state[roots[w*64+j]])value|=std::uint64_t(1)<<j;
        bitmaps[cell*stride+w]=value;
    }
}

__global__ void evaluate_level(oc::ProgramViewV1 p,const std::uint32_t* node_ids,
    std::uint64_t node_count,std::uint64_t cells,const std::uint8_t* before,
    const std::uint8_t* current,std::uint32_t* values,std::uint32_t* changed,bool incremental){
    const auto words=(cells+31)/32;
    const auto work=static_cast<std::uint64_t>(blockIdx.x)*4+threadIdx.x/32;
    if(work>=node_count*words)return;
    const auto node_id=node_ids[work/words];const auto word=work%words;const auto node=p.nodes[node_id];
    const auto lane=threadIdx.x%32;const auto cell=word*32+lane;
    const std::uint32_t valid=(word+1==words&&cells%32)?((1u<<(cells%32))-1):0xffffffffu;
    if(node.opcode==oc::OpcodeV1::predicate){
        const bool value=cell<cells && current[cell*p.predicate_count+node.predicate_index]==node.expected_value;
        const bool dirty=cell<cells && (!incremental || before[cell*p.predicate_count+node.predicate_index]!=current[cell*p.predicate_count+node.predicate_index]);
        const auto bits=__ballot_sync(0xffffffffu,value),delta=__ballot_sync(0xffffffffu,dirty);
        if(!lane){if(delta||!incremental)values[node_id*words+word]=bits;changed[node_id*words+word]=delta;}
    }else if(!lane){
        std::uint32_t delta=incremental?0:valid;
        if(incremental)for(unsigned j=0;j<node.child_count;++j)delta|=changed[(node.first_child+j)*words+word];
        changed[node_id*words+word]=delta;
        if(delta||!incremental){
            std::uint32_t value=node.opcode==oc::OpcodeV1::all_of?valid:0;
            for(unsigned j=0;j<node.child_count;++j){const auto child=values[(node.first_child+j)*words+word];
                if(node.opcode==oc::OpcodeV1::all_of)value&=child;
                else if(node.opcode==oc::OpcodeV1::any_of)value|=child;
                else value=~child;
            }
            values[node_id*words+word]=value&valid;
        }
    }
}

__global__ void expose_roots(oc::ProgramViewV1 p,std::uint64_t cells,const std::uint32_t* roots,
    const std::uint64_t* bitmaps,const std::uint32_t* planes,bool from_planes,
    bool ste,bool make_bytes,std::uint8_t* admitted,std::uint32_t* primal,
    std::uint32_t* derivative,const cg::cuda_workset_status_v1* native_status,std::uint32_t* status){
    if(native_status&&native_status->code){if(!blockIdx.x&&!threadIdx.x)atomicOr(status,status_invalid_descriptor);return;}
    const auto words=(cells+31)/32;
    const auto work=static_cast<std::uint64_t>(blockIdx.x)*4+threadIdx.x/32;
    if(work>=p.edge_count*words)return;
    const auto c=work/words,w=work%words;const auto lane=threadIdx.x%32;const auto cell=w*32+lane;
    const bool active=cell<cells && (from_planes?
        ((planes[roots[c]*words+w]>>lane)&1u):((bitmaps[cell*((p.edge_count+63)/64)+c/64]>>(c%64))&1u));
    const auto pv=__ballot_sync(0xffffffffu,active);
    const auto dv=__ballot_sync(0xffffffffu,cell<cells&&(active||ste));
    if(make_bytes&&cell<cells)admitted[cell*p.edge_count+c]=active;
    if(!lane){primal[c*words+w]=pv;derivative[c*words+w]=dv;}
}
} // namespace

struct ConditionDeviceFixture::Impl{
    std::vector<std::unique_ptr<Allocation>> allocations;std::size_t bytes=0;
    oc::ProgramViewV1 p{};cg::cuda_condition_program_v1 mapped{};
    std::uint64_t cells=0,source_nodes=0;const std::uint8_t* before=nullptr;const std::uint8_t* after=nullptr;
    std::uint8_t* scalar_values=nullptr;std::uint8_t* scalar_dirty=nullptr;
    std::uint32_t* planes=nullptr;std::uint32_t* dirty_planes=nullptr;
    std::uint64_t* bitmaps=nullptr;std::uint8_t* admitted=nullptr;
    std::uint32_t* primal=nullptr;std::uint32_t* derivative=nullptr;std::uint32_t* status=nullptr;
    cg::cuda_workset_status_v1* native_status=nullptr;
    std::vector<const std::uint32_t*> level_nodes;std::vector<std::uint64_t> level_counts;
    const std::uint64_t* level_offsets=nullptr;const std::uint32_t* flat_level_nodes=nullptr;
    GateRevisions revisions{};bool initialized=false,after_state=false,has_bytes=false,has_planes=false;ConditionProducer producer{};
    void* allocate(std::size_t n){allocations.emplace_back(new Allocation(n));bytes+=n;return allocations.back()->data;}
    template<class T>T* copy(const std::vector<T>& v,cudaStream_t stream){auto* out=static_cast<T*>(allocate(v.size()*sizeof(T)));
        if(!v.empty())check(cudaMemcpyAsync(out,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,stream),"upload condition metadata");return out;}
};

ConditionDeviceFixture::ConditionDeviceFixture(const WorkloadFixture& f,cudaStream_t stream):impl_(new Impl){
    auto& d=*impl_;d.cells=f.spec.cells;
    std::vector<oc::NodeV1> nodes;std::vector<oc::EdgeBindingV1> bindings;std::vector<std::uint32_t> roots;
    std::vector<std::vector<std::uint32_t>> levels;
    struct Bank { const ConditionDAG* dag;std::uint32_t base; };
    std::vector<Bank> banks;
    for(const auto& dag:f.programs){
        const auto same_bank=[&](const ConditionDAG& other){
            if(dag.nodes.size()!=other.nodes.size()||dag.levels!=other.levels)return false;
            for(std::size_t n=0;n<dag.nodes.size();++n){const auto& a=dag.nodes[n];const auto& b=other.nodes[n];
                if(a.op!=b.op||a.feature!=b.feature||a.first_child!=b.first_child||a.child_count!=b.child_count)return false;}
            return true;
        };
        auto base=static_cast<std::uint32_t>(nodes.size());bool found=false;
        for(const auto& bank:banks)if(same_bank(*bank.dag)){base=bank.base;found=true;break;}
        if(!found){
            if(nodes.size()+dag.nodes.size()>std::numeric_limits<std::uint32_t>::max())throw std::overflow_error("condition nodes exceed uint32");
            banks.push_back({&dag,base});
            for(const auto& n:dag.nodes){oc::NodeV1 node{};
                node.opcode=n.op==PredicateOp::feature?oc::OpcodeV1::predicate:n.op==PredicateOp::all?oc::OpcodeV1::all_of:n.op==PredicateOp::any?oc::OpcodeV1::any_of:oc::OpcodeV1::logical_not;
                node.predicate_index=n.feature;node.first_child=base+n.first_child;node.child_count=n.child_count;nodes.push_back(node);}
            levels.resize(std::max(levels.size(),dag.levels.size()));
            for(std::size_t level=0;level<dag.levels.size();++level)for(const auto n:dag.levels[level])levels[level].push_back(base+n);
        }
        const auto root=base+dag.root;bindings.push_back({roots.size(),root,0});roots.push_back(root);
    }
    if(!f.edge_roots.empty()){
        if(f.programs.size()!=1)throw std::invalid_argument("explicit condition roots require one shared bank");
        roots=f.edge_roots;bindings.clear();
        for(std::size_t edge=0;edge<roots.size();++edge)bindings.push_back({edge,roots[edge],0});
    }
    d.source_nodes=nodes.size();
    if(f.spec.role==WorkloadRole::complete_consumer){
        // Compact the demanded ancestor closure for BOTH matched producers.
        // Every retained node keeps its complete child range, so that range
        // remains contiguous after stable topological compaction. Original
        // biological/condition identities remain in the supplied fixture.
        std::vector<std::uint8_t> needed(nodes.size(),0);
        for(const auto root:roots){if(root>=nodes.size())throw std::invalid_argument("condition root outside bank");needed[root]=1;}
        for(std::size_t at=nodes.size();at;--at)if(needed[at-1]){
            const auto node=nodes[at-1];
            if(node.child_count && (node.first_child>=at-1 || node.child_count>at-1-node.first_child))
                throw std::invalid_argument("invalid condition child range");
            for(unsigned j=0;j<node.child_count;++j)needed[node.first_child+j]=1;
        }
        std::vector<std::uint32_t> remap(nodes.size(),UINT32_MAX);std::vector<oc::NodeV1> compact;
        for(std::size_t at=0;at<nodes.size();++at)if(needed[at]){remap[at]=compact.size();compact.push_back(nodes[at]);}
        for(auto& node:compact)if(node.child_count)node.first_child=remap[node.first_child];
        for(auto& level:levels){std::vector<std::uint32_t> retained;
            for(const auto node:level)if(needed[node])retained.push_back(remap[node]);level=std::move(retained);}
        for(auto& root:roots)root=remap[root];
        bindings.clear();for(std::size_t edge=0;edge<roots.size();++edge)bindings.push_back({edge,roots[edge],0});
        nodes=std::move(compact);
    }
    oc::ProgramViewV1 host{1,nodes.data(),nodes.size(),bindings.data(),bindings.size(),roots.size(),f.spec.features,0};
    if(oc::validate(host)!=oc::StatusV1::success)throw std::invalid_argument("invalid native condition study DAG");
    d.p=host;d.p.nodes=d.copy(nodes,stream);d.p.edges=d.copy(bindings,stream);
    d.mapped={d.p,d.copy(roots,stream),roots.size()};
    std::vector<std::uint64_t> level_offsets{0};std::vector<std::uint32_t> flat_level_nodes;
    for(const auto& level:levels){d.level_nodes.push_back(d.copy(level,stream));d.level_counts.push_back(level.size());}
    for(const auto& level:levels){flat_level_nodes.insert(flat_level_nodes.end(),level.begin(),level.end());level_offsets.push_back(flat_level_nodes.size());}
    d.level_offsets=d.copy(level_offsets,stream);d.flat_level_nodes=d.copy(flat_level_nodes,stream);
    d.before=d.copy(f.accessibility,stream);d.after=d.copy(f.accessibility_after,stream);
    const auto words=(d.cells+31)/32;
    d.scalar_values=static_cast<std::uint8_t*>(d.allocate(d.cells*d.p.node_count));
    d.scalar_dirty=static_cast<std::uint8_t*>(d.allocate(d.cells*d.p.node_count));
    d.planes=static_cast<std::uint32_t*>(d.allocate(d.p.node_count*words*4));
    d.dirty_planes=static_cast<std::uint32_t*>(d.allocate(d.p.node_count*words*4));
    d.bitmaps=static_cast<std::uint64_t*>(d.allocate(d.cells*((d.p.edge_count+63)/64)*8));
    d.admitted=static_cast<std::uint8_t*>(d.allocate(d.cells*d.p.edge_count));
    d.primal=static_cast<std::uint32_t*>(d.allocate(d.p.edge_count*words*4));
    d.derivative=static_cast<std::uint32_t*>(d.allocate(d.p.edge_count*words*4));
    d.native_status=static_cast<cg::cuda_workset_status_v1*>(d.allocate(sizeof(cg::cuda_workset_status_v1)));
    d.status=static_cast<std::uint32_t*>(d.allocate(4));
    check(cudaMemsetAsync(d.native_status,0,sizeof(cg::cuda_workset_status_v1),stream),"clear condition native status");
    check(cudaMemsetAsync(d.status,0,4,stream),"clear condition status");
    d.revisions.structure_epoch=1;d.revisions.mapping_revision=1;d.revisions.accessibility_generation=1;
    check(cudaStreamSynchronize(stream),"finish condition preparation");
}
ConditionDeviceFixture::~ConditionDeviceFixture()=default;

cudaError_t ConditionDeviceFixture::rebuild(ConditionProducer producer,bool after,GateProfile profile,bool make_bytes,cudaStream_t stream,bool make_planes)noexcept{
    auto& d=*impl_;const auto words=(d.cells+31)/32;const auto* input=after?d.after:d.before;
    const bool planes=producer==ConditionProducer::bitplanes||make_planes;
    if(profile==GateProfile::ste&&!planes)return cudaErrorInvalidValue;
    auto error=cudaMemsetAsync(d.status,0,4,stream);if(error!=cudaSuccess)return error;
    if(producer==ConditionProducer::bitmap_api){error=cudaMemsetAsync(d.native_status,0,sizeof(cg::cuda_workset_status_v1),stream);if(error!=cudaSuccess)return error;}
    if(d.cells&&d.p.edge_count){
        if(producer==ConditionProducer::root_mapped){
            if(d.p.node_count>24000)return cudaErrorInvalidValue;
            root_mapped_bytes<false><<<static_cast<unsigned>(d.cells),128,d.p.node_count,stream>>>(d.p,d.cells,d.level_offsets,d.flat_level_nodes,
                d.level_nodes.size(),d.before,input,d.scalar_values,d.mapped.root_node_by_edge_ordinal,make_bytes?d.admitted:nullptr);
            if(planes)expose_state_planes<<<static_cast<unsigned>((d.p.edge_count*words+3)/4),128,0,stream>>>(d.p,d.cells,d.mapped.root_node_by_edge_ordinal,d.scalar_values,profile==GateProfile::ste,d.primal,d.derivative);
        }else if(producer==ConditionProducer::bitmap_api){
            error=cg::cuda_evaluate_condition_bitmaps_v1(d.mapped,d.cells,input,d.cells*d.p.predicate_count,
                d.bitmaps,d.cells*((d.p.edge_count+63)/64),d.scalar_values,d.cells*d.p.node_count,d.native_status,stream);
            if(error!=cudaSuccess)return error;
        }else for(std::size_t level=0;level<d.level_nodes.size();++level)if(d.level_counts[level]){
            const auto blocks=static_cast<unsigned>((d.level_counts[level]*words+3)/4);
            evaluate_level<<<blocks,128,0,stream>>>(d.p,d.level_nodes[level],d.level_counts[level],d.cells,d.before,input,d.planes,d.dirty_planes,false);
        }
        if(producer!=ConditionProducer::root_mapped)expose_roots<<<static_cast<unsigned>((d.p.edge_count*words+3)/4),128,0,stream>>>(d.p,d.cells,d.mapped.root_node_by_edge_ordinal,d.bitmaps,d.planes,
            producer==ConditionProducer::bitplanes,profile==GateProfile::ste,make_bytes,d.admitted,d.primal,d.derivative,producer==ConditionProducer::bitmap_api?d.native_status:nullptr,d.status);
    }
    d.producer=producer;d.initialized=true;d.after_state=after;d.has_bytes=make_bytes;d.has_planes=planes;++d.revisions.accessibility_generation;
    return cudaGetLastError();
}
cudaError_t ConditionDeviceFixture::update(ConditionProducer producer,GateProfile profile,bool make_bytes,cudaStream_t stream,bool make_planes)noexcept{
    auto& d=*impl_;if(!d.initialized||producer!=d.producer)return cudaErrorInvalidValue;
    const bool planes=producer==ConditionProducer::bitplanes||make_planes;
    if(profile==GateProfile::ste&&!planes)return cudaErrorInvalidValue;
    const auto words=(d.cells+31)/32;
    const auto* old_input=d.after_state?d.after:d.before;const auto* new_input=d.after_state?d.before:d.after;
    if(d.cells&&d.p.edge_count){
        if(producer==ConditionProducer::root_mapped){
            if(d.p.node_count>24000)return cudaErrorInvalidValue;
            root_mapped_bytes<true><<<static_cast<unsigned>(d.cells),128,d.p.node_count*2,stream>>>(d.p,d.cells,d.level_offsets,d.flat_level_nodes,
                d.level_nodes.size(),old_input,new_input,d.scalar_values,d.mapped.root_node_by_edge_ordinal,make_bytes?d.admitted:nullptr);
            if(planes)expose_state_planes<<<static_cast<unsigned>((d.p.edge_count*words+3)/4),128,0,stream>>>(d.p,d.cells,d.mapped.root_node_by_edge_ordinal,d.scalar_values,profile==GateProfile::ste,d.primal,d.derivative);
        }else if(producer==ConditionProducer::bitmap_api)
            scalar_dependency_update<<<static_cast<unsigned>((d.cells+127)/128),128,0,stream>>>(d.p,d.cells,old_input,new_input,d.scalar_values,d.scalar_dirty,d.mapped.root_node_by_edge_ordinal,d.bitmaps);
        else for(std::size_t level=0;level<d.level_nodes.size();++level)if(d.level_counts[level])
            evaluate_level<<<static_cast<unsigned>((d.level_counts[level]*words+3)/4),128,0,stream>>>(d.p,d.level_nodes[level],d.level_counts[level],d.cells,old_input,new_input,d.planes,d.dirty_planes,true);
        if(producer!=ConditionProducer::root_mapped)expose_roots<<<static_cast<unsigned>((d.p.edge_count*words+3)/4),128,0,stream>>>(d.p,d.cells,d.mapped.root_node_by_edge_ordinal,d.bitmaps,d.planes,
            producer==ConditionProducer::bitplanes,profile==GateProfile::ste,make_bytes,d.admitted,d.primal,d.derivative,producer==ConditionProducer::bitmap_api?d.native_status:nullptr,d.status);
    }
    d.after_state=!d.after_state;d.has_bytes=make_bytes;d.has_planes=planes;++d.revisions.accessibility_generation;return cudaGetLastError();
}
GateView ConditionDeviceFixture::gates(GateProfile profile)const{
    GateView view{};const auto& d=*impl_;view.profile=profile;view.admitted=d.has_bytes?d.admitted:nullptr;view.primal_words=d.has_planes?d.primal:nullptr;view.derivative_words=d.has_planes?d.derivative:nullptr;
    view.words_per_condition=(d.cells+31)/32;view.revisions=d.revisions;return view;
}
std::uint32_t ConditionDeviceFixture::status(cudaStream_t stream)const{std::uint32_t value=0;check(cudaMemcpyAsync(&value,impl_->status,4,cudaMemcpyDeviceToHost,stream),"download condition status");check(cudaStreamSynchronize(stream),"finish condition status");return value;}
const std::uint32_t* ConditionDeviceFixture::device_status()const{return impl_->status;}
std::size_t ConditionDeviceFixture::device_bytes()const{return impl_->bytes;}
std::uint64_t ConditionDeviceFixture::evaluated_node_count()const{return impl_->p.node_count;}
std::uint64_t ConditionDeviceFixture::source_node_count()const{return impl_->source_nodes;}

bool run_condition_tests(cudaStream_t stream,std::string* error){
    std::string context="condition setup";
    try{
        for(const auto node_count:{7u,2304u})for(const auto change:{0u,1u,10u,100u}){
            auto spec=workload_manifest().front();spec.cells=35;spec.features=256;spec.groups=8;spec.dag_nodes=node_count;spec.predicate_change_percent=change;
            const auto host=materialize_workload(spec);ConditionDeviceFixture device(host,stream);
            const auto validate=[&](const std::vector<std::uint8_t>& expected){
                const auto gate=device.gates(GateProfile::supplied);std::vector<std::uint8_t> got(expected.size());
                if(gate.admitted){
                    check(cudaMemcpyAsync(got.data(),gate.admitted,got.size(),cudaMemcpyDeviceToHost,stream),"download root decisions");
                    check(cudaStreamSynchronize(stream),"finish root decision read");
                    if(got!=expected)throw std::runtime_error("condition DAG decisions differ from independent oracle");
                }
                if(device.status(stream))throw std::runtime_error("condition producer failed");
                if(!gate.primal_words){if(!gate.admitted)throw std::runtime_error("condition produced no representation");return;}
                const auto conditions=host.edge_roots.empty()?host.programs.size():host.edge_roots.size();
                std::vector<std::uint32_t> planes(conditions*((spec.cells+31)/32));
                check(cudaMemcpyAsync(planes.data(),gate.primal_words,planes.size()*4,cudaMemcpyDeviceToHost,stream),"download root planes");
                check(cudaStreamSynchronize(stream),"finish root planes");
                for(std::uint64_t c=0;c<conditions;++c)for(std::uint64_t cell=0;cell<spec.cells;++cell)
                    if(((planes[c*gate.words_per_condition+cell/32]>>(cell%32))&1u)!=expected[cell*conditions+c])throw std::runtime_error("condition root bit identity mismatch");
                for(std::uint64_t c=0;c<conditions;++c)if(planes[c*gate.words_per_condition+gate.words_per_condition-1]&~7u)throw std::runtime_error("condition tail mask mismatch");
            };
            for(const auto producer:{ConditionProducer::root_mapped,ConditionProducer::bitplanes,ConditionProducer::bitmap_api}){
                context="condition nodes="+std::to_string(node_count)+" change="+std::to_string(change)+" producer="+std::to_string(int(producer));
                check(device.rebuild(producer,false,GateProfile::supplied,true,stream,true),"rebuild before DAG");validate(host.predicates_before);
                check(device.update(producer,GateProfile::supplied,true,stream,true),"dependency DAG update");validate(host.predicates_after);
                check(device.update(producer,GateProfile::supplied,true,stream,true),"reverse dependency DAG update");validate(host.predicates_before);
                check(device.rebuild(producer,true,GateProfile::supplied,true,stream,true),"rebuild after DAG");validate(host.predicates_after);
            }
            check(device.rebuild(ConditionProducer::root_mapped,false,GateProfile::supplied,true,stream,false),"rebuild byte-only DAG");
            if(device.gates(GateProfile::supplied).primal_words||device.gates(GateProfile::supplied).derivative_words)throw std::runtime_error("byte-only producer exposed planes");
            validate(host.predicates_before);
            check(device.update(ConditionProducer::root_mapped,GateProfile::supplied,true,stream,false),"update byte-only DAG");validate(host.predicates_after);
            check(device.rebuild(ConditionProducer::root_mapped,false,GateProfile::supplied,false,stream,true),"rebuild plane-only root map");
            if(device.gates(GateProfile::supplied).admitted)throw std::runtime_error("plane-only root producer exposed bytes");validate(host.predicates_before);
        }
        return true;
    }catch(const std::exception& e){if(error)*error=context+": "+e.what();return false;}
}
} // namespace cgs
