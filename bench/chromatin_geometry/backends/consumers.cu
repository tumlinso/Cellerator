#include "../contract.hh"
#include "descriptor_debug.hh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <limits>

namespace cgs {
namespace {

// Study-private workspace, entirely owned by the caller. No numerical value is
// retained across calls: only the membership representation and its revisions.
struct Header { GateRevisions revisions; };
struct Layout {
    Header* header;
    std::uint64_t* primal_bitmap;
    std::uint64_t* derivative_bitmap;
    std::uint64_t* primal_work;
    std::uint64_t* derivative_work;
    std::uint64_t* primal_count;
    std::uint64_t* derivative_count;
    std::uint64_t* order;
    std::uint32_t* p_signature;
    std::uint32_t* d_signature;
    float* local_f;
    float* local_df;
};
bool is_cohort_backend(Backend backend){
    return backend==Backend::cohort_direct || backend==Backend::cohort_shared ||
           backend==Backend::cohort_direct_two_tile;
}
bool uses_program_rows(Backend backend){
    return is_cohort_backend(backend)||backend==Backend::masked_direct_two_tile;
}

bool multiply(std::size_t a, std::size_t b, std::size_t& out) {
    if (b && a > std::numeric_limits<std::size_t>::max()/b) return false;
    out=a*b; return true;
}
bool add(std::size_t& total, std::size_t n, std::size_t element) {
    std::size_t bytes=0;
    const auto max=std::numeric_limits<std::size_t>::max();
    if (!multiply(n,element,bytes) || total>max-15) return false;
    const auto aligned=(total+15)&~std::size_t(15);
    if(bytes>max-aligned)return false;
    total=aligned+bytes;return true;
}
template<class T> T* take(char*& at, std::size_t count) {
    at=reinterpret_cast<char*>((reinterpret_cast<std::uintptr_t>(at)+15)&~std::uintptr_t(15));
    T* result=reinterpret_cast<T*>(at); at+=count*sizeof(T); return result;
}
Layout layout(Backend backend, BodyKind body, const ProblemView& p, WorkspaceView w) {
    char* at=static_cast<char*>(w.data);
    Layout l{}; l.header=take<Header>(at,1);
    if (backend==Backend::bitmap) {
        const auto bits=p.batch*((p.topology.conditions+63)/64);
        const auto work=p.batch*p.topology.group_count;
        l.primal_bitmap=take<std::uint64_t>(at,bits);
        l.derivative_bitmap=take<std::uint64_t>(at,bits);
        l.primal_work=take<std::uint64_t>(at,work);
        l.derivative_work=take<std::uint64_t>(at,work);
        l.primal_count=take<std::uint64_t>(at,p.batch);
        l.derivative_count=take<std::uint64_t>(at,p.batch);
    }
    if (uses_program_rows(backend)) {
        const auto n=p.batch*p.neighborhoods.count;
        l.order=take<std::uint64_t>(at,n);
        l.p_signature=take<std::uint32_t>(at,n);
        l.d_signature=take<std::uint32_t>(at,n);
    }
    if (body==BodyKind::grouped_linear) {
        l.local_f=take<float>(at,p.batch*p.topology.output_count);
        l.local_df=take<float>(at,p.batch*p.topology.output_count);
    }
    return l;
}

__device__ bool revisions_equal(const GateRevisions& a, const GateRevisions& b) {
    return a.structure_epoch==b.structure_epoch && a.mapping_revision==b.mapping_revision &&
        a.accessibility_generation==b.accessibility_generation && a.score_generation==b.score_generation &&
        a.parameter_revision==b.parameter_revision && a.teacher_revision==b.teacher_revision &&
        a.temporal_generation==b.temporal_generation && a.primal_generation==b.primal_generation &&
        a.derivative_generation==b.derivative_generation && a.temperature_bits==b.temperature_bits &&
        a.threshold_bits==b.threshold_bits;
}
__global__ void save_header(Header* h, GateRevisions revision) { if (!threadIdx.x) h->revisions=revision; }
__global__ void check_header(const Header* h, GateRevisions revision, std::uint32_t* status) {
    if (!threadIdx.x && !revisions_equal(h->revisions,revision)) atomicOr(status,status_stale);
}
__device__ bool enabled(const std::uint32_t* status) { return *status==0; }
__device__ bool bit(const std::uint32_t* words, std::uint64_t stride,
                    std::uint64_t cell, std::uint64_t condition) {
    return ((words[condition*stride+cell/32]>>(cell%32))&1u)!=0;
}
__device__ float load(const void* ptr, DType dtype, std::uint64_t index) {
    return dtype==DType::fp16 ? __half2float(static_cast<const __half*>(ptr)[index])
                             : static_cast<const float*>(ptr)[index];
}
__device__ float rounded(float x, DType dtype) {
    return dtype==DType::fp16 ? __half2float(__float2half_rn(x)) : x;
}
__device__ void store(void* ptr, DType dtype, std::uint64_t index, float value) {
    if (dtype==DType::fp16) static_cast<__half*>(ptr)[index]=__float2half_rn(value);
    else static_cast<float*>(ptr)[index]=value;
}
// Retain the existing packed_gated_product.cu ordered warp scans for the
// conventional controls. No division is used, including when inputs are zero.
__device__ float ordered_prefix(float value,unsigned lane){
    for(unsigned distance=1;distance<32;distance<<=1){
        const auto left=__shfl_up_sync(0xffffffffu,value,distance);
        if(lane>=distance)value=left*value;
    }
    return value;
}
__device__ float ordered_suffix(float value,unsigned lane){
    for(unsigned distance=1;distance<32;distance<<=1){
        const auto right=__shfl_down_sync(0xffffffffu,value,distance);
        if(lane+distance<32)value*=right;
    }
    return value;
}
__device__ bool primal(Backend backend, const GateView& gates, const Layout& l,
                       const ProblemView& p, std::uint64_t cell, std::uint64_t condition) {
    if (backend==Backend::scalar) return gates.admitted[cell*p.topology.conditions+condition]!=0;
    if (backend==Backend::bitmap)
        return ((l.primal_bitmap[cell*((p.topology.conditions+63)/64)+condition/64]>>(condition%64))&1u)!=0;
    return bit(gates.primal_words,gates.words_per_condition,cell,condition);
}
__device__ bool derivative(Backend backend, const GateView& gates, const Layout& l,
                           const ProblemView& p, std::uint64_t cell, std::uint64_t condition) {
    if(gates.profile!=GateProfile::ste)return false;
    if (backend==Backend::bitmap)
        return ((l.derivative_bitmap[cell*((p.topology.conditions+63)/64)+condition/64]>>(condition%64))&1u)!=0;
    return bit(gates.derivative_words,gates.words_per_condition,cell,condition);
}
__device__ float qvalue(const GateView& gates,const ProblemView& p,
                        std::uint64_t cell,std::uint64_t condition) {
    return gates.profile==GateProfile::ste ? gates.probabilities[cell*p.topology.conditions+condition] : 1.f;
}

__global__ void materialize_bitmaps(ProblemView p, GateView gates, Layout l, std::uint32_t* status) {
    if (!enabled(status)) return;
    const auto cell=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (cell>=p.batch) return;
    const auto words=(p.topology.conditions+63)/64;
    for (std::uint64_t word=0;word<words;++word) {
        std::uint64_t pv=0,dv=0;
        for (unsigned j=0;j<64 && word*64+j<p.topology.conditions;++j) {
            const auto c=word*64+j;
            if (bit(gates.primal_words,gates.words_per_condition,cell,c)) pv|=std::uint64_t(1)<<j;
            if (gates.profile==GateProfile::ste && bit(gates.derivative_words,gates.words_per_condition,cell,c)) dv|=std::uint64_t(1)<<j;
        }
        l.primal_bitmap[cell*words+word]=pv;
        l.derivative_bitmap[cell*words+word]=dv;
    }
}
__global__ void materialize_worksets(ProblemView p, GateView gates, Layout l,std::uint32_t* status) {
    if (!enabled(status)) return;
    const auto cell=static_cast<std::uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(cell>=p.batch)return;
    std::uint64_t np=0,nd=0;
    for(std::uint64_t g=0;g<p.topology.group_count;++g) {
        const auto c=p.topology.groups[g].condition_index;
        if(primal(Backend::bitmap,gates,l,p,cell,c))l.primal_work[cell*p.topology.group_count+np++]=g;
        if(primal(Backend::bitmap,gates,l,p,cell,c)||derivative(Backend::bitmap,gates,l,p,cell,c))
            l.derivative_work[cell*p.topology.group_count+nd++]=g;
    }
    l.primal_count[cell]=np;l.derivative_count[cell]=nd;
}

__global__ void build_cohorts(ProblemView p,GateView gates,Layout l,unsigned tile,std::uint32_t* status) {
    if(!enabled(status))return;
    const auto nh=static_cast<std::uint64_t>(blockIdx.y);
    const auto base=static_cast<std::uint64_t>(blockIdx.x)*tile;
    const auto count=static_cast<unsigned>(min(static_cast<std::uint64_t>(tile),p.batch-base));
    const auto go=p.neighborhoods.group_offsets[nh],ge=p.neighborhoods.group_offsets[nh+1];
    if(ge<go || ge-go>32){if(!threadIdx.x)atomicOr(status,status_unsupported_shape);return;}
    const auto j=threadIdx.x;
    if(j<count){
        const auto cell=base+j;
        std::uint32_t pv=0,dv=0;
        for(auto k=go;k<ge;++k){
            const auto group=p.topology.groups[p.neighborhoods.groups[k]];
            if(bit(gates.primal_words,gates.words_per_condition,cell,group.condition_index))pv|=1u<<(k-go);
            if(gates.profile==GateProfile::ste&&bit(gates.derivative_words,gates.words_per_condition,cell,group.condition_index))dv|=1u<<(k-go);
        }
        l.order[nh*p.batch+cell]=cell;
        l.p_signature[nh*p.batch+cell]=pv;l.d_signature[nh*p.batch+cell]=dv;
    }
    __syncthreads();
    // Stable exact-key insertion sort inside one bounded tile. Its measured
    // construction cost is charged; hashes never establish cohort equality.
    if(!threadIdx.x)for(unsigned at=1;at<count;++at){
        const auto index=nh*p.batch+base+at;
        const auto cell=l.order[index];const auto pv=l.p_signature[index],dv=l.d_signature[index];
        unsigned to=at;
        while(to){const auto before=nh*p.batch+base+to-1;
            const auto bp=l.p_signature[before],bd=l.d_signature[before];
            if(bp<pv || (bp==pv && bd<=dv))break;
            l.order[before+1]=l.order[before];l.p_signature[before+1]=bp;l.d_signature[before+1]=bd;--to;
        }
        const auto dst=nh*p.batch+base+to;
        l.order[dst]=cell;l.p_signature[dst]=pv;l.d_signature[dst]=dv;
    }
}

// Contiguous-tile ablation: retain the same local P,D rows and identity map,
// without sorting cells into equal-program cohorts. No numerical data is read.
__global__ void build_masked_tiles(ProblemView p,GateView gates,Layout l,unsigned tile,std::uint32_t* status){
    if(!enabled(status))return;
    const auto nh=static_cast<std::uint64_t>(blockIdx.y);
    const auto base=static_cast<std::uint64_t>(blockIdx.x)*tile;
    const auto count=static_cast<unsigned>(min(static_cast<std::uint64_t>(tile),p.batch-base));
    const auto go=p.neighborhoods.group_offsets[nh],ge=p.neighborhoods.group_offsets[nh+1];
    if(ge<go||ge-go>32){if(!threadIdx.x)atomicOr(status,status_unsupported_shape);return;}
    const auto j=threadIdx.x;
    if(j<count){
        const auto cell=base+j;
        std::uint32_t pv=0,dv=0;
        for(auto k=go;k<ge;++k){
            const auto group=p.topology.groups[p.neighborhoods.groups[k]];
            if(bit(gates.primal_words,gates.words_per_condition,cell,group.condition_index))pv|=1u<<(k-go);
            if(gates.profile==GateProfile::ste&&bit(gates.derivative_words,gates.words_per_condition,cell,group.condition_index))dv|=1u<<(k-go);
        }
        const auto index=nh*p.batch+cell;
        l.order[index]=cell;l.p_signature[index]=pv;l.d_signature[index]=dv;
    }
}

struct Values {
    const ProblemView& p;std::uint64_t cell;const float* staged;
    const std::uint32_t* local;std::uint64_t feature_count;std::uint32_t p_mask;
    const std::uint32_t* membership;unsigned stages;
    __device__ float argument(std::uint64_t slot) const {
        if(!staged)return load(p.x,p.dtype,cell*p.topology.coordinates+p.topology.arguments[slot]);
        const auto k=local[slot];
        return staged[k];
    }
};

__device__ void product_cell(const ProblemView& p,const GateView& gates,
    const ResultView& out,const float* dy,const Values& values,
    std::uint64_t cell,std::uint64_t gid,bool admitted,bool deriv,bool backward,
    const float* staged_theta=nullptr) {
    const auto group=p.topology.groups[gid];
    if(!(backward ? admitted||deriv : admitted))return;
    const auto theta=staged_theta ? *staged_theta : p.theta[group.parameter_index];
    float product=1.f;
    for(std::uint64_t a=0;a<group.argument_count;++a)product*=values.argument(group.argument_begin+a);
    const auto q=admitted?qvalue(gates,p,cell,group.condition_index):0.f;
    if(!backward){
        const auto value=theta*q*product;
        for(std::uint64_t o=0;o<group.output_count;++o){const auto incidence=p.topology.outputs[group.output_begin+o];
            atomicAdd(out.y+cell*p.topology.coordinates+incidence.coordinate,value*incidence.scale);}
        return;
    }
    float upstream=0.f;
    for(std::uint64_t o=0;o<group.output_count;++o){const auto incidence=p.topology.outputs[group.output_begin+o];
        upstream+=dy[cell*p.topology.coordinates+incidence.coordinate]*incidence.scale;}
    if(deriv && out.dq)atomicAdd(out.dq+cell*p.topology.conditions+group.condition_index,upstream*theta*product);
    if(!admitted)return;
    if(out.dtheta)atomicAdd(out.dtheta+group.parameter_index,upstream*q*product);
    // An occurrence is a derivative slot, not a unique coordinate. Excluding
    // the slot without division handles zeros and repeated participants.
    for(std::uint64_t a=0;a<group.argument_count;++a){
        float other=1.f;for(std::uint64_t j=0;j<group.argument_count;++j)
            if(j!=a)other*=values.argument(group.argument_begin+j);
        atomicAdd(out.dx+cell*p.topology.coordinates+p.topology.arguments[group.argument_begin+a],upstream*theta*q*other);
    }
}

__device__ void linear_cell(const ProblemView& p,const GateView& gates,
    const ResultView& out,const float* dy,const Values& values,const Layout& l,
    std::uint64_t cell,std::uint64_t gid,bool admitted,bool deriv,bool backward,
    const float* staged_weights=nullptr) {
    const auto g=p.topology.groups[gid];
    if(!(backward?admitted||deriv:admitted))return;
    const auto I=g.argument_count,O=g.output_count;
    const auto wb=p.weight_offsets[g.parameter_index];
    const auto q=admitted?qvalue(gates,p,cell,g.condition_index):0.f;
    float gate_grad=0;
    for(std::uint64_t o=0;o<O;++o){
        float value=0;
        if(!backward||deriv)for(std::uint64_t i=0;i<I;++i){
            const auto weight=staged_weights?staged_weights[o*I+i]:load(p.weights,p.dtype,wb+o*I+i);
            value+=values.argument(g.argument_begin+i)*weight;
        }
        const float local=rounded(value,p.dtype);
        const auto incidence=p.topology.outputs[g.output_begin+o];
        if(!backward){
            l.local_f[cell*p.topology.output_count+g.output_begin+o]=local;
            atomicAdd(out.y+cell*p.topology.coordinates+incidence.coordinate,q*local*incidence.scale);
        }else{
            const auto upstream=dy[cell*p.topology.coordinates+incidence.coordinate]*incidence.scale;
            if(deriv)gate_grad+=upstream*local;
            l.local_df[cell*p.topology.output_count+g.output_begin+o]=admitted?rounded(upstream*q,p.dtype):0.f;
        }
    }
    if(!backward)return;
    if(deriv && out.dq)atomicAdd(out.dq+cell*p.topology.conditions+g.condition_index,gate_grad);
    for(std::uint64_t i=0;i<I;++i){
        float dx=0;
        if(admitted)for(std::uint64_t o=0;o<O;++o){
            const auto weight=staged_weights?staged_weights[o*I+i]:load(p.weights,p.dtype,wb+o*I+i);
            dx+=l.local_df[cell*p.topology.output_count+g.output_begin+o]*weight;
        }
        const auto value=rounded(dx,p.dtype);
        if(out.dx_group)store(out.dx_group,p.dtype,cell*p.topology.argument_count+g.argument_begin+i,value);
        if(admitted)atomicAdd(out.dx+cell*p.topology.coordinates+p.topology.arguments[g.argument_begin+i],value);
    }
}

template<bool Backward>
__global__ void direct_consumer(Backend backend,BodyKind body,ProblemView p,GateView gates,
    ResultView out,const float* dy,Layout l,std::uint32_t* status){
    if(!enabled(status))return;
    const unsigned warp=threadIdx.x/32,lane=threadIdx.x%32;
    const auto pair=static_cast<std::uint64_t>(blockIdx.x)*4+warp;
    const auto cell=pair/p.topology.group_count;
    auto gid=pair%p.topology.group_count;
    if(cell>=p.batch)return;
    if(backend==Backend::bitmap){
        const auto count=Backward?l.derivative_count[cell]:l.primal_count[cell];
        if(gid>=count)return;
        gid=(Backward?l.derivative_work:l.primal_work)[cell*p.topology.group_count+gid];
    }
    const auto c=p.topology.groups[gid].condition_index;
    const bool pv=primal(backend,gates,l,p,cell,c),dv=derivative(backend,gates,l,p,cell,c);
    // Every lane observes the same gate and group; exclusion is warp-uniform.
    if(!(Backward?pv||dv:pv))return;
    const auto group=p.topology.groups[gid];
    const auto q=pv?qvalue(gates,p,cell,c):0.f;
    if(body==BodyKind::ordered_product){
        float product=1,upstream=0,exclusive=1;
        if(group.argument_count<=32){
            const auto value=lane<group.argument_count?
                load(p.x,p.dtype,cell*p.topology.coordinates+p.topology.arguments[group.argument_begin+lane]):1.f;
            const auto prefix=ordered_prefix(value,lane);
            product=__shfl_sync(0xffffffffu,prefix,31);
            if(Backward&&pv){
                const auto suffix=ordered_suffix(value,lane);
                const auto left=__shfl_up_sync(0xffffffffu,prefix,1),right=__shfl_down_sync(0xffffffffu,suffix,1);
                exclusive=(lane==0?1.f:left)*(lane==31?1.f:right);
            }
        }
        if(!lane){
            if(group.argument_count>32)for(std::uint64_t a=0;a<group.argument_count;++a)
                product*=load(p.x,p.dtype,cell*p.topology.coordinates+p.topology.arguments[group.argument_begin+a]);
            if(Backward)for(std::uint64_t o=0;o<group.output_count;++o){
                const auto incidence=p.topology.outputs[group.output_begin+o];
                upstream+=dy[cell*p.topology.coordinates+incidence.coordinate]*incidence.scale;}
        }
        product=__shfl_sync(0xffffffffu,product,0);
        upstream=__shfl_sync(0xffffffffu,upstream,0);
        const auto theta=p.theta[group.parameter_index];
        if(!Backward){
            for(std::uint64_t o=lane;o<group.output_count;o+=32){const auto incidence=p.topology.outputs[group.output_begin+o];
                atomicAdd(out.y+cell*p.topology.coordinates+incidence.coordinate,theta*q*product*incidence.scale);}
        }else{
            if(!lane){
                if(dv&&out.dq)atomicAdd(out.dq+cell*p.topology.conditions+c,upstream*theta*product);
                if(pv&&out.dtheta)atomicAdd(out.dtheta+group.parameter_index,upstream*q*product);
            }
            if(pv){
                if(group.argument_count<=32){
                    if(lane<group.argument_count)atomicAdd(out.dx+cell*p.topology.coordinates+p.topology.arguments[group.argument_begin+lane],upstream*theta*q*exclusive);
                }else for(std::uint64_t a=lane;a<group.argument_count;a+=32){
                    float other=1;for(std::uint64_t j=0;j<group.argument_count;++j)if(j!=a)
                        other*=load(p.x,p.dtype,cell*p.topology.coordinates+p.topology.arguments[group.argument_begin+j]);
                    atomicAdd(out.dx+cell*p.topology.coordinates+p.topology.arguments[group.argument_begin+a],upstream*theta*q*other);
                }
            }
        }
    }else{
        const auto wb=p.weight_offsets[group.parameter_index];
        float gate_grad=0;
        for(std::uint64_t o=lane;o<group.output_count;o+=32){
            float value=0;
            if(!Backward||dv)for(std::uint64_t i=0;i<group.argument_count;++i)
                value+=load(p.x,p.dtype,cell*p.topology.coordinates+p.topology.arguments[group.argument_begin+i])*
                       load(p.weights,p.dtype,wb+o*group.argument_count+i);
            const auto local=rounded(value,p.dtype);
            const auto incidence=p.topology.outputs[group.output_begin+o];
            if(!Backward){
                l.local_f[cell*p.topology.output_count+group.output_begin+o]=local;
                atomicAdd(out.y+cell*p.topology.coordinates+incidence.coordinate,q*local*incidence.scale);
            }else{
                const auto upstream=dy[cell*p.topology.coordinates+incidence.coordinate]*incidence.scale;
                if(dv)gate_grad+=upstream*local;
                l.local_df[cell*p.topology.output_count+group.output_begin+o]=pv?rounded(upstream*q,p.dtype):0.f;
            }
        }
        if(Backward){
            for(unsigned shift=16;shift;shift>>=1)gate_grad+=__shfl_down_sync(0xffffffffu,gate_grad,shift);
            if(!lane&&dv&&out.dq)atomicAdd(out.dq+cell*p.topology.conditions+c,gate_grad);
            __syncwarp();
            for(std::uint64_t i=lane;i<group.argument_count;i+=32){
                float dx=0;
                if(pv)for(std::uint64_t o=0;o<group.output_count;++o)
                    dx+=l.local_df[cell*p.topology.output_count+group.output_begin+o]*load(p.weights,p.dtype,wb+o*group.argument_count+i);
                const auto value=rounded(dx,p.dtype);
                if(out.dx_group)store(out.dx_group,p.dtype,cell*p.topology.argument_count+group.argument_begin+i,value);
                if(pv)atomicAdd(out.dx+cell*p.topology.coordinates+p.topology.arguments[group.argument_begin+i],value);
            }
        }
    }
}

template<bool Backward>
__global__ void cohort_consumer(BodyKind body,ProblemView p,GateView gates,ResultView out,
    const float* dy,Layout l,unsigned tile,unsigned stages,std::uint32_t* status){
    if(!enabled(status))return;
    const auto nh=static_cast<std::uint64_t>(blockIdx.y);
    const auto base=static_cast<std::uint64_t>(blockIdx.x)*tile;
    const auto count=static_cast<unsigned>(min(static_cast<std::uint64_t>(tile),p.batch-base));
    const auto gb=p.neighborhoods.group_offsets[nh],ge=p.neighborhoods.group_offsets[nh+1];
    const auto fb=p.neighborhoods.feature_offsets[nh];
    // One warp handles an exact (P,D) cohort. Lane identity is a cell within
    // that cohort, making the group decision uniform even for tiny cohorts.
    for(unsigned cohort=threadIdx.x/32;cohort<count;cohort+=blockDim.x/32){
        const auto index=nh*p.batch+base+cohort;
        const auto pv=l.p_signature[index],dv=l.d_signature[index];
        if(cohort && pv==l.p_signature[index-1] && dv==l.d_signature[index-1])continue;
        unsigned end=cohort+1;while(end<count && l.p_signature[nh*p.batch+base+end]==pv && l.d_signature[nh*p.batch+base+end]==dv)++end;
        for(unsigned j=cohort+threadIdx.x%32;j<end;j+=32){
            const auto cell=l.order[nh*p.batch+base+j];
            Values values{p,cell,nullptr,
                p.neighborhoods.argument_local_indices,p.neighborhoods.max_features,pv,
                p.neighborhoods.feature_group_masks+fb,stages};
            for(auto k=gb;k<ge;++k){const auto local=k-gb,gid=p.neighborhoods.groups[k];
                const bool pass=(pv&(1u<<local))!=0,deriv=(dv&(1u<<local))!=0;
                if(body==BodyKind::ordered_product)product_cell(p,gates,out,dy,values,cell,gid,pass,deriv,Backward);
                else linear_cell(p,gates,out,dy,values,l,cell,gid,pass,deriv,Backward);
            }
        }
    }
}

// Matched ownership control for shared_consumer: same two successive cell
// tiles per CTA and same per-tile cohort assignment, with cached global
// participant/parameter loads instead of shared staging. Keep the measured
// one-tile cohort_consumer and shared_consumer implementations untouched.
template<bool Backward>
__device__ void consume_cached_cell_tile(BodyKind body,const ProblemView& p,const GateView& gates,
    const ResultView& out,const float* dy,const Layout& l,std::uint64_t nh,std::uint64_t base,
    unsigned tile){
    if(base>=p.batch)return;
    const auto count=static_cast<unsigned>(min(static_cast<std::uint64_t>(tile),p.batch-base));
    const auto gb=p.neighborhoods.group_offsets[nh],ge=p.neighborhoods.group_offsets[nh+1];
    const auto fb=p.neighborhoods.feature_offsets[nh];
    for(unsigned cohort=threadIdx.x/32;cohort<count;cohort+=blockDim.x/32){
        const auto index=nh*p.batch+base+cohort;
        const auto pv=l.p_signature[index],dv=l.d_signature[index];
        if(cohort&&pv==l.p_signature[index-1]&&dv==l.d_signature[index-1])continue;
        unsigned end=cohort+1;
        while(end<count&&l.p_signature[nh*p.batch+base+end]==pv&&l.d_signature[nh*p.batch+base+end]==dv)++end;
        for(unsigned j=cohort+threadIdx.x%32;j<end;j+=32){
            const auto cell=l.order[nh*p.batch+base+j];
            Values values{p,cell,nullptr,p.neighborhoods.argument_local_indices,
                p.neighborhoods.max_features,pv,p.neighborhoods.feature_group_masks+fb,1};
            for(auto k=gb;k<ge;++k){const auto local=k-gb,gid=p.neighborhoods.groups[k];
                const bool pass=(pv&(1u<<local))!=0,deriv=(dv&(1u<<local))!=0;
                if(body==BodyKind::ordered_product)product_cell(p,gates,out,dy,values,cell,gid,pass,deriv,Backward);
                else linear_cell(p,gates,out,dy,values,l,cell,gid,pass,deriv,Backward);
            }
        }
    }
}
template<bool Backward>
__global__ void cached_two_tile_consumer(BodyKind body,ProblemView p,GateView gates,ResultView out,
    const float* dy,Layout l,unsigned tile,std::uint32_t* status){
    if(!enabled(status))return;
    const auto nh=static_cast<std::uint64_t>(blockIdx.y);
    const auto base=static_cast<std::uint64_t>(blockIdx.x)*tile*2;
    consume_cached_cell_tile<Backward>(body,p,gates,out,dy,l,nh,base,tile);
    __syncthreads();
    consume_cached_cell_tile<Backward>(body,p,gates,out,dy,l,nh,base+tile,tile);
}

template<bool Backward>
__device__ void consume_masked_cell_tile(BodyKind body,const ProblemView& p,const GateView& gates,
    const ResultView& out,const float* dy,const Layout& l,std::uint64_t nh,std::uint64_t base,
    unsigned tile){
    if(base>=p.batch)return;
    const auto count=static_cast<unsigned>(min(static_cast<std::uint64_t>(tile),p.batch-base));
    const auto gb=p.neighborhoods.group_offsets[nh],ge=p.neighborhoods.group_offsets[nh+1];
    const auto fb=p.neighborhoods.feature_offsets[nh];
    // Lanes own contiguous cells. P,D may differ within a warp, so the helper's
    // exclusion branch is intentionally lane-local rather than cohort-uniform.
    for(unsigned j=threadIdx.x;j<count;j+=blockDim.x){
        const auto cell=base+j,index=nh*p.batch+cell;
        const auto pv=l.p_signature[index],dv=l.d_signature[index];
        Values values{p,cell,nullptr,p.neighborhoods.argument_local_indices,
            p.neighborhoods.max_features,pv,p.neighborhoods.feature_group_masks+fb,1};
        for(auto k=gb;k<ge;++k){const auto local=k-gb,gid=p.neighborhoods.groups[k];
            const bool pass=(pv&(1u<<local))!=0,deriv=(dv&(1u<<local))!=0;
            if(body==BodyKind::ordered_product)product_cell(p,gates,out,dy,values,cell,gid,pass,deriv,Backward);
            else linear_cell(p,gates,out,dy,values,l,cell,gid,pass,deriv,Backward);
        }
    }
}
template<bool Backward>
__global__ void masked_two_tile_consumer(BodyKind body,ProblemView p,GateView gates,ResultView out,
    const float* dy,Layout l,unsigned tile,std::uint32_t* status){
    if(!enabled(status))return;
    const auto nh=static_cast<std::uint64_t>(blockIdx.y);
    const auto base=static_cast<std::uint64_t>(blockIdx.x)*tile*2;
    consume_masked_cell_tile<Backward>(body,p,gates,out,dy,l,nh,base,tile);
    __syncthreads();
    consume_masked_cell_tile<Backward>(body,p,gates,out,dy,l,nh,base+tile,tile);
}

template<bool Backward>
__device__ void stage_cell_tile(const ProblemView& p,const Layout& l,std::uint64_t nh,
    std::uint64_t base,unsigned tile,std::uint64_t fb,std::uint64_t features,
    float* buffer,unsigned start,unsigned stride){
    if(base>=p.batch)return;
    const auto count=min(static_cast<std::uint64_t>(tile),p.batch-base);
    for(std::uint64_t at=start;at<count*features;at+=stride){
        const auto j=at/features,k=at%features,index=nh*p.batch+base+j;
        const auto pv=l.p_signature[index],dv=Backward?l.d_signature[index]:0;
        if(p.neighborhoods.feature_group_masks[fb+k]&(pv|dv)){
            const auto cell=l.order[index];
            buffer[j*p.neighborhoods.max_features+k]=load(p.x,p.dtype,cell*p.topology.coordinates+p.neighborhoods.features[fb+k]);
        }
    }
}
template<bool Backward>
__device__ void consume_cell_tile(BodyKind body,const ProblemView& p,const GateView& gates,
    const ResultView& out,const float* dy,const Layout& l,std::uint64_t nh,std::uint64_t base,
    unsigned tile,const float* buffer,const float* parameters){
    if(base>=p.batch)return;
    const auto count=static_cast<unsigned>(min(static_cast<std::uint64_t>(tile),p.batch-base));
    const auto gb=p.neighborhoods.group_offsets[nh],ge=p.neighborhoods.group_offsets[nh+1];
    const auto fb=p.neighborhoods.feature_offsets[nh];
    for(unsigned cohort=threadIdx.x/32;cohort<count;cohort+=blockDim.x/32){
        const auto index=nh*p.batch+base+cohort;const auto pv=l.p_signature[index],dv=l.d_signature[index];
        if(cohort&&pv==l.p_signature[index-1]&&dv==l.d_signature[index-1])continue;
        unsigned end=cohort+1;while(end<count&&l.p_signature[nh*p.batch+base+end]==pv&&l.d_signature[nh*p.batch+base+end]==dv)++end;
        for(unsigned j=cohort+threadIdx.x%32;j<end;j+=32){
            const auto cell=l.order[nh*p.batch+base+j];
            Values values{p,cell,buffer+j*p.neighborhoods.max_features,p.neighborhoods.argument_local_indices,
                p.neighborhoods.max_features,pv,p.neighborhoods.feature_group_masks+fb,1};
            std::uint64_t weight_offset=0;
            for(auto k=gb;k<ge;++k){const auto local=k-gb,gid=p.neighborhoods.groups[k];
                const bool pass=(pv&(1u<<local))!=0,deriv=(dv&(1u<<local))!=0;
                const float* parameter=parameters+(body==BodyKind::ordered_product?local:weight_offset);
                if(body==BodyKind::ordered_product)product_cell(p,gates,out,dy,values,cell,gid,pass,deriv,Backward,parameter);
                else linear_cell(p,gates,out,dy,values,l,cell,gid,pass,deriv,Backward,parameter);
                const auto group=p.topology.groups[gid];weight_offset+=group.argument_count*group.output_count;
            }
        }
    }
}

// Both variants process two successive cell tiles and reuse exactly the same
// parameter stage. With two buffers, warp0 copies the next demanded tile while
// the remaining warps consume this tile; warp0 then consumes its own cohorts.
// SM70 uses ordinary global loads/shared stores, not asynchronous copy ISA.
template<bool Backward>
__global__ void shared_consumer(BodyKind body,ProblemView p,GateView gates,ResultView out,
    const float* dy,Layout l,unsigned tile,unsigned stages,std::uint32_t* status){
    if(!enabled(status))return;
    const auto nh=static_cast<std::uint64_t>(blockIdx.y),base=static_cast<std::uint64_t>(blockIdx.x)*tile*2;
    const auto gb=p.neighborhoods.group_offsets[nh],ge=p.neighborhoods.group_offsets[nh+1];
    const auto fb=p.neighborhoods.feature_offsets[nh],features=p.neighborhoods.feature_offsets[nh+1]-fb;
    extern __shared__ float shared[];
    auto* parameter=shared+tile*p.neighborhoods.max_features*stages;
    __shared__ std::uint32_t demand;
    if(!threadIdx.x){std::uint32_t bits=0;
        const auto end=min(p.batch,base+tile*2);
        for(auto j=base;j<end;++j)bits|=l.p_signature[nh*p.batch+j]|(Backward?l.d_signature[nh*p.batch+j]:0);
        demand=bits;
    }
    __syncthreads();
    for(auto k=gb+threadIdx.x;k<ge;k+=blockDim.x){const auto local=k-gb;
        const auto group=p.topology.groups[p.neighborhoods.groups[k]];
        if(!(demand&(1u<<local)))continue;
        if(body==BodyKind::ordered_product)parameter[local]=p.theta[group.parameter_index];
        else{std::uint64_t offset=0;for(auto before=gb;before<k;++before){const auto prior=p.topology.groups[p.neighborhoods.groups[before]];
                offset+=prior.argument_count*prior.output_count;}
            const auto source=p.weight_offsets[group.parameter_index];
            for(std::uint64_t w=0;w<group.argument_count*group.output_count;++w)parameter[offset+w]=load(p.weights,p.dtype,source+w);
        }
    }
    stage_cell_tile<Backward>(p,l,nh,base,tile,fb,features,shared,threadIdx.x,blockDim.x);
    __syncthreads();
    if(stages==2&&base+tile<p.batch&&threadIdx.x<32)
        stage_cell_tile<Backward>(p,l,nh,base+tile,tile,fb,features,shared+tile*p.neighborhoods.max_features,threadIdx.x,32);
    consume_cell_tile<Backward>(body,p,gates,out,dy,l,nh,base,tile,shared,parameter);
    __syncthreads();
    if(stages==1){
        stage_cell_tile<Backward>(p,l,nh,base+tile,tile,fb,features,shared,threadIdx.x,blockDim.x);
        __syncthreads();
    }
    consume_cell_tile<Backward>(body,p,gates,out,dy,l,nh,base+tile,tile,
        shared+(stages==2?tile*p.neighborhoods.max_features:0),parameter);
}

__global__ void linear_weight_gradient(Backend backend,ProblemView p,GateView gates,ResultView out,Layout l,
    std::uint32_t* status){
    if(!enabled(status))return;
    const auto gid=static_cast<std::uint64_t>(blockIdx.x);
    const auto g=p.topology.groups[gid];const auto n=g.argument_count*g.output_count;
    for(std::uint64_t w=threadIdx.x;w<n;w+=blockDim.x){
        const auto o=w/g.argument_count,i=w%g.argument_count;
        float sum=0;
        for(std::uint64_t cell=0;cell<p.batch;++cell){
            if(!primal(backend,gates,l,p,cell,g.condition_index))continue;
            sum+=l.local_df[cell*p.topology.output_count+g.output_begin+o]*
                load(p.x,p.dtype,cell*p.topology.coordinates+p.topology.arguments[g.argument_begin+i]);
        }
        const auto value=rounded(sum,p.dtype);
        if(out.dweights_group)store(out.dweights_group,p.dtype,p.group_weight_offsets[gid]+w,value);
        if(out.dweights)atomicAdd(out.dweights+p.weight_offsets[g.parameter_index]+w,value);
    }
}

bool valid(Backend backend,BodyKind body,const ProblemView& p,const GateView& g,
    BackendOptions options,WorkspaceView w){
    if(p.include_baseline)return false; // this study measures mechanism terms only
    std::size_t required=0;
    if(!workspace_bytes(backend,body,p,options,&required)||!w.data||w.bytes<required||!w.status)return false;
    if(p.batch && p.topology.group_count && !p.topology.groups)return false;
    if(p.topology.argument_count && !p.topology.arguments)return false;
    if(p.topology.output_count && !p.topology.outputs)return false;
    if(p.batch && p.topology.argument_count && !p.x)return false;
    if(p.topology.conditions && p.batch){
        if(backend!=Backend::scalar&&!g.primal_words)return false;
        if(g.profile==GateProfile::ste&&!g.derivative_words)return false;
        if((backend!=Backend::scalar||g.profile==GateProfile::ste)&&g.words_per_condition!=(p.batch+31)/32)return false;
    }
    if(backend==Backend::scalar && p.batch && p.topology.conditions && !g.admitted)return false;
    if(g.profile==GateProfile::ste && p.batch && p.topology.conditions && !g.probabilities)return false;
    if(body==BodyKind::ordered_product && p.topology.parameters && !p.theta)return false;
    if(body==BodyKind::grouped_linear && p.topology.group_count && (!p.weights||!p.weight_offsets||!p.group_weight_offsets))return false;
    if(uses_program_rows(backend)){const auto& n=p.neighborhoods;
        if(n.count && (!n.group_offsets||!n.feature_offsets))return false;
        if(n.group_memberships&&!n.groups)return false;
        if(n.feature_memberships&&(!n.features||!n.feature_group_masks))return false;
        if(p.topology.argument_count&&!n.argument_local_indices)return false;
    }
    return true;
}
cudaError_t launch(bool backward_mode,Backend backend,BodyKind body,const ProblemView& p,
    const GateView& g,const float* dy,BackendOptions options,ResultView out,WorkspaceView w,cudaStream_t stream){
    if(!valid(backend,body,p,g,options,w))return cudaErrorInvalidValue;
    const auto l=layout(backend,body,p,w);
    check_header<<<1,1,0,stream>>>(l.header,g.revisions,w.status);
    if(!p.batch||!p.topology.group_count)return cudaGetLastError();
    if(backward_mode && (!dy||!out.dx))return cudaErrorInvalidValue;
    if(!backward_mode && !out.y)return cudaErrorInvalidValue;
    if(uses_program_rows(backend)){
        const auto tiles_per_block=backend==Backend::cohort_direct?1u:2u;
        const dim3 grid(static_cast<unsigned>((p.batch+options.cell_tile*tiles_per_block-1)/(options.cell_tile*tiles_per_block)),static_cast<unsigned>(p.neighborhoods.count));
        const auto local_parameters=body==BodyKind::ordered_product?p.neighborhoods.max_groups:p.neighborhoods.max_group_weights;
        const auto smem=backend==Backend::cohort_shared?
            sizeof(float)*(options.cell_tile*p.neighborhoods.max_features*options.stage_count+local_parameters):0;
        if(backward_mode){
            if(backend==Backend::cohort_shared)shared_consumer<true><<<grid,128,smem,stream>>>(body,p,g,out,dy,l,options.cell_tile,options.stage_count,w.status);
            else if(backend==Backend::masked_direct_two_tile)masked_two_tile_consumer<true><<<grid,128,0,stream>>>(body,p,g,out,dy,l,options.cell_tile,w.status);
            else if(backend==Backend::cohort_direct_two_tile)cached_two_tile_consumer<true><<<grid,128,0,stream>>>(body,p,g,out,dy,l,options.cell_tile,w.status);
            else cohort_consumer<true><<<grid,128,0,stream>>>(body,p,g,out,dy,l,options.cell_tile,options.stage_count,w.status);
        }else{
            if(backend==Backend::cohort_shared)shared_consumer<false><<<grid,128,smem,stream>>>(body,p,g,out,dy,l,options.cell_tile,options.stage_count,w.status);
            else if(backend==Backend::masked_direct_two_tile)masked_two_tile_consumer<false><<<grid,128,0,stream>>>(body,p,g,out,dy,l,options.cell_tile,w.status);
            else if(backend==Backend::cohort_direct_two_tile)cached_two_tile_consumer<false><<<grid,128,0,stream>>>(body,p,g,out,dy,l,options.cell_tile,w.status);
            else cohort_consumer<false><<<grid,128,0,stream>>>(body,p,g,out,dy,l,options.cell_tile,options.stage_count,w.status);
        }
    }else{
        const auto blocks=static_cast<unsigned>((p.batch*p.topology.group_count+3)/4);
        if(backward_mode)direct_consumer<true><<<blocks,128,0,stream>>>(backend,body,p,g,out,dy,l,w.status);
        else direct_consumer<false><<<blocks,128,0,stream>>>(backend,body,p,g,out,dy,l,w.status);
    }
    if(backward_mode && body==BodyKind::grouped_linear)
        linear_weight_gradient<<<static_cast<unsigned>(p.topology.group_count),128,0,stream>>>(backend,p,g,out,l,w.status);
    return cudaGetLastError();
}
} // namespace

bool workspace_bytes(Backend backend,BodyKind body,const ProblemView& p,BackendOptions options,std::size_t* bytes) noexcept {
    if(!bytes || (options.cell_tile!=32&&options.cell_tile!=64&&options.cell_tile!=128) ||
       (options.stage_count!=1&&options.stage_count!=2))return false;
    const auto max=std::numeric_limits<std::size_t>::max();
    if(p.batch>max || p.topology.group_count>max || p.topology.conditions>max || p.topology.output_count>max)return false;
    // Bound CUDA grid dimensions and all base logical extents before any
    // backend-specific arithmetic, including backends needing no arrays.
    if(p.batch>UINT32_MAX || p.topology.group_count>UINT32_MAX || p.topology.conditions>UINT32_MAX || p.neighborhoods.count>65535)return false;
    std::size_t extent=0;
    if(!multiply(p.batch,p.topology.group_count,extent)||extent>std::uint64_t(INT32_MAX)*4)return false;
    if(!multiply(p.batch,p.topology.coordinates,extent)||!multiply(p.batch,p.topology.conditions,extent)||
       !multiply(p.batch,p.topology.argument_count,extent))return false;
    std::size_t n=0,count=0;
    if(!add(n,1,sizeof(Header)))return false;
    if(backend==Backend::bitmap){
        if(!multiply(p.batch,(p.topology.conditions+63)/64,count))return false;
        if(!add(n,count,8)||!add(n,count,8))return false;
        if(!multiply(p.batch,p.topology.group_count,count))return false;
        if(!add(n,count,8)||!add(n,count,8)||!add(n,p.batch,8)||!add(n,p.batch,8))return false;
    }
    if(uses_program_rows(backend)){
        if(p.neighborhoods.max_groups>32 || !multiply(p.batch,p.neighborhoods.count,count))return false;
        if(!add(n,count,8)||!add(n,count,4)||!add(n,count,4))return false;
        if(backend==Backend::cohort_shared){
            std::size_t stage=0,params=body==BodyKind::ordered_product?p.neighborhoods.max_groups:p.neighborhoods.max_group_weights;
            if(!multiply(options.cell_tile,p.neighborhoods.max_features,stage)||!multiply(stage,options.stage_count,stage))return false;
            if(params>max-stage || stage+params>12000)return false; // <48KiB, including static shared metadata
        }
    }
    if(body==BodyKind::grouped_linear){
        if(!multiply(p.batch,p.topology.output_count,count)||!add(n,count,4)||!add(n,count,4))return false;
    }
    if(n>max-15)return false;*bytes=n+15;return true;
}

cudaError_t prepare(Backend backend,BodyKind body,const ProblemView& p,const GateView& g,
    BackendOptions options,WorkspaceView w,cudaStream_t stream) noexcept {
    if(!valid(backend,body,p,g,options,w))return cudaErrorInvalidValue;
    const auto l=layout(backend,body,p,w);
    save_header<<<1,1,0,stream>>>(l.header,g.revisions);
    if(!p.batch)return cudaGetLastError();
    if(backend==Backend::bitmap){
        const auto blocks=static_cast<unsigned>((p.batch+127)/128);
        materialize_bitmaps<<<blocks,128,0,stream>>>(p,g,l,w.status);
        materialize_worksets<<<blocks,128,0,stream>>>(p,g,l,w.status);
    }else if(uses_program_rows(backend)&&p.neighborhoods.count){
        const dim3 grid(static_cast<unsigned>((p.batch+options.cell_tile-1)/options.cell_tile),static_cast<unsigned>(p.neighborhoods.count));
        if(backend==Backend::masked_direct_two_tile)build_masked_tiles<<<grid,128,0,stream>>>(p,g,l,options.cell_tile,w.status);
        else build_cohorts<<<grid,128,0,stream>>>(p,g,l,options.cell_tile,w.status);
    }
    return cudaGetLastError();
}
cudaError_t forward(Backend backend,BodyKind body,const ProblemView& p,const GateView& g,
    BackendOptions options,ResultView out,WorkspaceView w,cudaStream_t stream) noexcept {
    return launch(false,backend,body,p,g,nullptr,options,out,w,stream);
}
cudaError_t backward(Backend backend,BodyKind body,const ProblemView& p,const GateView& g,
    const float* dy,BackendOptions options,ResultView out,WorkspaceView w,cudaStream_t stream) noexcept {
    return launch(true,backend,body,p,g,dy,options,out,w,stream);
}
bool inspect_descriptor(Backend backend,BodyKind body,const ProblemView& p,
    BackendOptions options,WorkspaceView w,DescriptorDebugView* out)noexcept{
    std::size_t bytes=0;if(!out||!workspace_bytes(backend,body,p,options,&bytes)||!w.data||w.bytes<bytes)return false;
    const auto l=layout(backend,body,p,w);
    *out={l.primal_bitmap,l.derivative_bitmap,l.primal_work,l.derivative_work,l.primal_count,l.derivative_count,
        l.order,l.p_signature,l.d_signature,(p.topology.conditions+63)/64,p.topology.group_count,
        uses_program_rows(backend)?p.batch*p.neighborhoods.count:0};
    return true;
}
} // namespace cgs
