#include <Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/hybrid_gradient.cuh>
#include <Cellerator/compute/architecture/providers/nvidia/sm70/edge_value_gradient/gradient_pack.cuh>
#include <algorithm>
#include <limits>
#include <new>
namespace cellerator::compute::architecture::providers::nvidia::sm70::edge_value_gradient {
struct hybrid_gradient {
    contract::support_view_v1 support{};
    cudaStream_t stream=nullptr;
    contract::edge_ref_v1 *residual=nullptr;
    std::uint32_t *gather=nullptr, *slots=nullptr;
    contract::rectangular_tile_v1 *descriptors=nullptr;
    __half *panels=nullptr;
    float *scores=nullptr;
    contract::prepared_rectangular_v1 rectangular;
    hybrid_report report;
    bool poisoned=false;
};
namespace {
__global__ void extract_kernel(const float *scores,const std::uint32_t *slots,
    std::uint32_t count,float *output) {
    for(std::uint64_t i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
        i<count;i+=std::uint64_t(gridDim.x)*blockDim.x)
        if(slots[i]!=absent_slot)output[slots[i]]=scores[i];
}
bool overlap(const void *a,std::uint64_t an,const void *b,std::uint64_t bn) {
    if(!an||!bn)return false;
    auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    auto max=std::numeric_limits<std::uintptr_t>::max();
    return !a||!b||x>max-an||y>max-bn||(x<y+bn&&y<x+an);
}
}
void destroy_hybrid_gradient(hybrid_gradient *p) noexcept {
    if(!p)return;
    cudaStreamSynchronize(p->stream);
    cudaFree(p->residual);cudaFree(p->gather);cudaFree(p->slots);
    cudaFree(p->descriptors);cudaFree(p->panels);cudaFree(p->scores);delete p;
}
hybrid_report inspect_hybrid_gradient(const hybrid_gradient &p) noexcept{return p.report;}
bool overlaps_hybrid_storage(const hybrid_gradient& p,const void* data,std::uint64_t size) noexcept {
    const void* owned[]={p.panels,p.scores,p.gather,p.slots,p.descriptors,p.residual};
    const std::uint64_t bytes[]={p.report.tile_count*1024,p.report.tile_count*1024,
        p.report.tile_count*128,p.report.tile_count*1024,p.report.tile_count*sizeof(contract::rectangular_tile_v1),
        p.report.residual_count*sizeof(contract::edge_ref_v1)};
    for(unsigned i=0;i<6;++i)if(overlap(data,size,owned[i],bytes[i]))return true;
    return false;
}
contract::status_v1 prepare_hybrid_gradient(const contract::edge_ref_v1 *host,
    contract::support_view_v1 support,const tile_hint *hints,std::uint64_t hint_count,
    std::uint64_t limit,cudaStream_t stream,hybrid_gradient **out,std::uint64_t scratch_byte_limit) noexcept {
    using contract::status_v1;
    if(!out||*out|| (support.local_edge_count&&(!host||!support.edges)))return status_v1::invalid_argument;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess) return status_v1::cuda_failure;
    if (capture != cudaStreamCaptureStatusNone) return status_v1::unsupported;
    if(!limit)limit=256u*1024u*1024u;
    hybrid_gradient *p=nullptr;
    try {
        const std::uint64_t count=support.local_edge_count;
        if(count*sizeof(gradient_edge)>=limit)return status_v1::unsupported;
        std::vector<gradient_edge> edges(count);
        for(std::uint32_t i=0;i<count;++i){
            if(host[i].logical_output_local!=i)return status_v1::invalid_argument;
            edges[i]={host[i].source_local,host[i].destination_local,i};
        }
        gradient_cover cover;
        auto covered=prepare_gradient_cover(edges.data(),count,support.source_count,
            support.destination_count,hints,hint_count,limit-count*sizeof(gradient_edge),cover);
        if(covered!=cover_status::success)return covered==cover_status::invalid_argument
            ?status_v1::invalid_argument:status_v1::unsupported;
        const auto tiles=cover.tiles.size(),residual=cover.residual.size();
        const std::uint64_t persistent=residual*sizeof(contract::edge_ref_v1)
            +tiles*(288u*sizeof(std::uint32_t)+sizeof(contract::rectangular_tile_v1));
        const std::uint64_t scratch=tiles*256u*(2*sizeof(__half)+sizeof(float));
        const std::uint64_t preparation=cover.preparation_byte_bound+count*sizeof(gradient_edge)
            +persistent*2+scratch+sizeof(hybrid_gradient);
        if(preparation>limit || scratch>scratch_byte_limit)return status_v1::unsupported;
        p=new hybrid_gradient;p->support=support;p->stream=stream;
        p->report.tile_count=tiles;p->report.residual_count=residual;
        p->report.persistent_bytes=persistent+sizeof(hybrid_gradient);
        p->report.scratch_bytes=scratch;p->report.preparation_byte_bound=preparation;
        auto alloc=[&](auto **ptr,std::size_t bytes){return !bytes||cudaMalloc(ptr,bytes)==cudaSuccess;};
        if(!alloc(&p->residual,residual*sizeof(contract::edge_ref_v1))
            ||!alloc(&p->gather,tiles*32*sizeof(std::uint32_t))
            ||!alloc(&p->slots,tiles*256*sizeof(std::uint32_t))
            ||!alloc(&p->descriptors,tiles*sizeof(contract::rectangular_tile_v1))
            ||!alloc(&p->panels,tiles*512*sizeof(__half))
            ||!alloc(&p->scores,tiles*256*sizeof(float))){destroy_hybrid_gradient(p);return status_v1::cuda_failure;}
        std::vector<contract::edge_ref_v1> refs(residual);
        for(std::size_t i=0;i<residual;++i){auto e=cover.residual[i];refs[i]={e.source,e.destination,e.physical};}
        std::vector<std::uint32_t> gather(tiles*32,absent_slot),slots(tiles*256,absent_slot);
        std::vector<contract::rectangular_tile_v1> descriptors(tiles);
        for(std::size_t t=0;t<tiles;++t){auto &tile=cover.tiles[t];
            for(unsigned i=0;i<tile.gather.source_count;++i)gather[t*16+i]=tile.gather.sources[i];
            for(unsigned i=0;i<tile.gather.destination_count;++i)gather[tiles*16+t*16+i]=tile.gather.destinations[i];
            for(unsigned i=0;i<256;++i)slots[t*256+i]=tile.physical_slots[i];
            descriptors[t]={unsigned(t*16),unsigned(t*16),unsigned(t*256)};
        }
        auto upload=[&](void *dst,const void *src,std::size_t bytes){return !bytes||cudaMemcpyAsync(dst,src,bytes,cudaMemcpyHostToDevice,stream)==cudaSuccess;};
        if(!upload(p->residual,refs.data(),refs.size()*sizeof(refs[0]))
            ||!upload(p->gather,gather.data(),gather.size()*sizeof(gather[0]))
            ||!upload(p->slots,slots.data(),slots.size()*sizeof(slots[0]))){destroy_hybrid_gradient(p);return status_v1::cuda_failure;}
        if(tiles){contract::rectangular_request_v1 request{};
            request.tile_count=tiles;request.dense={p->panels,p->panels+tiles*256,16};
            request.source_count=request.destination_count=tiles*16;
            request.source_stride=request.destination_stride=16;
            request.source_capacity=request.destination_capacity=tiles*256;
            request.projection_output=p->scores;request.output_capacity=tiles*256;request.stream=stream;
            auto prepared=contract::prepare_rectangular_v1(request,descriptors.data(),p->descriptors,tiles,p->rectangular);
            if(prepared!=status_v1::success){destroy_hybrid_gradient(p);return prepared;}
        }
        if(cudaStreamSynchronize(stream)!=cudaSuccess){destroy_hybrid_gradient(p);return status_v1::cuda_failure;}
        *out=p;return status_v1::success;
    }catch(const std::bad_alloc&){destroy_hybrid_gradient(p);return status_v1::unsupported;}
}
contract::status_v1 enqueue_hybrid_gradient(hybrid_gradient &p,
    const relation_gradient_request &r,bool hybrid) noexcept {
    using contract::status_v1;
    if(p.poisoned)return status_v1::cuda_failure;
    if(r.stream!=p.stream||r.support.edges!=p.support.edges
        ||r.support.local_edge_count!=p.support.local_edge_count
        ||r.support.source_count!=p.support.source_count
        ||r.support.destination_count!=p.support.destination_count)return status_v1::invalid_argument;
    if(hybrid&&(!r.half_rounded||!p.report.tile_count))return status_v1::unsupported;
    const std::uint64_t nx=std::uint64_t(p.support.source_count)*16,ny=std::uint64_t(p.support.destination_count)*16;
    if(r.source_capacity<nx||r.cotangent_capacity<ny||r.output_capacity<p.support.local_edge_count
        ||(nx&&!r.source)||(ny&&!r.cotangent)||(p.support.local_edge_count&&!r.output)
        ||(r.half_rounded&&(r.source_scratch_capacity<nx||r.cotangent_scratch_capacity<ny
            ||(nx&&!r.source_scratch)||(ny&&!r.cotangent_scratch))))return status_v1::invalid_argument;
    if (reinterpret_cast<std::uintptr_t>(r.source)%alignof(float)
        || reinterpret_cast<std::uintptr_t>(r.cotangent)%alignof(float)
        || reinterpret_cast<std::uintptr_t>(r.output)%alignof(float)
        || (r.half_rounded && (reinterpret_cast<std::uintptr_t>(r.source_scratch)%32u
            || reinterpret_cast<std::uintptr_t>(r.cotangent_scratch)%32u)))
        return status_v1::invalid_argument;
    const void *external[]={r.source,r.cotangent,r.output,r.source_scratch,r.cotangent_scratch};
    const std::uint64_t sizes[]={nx*4,ny*4,std::uint64_t(p.support.local_edge_count)*4,r.half_rounded?nx*2:0,r.half_rounded?ny*2:0};
    for(unsigned i=0;i<5;++i)for(unsigned j=i+1;j<5;++j)
        if((i>=2||j>=2)&&overlap(external[i],sizes[i],external[j],sizes[j]))return status_v1::invalid_argument;
    for (unsigned i=0;i<5;++i)
        if (overlap(external[i], sizes[i], p.support.edges,
            std::uint64_t(p.support.local_edge_count)*sizeof(contract::edge_ref_v1)))
            return status_v1::invalid_argument;
    const void *owned[]={p.panels,p.scores,p.gather,p.slots,p.descriptors,p.residual};
    const std::uint64_t bytes[]={p.report.tile_count*1024,p.report.tile_count*1024,
        p.report.tile_count*128,p.report.tile_count*1024,p.report.tile_count*sizeof(contract::rectangular_tile_v1),p.report.residual_count*sizeof(contract::edge_ref_v1)};
    for(unsigned i=0;i<5;++i)for(unsigned j=0;j<6;++j)
        if(overlap(external[i],sizes[i],owned[j],bytes[j]))return status_v1::invalid_argument;
    if(!p.support.local_edge_count)return status_v1::success;
    auto checked=[&](status_v1 s){if(s!=status_v1::success)p.poisoned=true;return s;};
    if(!hybrid){auto s=enqueue_relation_gradient(r);if(s!=status_v1::success)return checked(s);
        ++p.report.sparse_launches;if(r.half_rounded){++p.report.pack_refreshes;p.report.pack_launches+=2;}return s;}
    pack_request a{r.source,p.support.source_count,16,r.source_capacity,r.source_scratch,
        p.support.source_count,16,r.source_scratch_capacity,nullptr,0,true,r.stream};
    pack_request b{r.cotangent,p.support.destination_count,16,r.cotangent_capacity,r.cotangent_scratch,
        p.support.destination_count,16,r.cotangent_scratch_capacity,nullptr,0,true,r.stream};
    auto s=enqueue_gradient_pack(a);if(s!=status_v1::success)return checked(s);++p.report.pack_launches;
    s=enqueue_gradient_pack(b);if(s!=status_v1::success)return checked(s);++p.report.pack_launches;
    const auto tiles=p.report.tile_count;
    a.output=p.panels;a.output_rows=tiles*16;a.output_capacity=tiles*256;a.gather_ids=p.gather;a.gather_capacity=tiles*16;
    b.output=p.panels+tiles*256;b.output_rows=tiles*16;b.output_capacity=tiles*256;b.gather_ids=p.gather+tiles*16;b.gather_capacity=tiles*16;
    s=enqueue_gradient_pack(a);if(s!=status_v1::success)return checked(s);++p.report.pack_launches;
    s=enqueue_gradient_pack(b);if(s!=status_v1::success)return checked(s);++p.report.pack_launches;++p.report.pack_refreshes;
    s=contract::enqueue_rectangular_mma_residual_v1(p.rectangular);if(s!=status_v1::success)return checked(s);++p.report.wmma_launches;
    const unsigned count=tiles*256;
    extract_kernel<<<unsigned(std::min<std::uint64_t>((std::uint64_t(count)+127)/128,65535)),128,0,r.stream>>>(p.scores,p.slots,count,r.output);
    if(cudaGetLastError()!=cudaSuccess)return checked(status_v1::cuda_failure);++p.report.extraction_launches;
    if(p.report.residual_count){contract::launch_request_v1 residual{};
        residual.support={p.residual,0,unsigned(p.report.residual_count),p.support.source_count,p.support.destination_count};
        residual.dense={r.source_scratch,r.cotangent_scratch,16};residual.output=r.output;
        residual.output_order=contract::output_order_v1::logical_edge;residual.stream=r.stream;
        s=contract::enqueue_sparse_v1(residual);if(s!=status_v1::success)return checked(s);++p.report.residual_launches;
    }
    return status_v1::success;
}
}
