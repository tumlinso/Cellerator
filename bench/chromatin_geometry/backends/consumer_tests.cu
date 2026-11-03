#include "consumer_fixture.hh"
#include "descriptor_debug.hh"
#include "../reference/reference.hh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

namespace cgs {
namespace {
void check(cudaError_t error,const char* operation){
    if(error!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(error));
}
struct Allocation {
    void* data=nullptr;std::size_t bytes=0;
    explicit Allocation(std::size_t n):bytes(n){if(n)check(cudaMalloc(&data,n),"allocate consumer fixture");}
    ~Allocation(){if(data)cudaFree(data);}
};
void compare(const std::vector<float>& actual,const std::vector<float>& expected,const char* name,float tolerance=reference::product_abs_tolerance){
    if(actual.size()!=expected.size())throw std::runtime_error(std::string(name)+" size mismatch");
    for(std::size_t i=0;i<actual.size();++i)
        if(!std::isfinite(actual[i]) || std::abs(actual[i]-expected[i])>tolerance*(1+std::abs(expected[i])))
            throw std::runtime_error(std::string(name)+" mismatch at "+std::to_string(i)+" actual="+std::to_string(actual[i])+" expected="+std::to_string(expected[i]));
}
void compare_bound(const std::vector<float>& actual,const std::vector<float>& expected,const std::vector<float>& bounds,const char* name){
    if(actual.size()!=expected.size()||bounds.size()!=expected.size())throw std::runtime_error(std::string(name)+" bound size mismatch");
    for(std::size_t i=0;i<actual.size();++i)if(!reference::within_absolute_bound(actual[i],expected[i],bounds[i]))
        throw std::runtime_error(std::string(name)+" compound bound mismatch at "+std::to_string(i)+" actual="+std::to_string(actual[i])+" expected="+std::to_string(expected[i])+" bound="+std::to_string(bounds[i]));
}
template<class T>std::vector<T> debug_read(const T* source,std::size_t count,cudaStream_t stream){
    std::vector<T> out(count);if(count)check(cudaMemcpyAsync(out.data(),source,count*sizeof(T),cudaMemcpyDeviceToHost,stream),"read execution descriptor");
    check(cudaStreamSynchronize(stream),"finish descriptor read");return out;
}
bool host_derivative(const ConsumerHostFixture& h,std::uint64_t cell,std::uint64_t condition){
    return h.profile==GateProfile::ste&&((h.derivative[condition*((h.batch+31)/32)+cell/32]>>(cell%32))&1u);
}
void validate_descriptor(const ConsumerHostFixture& h,Backend backend,const ProblemView& p,
    BackendOptions options,WorkspaceView workspace,cudaStream_t stream){
    // Read the actual device identity/address maps too. Numerically identical
    // participants or repeated rules must not hide an incorrect identity map.
    const auto groups=debug_read(p.topology.groups,h.groups.size(),stream);
    for(std::size_t j=0;j<groups.size();++j){const auto& a=groups[j];const auto& b=h.groups[j];
        if(a.argument_begin!=b.argument_begin||a.argument_count!=b.argument_count||a.output_begin!=b.output_begin||
           a.output_count!=b.output_count||a.parameter_index!=b.parameter_index||a.condition_index!=b.condition_index||a.logical_id!=b.logical_id)
            throw std::runtime_error("device group identity/address map differs");}
    if(debug_read(p.topology.arguments,h.arguments.size(),stream)!=h.arguments)throw std::runtime_error("device ordered participant map differs");
    const auto outputs=debug_read(p.topology.outputs,h.outputs.size(),stream);
    for(std::size_t j=0;j<outputs.size();++j)if(outputs[j].coordinate!=h.outputs[j].coordinate||
        std::memcmp(&outputs[j].scale,&h.outputs[j].scale,sizeof(float)))throw std::runtime_error("device ordered output incidence differs");
    if(backend==Backend::cohort_direct||backend==Backend::cohort_shared||backend==Backend::cohort_direct_two_tile||backend==Backend::masked_direct_two_tile){const auto& n=p.neighborhoods;
        if(debug_read(n.group_offsets,h.neighborhood_group_offsets.size(),stream)!=h.neighborhood_group_offsets||
           debug_read(n.groups,h.neighborhood_groups.size(),stream)!=h.neighborhood_groups||
           debug_read(n.feature_offsets,h.neighborhood_feature_offsets.size(),stream)!=h.neighborhood_feature_offsets||
           debug_read(n.features,h.neighborhood_features.size(),stream)!=h.neighborhood_features||
           debug_read(n.feature_group_masks,h.feature_group_masks.size(),stream)!=h.feature_group_masks||
           debug_read(n.argument_local_indices,h.argument_local_indices.size(),stream)!=h.argument_local_indices)
            throw std::runtime_error("device neighborhood identity/address map differs");
    }
    DescriptorDebugView d{};
    if(!inspect_descriptor(backend,h.body,p,options,workspace,&d))throw std::runtime_error("cannot inspect execution descriptor");
    if(backend==Backend::bitmap){
        const auto pv=debug_read(d.primal_bitmap,h.batch*d.bitmap_words_per_cell,stream);
        const auto dv=debug_read(d.derivative_bitmap,h.batch*d.bitmap_words_per_cell,stream);
        const auto pc=debug_read(d.primal_count,h.batch,stream),dc=debug_read(d.derivative_count,h.batch,stream);
        for(std::uint64_t cell=0;cell<h.batch;++cell){
            for(std::uint64_t word=0;word<d.bitmap_words_per_cell;++word){
                std::uint64_t expected_p=0,expected_d=0;
                for(unsigned bit=0;bit<64&&word*64+bit<h.conditions;++bit){const auto c=word*64+bit;
                    if(h.admitted[cell*h.conditions+c])expected_p|=std::uint64_t(1)<<bit;
                    if(host_derivative(h,cell,c))expected_d|=std::uint64_t(1)<<bit;
                }
                if(pv[cell*d.bitmap_words_per_cell+word]!=expected_p||dv[cell*d.bitmap_words_per_cell+word]!=expected_d)
                    throw std::runtime_error("execution bitmap differs bit-for-bit");
            }
            std::vector<std::uint64_t> expected_p,expected_d;
            for(std::uint64_t g=0;g<h.groups.size();++g){const auto c=h.groups[g].condition_index;
                const bool pass=h.admitted[cell*h.conditions+c]!=0;
                if(pass)expected_p.push_back(g);
                if(pass||host_derivative(h,cell,c))expected_d.push_back(g);
            }
            if(pc[cell]!=expected_p.size()||dc[cell]!=expected_d.size())throw std::runtime_error("execution workset count differs");
            const auto pw=debug_read(d.primal_work+cell*d.work_capacity_per_cell,pc[cell],stream);
            const auto dw=debug_read(d.derivative_work+cell*d.work_capacity_per_cell,dc[cell],stream);
            if(pw!=expected_p||dw!=expected_d)throw std::runtime_error("execution group-ordinal workset differs");
            std::vector<std::uint64_t> actual_raw,expected_raw;
            for(const auto ordinal:dw)actual_raw.push_back(h.groups.at(ordinal).logical_id);
            for(const auto ordinal:expected_d)expected_raw.push_back(h.groups.at(ordinal).logical_id);
            if(actual_raw!=expected_raw)throw std::runtime_error("execution workset raw identities differ");
        }
    }else if(backend==Backend::cohort_direct||backend==Backend::cohort_shared||backend==Backend::cohort_direct_two_tile||backend==Backend::masked_direct_two_tile){
        const auto order=debug_read(d.cohort_order,d.cohort_entries,stream);
        const auto pv=debug_read(d.primal_signature,d.cohort_entries,stream),dv=debug_read(d.derivative_signature,d.cohort_entries,stream);
        const auto neighborhoods=h.neighborhood_group_offsets.size()-1;
        for(std::uint64_t nh=0;nh<neighborhoods;++nh)for(std::uint64_t base=0;base<h.batch;base+=options.cell_tile){
            using Record=std::tuple<std::uint32_t,std::uint32_t,std::uint64_t>;
            std::vector<Record> expected;
            const auto end=std::min(h.batch,base+options.cell_tile);
            for(auto cell=base;cell<end;++cell){std::uint32_t ep=0,ed=0;
                const auto gb=h.neighborhood_group_offsets[nh],ge=h.neighborhood_group_offsets[nh+1];
                for(auto k=gb;k<ge;++k){const auto gid=h.neighborhood_groups[k],c=h.groups[gid].condition_index;
                    if(h.admitted[cell*h.conditions+c])ep|=1u<<(k-gb);
                    if(host_derivative(h,cell,c))ed|=1u<<(k-gb);
                }
                expected.emplace_back(ep,ed,cell);
            }
            if(backend!=Backend::masked_direct_two_tile)std::stable_sort(expected.begin(),expected.end(),[](const auto& a,const auto& b){
                return std::tie(std::get<0>(a),std::get<1>(a))<std::tie(std::get<0>(b),std::get<1>(b));});
            std::vector<std::uint64_t> permutation;
            for(std::size_t j=0;j<expected.size();++j){const auto at=nh*h.batch+base+j;
                if(order[at]!=std::get<2>(expected[j])||pv[at]!=std::get<0>(expected[j])||dv[at]!=std::get<1>(expected[j]))
                    throw std::runtime_error("exact local program/order differs");
                permutation.push_back(order[at]);
                // Resolve the encoded membership back to raw group identity.
                std::vector<std::uint64_t> actual_p,actual_d,expected_p,expected_d;
                const auto gb=h.neighborhood_group_offsets[nh],ge=h.neighborhood_group_offsets[nh+1];
                const auto cell=order[at];
                for(auto k=gb;k<ge;++k){const auto gid=h.neighborhood_groups[k],raw=h.groups[gid].logical_id,c=h.groups[gid].condition_index;
                    if(pv[at]&(1u<<(k-gb)))actual_p.push_back(raw);
                    if((pv[at]|dv[at])&(1u<<(k-gb)))actual_d.push_back(raw);
                    if(h.admitted[cell*h.conditions+c])expected_p.push_back(raw);
                    if(h.admitted[cell*h.conditions+c]||host_derivative(h,cell,c))expected_d.push_back(raw);
                }
                if(actual_p!=expected_p||actual_d!=expected_d)throw std::runtime_error("cohort raw group membership differs");
            }
            std::sort(permutation.begin(),permutation.end());
            for(std::size_t j=0;j<permutation.size();++j)if(permutation[j]!=base+j)throw std::runtime_error("cohort cell identities are not a permutation");
        }
    }
}
} // namespace

struct ConsumerDeviceFixture::Impl {
    std::vector<std::unique_ptr<Allocation>> allocations;
    ProblemView p{};GateView g{};ResultView r{};const float* dy=nullptr;std::uint32_t* status=nullptr;
    BodyKind body{};
    void* allocate(std::size_t n){allocations.emplace_back(new Allocation(n));return allocations.back()->data;}
    template<class T>T* copy(const std::vector<T>& values,cudaStream_t stream){
        auto* result=static_cast<T*>(allocate(values.size()*sizeof(T)));
        if(!values.empty())check(cudaMemcpyAsync(result,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice,stream),"upload consumer fixture");
        return result;
    }
    template<class T>T* output(std::size_t n){return static_cast<T*>(allocate(n*sizeof(T)));}
};
ConsumerDeviceFixture::ConsumerDeviceFixture(const ConsumerHostFixture& h,cudaStream_t stream):impl_(new Impl){
    auto& d=*impl_;d.p=h.problem();d.g=h.gates();d.body=h.body;
    d.p.topology.groups=d.copy(h.groups,stream);d.p.topology.arguments=d.copy(h.arguments,stream);d.p.topology.outputs=d.copy(h.outputs,stream);
    d.p.x=h.dtype==DType::fp16?static_cast<void*>(d.copy(h.x16,stream)):d.copy(h.x32,stream);
    d.p.theta=d.copy(h.theta,stream);
    d.p.weights=h.dtype==DType::fp16?static_cast<void*>(d.copy(h.weights16,stream)):d.copy(h.weights32,stream);
    d.p.weight_offsets=d.copy(h.weight_offsets,stream);d.p.group_weight_offsets=d.copy(h.group_weight_offsets,stream);
    d.p.neighborhoods.group_offsets=d.copy(h.neighborhood_group_offsets,stream);d.p.neighborhoods.groups=d.copy(h.neighborhood_groups,stream);
    d.p.neighborhoods.feature_offsets=d.copy(h.neighborhood_feature_offsets,stream);d.p.neighborhoods.features=d.copy(h.neighborhood_features,stream);
    d.p.neighborhoods.feature_group_masks=d.copy(h.feature_group_masks,stream);d.p.neighborhoods.argument_local_indices=d.copy(h.argument_local_indices,stream);
    d.g.probabilities=d.copy(h.probabilities,stream);d.g.admitted=d.copy(h.admitted,stream);
    d.g.primal_words=d.copy(h.primal,stream);d.g.derivative_words=d.copy(h.derivative,stream);
    d.dy=d.copy(h.dy,stream);
    d.r.y=d.output<float>(h.batch*h.coordinates);d.r.dx=d.output<float>(h.batch*h.coordinates);
    d.r.dtheta=d.output<float>(h.parameters);d.r.dq=d.output<float>(h.batch*h.conditions);
    d.r.dweights=d.output<float>(h.weights32.size());
    const auto element=h.dtype==DType::fp16?2u:4u;
    d.r.dx_group=d.allocate(h.batch*h.arguments.size()*element);
    d.r.dweights_group=d.allocate(d.p.group_weight_count*element);d.status=d.output<std::uint32_t>(1);
    clear_results(stream);check(cudaStreamSynchronize(stream),"finish consumer fixture upload");
}
ConsumerDeviceFixture::~ConsumerDeviceFixture()=default;
ProblemView ConsumerDeviceFixture::problem()const{return impl_->p;}
GateView ConsumerDeviceFixture::gates()const{return impl_->g;}
ResultView ConsumerDeviceFixture::results()const{return impl_->r;}
const float* ConsumerDeviceFixture::dy()const{return impl_->dy;}
WorkspaceView ConsumerDeviceFixture::workspace(Backend backend,BackendOptions options){
    std::size_t bytes=0;if(!workspace_bytes(backend,impl_->body,impl_->p,options,&bytes))throw std::invalid_argument("unsupported consumer workspace shape");
    return {impl_->allocate(bytes),bytes,impl_->status};
}
void ConsumerDeviceFixture::clear_results(cudaStream_t stream){
    auto& d=*impl_;const auto& p=d.p;
    const auto zero=[&](void* ptr,std::size_t bytes){if(bytes)check(cudaMemsetAsync(ptr,0,bytes,stream),"clear consumer result");};
    zero(d.r.y,p.batch*p.topology.coordinates*4);zero(d.r.dx,p.batch*p.topology.coordinates*4);
    zero(d.r.dtheta,p.topology.parameters*4);zero(d.r.dq,p.batch*p.topology.conditions*4);zero(d.r.dweights,p.weight_count*4);
    zero(d.r.dx_group,p.batch*p.topology.argument_count*(p.dtype==DType::fp16?2:4));
    zero(d.r.dweights_group,p.group_weight_count*(p.dtype==DType::fp16?2:4));zero(d.status,4);
}
ConsumerResults ConsumerDeviceFixture::download(cudaStream_t stream)const{
    const auto& d=*impl_;const auto& p=d.p;ConsumerResults out;
    const auto read=[&](std::vector<float>& dst,const float* src,std::size_t n){
        dst.resize(n);if(n)check(cudaMemcpyAsync(dst.data(),src,n*4,cudaMemcpyDeviceToHost,stream),"download consumer result");};
    read(out.y,d.r.y,p.batch*p.topology.coordinates);read(out.dx,d.r.dx,p.batch*p.topology.coordinates);
    read(out.dtheta,d.r.dtheta,p.topology.parameters);read(out.dq,d.r.dq,p.batch*p.topology.conditions);read(out.dweights,d.r.dweights,p.weight_count);
    check(cudaMemcpyAsync(&out.status,d.status,4,cudaMemcpyDeviceToHost,stream),"download consumer status");
    if(p.dtype==DType::fp32){read(out.dx_group,static_cast<const float*>(d.r.dx_group),p.batch*p.topology.argument_count);
        read(out.dweights_group,static_cast<const float*>(d.r.dweights_group),p.group_weight_count);}
    else{
        std::vector<std::uint16_t> dx(p.batch*p.topology.argument_count),dw(p.group_weight_count);
        if(!dx.empty())check(cudaMemcpyAsync(dx.data(),d.r.dx_group,dx.size()*2,cudaMemcpyDeviceToHost,stream),"download local dx");
        if(!dw.empty())check(cudaMemcpyAsync(dw.data(),d.r.dweights_group,dw.size()*2,cudaMemcpyDeviceToHost,stream),"download local dw");
        check(cudaStreamSynchronize(stream),"finish half result downloads");
        out.dx_group.reserve(dx.size());for(auto x:dx)out.dx_group.push_back(reference::half_bits_to_float(x));
        out.dweights_group.reserve(dw.size());for(auto x:dw)out.dweights_group.push_back(reference::half_bits_to_float(x));
    }
    check(cudaStreamSynchronize(stream),"finish consumer downloads");return out;
}

bool run_consumer_tests(cudaStream_t stream,std::string* error){
    std::string context="consumer setup";
    try{
        for(const auto cells:{35u,65u})
        for(const auto body:{BodyKind::ordered_product,BodyKind::grouped_linear})
        for(const auto dtype:{DType::fp16,DType::fp32})
        for(const auto profile:{GateProfile::supplied,GateProfile::ste,GateProfile::teacher}){
            auto spec=workload_manifest().front();spec.cells=cells;spec.features=96;spec.groups=8;spec.overlap_pattern=OverlapPattern::high;
            auto h=make_consumer_fixture(materialize_workload(spec),body,dtype,profile);
            for(std::size_t g=0;g<h.groups.size();++g)h.groups[g].logical_id=1009+17*((g*5)%h.groups.size());
            if(h.outputs.size()>1){h.outputs[1].coordinate=h.outputs[0].coordinate;h.outputs[1].scale=-0.5f;}
            // Explicit zero/repeated slot and arity boundary probes retain
            // source order. All groups share the same bounded shape here.
            if(body==BodyKind::ordered_product){
                const auto old=h.arguments;h.arguments.clear();
                for(std::size_t gid=0;gid<h.groups.size();++gid){auto& group=h.groups[gid];
                    const auto count=gid==0?0:gid==1?32:gid==2?33:3;
                    group.argument_begin=h.arguments.size();group.argument_count=count;
                    for(int a=0;a<count;++a)h.arguments.push_back(a%3);
                }
                h.neighborhood_features={0,1,2};h.feature_group_masks={0xffu,0xffu,0xffu};
                h.neighborhood_feature_offsets={0,3};h.argument_local_indices.resize(h.arguments.size());
                for(std::size_t a=0;a<h.arguments.size();++a)h.argument_local_indices[a]=h.arguments[a];h.max_features=3;
            }
            const auto hp=h.problem();const auto hg=h.gates();
            std::vector<float> ey,ex,et,eq,ew,elx,elw;reference::LinearValidationBounds bounds;
            if(body==BodyKind::ordered_product){
                ey=reference::product_forward(hp,hg).y;const auto b=reference::product_backward(hp,hg,h.dy.data());
                ex=b.dx;et=b.dtheta;eq=b.dq;
            }else{
                const auto f=reference::grouped_linear_forward(hp,hg);const auto b=reference::grouped_linear_backward(hp,hg,h.dy.data(),f);
                ey=f.y;ex=b.dx;eq=b.dq;ew=b.dweights;elx=b.group_dx;elw=b.group_dw;
                bounds=reference::grouped_linear_validation_bounds(hp,hg,h.dy.data(),f,b);
            }
            ConsumerDeviceFixture device(h,stream);
            for(const auto backend:{Backend::scalar,Backend::bitmap,Backend::direct,Backend::cohort_direct,Backend::cohort_shared,Backend::cohort_direct_two_tile,Backend::masked_direct_two_tile})
            for(const auto tile:{32u,64u,128u})for(const auto stages:{1u,2u}){
                if(cells==65&&((backend!=Backend::cohort_direct_two_tile&&backend!=Backend::masked_direct_two_tile)||stages!=1))continue;
                if((backend==Backend::scalar||backend==Backend::bitmap||backend==Backend::direct)&&(tile!=32||stages!=1))continue;
                BackendOptions options{tile,stages};std::size_t bytes=0;
                if(!workspace_bytes(backend,body,device.problem(),options,&bytes))continue;
                context="cells="+std::to_string(cells)+" body="+std::to_string(int(body))+" dtype="+std::to_string(int(dtype))+" gate="+std::to_string(int(profile))+" backend="+std::to_string(int(backend))+" tile="+std::to_string(tile)+" stages="+std::to_string(stages);
                auto w=device.workspace(backend,options);device.clear_results(stream);
                auto gates=device.gates();
                if(backend==Backend::scalar){gates.primal_words=nullptr;
                    if(profile!=GateProfile::ste){gates.derivative_words=nullptr;gates.words_per_condition=0;}}
                check(prepare(backend,body,device.problem(),gates,options,w,stream),"prepare test consumer");
                validate_descriptor(h,backend,device.problem(),options,w,stream);
                check(forward(backend,body,device.problem(),gates,options,device.results(),w,stream),"forward test consumer");
                check(backward(backend,body,device.problem(),gates,device.dy(),options,device.results(),w,stream),"backward test consumer");
                const auto got=device.download(stream);if(got.status)throw std::runtime_error("consumer status="+std::to_string(got.status));
                if(body==BodyKind::ordered_product){compare(got.y,ey,"consumer y");compare(got.dx,ex,"consumer dx");compare(got.dq,eq,"consumer dq");compare(got.dtheta,et,"consumer dtheta");}
                else{compare_bound(got.y,ey,bounds.y,"consumer y");compare_bound(got.dx,ex,bounds.dx,"consumer dx");compare_bound(got.dq,eq,bounds.dq,"consumer dq");
                    compare_bound(got.dweights,ew,bounds.dweights,"consumer dweights");compare_bound(got.dx_group,elx,bounds.group_dx,"consumer localdx");compare_bound(got.dweights_group,elw,bounds.group_dw,"consumer localdw");}
                auto stale=gates;++stale.revisions.accessibility_generation;
                check(forward(backend,body,device.problem(),stale,options,device.results(),w,stream),"enqueue stale guard");
                if(!(device.download(stream).status&status_stale))throw std::runtime_error("consumer stale revision accepted");
                if(backend==Backend::masked_direct_two_tile){
                    device.clear_results(stream);
                    auto stale_mapping=gates;++stale_mapping.revisions.mapping_revision;
                    check(forward(backend,body,device.problem(),stale_mapping,options,device.results(),w,stream),"enqueue stale mapping guard");
                    if(!(device.download(stream).status&status_stale))throw std::runtime_error("consumer stale mapping accepted");
                }
            }
        }
        // Poison every dormant STE participant and parameter in forward.
        // D remains nonempty, so a consumer accidentally using P|D in forward
        // produces NaN instead of the required zero output.
        for(const auto body:{BodyKind::ordered_product,BodyKind::grouped_linear}){
            context="excluded forward poison body="+std::to_string(int(body));
            auto spec=workload_manifest().front();spec.cells=35;spec.features=96;spec.groups=8;
            auto h=make_consumer_fixture(materialize_workload(spec),body,DType::fp16,GateProfile::ste);
            std::fill(h.admitted.begin(),h.admitted.end(),0);std::fill(h.primal.begin(),h.primal.end(),0);
            std::fill(h.x16.begin(),h.x16.end(),0x7e00u);std::fill(h.weights16.begin(),h.weights16.end(),0x7e00u);
            std::fill(h.theta.begin(),h.theta.end(),std::numeric_limits<float>::quiet_NaN());
            ConsumerDeviceFixture device(h,stream);
            for(const auto backend:{Backend::scalar,Backend::bitmap,Backend::direct,Backend::cohort_direct,Backend::cohort_shared,Backend::cohort_direct_two_tile,Backend::masked_direct_two_tile}){
                const BackendOptions options{32,2};auto w=device.workspace(backend,options);device.clear_results(stream);
                check(prepare(backend,body,device.problem(),device.gates(),options,w,stream),"prepare poisoned consumer");
                validate_descriptor(h,backend,device.problem(),options,w,stream);
                check(forward(backend,body,device.problem(),device.gates(),options,device.results(),w,stream),"forward poisoned consumer");
                const auto result=device.download(stream);if(result.status)throw std::runtime_error("poison forward status");
                for(const auto y:result.y)if(y!=0)throw std::runtime_error("excluded forward consumed poisoned input");
            }
        }
        // A completely empty product has no argument allocation at all.
        {
            context="all-zero-arity product";
            auto spec=workload_manifest().front();spec.cells=3;spec.features=96;spec.groups=8;
            auto h=make_consumer_fixture(materialize_workload(spec),BodyKind::ordered_product,DType::fp16,GateProfile::ste);
            h.arguments.clear();h.argument_local_indices.clear();h.x16.clear();h.x32.clear();
            for(auto& g:h.groups){g.argument_begin=0;g.argument_count=0;}
            h.neighborhood_features.clear();h.feature_group_masks.clear();h.neighborhood_feature_offsets={0,0};h.max_features=0;
            std::fill(h.dy.begin(),h.dy.end(),1.f);
            for(std::size_t c=0;c<h.conditions;++c){h.admitted[c]=0;h.primal[c*((h.batch+31)/32)]&=~1u;}
            const auto expected=reference::product_forward(h.problem(),h.gates());
            const auto gradient=reference::product_backward(h.problem(),h.gates(),h.dy.data());
            ConsumerDeviceFixture device(h,stream);
            for(const auto backend:{Backend::scalar,Backend::bitmap,Backend::direct,Backend::cohort_direct,Backend::cohort_shared,Backend::cohort_direct_two_tile,Backend::masked_direct_two_tile}){
                const BackendOptions options{32,2};auto w=device.workspace(backend,options);device.clear_results(stream);
                check(prepare(backend,BodyKind::ordered_product,device.problem(),device.gates(),options,w,stream),"prepare empty product");
                validate_descriptor(h,backend,device.problem(),options,w,stream);
                check(forward(backend,BodyKind::ordered_product,device.problem(),device.gates(),options,device.results(),w,stream),"forward empty product");
                check(backward(backend,BodyKind::ordered_product,device.problem(),device.gates(),device.dy(),options,device.results(),w,stream),"backward empty product");
                const auto result=device.download(stream);if(result.status)throw std::runtime_error("empty product status");
                compare(result.y,expected.y,"empty product y");compare(result.dq,gradient.dq,"empty product dq");compare(result.dtheta,gradient.dtheta,"empty product dtheta");
                bool dormant_nonzero=false;for(std::size_t cell=0;cell<h.batch;++cell)for(std::size_t c=0;c<h.conditions;++c)
                    if(!h.admitted[cell*h.conditions+c]&&result.dq[cell*h.conditions+c]!=0)dormant_nonzero=true;
                if(!dormant_nonzero)throw std::runtime_error("dormant STE fixture has no nonzero dq");
            }
        }
        {
            context="empty batch";
            auto spec=workload_manifest().front();spec.cells=3;spec.features=96;spec.groups=8;
            auto h=make_consumer_fixture(materialize_workload(spec),BodyKind::ordered_product,DType::fp16,GateProfile::ste);
            h.batch=0;h.x16.clear();h.x32.clear();h.dy.clear();h.admitted.clear();h.probabilities.clear();h.primal.clear();h.derivative.clear();
            ConsumerDeviceFixture device(h,stream);
            for(const auto backend:{Backend::scalar,Backend::bitmap,Backend::direct,Backend::cohort_direct,Backend::cohort_shared,Backend::cohort_direct_two_tile,Backend::masked_direct_two_tile}){
                const BackendOptions options{32,1};auto w=device.workspace(backend,options);device.clear_results(stream);
                check(prepare(backend,BodyKind::ordered_product,device.problem(),device.gates(),options,w,stream),"prepare empty batch");
                validate_descriptor(h,backend,device.problem(),options,w,stream);
                check(forward(backend,BodyKind::ordered_product,device.problem(),device.gates(),options,device.results(),w,stream),"forward empty batch");
                check(backward(backend,BodyKind::ordered_product,device.problem(),device.gates(),device.dy(),options,device.results(),w,stream),"backward empty batch");
                const auto result=device.download(stream);if(result.status||!result.y.empty()||!result.dx.empty()||!result.dq.empty())throw std::runtime_error("empty batch result differs");
            }
        }
        return true;
    }catch(const std::exception& e){if(error)*error=context+": "+e.what();return false;}
}

// Narrow, bounded sanitizer qualification: identical cached two-tile ownership
// and supplied Boolean product arithmetic, with independent scalar outputs,
// derivatives and bit-exact descriptors. This is separate from the broader
// matrix to keep each of four sanitizer passes within the probe's budget.
bool run_grouping_probe_qualification(cudaStream_t stream, std::string* error) {
    std::string context = "qualification setup";
    try {
        for (int scenario = 0; scenario < 4; ++scenario) {
            // 65 cells straddles both 32-cell tiles and a second two-tile CTA.
            auto spec = workload_manifest().front();
            spec.cells = 65; spec.features = 96; spec.groups = 8; spec.condition_edges = 8;
            spec.role = WorkloadRole::complete_consumer;
            auto h = make_consumer_fixture(materialize_workload(spec), BodyKind::ordered_product,
                                           DType::fp16, GateProfile::supplied);
            context = scenario == 0 ? "tail-65-arity-repeated-zero" : scenario == 1 ?
                "zero-arity" : scenario == 2 ? "poisoned-excluded" : "empty-batch";
            for (std::size_t g=0; g<h.groups.size(); ++g) h.groups[g].logical_id = 1009 + 17*((g*5)%h.groups.size());
            // Explicit zero and repeated input values with 0, 32 and 33 ordered slots.
            h.arguments.clear();
            for (std::size_t gid=0; gid<h.groups.size(); ++gid) {
                auto& g=h.groups[gid]; g.argument_begin=h.arguments.size();
                g.argument_count = scenario == 1 ? 0 : gid == 0 ? 0 : gid == 1 ? 32 : gid == 2 ? 33 : 3;
                for (std::uint64_t a=0; a<g.argument_count; ++a) h.arguments.push_back(a%3);
            }
            h.neighborhood_features = scenario == 1 ? std::vector<std::uint64_t>{} : std::vector<std::uint64_t>{0,1,2};
            h.feature_group_masks = scenario == 1 ? std::vector<std::uint32_t>{} : std::vector<std::uint32_t>{0xfeu,0xfeu,0xfeu};
            h.neighborhood_feature_offsets = {0, h.neighborhood_features.size()};
            h.argument_local_indices.resize(h.arguments.size());
            for (std::size_t a=0; a<h.arguments.size(); ++a) h.argument_local_indices[a]=h.arguments[a];
            h.max_features=h.neighborhood_features.size();
            // Half the cells have an exact zero participant; the others keep
            // nonzero repeated operands at the arity boundary.
            if (scenario == 0) {
                for (std::uint64_t cell=0; cell<h.batch; cell+=2) h.x16[cell*h.coordinates]=0;
                for (const auto gid : {1u, 2u}) {
                    unsigned admitted_zero = 0, admitted_nonzero = 0;
                    for (std::uint64_t cell=0; cell<h.batch; ++cell) {
                        if (!h.admitted[cell*h.conditions+h.groups[gid].condition_index]) continue;
                        if (h.x16[cell*h.coordinates] == 0) ++admitted_zero;
                        else ++admitted_nonzero;
                    }
                    if (!admitted_zero || !admitted_nonzero)
                        throw std::runtime_error("arity boundary lacks admitted zero and nonzero participants");
                }
            }
            if (scenario == 2) {
                std::fill(h.admitted.begin(), h.admitted.end(), 0);
                std::fill(h.primal.begin(), h.primal.end(), 0);
                std::fill(h.derivative.begin(), h.derivative.end(), 0);
                std::fill(h.x16.begin(), h.x16.end(), 0x7e00u);
                std::fill(h.theta.begin(), h.theta.end(), std::numeric_limits<float>::quiet_NaN());
            }
            if (scenario == 3) {
                h.batch=0; h.x16.clear(); h.x32.clear(); h.dy.clear(); h.admitted.clear();
                h.probabilities.clear(); h.primal.clear(); h.derivative.clear();
            }
            const auto expected = reference::product_forward(h.problem(), h.gates());
            const auto gradient = reference::product_backward(h.problem(), h.gates(), h.dy.data());
            ConsumerDeviceFixture device(h,stream);
            for (const auto backend : {Backend::masked_direct_two_tile, Backend::cohort_direct_two_tile}) {
                const BackendOptions options{32,1};
                auto workspace=device.workspace(backend,options);
                device.clear_results(stream);
                check(prepare(backend,h.body,device.problem(),device.gates(),options,workspace,stream),"prepare qualification");
                validate_descriptor(h,backend,device.problem(),options,workspace,stream);
                check(forward(backend,h.body,device.problem(),device.gates(),options,device.results(),workspace,stream),"forward qualification");
                check(backward(backend,h.body,device.problem(),device.gates(),device.dy(),options,device.results(),workspace,stream),"backward qualification");
                const auto result=device.download(stream);
                if (result.status) throw std::runtime_error("qualification status="+std::to_string(result.status));
                compare(result.y,expected.y,"qualification y"); compare(result.dx,gradient.dx,"qualification dx");
                compare(result.dtheta,gradient.dtheta,"qualification dtheta"); compare(result.dq,gradient.dq,"qualification dq");
                if (scenario == 0) {
                    auto stale=device.gates(); ++stale.revisions.accessibility_generation;
                    check(forward(backend,h.body,device.problem(),stale,options,device.results(),workspace,stream),"stale qualification");
                    if (!(device.download(stream).status & status_stale))
                        throw std::runtime_error("stale accessibility descriptor accepted");
                    device.clear_results(stream);
                    stale=device.gates(); ++stale.revisions.mapping_revision;
                    check(forward(backend,h.body,device.problem(),stale,options,device.results(),workspace,stream),"stale mapping qualification");
                    if (!(device.download(stream).status & status_stale))
                        throw std::runtime_error("stale mapping descriptor accepted");
                }
            }
        }
        return true;
    } catch (const std::exception& e) { if (error) *error=context+": "+e.what(); return false; }
}
} // namespace cgs
