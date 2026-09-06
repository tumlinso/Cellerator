#include <Cellerator/compute/operation/relation_update.hh>
#include "reference_math.hh"
#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/contract_portfolio_v1.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
namespace ce=cellerator::compute::relation;
namespace ex=cellerator::execution;
namespace ref=ru1_reference;
namespace {
std::uint64_t checks=0;
void require(bool ok,const char* message) {
    ++checks;if(!ok){std::cerr<<message<<'\n';std::exit(1);}
}
void gpu(cudaError_t code) {
    if(code!=cudaSuccess){std::cerr<<cudaGetErrorString(code)<<'\n';std::exit(1);}
}
void core(ce::status code) { if(!code){std::cerr<<code.message<<'\n';std::exit(1);} }
struct stream_owner {
    cudaStream_t value{};
    stream_owner(){gpu(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~stream_owner(){cudaStreamDestroy(value);}
};
template<class T> struct device_buffer {
    T* data=nullptr;std::size_t count=0;
    explicit device_buffer(std::size_t n):count(n){if(n)gpu(cudaMalloc(&data,n*sizeof(T)));}
    ~device_buffer(){if(data)cudaFree(data);}
    device_buffer(const device_buffer&)=delete;
    void upload(const std::vector<T>& values,cudaStream_t stream){require(values.size()==count,"upload shape");if(count)gpu(cudaMemcpyAsync(data,values.data(),count*sizeof(T),cudaMemcpyHostToDevice,stream));}
    std::vector<T> download(cudaStream_t stream){std::vector<T> result(count);if(count)gpu(cudaMemcpyAsync(result.data(),data,count*sizeof(T),cudaMemcpyDeviceToHost,stream));gpu(cudaStreamSynchronize(stream));return result;}
};
struct pair_owner {
    ce::prepared_relation_pair* value=nullptr;
    ~pair_owner(){ce::destroy(value);}
};
ce::axis_descriptor axis(unsigned id,unsigned extent) {
    ce::axis_descriptor a{};a.extent=extent;
    a.identity.header={ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,sizeof(a.identity)};
    a.identity.domain={id,0xf000000000000001ull};a.identity.order={id,0xe000000000000001ull};
    a.identity.geometry={id,0xd000000000000001ull};a.identity.partition={id,0xc000000000000001ull};return a;
}
constexpr unsigned width=16,guard=8;
constexpr float sentinel=1234567.0f;
void guards(const std::vector<float>& values) {
    for(unsigned i=0;i<guard;++i)require(values[i]==sentinel && values[values.size()-1-i]==sentinel,"provider wrote outside declared output");
}
ref::support fixture(unsigned which) {
    ref::support graph{};
    if(which<=1){graph.sources=20;graph.destinations=19;
        for(unsigned d=0;d<16;++d)for(unsigned s=16;s>0;--s)graph.edges.push_back({d,s-1});
        if(which==1)graph.edges.insert(graph.edges.end(),{{16,16},{16,3},{16,0},{17,18},{17,16},{17,2}});
    } else if(which==2){graph.sources=67;graph.destinations=35;
        for(unsigned d=0;d<34;++d)if(d%4)graph.edges.insert(graph.edges.end(),{{d,(d*7+33)%67},{d,(d*7)%67}});
    } else if(which==3){graph.sources=257;graph.destinations=35;
        for(unsigned d=0;d<34;++d)if(d%3)for(unsigned s=257;s>0;--s)graph.edges.push_back({d,s-1});
    } else {graph.sources=7;graph.destinations=5;}
    return graph;
}
void run_fixture(unsigned fixture_id,bool rounded,bool hybrid) {
    const auto graph=fixture(fixture_id);
    std::vector<unsigned> offsets(graph.destinations+1),sources;
    for(const auto e:graph.edges){++offsets[e.destination+1];sources.push_back(e.source);}
    std::partial_sum(offsets.begin(),offsets.end(),offsets.begin());
    ce::relation_calculus_descriptor semantic{};
    semantic.forward.topology={{71,0xf000000000000003ull},{3},axis(10,graph.sources),axis(20,graph.destinations),{31,0xb000000000000001ull},graph.edges.size()};
    semantic.forward.dense_width=width;semantic.transpose=semantic.forward;
    semantic.transpose.direction=ce::orientation::transpose;
    semantic.gradient=rounded?ce::gradient_arithmetic::round_operands_f16_rne:ce::gradient_arithmetic::full_f32;
    stream_owner stream;pair_owner pair;
    core(ce::prepare_relation_pair(semantic.forward,semantic.transpose,{offsets.data(),offsets.size(),sources.data(),sources.size()},
        {0,128ull<<20},stream.value,&pair.value));
    if(!rounded) {
        const auto illegal=ce::prepare_relation_gradient(*pair.value,semantic,{ce::gradient_route::force_hybrid,128ull<<20},stream.value);
        require(illegal.code==ce::status_code::unsupported_semantics || illegal.code==ce::status_code::unsupported_numeric_policy,
            "forced hybrid silently changed full-f32 semantics");
        ce::relation_update_report rejected{};core(ce::inspect_updates(*pair.value,&rejected));
        require(rejected.gradient_preparations==0 && rejected.wmma_launches==0,"ineligible hybrid request changed prepared state");
    }
    core(ce::prepare_relation_gradient(*pair.value,semantic,{hybrid?ce::gradient_route::force_hybrid:ce::gradient_route::force_sparse,128ull<<20},stream.value));
    ce::edge_layout_view layout{};core(ce::inspect_edge_layout(*pair.value,&layout));
    require(layout.count==graph.edges.size(),"edge layout count");
    std::vector<bool> seen(graph.edges.size());bool shuffled=false;
    for(unsigned logical=0;logical<layout.count;++logical){const auto physical=layout.logical_to_physical[logical];
        require(physical<layout.count && !seen[physical],"edge layout not bijective");seen[physical]=true;shuffled|=physical!=logical;}
    if(fixture_id<=1)require(shuffled,"fixture failed to exercise physical permutation");
    std::vector<std::uint16_t> bits(graph.edges.size());std::vector<double> weights(graph.edges.size());
    for(unsigned i=0;i<bits.size();++i){bits[i]=ref::half_bits(float(int((i*17)%31)-15)/64.0f);weights[i]=ref::half_value(bits[i]);}
    device_buffer<std::uint16_t> device_weights(bits.size());device_weights.upload(bits,stream.value);
    const auto& topology=semantic.forward.topology;
    core(ce::publish_values(*pair.value,{device_weights.data,bits.size(),topology.identity,topology.epoch,topology.logical_edge_order,{1},0},stream.value));
    double worst_dense=0,worst_gradient=0,quantization_difference=0;
    device_buffer<float> first_input(graph.sources*width),second_input(graph.sources*width);
    device_buffer<float> first_cotangent(graph.destinations*width),second_cotangent(graph.destinations*width);
    require(first_input.data!=second_input.data && first_cotangent.data!=second_cotangent.data,"input allocations must actually differ");
    for(unsigned round=1;round<=2;++round){
        std::vector<float> x(graph.sources*width),cot(graph.destinations*width);
        for(unsigned i=0;i<x.size();++i)x[i]=float(int((i*13+round*7)%61)-30)/37.0f;
        for(unsigned i=0;i<cot.size();++i)cot[i]=float(int((i*19+round*11)%67)-33)/43.0f;
        // All device pointers are independently rebound, while topology and
        // weight generation remain unchanged. Inputs are deliberately not half.
        auto& dx=round==1?first_input:second_input;
        auto& dc=round==1?first_cotangent:second_cotangent;
        dx.upload(x,stream.value);dc.upload(cot,stream.value);
        device_buffer<float> dy(graph.destinations*width+2*guard),da(graph.sources*width+2*guard),dg(graph.edges.size()+2*guard);
        dy.upload(std::vector<float>(dy.count,sentinel),stream.value);da.upload(std::vector<float>(da.count,sentinel),stream.value);dg.upload(std::vector<float>(dg.count,sentinel),stream.value);
        ce::device_state_view input{dx.data,x.size(),topology.source,0},cotangent{dc.data,cot.size(),topology.destination,0};
        core(ce::enqueue(*pair.value,semantic.forward,input,{dy.data+guard,graph.destinations*width,topology.destination,0},{1},stream.value));
        core(ce::enqueue(*pair.value,semantic.transpose,cotangent,{da.data+guard,graph.sources*width,topology.source,0},{1},stream.value));
        ce::gradient_stamp stamp{};
        core(ce::enqueue_edge_gradient(*pair.value,semantic,input,cotangent,{101,round},{202,round},{1},
            {dg.data+guard,graph.edges.size(),topology.identity,topology.epoch,layout.order,0},&stamp,stream.value));
        const auto y=dy.download(stream.value),adjoint=da.download(stream.value),gradient=dg.download(stream.value);
        guards(y);guards(adjoint);guards(gradient);
        const auto yref=ref::forward(graph,weights,x,width),aref=ref::transpose(graph,weights,cot,width);
        const auto gref=ref::edge_gradient(graph,x,cot,width,rounded),full=ref::edge_gradient(graph,x,cot,width,false);
        std::vector<double> yabs(yref.size()),aabs(aref.size());std::vector<unsigned> yterms(graph.destinations),aterms(graph.sources);
        for(unsigned e=0;e<graph.edges.size();++e){const auto edge=graph.edges[e];++yterms[edge.destination];++aterms[edge.source];
            for(unsigned k=0;k<width;++k){yabs[edge.destination*width+k]+=std::abs(weights[e]*x[edge.source*width+k]);aabs[edge.source*width+k]+=std::abs(weights[e]*cot[edge.destination*width+k]);}}
        for(unsigned i=0;i<yref.size();++i){const double error=std::abs(y[i+guard]-yref[i]);worst_dense=std::max(worst_dense,error);
            require(std::isfinite(y[i+guard]) && error<=ref::reduction_bound(yterms[i/width],yabs[i]),"forward differs from independent conditioned oracle");}
        for(unsigned i=0;i<aref.size();++i){const double error=std::abs(adjoint[i+guard]-aref[i]);worst_dense=std::max(worst_dense,error);
            require(std::isfinite(adjoint[i+guard]) && error<=ref::reduction_bound(aterms[i/width],aabs[i]),"transpose differs from independent conditioned oracle");}
        for(unsigned e=0;e<gref.size();++e){const auto edge=graph.edges[e];double absolute=0;
            for(unsigned k=0;k<width;++k){const auto a=x[edge.source*width+k],b=cot[edge.destination*width+k];absolute+=std::abs(double(rounded?ref::half_round(a):a)*(rounded?ref::half_round(b):b));}
            const double error=std::abs(gradient[guard+layout.logical_to_physical[e]]-gref[e]);worst_gradient=std::max(worst_gradient,error);
            require(std::isfinite(gradient[guard+layout.logical_to_physical[e]]) && error<=ref::reduction_bound(width,absolute),"logical edge gradient differs or is mapped to wrong slot");
            quantization_difference=std::max(quantization_difference,std::abs(full[e]-gref[e]));
        }
        require(stamp.forward_generation.value==1 && stamp.input.version==round && stamp.cotangent.version==round,"gradient provenance lost across rebinding");
    }
    ce::relation_update_report report{};core(ce::inspect_updates(*pair.value,&report));
    require(report.relation.topology_preparations==1 && report.gradient_preparations==1 && report.relation.value_refreshes==1,"rebinding rebuilt topology or values");
    require(report.relation.accepted_forward_launches==2 && report.relation.accepted_transpose_launches==2 && report.gradient_launches==2,"actual core stages not reached");
    require(report.implicit_canonicalizations==0,"implicit canonical-order traffic");
    if(!graph.edges.empty()) {
        if(hybrid){require(report.wmma_launches>=2,"forced hybrid did not execute WMMA");if(fixture_id==1)require(report.residual_launches>=2,"mixed fixture did not execute sparse residual");}
        else require(report.sparse_launches>=2 && report.wmma_launches==0,"forced sparse did not execute sparse provider");
        if(rounded)require(quantization_difference>1e-5 && quantization_difference>10*worst_gradient,"fixture cannot distinguish rounding from accumulation error");
    }
    std::cout<<"fixture="<<fixture_id<<" profile="<<(rounded?"half-rounded":"full-f32")<<" route="<<(hybrid?"hybrid":"sparse")
        <<" edges="<<graph.edges.size()<<" max_dense_error="<<worst_dense<<" max_gradient_error="<<worst_gradient<<" operand_quantization_difference="<<quantization_difference
        <<" wmma_launches="<<report.wmma_launches<<" residual_launches="<<report.residual_launches<<" sparse_launches="<<report.sparse_launches<<'\n';
}
void illegal_direct_wmma() {
    namespace contract=cellerator::compute::architecture::providers::nvidia::sm70::contract;
    stream_owner stream;
    device_buffer<__half> a(1024),b(1024);
    device_buffer<float> output(256);
    device_buffer<contract::rectangular_tile_v1> tile_storage(1);
    output.upload(std::vector<float>(256,sentinel),stream.value);
    contract::rectangular_request_v1 request{};
    request.dense={a.data,b.data,17};request.tile_count=1;
    request.source_count=request.destination_count=32;
    request.source_stride=request.destination_stride=17;
    request.source_capacity=request.destination_capacity=1024;
    request.output_capacity=256;request.projection_output=output.data;request.stream=stream.value;
    contract::rectangular_tile_v1 tile{0,0,0};
    contract::prepared_rectangular_v1 prepared{};
    require(contract::prepare_rectangular_v1(request,&tile,tile_storage.data,1,prepared)==contract::status_v1::invalid_argument,
        "K17/ldm17 must fail before any WMMA prefix");
    request.dense.dense_width=16;request.source_stride=request.destination_stride=24;tile.source_begin_local=1;
    require(contract::prepare_rectangular_v1(request,&tile,tile_storage.data,1,prepared)==contract::status_v1::invalid_argument,
        "derived unaligned panel must fail");
    tile.source_begin_local=0;request.source_stride=request.destination_stride=32;request.dense.source=a.data+1;
    require(contract::prepare_rectangular_v1(request,&tile,tile_storage.data,1,prepared)==contract::status_v1::invalid_argument,
        "unaligned panel base must fail");
    for(auto value:output.download(stream.value))require(value==sentinel,"illegal WMMA request modified output");
}

}
int main(){
    gpu(cudaSetDevice(0));cudaDeviceProp device{};gpu(cudaGetDeviceProperties(&device,0));require(device.major==7 && device.minor==0,"actual sm70 device required");
    illegal_direct_wmma();
    for(unsigned f=0;f<5;++f)for(bool rounded:{false,true})run_fixture(f,rounded,false);
    run_fixture(0,true,true);run_fixture(1,true,true);
    std::cout<<checks<<" independent GPU numerical checks passed on "<<device.name<<'\n';
}
