// Post-epic acceptance example. Normal mode requires the API implemented by Semantic Spine v1.
// No fake backend is supplied. Reference-only mode checks fixture arithmetic, not Cellerator.
#include "fixture.hh"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef CELLERATOR_SPINE_REFERENCE_ONLY
#include <Cellerator/compute/operation/prepared_relation.hh>
#include <cuda_runtime_api.h>
#include <memory>
#endif

namespace sd=spine_demo;
namespace {
[[maybe_unused]] void demand(bool value,const char* what) { if(!value) throw std::runtime_error(what); }
[[maybe_unused]] void print_result(const char* label,const float* actual,const double* expected,
                  const char* const* names,std::size_t count) {
    std::cout << label << '\n';
    for(std::size_t i=0;i<count;++i) {
        if(!sd::near(actual[i],expected[i]))
            throw std::runtime_error(std::string("numerical mismatch at ")+names[i]);
        std::cout << "  " << names[i] << ": " << actual[i] << '\n';
    }
}
#ifndef CELLERATOR_SPINE_REFERENCE_ONLY
namespace ce=cellerator::compute::relation;
namespace ex=cellerator::execution;
void check(ce::status s,const char* what) {
    if(!s) throw std::runtime_error(std::string(what)+": "+(s.message?s.message:"unspecified error"));
}
void cuda_check(cudaError_t s,const char* what) {
    if(s!=cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(s));
}
struct stream_owner {
    cudaStream_t stream{};
    stream_owner(){cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create stream");}
    ~stream_owner(){if(stream)cudaStreamDestroy(stream);}
    stream_owner(const stream_owner&)=delete;
    stream_owner& operator=(const stream_owner&)=delete;
};
template<class T> struct device_array {
    T* data=nullptr;
    explicit device_array(std::size_t n){cuda_check(cudaMalloc(reinterpret_cast<void**>(&data),n*sizeof(T)),"allocate");}
    ~device_array(){if(data)cudaFree(data);}
    device_array(const device_array&)=delete;
    device_array& operator=(const device_array&)=delete;
};
ce::axis_descriptor axis(std::uint64_t base,std::uint64_t extent) {
    ex::persistent_axis_identity id{};
    id.header={ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,sizeof(id)};
    id.domain={base,1};id.order={base+1,1};id.geometry={base+2,1};id.partition={base+3,1};
    return {id,extent};
}
template<class T> void upload(T* dst,const T* src,std::size_t n,cudaStream_t stream) {
    cuda_check(cudaMemcpyAsync(dst,src,n*sizeof(T),cudaMemcpyHostToDevice,stream),"upload");
}
void readback(float* dst,const float* src,std::size_t n,cudaStream_t stream) {
    cuda_check(cudaMemcpyAsync(dst,src,n*sizeof(float),cudaMemcpyDeviceToHost,stream),"read result");
    cuda_check(cudaStreamSynchronize(stream),"finish observed result");
}
int device_main(int device,bool require_sm70) {
    int count=0;cuda_check(cudaGetDeviceCount(&count),"enumerate CUDA devices");
    demand(device>=0 && device<count,"requested CUDA device unavailable; this is not a skip pass");
    cuda_check(cudaSetDevice(device),"select CUDA device");
    cudaDeviceProp properties{};cuda_check(cudaGetDeviceProperties(&properties,device),"query device");
    demand(!require_sm70 || (properties.major==7 && properties.minor==0),"sm70 acceptance requires an actual V100-class sm70 device");
    std::cout << "Device: " << properties.name << " (sm_" << properties.major << properties.minor << ")\n";
    stream_owner stream;
    device_array<std::uint16_t> weights(sd::edges);
    device_array<float> regulator_state(sd::regulators),gene_input(sd::genes),gene_result(sd::genes),regulator_result(sd::regulators);

    ce::topology_descriptor topology{};
    topology.identity={0x1001,0x535031};topology.epoch={1};
    topology.source=axis(0x2000,sd::regulators);topology.destination=axis(0x3000,sd::genes);
    topology.logical_edge_order={0x4001,0x535031};topology.edge_count=sd::edges;
    ce::operation_descriptor forward{};forward.topology=topology;
    forward.direction=ce::orientation::forward;
    ce::operation_descriptor transpose=forward;transpose.direction=ce::orientation::transpose;
    check(ce::validate(forward),"validate forward");check(ce::validate(transpose),"validate transpose");
    ce::prepared_relation_pair* raw=nullptr;
    check(ce::prepare_relation_pair(forward,transpose,
        {sd::row_offsets,sd::genes+1,sd::sources,sd::edges},
        {device,64u*1024u*1024u},stream.stream,&raw),"prepare one relation pair");
    demand(raw!=nullptr,"prepare returned no pair");
    std::unique_ptr<ce::prepared_relation_pair,decltype(&ce::destroy)> prepared(raw,&ce::destroy);
    ce::preparation_report initial{};check(ce::inspect(*prepared,&initial),"inspect initial preparation");
    demand(initial.topology_preparations==1,"one topology preparation expected");
    demand(ex::valid_identity(initial.forward_projection) && ex::valid_identity(initial.transpose_projection),"missing physical projection identity");

    auto publish=[&](const std::uint16_t* values,std::uint64_t generation) {
        upload(weights.data,values,sd::edges,stream.stream);
        ce::device_values_binding b{weights.data,sd::edges,topology.identity,topology.epoch,
                                    topology.logical_edge_order,{generation},device};
        check(ce::publish_values(*prepared,b,stream.stream),"publish value generation");
    };
    auto apply=[&](const float* state,const float* reference_weights,std::uint64_t generation,const char* label) {
        upload(regulator_state.data,state,sd::regulators,stream.stream);
        check(ce::enqueue(*prepared,forward,
            {regulator_state.data,sd::regulators,topology.source,device},
            {gene_result.data,sd::genes,topology.destination,device},{generation},stream.stream),"enqueue forward");
        float actual[sd::genes];double expected[sd::genes];
        sd::forward_reference(reference_weights,state,expected);
        readback(actual,gene_result.data,sd::genes,stream.stream);
        print_result(label,actual,expected,sd::gene_names,sd::genes);
    };
    auto reverse=[&](const float* reference_weights,std::uint64_t generation,const char* label) {
        upload(gene_input.data,sd::gene_signal,sd::genes,stream.stream);
        check(ce::enqueue(*prepared,transpose,
            {gene_input.data,sd::genes,topology.destination,device},
            {regulator_result.data,sd::regulators,topology.source,device},{generation},stream.stream),"enqueue transpose");
        float actual[sd::regulators];double expected[sd::regulators];
        sd::transpose_reference(reference_weights,sd::gene_signal,expected);
        readback(actual,regulator_result.data,sd::regulators,stream.stream);
        print_result(label,actual,expected,sd::regulator_names,sd::regulators);
    };
    publish(sd::half_weights_1,1);
    apply(sd::state_a,sd::weights_1,1,"Forward: state A, generation 1");
    apply(sd::state_b,sd::weights_1,1,"Forward: state B, same prepared topology");
    reverse(sd::weights_1,1,"Transpose: gene signal to regulator contributions (not an inverse)");
    publish(sd::half_weights_2,2);
    apply(sd::state_a,sd::weights_2,2,"Forward: changed weights, generation 2");
    reverse(sd::weights_2,2,"Transpose: same new value generation");

    float sentinel[sd::genes],after[sd::genes];
    for(float& v:sentinel)v=-1234.5f;
    upload(gene_result.data,sentinel,sd::genes,stream.stream);
    const ce::status stale=ce::enqueue(*prepared,forward,
        {regulator_state.data,sd::regulators,topology.source,device},
        {gene_result.data,sd::genes,topology.destination,device},{1},stream.stream);
    demand(stale.code==ce::status_code::stale_generation,"stale values must be rejected before launch");
    readback(after,gene_result.data,sd::genes,stream.stream);
    for(std::size_t i=0;i<sd::genes;++i)demand(after[i]==sentinel[i],"rejected call modified output");
    ce::preparation_report final{};check(ce::inspect(*prepared,&final),"inspect final state");
    demand(final.topology_preparations==1 && final.value_refreshes==2,"topology was rebuilt or refresh accounting is wrong");
    demand(final.latest_enqueued_generation.value==2,"wrong current generation");
    demand(final.accepted_forward_launches==3 && final.accepted_transpose_launches==2,"wrong actual launch accounting");
    demand(ex::same_identity(initial.forward_projection,final.forward_projection) && ex::same_identity(initial.transpose_projection,final.transpose_projection),"refresh changed topology projections");
    demand(final.forward_candidate && *final.forward_candidate && final.transpose_candidate && *final.transpose_candidate,"missing actual candidate attribution");
    std::cout << "Bound implementations: " << final.forward_candidate << " / " << final.transpose_candidate << '\n'
              << "Topology preparations: " << final.topology_preparations << "; value refreshes: " << final.value_refreshes << '\n'
              << "SPINE_DEMO_GPU_PASS: forward, transpose, reuse, generation change, stale rejection\n"
              << "Not a full .cell compiler or a performance superiority claim.\n";
    return 0;
}
#endif
} // namespace
int main(int argc,char** argv) {
    try {
        sd::check_fixture();
#ifdef CELLERATOR_SPINE_REFERENCE_ONLY
        (void)argv;
        if(argc!=1) throw std::runtime_error("reference-only build takes no device arguments");
        double y[sd::genes],z[sd::regulators];
        sd::forward_reference(sd::weights_2,sd::state_a,y);
        sd::transpose_reference(sd::weights_2,sd::gene_signal,z);
        std::cout << "REFERENCE_ONLY_FIXTURE_PASS: no Cellerator or GPU execution was tested.\n";
        std::cout << "Generation-2 forward:";for(double x:y)std::cout << ' ' << x;
        std::cout << "\nGeneration-2 transpose:";for(double x:z)std::cout << ' ' << x;
        std::cout << '\n';
        return 0;
#else
        int device=0;bool require_sm70=false;
        for(int i=1;i<argc;++i) {
            if(std::strcmp(argv[i],"--require-sm70")==0)require_sm70=true;
            else if(std::strcmp(argv[i],"--device")==0 && i+1<argc) {
                const std::string value=argv[++i];std::size_t used=0;
                device=std::stoi(value,&used);demand(used==value.size(),"invalid device number");
            } else throw std::runtime_error("usage: ceSemanticSpineDemo [--device N] [--require-sm70]");
        }
        return device_main(device,require_sm70);
#endif
    } catch(const std::exception& e) {
        std::cerr << "Semantic Spine demo FAILED: " << e.what() << '\n';
        return 1;
    }
}
