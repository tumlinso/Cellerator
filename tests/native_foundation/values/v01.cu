#include <Cellerator/compute/operation/prepared_relation.hh>
#include <Cellerator/compute/operation/relation_update.hh>
#include <cuda_fp16.h>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace rel=cellerator::compute::relation;
namespace ex=cellerator::execution;
void require(bool b,const char* why){if(!b)throw std::runtime_error(why);}
void ok(rel::status s){require(static_cast<bool>(s),s.message);}
void cuda_ok(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
rel::axis_descriptor axis(std::uint64_t id){return {{{ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,sizeof(ex::persistent_axis_identity)},{id,1},{id,2},{id,3},{id,4}},2};}
struct storage {
    cudaStream_t a{},b{};rel::prepared_relation_pair *first{},*second{};
    __half* weights{};float *input{},*output{};
    ~storage(){if(first)rel::destroy(first);if(second)rel::destroy(second);if(weights)cudaFree(weights);if(input)cudaFree(input);if(output)cudaFree(output);if(a)cudaStreamDestroy(a);if(b)cudaStreamDestroy(b);}
};
void test(int width){
    storage owner;cuda_ok(cudaStreamCreate(&owner.a));cuda_ok(cudaStreamCreate(&owner.b));
    rel::operation_descriptor forward;forward.topology={{40,1},{1},axis(1),axis(2),{41,1},3};forward.dense_width=width;
    auto transpose=forward;transpose.direction=rel::orientation::transpose;
    const std::array<std::uint32_t,3> offsets{0,2,3},sources{0,1,0};
    ok(rel::prepare_relation_pair(forward,transpose,{offsets.data(),offsets.size(),sources.data(),sources.size()},{0,0},owner.a,&owner.first));
    ok(rel::create_relation_instance(*owner.first,owner.b,&owner.second));
    rel::preparation_report one{},two{};ok(rel::inspect(*owner.first,&one));ok(rel::inspect(*owner.second,&two));
    require(one.structural_preparation_id==two.structural_preparation_id && one.structural_preparation_id!=0,"one actual counted preparation");
    require(one.structural_instance_count==2 && two.structural_instance_count==2,"two shared owners");
    require(one.shared_structural_bytes==two.shared_structural_bytes && one.shared_structural_bytes>0,"one charged immutable footprint");
    require(one.instance_value_bytes==6 && two.instance_value_bytes==6,"independent instance value allocation");
    rel::edge_layout_view layout1{},layout2{};ok(rel::inspect_edge_layout(*owner.first,&layout1));ok(rel::inspect_edge_layout(*owner.second,&layout2));
    require(layout1.logical_to_physical==layout2.logical_to_physical,"host bijection storage must be shared, not copied");
    cuda_ok(cudaMalloc(&owner.weights,3*sizeof(__half)));cuda_ok(cudaMalloc(&owner.input,2*width*sizeof(float)));cuda_ok(cudaMalloc(&owner.output,2*width*sizeof(float)));
    std::vector<float> input(2*width);for(int c=0;c<width;++c){input[c]=1.f+c*.125f;input[width+c]=2.f-c*.0625f;}
    cuda_ok(cudaMemcpy(owner.input,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice));
    auto publish=[&](rel::prepared_relation_pair& pair,cudaStream_t stream,float scale,std::uint64_t generation){
        std::array<__half,3> values{__float2half(2*scale),__float2half(-scale),__float2half(3*scale)};
        cuda_ok(cudaMemcpy(owner.weights,values.data(),sizeof(values),cudaMemcpyHostToDevice));
        ok(rel::publish_values(pair,{owner.weights,3,forward.topology.identity,forward.topology.epoch,forward.topology.logical_edge_order,{generation},0},stream));
        cuda_ok(cudaStreamSynchronize(stream));
    };
    auto check=[&](rel::prepared_relation_pair& pair,cudaStream_t stream,const rel::operation_descriptor& op,float scale,std::uint64_t generation){
        ok(rel::enqueue(pair,op,{owner.input,std::uint64_t(input.size()),rel::input_axis(op),0},
             {owner.output,std::uint64_t(input.size()),rel::result_axis(op),0},{generation},stream));
        cuda_ok(cudaStreamSynchronize(stream));std::vector<float> got(input.size());
        cuda_ok(cudaMemcpy(got.data(),owner.output,got.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for(int c=0;c<width;++c){
            const double x=input[c],y=input[width+c];
            const double first=scale*(op.direction==rel::orientation::forward?2*x-y:2*x+3*y);
            const double second=scale*(op.direction==rel::orientation::forward?3*x:-x);
            require(std::abs(got[c]-first)<1e-5 && std::abs(got[width+c]-second)<1e-5,"retained forward/transpose independent formulas");
        }
    };
    publish(*owner.first,owner.a,1,1);publish(*owner.second,owner.b,2,1);
    check(*owner.first,owner.a,forward,1,1);check(*owner.second,owner.b,forward,2,1);
    check(*owner.first,owner.a,transpose,1,1);check(*owner.second,owner.b,transpose,2,1);
    publish(*owner.first,owner.a,3,2);check(*owner.second,owner.b,forward,2,1);
    ok(rel::close_relation_pair(&owner.first));require(owner.first==nullptr,"close first instance");
    // Reuse freed candidate metadata storage to expose accidental source pointers.
    std::vector<std::vector<unsigned char>> churn(64,std::vector<unsigned char>(4096,0xa5));
    ok(rel::inspect(*owner.second,&two));require(two.structural_instance_count==1,"surviving owner retains topology");
    check(*owner.second,owner.b,forward,2,1);check(*owner.second,owner.b,transpose,2,1);
    ok(rel::close_relation_pair(&owner.second));
}
int main()try{
    int count=0;cuda_ok(cudaGetDeviceCount(&count));require(count==1,"exactly one leased visible device required");
    cudaDeviceProp properties{};cuda_ok(cudaGetDeviceProperties(&properties,0));require(properties.major==7 && properties.minor==0,"real sm70 device required");
    test(1);test(16);
    std::cout<<"{\"task\":\"CE-NF1-V01\",\"actual_cuda\":true,\"shared_structure\":true,\"independent_instances\":2,\"widths\":[1,16]}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
