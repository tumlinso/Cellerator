#include <Cellerator/compute/operation/differential/local_arithmetic.hh>
#include <cuda_runtime_api.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace df = cellerator::compute::differential;
namespace nf = df::nf1;
namespace nn = cellerator::compute::native_numeric;
namespace ex = cellerator::execution;
namespace pg = ex::program;
void check(bool value) { if (!value) std::abort(); }
void gpu(cudaError_t value) { check(value == cudaSuccess); }
nf::generation_stamp stamp(std::uint64_t id, std::uint64_t generation) {
    return {{id, 1}, {generation}};
}
nf::primal_record primal(const nf::identity& definition) {
    nf::primal_record result{};
    result.instance.prepared = {definition, {2, 1}, {1}, {3, 1}};
    result.instance.state = stamp(4, 7);
    result.instance.parameters = stamp(5, 11);
    return result;
}
struct primitive {
    std::array<nf::operand_signature, 2> inputs{};
    nf::output_signature output{};
    df::local_block block{};
    primitive(nn::local_operation operation, std::size_t count) {
        inputs = {nf::operand_signature{{10,1},{},count,ex::numeric_type::f32},
                  nf::operand_signature{{11,1},{},count,ex::numeric_type::f32}};
        output.operand = {{12,1},{},count,ex::numeric_type::f32}; output.assembly_owner = {13,1};
        nf::operation_contract contract{}; contract.definition = {100 + std::uint64_t(operation),1};
        contract.arguments = {inputs.data(), nn::local_arity(operation)}; contract.outputs = {&output,1};
        contract.numeric = {ex::numeric_type::f32,ex::numeric_type::f32,ex::numeric_type::f32,
            ex::numeric_type::f32,ex::numeric_type::f32,ex::numeric_type::f32};
        contract.capabilities = nf::forward | nf::jvp | nf::vjp | nf::second_direction;
        check(df::make_local_device_block(operation, contract, block) == nf::status::success);
    }
};
template<nf::capability Action> pg::program_status run(primitive& primitive,
    df::response_binding<float>& response, cudaStream_t stream) {
    pg::prepared_stage_v2 stage{};
    check(nf::bind_compiled_stage(primitive.block.block, Action, &primitive.block, 1, 1, 0, stage) == nf::status::success);
    pg::prepared_program_v2 program{2,0,&stage,1,nullptr,0};
    pg::launch_binding_v2 launch{}; launch.input = &response;
    return pg::execute_prepared_program_v2(program, &launch, 1, stream);
}
int main() {
    cudaDeviceProp prop{}; gpu(cudaSetDevice(0)); gpu(cudaGetDeviceProperties(&prop, 0)); check(prop.major == 7 && prop.minor == 0);
    constexpr std::size_t n = 33;
    primitive multiply(nn::local_operation::multiply, n);
    std::vector<float> left(n), right(n), dl(n), dr(n), cotangent(n), result(n), left_adj(n), right_adj(n);
    for (std::size_t i=0;i<n;++i) { left[i]=.2f+i*.01f; right[i]=-.3f+i*.02f; dl[i]=.05f; dr[i]=-.04f; cotangent[i]=.25f+i*.001f; }
    float *dleft{},*dright{},*ddl{},*ddr{},*dcot{},*dresult{},*dleft_adj{},*dright_adj{};
    for (auto address : {&dleft,&dright,&ddl,&ddr,&dcot,&dresult,&dleft_adj,&dright_adj}) gpu(cudaMalloc(address,n*sizeof(float)));
    auto upload=[&](float* target,const std::vector<float>& source){gpu(cudaMemcpy(target,source.data(),n*sizeof(float),cudaMemcpyHostToDevice));};
    upload(dleft,left);upload(dright,right);upload(ddl,dl);upload(ddr,dr);upload(dcot,cotangent);
    cudaStream_t stream{};gpu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    df::response_binding<float> response{};
    response.device={dleft,dright,ddl,ddr,dcot,dresult,dleft_adj,dright_adj,n};
    response.live_primal=primal(multiply.block.block.contract.definition);
    response.request.primal=response.live_primal;response.request.action=nf::jvp;
    response.request.direction_domain=multiply.inputs[0];response.request.response_domain=multiply.output.operand;
    response.direction=multiply.inputs[0];response.response=multiply.output.operand;
    check(run<nf::jvp>(multiply,response,stream)==pg::program_status::success);gpu(cudaStreamSynchronize(stream));
    gpu(cudaMemcpy(result.data(),dresult,n*sizeof(float),cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<n;++i) check(std::abs(result[i]-(right[i]*dl[i]+left[i]*dr[i]))<2e-6f);
    std::fill(dr.begin(),dr.end(),0);upload(ddr,dr);
    response.request.direction_domain=multiply.inputs[1];response.direction=multiply.inputs[1];
    check(run<nf::jvp>(multiply,response,stream)==pg::program_status::success);gpu(cudaStreamSynchronize(stream));
    gpu(cudaMemcpy(result.data(),dresult,n*sizeof(float),cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<n;++i) check(std::abs(result[i]-right[i]*dl[i])<2e-6f);
    for(std::size_t i=0;i<n;++i) dr[i]=-.04f;
    std::fill(dl.begin(),dl.end(),0);upload(ddl,dl);upload(ddr,dr);
    check(run<nf::jvp>(multiply,response,stream)==pg::program_status::success);gpu(cudaStreamSynchronize(stream));
    gpu(cudaMemcpy(result.data(),dresult,n*sizeof(float),cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<n;++i) check(std::abs(result[i]-left[i]*dr[i])<2e-6f);
    for(std::size_t i=0;i<n;++i) dl[i]=.05f;
    upload(ddl,dl);
    response.request.action=nf::vjp;response.request.direction_domain=multiply.output.operand;
    response.request.response_domain=multiply.inputs[0];response.direction=multiply.output.operand;response.response=multiply.inputs[0];
    check(run<nf::vjp>(multiply,response,stream)==pg::program_status::success);gpu(cudaStreamSynchronize(stream));
    gpu(cudaMemcpy(left_adj.data(),dleft_adj,n*sizeof(float),cudaMemcpyDeviceToHost));gpu(cudaMemcpy(right_adj.data(),dright_adj,n*sizeof(float),cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<n;++i) check(std::abs(left_adj[i]-right[i]*cotangent[i])<2e-6f && std::abs(right_adj[i]-left[i]*cotangent[i])<2e-6f);
    // One primal may occupy repeated argument roles.  The program leaves their
    // separate adjoint contributions for the caller to assemble.
    response.device.right=dleft;
    check(run<nf::vjp>(multiply,response,stream)==pg::program_status::success);gpu(cudaStreamSynchronize(stream));
    gpu(cudaMemcpy(left_adj.data(),dleft_adj,n*sizeof(float),cudaMemcpyDeviceToHost));gpu(cudaMemcpy(right_adj.data(),dright_adj,n*sizeof(float),cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<n;++i) check(std::abs(left_adj[i]-left[i]*cotangent[i])<2e-6f && std::abs(right_adj[i]-left[i]*cotangent[i])<2e-6f);
    response.device.right=dright;
    response.request.action=nf::second_direction;response.request.direction_domain=multiply.inputs[0];response.request.response_domain=multiply.output.operand;response.direction=multiply.inputs[0];response.response=multiply.output.operand;
    check(run<nf::second_direction>(multiply,response,stream)==pg::program_status::success);gpu(cudaStreamSynchronize(stream));
    gpu(cudaMemcpy(result.data(),dresult,n*sizeof(float),cudaMemcpyDeviceToHost));
    for(std::size_t i=0;i<n;++i) check(std::abs(result[i]-2*dl[i]*dr[i])<2e-6f);
    std::fill(result.begin(),result.end(),91.f);upload(dresult,result);
    auto stale=response; ++stale.request.primal.instance.state.generation.value;
    check(run<nf::second_direction>(multiply,stale,stream)==pg::program_status::launch_failed);
    gpu(cudaStreamSynchronize(stream));
    gpu(cudaMemcpy(result.data(),dresult,n*sizeof(float),cudaMemcpyDeviceToHost));
    for(float value:result) check(value==91.f);
    cudaStreamDestroy(stream);
    for (auto address : {dleft,dright,ddl,ddr,dcot,dresult,dleft_adj,dright_adj}) cudaFree(address);
    std::cout << "D02 real CUDA width33 state/parameter JVP, VJP, second action and stale primal refusal passed\n";
}
