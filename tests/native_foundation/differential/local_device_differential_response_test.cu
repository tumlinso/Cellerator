#include <Cellerator/compute/operation/differential/local_arithmetic.hh>
#include <cuda_runtime_api.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
namespace df=cellerator::compute::differential; namespace nf=df::nf1; namespace nn=cellerator::compute::native_numeric; namespace ex=cellerator::execution; namespace pg=ex::program;
void check(bool v){if(!v) std::abort();} void gpu(cudaError_t v){check(v==cudaSuccess);}
nf::generation_stamp stamp(std::uint64_t id,std::uint64_t g){return {{id,1},{g}};}
nf::primal_record primal(const nf::identity& d){nf::primal_record p{}; p.instance.prepared={d,{2,1},{1},{3,1}};p.instance.state=stamp(4,7);p.instance.parameters=stamp(5,11);return p;}
struct vector { nn::resident_vector value{}; vector(std::size_t n,cudaStream_t s,const std::vector<float>& h,std::uint64_t g=1){gpu(nn::allocate(&value,n,nn::device_representation::f32,0));gpu(nn::upload(value,h.data(),n,{g},s));} ~vector(){nn::release(&value);} vector(const vector&)=delete; };
struct primitive { std::array<nf::operand_signature,2> in{}; nf::output_signature out{}; df::local_block block{};
 primitive(nn::local_operation op,std::size_t n,df::local_primal_owners owners){in={nf::operand_signature{{10,1},{},n,ex::numeric_type::f32},nf::operand_signature{{11,1},{},n,ex::numeric_type::f32}};out.operand={{12,1},{},n,ex::numeric_type::f32};out.assembly_owner={13,1};nf::operation_contract c{};c.definition={100+std::uint64_t(op),1};c.arguments={in.data(),nn::local_arity(op)};c.outputs={&out,1};c.numeric={ex::numeric_type::f32,ex::numeric_type::f32,ex::numeric_type::f32,ex::numeric_type::f32,ex::numeric_type::f32,ex::numeric_type::f32};c.capabilities=nf::forward|nf::jvp|nf::vjp|nf::second_direction;owners.identity=primal(c.definition);check(df::make_local_device_block(op,c,owners,block)==nf::status::success);}
};
template<nf::capability A> pg::program_status run(primitive& p,df::response_binding<float>& r,cudaStream_t s){pg::prepared_stage_v2 stage{};check(df::make_local_device_stage(p.block,A,1,1,0,stage)==nf::status::success);pg::prepared_program_v2 program{2,0,&stage,1,nullptr,0};pg::launch_binding_v2 launch{};launch.input=&r;return pg::execute_prepared_program_v2(program,&launch,1,s);}
df::response_binding<float> response(const primitive& p,nf::capability a,const nn::resident_vector* dl,const nn::resident_vector* dr,const nn::resident_vector* cot,const nn::resident_vector* out,const nn::resident_vector* la,const nn::resident_vector* ra){df::response_binding<float> r{};r.left_direction=dl;r.right_direction=dr;r.cotangent=cot;r.output=out;r.left_adjoint=la;r.right_adjoint=ra;r.count=out?out->elements:(la?la->elements:0);r.request.action=a;r.request.primal=primal(p.block.block.contract.definition);r.request.direction_domain=(a==nf::vjp?p.out.operand:p.in[0]);r.request.response_domain=(a==nf::vjp?p.in[0]:p.out.operand);r.direction=r.request.direction_domain;r.response=r.request.response_domain;return r;}
void get(const vector& d,std::vector<float>& h,cudaStream_t s){gpu(nn::download(d.value,h.data(),h.size(),s));gpu(cudaStreamSynchronize(s));}
int main(){gpu(cudaSetDevice(0));cudaDeviceProp prop{};gpu(cudaGetDeviceProperties(&prop,0));check(prop.major==7&&prop.minor==0);constexpr std::size_t n=33;cudaStream_t stream{};gpu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 std::vector<float> left(n),right(n),dl(n,.05f),dr(n,-.04f),cot(n),zero(n),got(n),la(n),ra(n);for(std::size_t i=0;i<n;++i){left[i]=.2f+i*.01f;right[i]=-.3f+i*.02f;cot[i]=.25f+i*.001f;}
 std::vector<float> state_sentinel(n,73.f),parameter_sentinel(n,71.f);
 vector L(n,stream,left,7),R(n,stream,right,11),DL(n,stream,dl),DR(n,stream,dr),C(n,stream,cot),O(n,stream,zero),LA(n,stream,zero),RA(n,stream,zero),S(n,stream,state_sentinel,17),P(n,stream,parameter_sentinel,19);gpu(cudaStreamSynchronize(stream));
 auto owners=[&](const nn::resident_vector* right_owner,const nn::resident_vector* state,const nn::resident_vector* parameters){df::local_primal_owners o{};o.left=&L.value;o.right=right_owner;o.state=state;o.parameters=parameters;o.stream=stream;o.identity=primal({});return o;};
 primitive mul(nn::local_operation::multiply,n,owners(&R.value,&L.value,&R.value)), add(nn::local_operation::add,n,owners(&R.value,&L.value,&R.value)), tanh_op(nn::local_operation::tanh,n,owners(nullptr,&L.value,&R.value)), protected_mul(nn::local_operation::multiply,n,owners(&R.value,&S.value,&P.value));
 auto jmul=response(mul,nf::jvp,&DL.value,&DR.value,nullptr,&O.value,nullptr,nullptr);check(run<nf::jvp>(mul,jmul,stream)==pg::program_status::success);get(O,got,stream);for(std::size_t i=0;i<n;++i)check(std::abs(got[i]-(right[i]*dl[i]+left[i]*dr[i]))<2e-6f);
 auto jadd=response(add,nf::jvp,&DL.value,&DR.value,nullptr,&O.value,nullptr,nullptr);check(run<nf::jvp>(add,jadd,stream)==pg::program_status::success);get(O,got,stream);for(auto v:got)check(std::abs(v-.01f)<2e-6f);
 auto jtanh=response(tanh_op,nf::jvp,&DL.value,nullptr,nullptr,&O.value,nullptr,nullptr);check(run<nf::jvp>(tanh_op,jtanh,stream)==pg::program_status::success);get(O,got,stream);for(std::size_t i=0;i<n;++i){float y=std::tanh(left[i]);check(std::abs(got[i]-(1-y*y)*dl[i])<2e-6f);}
 auto vmul=response(mul,nf::vjp,nullptr,nullptr,&C.value,nullptr,&LA.value,&RA.value);check(run<nf::vjp>(mul,vmul,stream)==pg::program_status::success);get(LA,la,stream);get(RA,ra,stream);for(std::size_t i=0;i<n;++i)check(std::abs(la[i]-right[i]*cot[i])<2e-6f&&std::abs(ra[i]-left[i]*cot[i])<2e-6f);
 auto vadd=response(add,nf::vjp,nullptr,nullptr,&C.value,nullptr,&LA.value,&RA.value);check(run<nf::vjp>(add,vadd,stream)==pg::program_status::success);get(LA,la,stream);get(RA,ra,stream);for(std::size_t i=0;i<n;++i)check(std::abs(la[i]-cot[i])<2e-6f&&std::abs(ra[i]-cot[i])<2e-6f);
 auto vtanh=response(tanh_op,nf::vjp,nullptr,nullptr,&C.value,nullptr,&LA.value,nullptr);check(run<nf::vjp>(tanh_op,vtanh,stream)==pg::program_status::success);get(LA,la,stream);for(std::size_t i=0;i<n;++i){float y=std::tanh(left[i]);check(std::abs(la[i]-(1-y*y)*cot[i])<2e-6f);}
 auto smul=response(mul,nf::second_direction,&DL.value,&DR.value,nullptr,&O.value,nullptr,nullptr);check(run<nf::second_direction>(mul,smul,stream)==pg::program_status::success);get(O,got,stream);for(auto v:got)check(std::abs(v-2*.05f*-.04f)<2e-6f);
 auto sadd=response(add,nf::second_direction,&DL.value,&DR.value,nullptr,&O.value,nullptr,nullptr);check(run<nf::second_direction>(add,sadd,stream)==pg::program_status::success);get(O,got,stream);for(auto v:got)check(std::abs(v)<2e-6f);
 auto stanh=response(tanh_op,nf::second_direction,&DL.value,nullptr,nullptr,&O.value,nullptr,nullptr);check(run<nf::second_direction>(tanh_op,stanh,stream)==pg::program_status::success);get(O,got,stream);for(std::size_t i=0;i<n;++i){float y=std::tanh(left[i]);check(std::abs(got[i]-(-2*y*(1-y*y)*.05f*.05f))<2e-6f);}
 // Binary right is mandatory and write ranges must not overlap a primal, direction, or another VJP result.
 auto no_right=jmul; no_right.right_direction=nullptr;check(run<nf::jvp>(mul,no_right,stream)==pg::program_status::launch_failed);auto alias=jmul;alias.output=&L.value;check(run<nf::jvp>(mul,alias,stream)==pg::program_status::launch_failed);auto vadj=vmul;vadj.right_adjoint=&LA.value;check(run<nf::vjp>(mul,vadj,stream)==pg::program_status::launch_failed);
 auto state_alias=response(protected_mul,nf::jvp,&DL.value,&DR.value,nullptr,&S.value,nullptr,nullptr);check(run<nf::jvp>(protected_mul,state_alias,stream)==pg::program_status::launch_failed);get(S,got,stream);for(float v:got)check(v==73.f);
 auto parameter_alias=response(protected_mul,nf::vjp,nullptr,nullptr,&C.value,nullptr,&P.value,&RA.value);check(run<nf::vjp>(protected_mul,parameter_alias,stream)==pg::program_status::launch_failed);get(P,got,stream);for(float v:got)check(v==71.f);
 // VJP does not require directions: the kernel must never dereference them on this action.
 check(vmul.left_direction==nullptr&&vmul.right_direction==nullptr);
 // Uploading the bound state owner changes the resident generation. A copied old request must refuse without writes.
 std::fill(got.begin(),got.end(),91.f);gpu(nn::upload(O.value,got.data(),n,{5},stream));gpu(nn::upload(L.value,left.data(),n,{8},stream));auto stale=smul;check(run<nf::second_direction>(mul,stale,stream)==pg::program_status::launch_failed);get(O,got,stream);for(float v:got)check(v==91.f);
 cudaStreamDestroy(stream);std::cout<<"D02 real CUDA width33 add/multiply/tanh JVP/VJP/second, resident-stale and alias refusal passed\n";
}
