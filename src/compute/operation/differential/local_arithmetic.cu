#include <Cellerator/compute/operation/differential/local_arithmetic.hh>
#include <cuda_runtime_api.h>
#include <limits>

namespace cellerator::compute::differential {
namespace {
using op = numeric::local_operation;
namespace pg = execution::program;
template<class T> __global__ void action_kernel(op operation, local_device_binding<T> b, nf1::capability capability) {
    const auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= b.count) return;
    const T a = b.left[i]; const T c = operation == op::tanh ? T{} : b.right[i];
    T da{}, db{};
    switch (operation) { case op::add: da=T{1}; db=T{1}; break; case op::multiply: da=c; db=a; break;
    case op::tanh: { const T y=tanh(a); da=T{1}-y*y; break; } }
    if (capability == nf1::jvp) b.output[i]=da*b.left_direction[i]+(operation==op::tanh?T{}:db*b.right_direction[i]);
    else if (capability == nf1::vjp) { b.left_adjoint[i]=da*b.cotangent[i]; if(operation!=op::tanh) b.right_adjoint[i]=db*b.cotangent[i]; }
    else { T second{}; if(operation==op::multiply) second=T{2}*b.left_direction[i]*b.right_direction[i];
           else if(operation==op::tanh) { const T y=tanh(a); second=T{-2}*y*(T{1}-y*y)*b.left_direction[i]*b.left_direction[i]; }
           b.output[i]=second; }
}
bool valid_owner(const numeric::resident_vector* v, std::uint64_t n, int device) {
    return v && v->data && v->elements==n && v->representation==numeric::device_representation::f32 && v->device_ordinal==device;
}
bool separate(const numeric::resident_vector* output, std::initializer_list<const numeric::resident_vector*> inputs) {
    if (!output) return false; for (const auto* input : inputs) if (input && input->data==output->data) return false; return true;
}
nf1::primal_record current_primal(const local_primal_owners& owners) {
    auto result=owners.identity; result.instance.state.generation=owners.state->generation; result.instance.parameters.generation=owners.parameters->generation; return result;
}
bool valid_primal_owners(const local_primal_owners& o, bool binary, std::uint64_t n) {
    if (!o.left || !o.stream || !valid_owner(o.left,n,o.left->device_ordinal) || !valid_owner(o.state,n,o.left->device_ordinal) || !valid_owner(o.parameters,n,o.left->device_ordinal)) return false;
    return !binary || valid_owner(o.right,n,o.left->device_ordinal);
}
template<class T> bool complete(const local_block& block, const response_binding<T>& r, nf1::capability action, bool binary, void* stream, local_device_binding<T>& d) {
    const auto& o=block.device_primal; if (!r.count || stream!=o.stream || !valid_primal_owners(o,binary,r.count)) return false;
    const int device=o.left->device_ordinal;
    if(action==nf1::vjp) {
        if(r.output || r.left_direction || r.right_direction || !valid_owner(r.cotangent,r.count,device)||!valid_owner(r.left_adjoint,r.count,device)||(binary&&!valid_owner(r.right_adjoint,r.count,device))||
           !separate(r.left_adjoint,{o.left,o.right,r.cotangent,r.left_direction,r.right_direction}) || (binary&&(!separate(r.right_adjoint,{o.left,o.right,r.cotangent,r.left_direction,r.right_direction,r.left_adjoint})||r.left_adjoint->data==r.right_adjoint->data))) return false;
        d={static_cast<const T*>(o.left->data),binary?static_cast<const T*>(o.right->data):nullptr,nullptr,nullptr,static_cast<const T*>(r.cotangent->data),nullptr,static_cast<T*>(r.left_adjoint->data),binary?static_cast<T*>(r.right_adjoint->data):nullptr,r.count}; return true;
    }
    if(r.cotangent || r.left_adjoint || r.right_adjoint || !valid_owner(r.output,r.count,device)||!valid_owner(r.left_direction,r.count,device)||(binary&&!valid_owner(r.right_direction,r.count,device))||
       !separate(r.output,{o.left,o.right,r.left_direction,r.right_direction})) return false;
    d={static_cast<const T*>(o.left->data),binary?static_cast<const T*>(o.right->data):nullptr,static_cast<const T*>(r.left_direction->data),binary?static_cast<const T*>(r.right_direction->data):nullptr,nullptr,static_cast<T*>(r.output->data),nullptr,nullptr,r.count}; return true;
}
template<class T, nf1::capability Capability> pg::program_status device_callback(const void* state, const pg::launch_binding_v2& launch, void* stream) noexcept {
    if(!state||!launch.input||launch.output||launch.values||!stream) return pg::program_status::invalid_argument;
    const auto& block=*static_cast<const local_block*>(state); const auto& response=*static_cast<const response_binding<T>*>(launch.input);
    if(response.request.action!=Capability || nf1::validate_derivative(block.block.contract,response.request,current_primal(block.device_primal),response.direction,response.response)!=nf1::status::success) return pg::program_status::invalid_argument;
    const auto arity=numeric::local_arity(block.operation); local_device_binding<T> d{};
    if(!arity||!complete(block,response,Capability,arity==2,stream,d)||d.count>std::uint64_t(std::numeric_limits<unsigned>::max())*256u) return pg::program_status::invalid_argument;
    action_kernel<<<static_cast<unsigned>((d.count+255u)/256u),256,0,static_cast<cudaStream_t>(stream)>>>(block.operation,d,Capability);
    return cudaGetLastError()==cudaSuccess?pg::program_status::success:pg::program_status::launch_failed;
}
}
nf1::status make_local_device_block(op operation, const nf1::operation_contract& contract, const local_primal_owners& owners, local_block& output) noexcept {
    const auto arity=numeric::local_arity(operation); const auto n=contract.arguments.empty()?0:contract.arguments[0].element_count;
    if(!arity||!valid_primal_owners(owners,arity==2,n)) return nf1::status::invalid_binding;
    if(!(contract.capabilities&nf1::second_direction)) return nf1::status::unsupported_capability;
    auto first_order=contract; first_order.capabilities&=~nf1::second_direction;
    const auto status=make_local_block(operation,first_order,output); if(status!=nf1::status::success) return status;
    output.block.contract=contract; output.device_primal=owners;
    output.block.jvp_launch=device_callback<float,nf1::jvp>; output.block.vjp_launch=device_callback<float,nf1::vjp>; output.block.second_launch=device_callback<float,nf1::second_direction>;
    if(contract.numeric.state_storage!=execution::numeric_type::f32) return nf1::status::unsupported_capability;
    return nf1::validate_compiled_block(output.block);
}
} // namespace cellerator::compute::differential
