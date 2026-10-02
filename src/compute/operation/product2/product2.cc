#include "Cellerator/compute/operation/product2/product2.hh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
namespace cellerator::compute::product2 {
namespace {
struct range { uintptr_t start,end; };
bool extent(const void *p,uint64_t count,uint64_t need,range& r) noexcept {
 if(count<need || (need && (!p || reinterpret_cast<uintptr_t>(p)%alignof(float))) || count>SIZE_MAX/sizeof(float)) return false;
 auto start=reinterpret_cast<uintptr_t>(p); auto bytes=count*sizeof(float);
 if(bytes>UINTPTR_MAX-start)return false;
 r={start,start+bytes}; return true;
}
bool overlaps(range a,range b) noexcept {return a.start<a.end && b.start<b.end && a.start<b.end && b.start<a.end;}
using program_status=execution::program::program_status;
program_status stage_admit(const void* s,const execution::program::launch_binding_v2& b,void* stream) noexcept {
 if(!s || !b.input || stream)return program_status::invalid_argument;
 const auto& state=*static_cast<const stage_state*>(s);
 if(!state.owner)return program_status::invalid_argument;
 return state.owner->admit(*static_cast<const bound_view*>(b.input),state.op)==CE_PRODUCT2_SUCCESS?program_status::success:program_status::invalid_argument;
}
program_status stage_launch(const void* s,const execution::program::launch_binding_v2& b,void* stream) noexcept {
 if(stage_admit(s,b,stream)!=program_status::success)return program_status::invalid_argument;
 const auto& state=*static_cast<const stage_state*>(s);
 return state.owner->execute(*static_cast<const bound_view*>(b.input),state.op)==CE_PRODUCT2_SUCCESS?program_status::success:program_status::launch_failed;
}
}
prepared_owner::prepared_owner(const descriptor& d):topology_(d) {
 if(d.binds_axes && (d.input_axis.extent!=d.input_count || d.product_axis.extent!=d.product_count || d.coefficient_axis.extent!=d.product_count || execution::validate_persistent_axis_identity(d.input_axis.identity)!=execution::biological_validation_code::ok || execution::validate_persistent_axis_identity(d.product_axis.identity)!=execution::biological_validation_code::ok || execution::validate_persistent_axis_identity(d.coefficient_axis.identity)!=execution::biological_validation_code::ok))throw std::invalid_argument("product2 biological axes");
 if(d.product_count>SIZE_MAX/sizeof(int64_t) || d.a_count>SIZE_MAX/sizeof(int64_t) || d.b_count>SIZE_MAX/sizeof(int64_t) || d.input_count>SIZE_MAX/sizeof(float) || d.product_count*sizeof(int64_t)>UINTPTR_MAX-reinterpret_cast<uintptr_t>(d.a) || d.product_count*sizeof(int64_t)>UINTPTR_MAX-reinterpret_cast<uintptr_t>(d.b))throw std::length_error("product2 count overflow");
 if(d.a_count<d.product_count || d.b_count<d.product_count || (d.product_count && (!d.a || !d.b || reinterpret_cast<uintptr_t>(d.a)%alignof(int64_t) || reinterpret_cast<uintptr_t>(d.b)%alignof(int64_t))))throw std::invalid_argument("product2 index extent");
 if(d.product_count) {a_.assign(d.a,d.a+d.product_count);b_.assign(d.b,d.b+d.product_count);}
 for(uint64_t i=0;i<d.product_count;++i)if(a_[i]<0 || b_[i]<0 || uint64_t(a_[i])>=d.input_count || uint64_t(b_[i])>=d.input_count)throw std::out_of_range("product2 index");
 topology_.a=a_.data();topology_.b=b_.data();topology_.a_count=d.product_count;topology_.b_count=d.product_count;
}
ce_product2_status prepared_owner::admit_numeric(const ce_product2_binding& v,operation op) const noexcept {
 if(v.expected_structure_generation!=topology_.structure_generation || v.current_structure_generation!=v.expected_structure_generation || v.expected_value_generation!=v.current_value_generation || v.expected_parameter_generation!=v.current_parameter_generation)return CE_PRODUCT2_STALE_GENERATION;
 if(op!=operation::forward && op!=operation::vjp && op!=operation::jvp)return CE_PRODUCT2_INVALID_ARGUMENT;
 const uint64_t n=topology_.input_count,m=topology_.product_count;
 const uint64_t counts[]={v.x_count,v.k_count,v.y_count,v.g_count,v.dx_count,v.dk_count,v.dy_count,v.gx_count,v.gk_count};
 for(auto c:counts)if(c>SIZE_MAX/sizeof(float))return CE_PRODUCT2_OVERFLOW;
 range inputs[5]{},outputs[2]{}; unsigned ni=2,no=0;
 if(!extent(v.x,v.x_count,n,inputs[0]) || !extent(v.k,v.k_count,m,inputs[1]))return CE_PRODUCT2_INVALID_ARGUMENT;
 if(op==operation::forward){no=1;if(!extent(v.y,v.y_count,m,outputs[0]))return CE_PRODUCT2_INVALID_ARGUMENT;}
 if(op==operation::vjp){ni=3;no=2;if(!extent(v.g,v.g_count,m,inputs[2]) || !extent(v.gx,v.gx_count,n,outputs[0]) || !extent(v.gk,v.gk_count,m,outputs[1]))return CE_PRODUCT2_INVALID_ARGUMENT;}
 if(op==operation::jvp){ni=4;no=1;if(!extent(v.dx,v.dx_count,n,inputs[2]) || !extent(v.dk,v.dk_count,m,inputs[3]) || !extent(v.dy,v.dy_count,m,outputs[0]))return CE_PRODUCT2_INVALID_ARGUMENT;}
 for(unsigned o=0;o<no;++o){for(unsigned i=0;i<ni;++i)if(overlaps(outputs[o],inputs[i]))return CE_PRODUCT2_ALIAS;for(unsigned p=0;p<o;++p)if(overlaps(outputs[o],outputs[p]))return CE_PRODUCT2_ALIAS;}
 return CE_PRODUCT2_SUCCESS;
}
ce_product2_status prepared_owner::admit(const bound_view& v,operation op) const noexcept {
 if(topology_.binds_axes) {
  namespace nf1=::cellerator::compute::operation::nf1;
  if(execution::validate_persistent_axis_identity(v.input_axis)!=execution::biological_validation_code::ok || execution::validate_persistent_axis_identity(v.product_axis)!=execution::biological_validation_code::ok || execution::validate_persistent_axis_identity(v.coefficient_axis)!=execution::biological_validation_code::ok || !nf1::same_axis(topology_.input_axis.identity,v.input_axis) || !nf1::same_axis(topology_.product_axis.identity,v.product_axis) || !nf1::same_axis(topology_.coefficient_axis.identity,v.coefficient_axis))return CE_PRODUCT2_INVALID_ARGUMENT;
 }
 return admit_numeric(v.numeric,op);
}
ce_product2_status prepared_owner::execute_numeric(const ce_product2_binding& v,operation op) const noexcept {
 auto status=admit_numeric(v,op);if(status!=CE_PRODUCT2_SUCCESS)return status;
 const auto n=topology_.input_count,m=topology_.product_count;
 if(op==operation::vjp && n)std::fill(v.gx,v.gx+n,0.f);
 for(uint64_t i=0;i<m;++i){const auto a=a_[i],b=b_[i];const float xa=v.x[a],xb=v.x[b];
  if(op==operation::forward)v.y[i]=(v.k[i]*xa)*xb;
  else if(op==operation::vjp){v.gx[a]+=(v.g[i]*v.k[i])*xb;v.gx[b]+=(v.g[i]*v.k[i])*xa;v.gk[i]=(v.g[i]*xa)*xb;}
  else v.dy[i]=v.k[i]*std::fma(v.dx[a],xb,xa*v.dx[b])+(v.dk[i]*xa)*xb;
 }
 return CE_PRODUCT2_SUCCESS;
}
ce_product2_status prepared_owner::admit(const ce_product2_binding& v,operation op) const noexcept {
 return topology_.binds_axes?CE_PRODUCT2_INVALID_ARGUMENT:admit_numeric(v,op);
}
ce_product2_status prepared_owner::execute(const ce_product2_binding& v,operation op) const noexcept {
 return topology_.binds_axes?CE_PRODUCT2_INVALID_ARGUMENT:execute_numeric(v,op);
}
ce_product2_status prepared_owner::execute(const bound_view& v,operation op) const noexcept {
 auto status=admit(v,op);return status==CE_PRODUCT2_SUCCESS?execute_numeric(v.numeric,op):status;
}
execution::program::prepared_stage_v2 make_stage(const stage_state& s,uint64_t id,uint32_t index) noexcept {
 execution::program::prepared_stage_v2 out{};out.stable_stage_id=id;out.candidate_id=6;out.prepared_state=&s;out.binding_index=index;out.admit=stage_admit;out.launch=stage_launch;return out;
}
}
struct ce_product2_context {cellerator::compute::product2::prepared_owner owner;explicit ce_product2_context(const cellerator::compute::product2::descriptor&d):owner(d){}};
extern "C" ce_product2_status ce_product2_create(uint64_t n,uint64_t m,const int64_t*a,uint64_t ac,const int64_t*b,uint64_t bc,uint64_t gen,ce_product2_context**out){
 if(!out)return CE_PRODUCT2_INVALID_ARGUMENT;
 // Never overwrite an existing context handle on rejection.
 if(n>SIZE_MAX/sizeof(float) || m>SIZE_MAX/sizeof(int64_t) || ac>SIZE_MAX/sizeof(int64_t) || bc>SIZE_MAX/sizeof(int64_t) || m*sizeof(int64_t)>UINTPTR_MAX-reinterpret_cast<uintptr_t>(a) || m*sizeof(int64_t)>UINTPTR_MAX-reinterpret_cast<uintptr_t>(b))return CE_PRODUCT2_OVERFLOW;
 if(ac<m || bc<m || (m && (!a || !b || reinterpret_cast<uintptr_t>(a)%alignof(int64_t) || reinterpret_cast<uintptr_t>(b)%alignof(int64_t))))return CE_PRODUCT2_INVALID_ARGUMENT;
 for(uint64_t i=0;i<m;++i)if(a[i]<0 || b[i]<0 || uint64_t(a[i])>=n || uint64_t(b[i])>=n)return CE_PRODUCT2_INVALID_INDEX;
 try{*out=new ce_product2_context({n,m,gen,a,b,ac,bc});return CE_PRODUCT2_SUCCESS;}
 catch(const std::length_error&){return CE_PRODUCT2_OVERFLOW;}catch(...){return CE_PRODUCT2_BACKEND_ERROR;}
}
extern "C" void ce_product2_destroy(ce_product2_context*c){delete c;}
#define CE_EXECUTE(NAME,OP) extern "C" ce_product2_status ce_product2_##NAME(const ce_product2_context*c,const ce_product2_binding*b){return c&&b?c->owner.execute(*b,cellerator::compute::product2::operation::OP):CE_PRODUCT2_INVALID_ARGUMENT;}
CE_EXECUTE(forward,forward)
CE_EXECUTE(vjp,vjp)
CE_EXECUTE(jvp,jvp)

#ifdef CELLERATOR_PRODUCT2_HAS_CUDA
namespace cellerator::compute::product2 {
namespace {
execution::program::program_status cuda_admit_stage(const void*s,const execution::program::launch_binding_v2&b,void*stream) noexcept {
 if(!s || !b.input)return execution::program::program_status::invalid_argument;
 auto& state=*static_cast<const cuda_stage_state*>(s);
 const auto& view=*static_cast<const bound_view*>(b.input);
 if(!state.owner || state.owner->admit(view,state.op)!=CE_PRODUCT2_SUCCESS)return execution::program::program_status::invalid_argument;
 const auto& topology=state.owner->topology();
 if(ce_product2_cuda_matches(state.context,topology.input_count,topology.product_count,topology.a,topology.a_count,topology.b,topology.b_count,topology.structure_generation)!=CE_PRODUCT2_SUCCESS)return execution::program::program_status::invalid_argument;
 return ce_product2_cuda_admit(state.context,&view.numeric,stream,static_cast<int>(state.op))==CE_PRODUCT2_SUCCESS?execution::program::program_status::success:execution::program::program_status::invalid_argument;
}
execution::program::program_status cuda_launch_stage(const void*s,const execution::program::launch_binding_v2&b,void*stream) noexcept {
 if(cuda_admit_stage(s,b,stream)!=execution::program::program_status::success)return execution::program::program_status::invalid_argument;
 auto& state=*static_cast<const cuda_stage_state*>(s);auto*v=&static_cast<const bound_view*>(b.input)->numeric;
 auto result=state.op==operation::forward?ce_product2_cuda_forward(state.context,v,stream):state.op==operation::vjp?ce_product2_cuda_vjp(state.context,v,stream):ce_product2_cuda_jvp(state.context,v,stream);
 return result==CE_PRODUCT2_SUCCESS?execution::program::program_status::success:execution::program::program_status::launch_failed;
}
}
execution::program::prepared_stage_v2 make_cuda_stage(const cuda_stage_state&s,uint64_t id,uint32_t index) noexcept {
 execution::program::prepared_stage_v2 out{};out.stable_stage_id=id;out.candidate_id=6;out.prepared_state=&s;out.binding_index=index;out.admit=cuda_admit_stage;out.launch=cuda_launch_stage;return out;
}
}
#endif
