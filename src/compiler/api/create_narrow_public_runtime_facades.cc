#include <Cellerator/compiler/api/create_narrow_public_runtime_facades_v1.hh>
namespace cellerator::compiler::api::v1 {
runtime_status_v1 execute_v1(const runtime_facade_v1& f,const runtime_relation_v1& r,const runtime_value_plane_v1& v,const runtime_launch_v1& l) noexcept {
 if(!f.execute||r.source.extent==0||r.destination.extent==0)return runtime_status_v1::invalid_identity;
 return f.execute(r,v,l,f.user_data);
}
}
