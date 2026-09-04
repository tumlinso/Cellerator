#include <Cellerator/sdk/create_narrow_public_runtime_facades_v1.hh>
#include <Cellerator/sdk/define_cpp_compiler_session_api_v1.hh>
#include <cassert>
namespace api=cellerator::compiler::api::v1;
api::runtime_status_v1 run(const api::runtime_relation_v1&,const api::runtime_value_plane_v1& v,const api::runtime_launch_v1&,void*)noexcept{return v.generation?api::runtime_status_v1::ok:api::runtime_status_v1::stale_generation;}
int main(){api::runtime_facade_v1 f{run,nullptr};api::runtime_relation_v1 r{{1,2,4},{3,4,8},5,1,9};api::runtime_value_plane_v1 v{nullptr,0,7};assert(api::execute_v1(f,r,v,{})==api::runtime_status_v1::ok);api::compiler_embedding_options_v1 opts{};(void)opts;}
