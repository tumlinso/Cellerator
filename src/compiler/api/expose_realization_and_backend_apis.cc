#include <Cellerator/compiler/api/expose_realization_and_backend_apis_v1.hh>
namespace cellerator::compiler::api::v1 {
bool backend_registry_v1::add(backend_v1 b){if(b.name.empty()||!b.emit||find(b.name))return false;entries_.push_back(std::move(b));return true;}
const backend_v1* backend_registry_v1::find(const std::string&n)const noexcept{for(auto&e:entries_)if(e.name==n)return&e;return nullptr;}
bool emit_object_v1(const backend_v1&b,const target_description_v1&t,physical_ir_v1 ir,generated_artifact_v1&o)noexcept{if(!b.emit)return false;if(b.fragment&&!b.fragment(ir,b.user_data))return false;return b.emit(t,ir,o,b.user_data);}
}
