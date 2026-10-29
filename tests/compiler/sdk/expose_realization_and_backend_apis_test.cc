#include <Cellerator/sdk/expose_realization_and_backend_apis_v1.hh>
#include <cassert>
namespace ca=cellerator::compiler::api::v1;namespace{bool hook(ca::physical_ir_v1&i,void*)noexcept{i.operations.push_back("cpu.fragment");return true;}bool emit(const ca::target_description_v1&t,const ca::physical_ir_v1&i,ca::generated_artifact_v1&o,void*)noexcept{o.kind="object/"+t.architecture;o.bytes={static_cast<std::uint8_t>(i.operations.size())};return true;}}
int main(){ca::backend_registry_v1 r;assert(r.add({"external-cpu",emit,hook,nullptr}));auto*b=r.find("external-cpu");ca::generated_artifact_v1 o;assert(ca::emit_object_v1(*b,{"cpu","x86_64"},{{"load"},{"sample.cell:4"}},o));assert(o.kind=="object/cpu"&&o.bytes[0]==2);}
