#include <Cellerator/sdk/expose_pass_reflection_and_extension_apis_v1.hh>
#include <cassert>
int main(){auto s=cellerator::compiler::api::v1::programmable_compiler_surface();assert(s.abi_version==1&&s.passes&&s.reflection&&s.extensions&&s.same_compilation&&s.explicit_trust);cellerator::compiler::pass::v1::extension_registry_v1 registry;assert(registry.register_namespace({"external",1,{}})==cellerator::compiler::pass::v1::extension_registration_status_v1::success);}
