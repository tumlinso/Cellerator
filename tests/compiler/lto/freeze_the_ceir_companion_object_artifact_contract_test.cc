#include <Cellerator/compiler/lto/freeze_the_ceir_companion_object_artifact_contract_v1.hh>
#include <cassert>
using namespace cellerator::compiler::lto::v1;
int main(){for(auto f:{object_format_v1::elf,object_format_v1::mach_o,object_format_v1::coff,object_format_v1::archive,object_format_v1::sidecar}){ceir_companion_artifact_v1 a;a.format=f;a.semantic_summary={1,1};a.planning_summary={1,2};a.profile_reference={1,3};a.toolchain={1,4};a.content_hash[0]=1;a.fields={{{2,1},"field"}};a.provenance={{{3,1},{4,1}}};a.placement=f==object_format_v1::sidecar?"file.ceir":"section";assert(validate_companion_artifact_v1(a)==companion_status_v1::valid);a.content_hash={};assert(validate_companion_artifact_v1(a)==companion_status_v1::missing_hash);}}
