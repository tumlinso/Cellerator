#include <Cellerator/compiler/tooling/implement_host_only_no_profile_editing_behavior_v1.hh>
namespace Cellerator::compiler::tooling {
std::vector<editor_capability_status_v1> editor_capabilities_v1(bool cuda,bool profile){return {
 {editor_capability_v1::syntax,true,""},{editor_capability_v1::cxx_semantics,true,""},{editor_capability_v1::ast,true,""},{editor_capability_v1::structural_ceir,true,""},
 {editor_capability_v1::profile_analysis,profile,profile?"":"profile not loaded"},
 {editor_capability_v1::cuda_analysis,cuda,cuda?"":"CUDA unavailable"}};}
} // namespace Cellerator::compiler::tooling
