#include <Cellerator/compiler/lto/freeze_the_ceir_companion_object_artifact_contract_v1.hh>
#include <algorithm>
#include <set>
namespace cellerator::compiler::lto::v1 {namespace{bool valid(artifact_identity_v1 i){return i.high||i.low;}}
companion_status_v1 validate_companion_artifact_v1(const ceir_companion_artifact_v1&a)noexcept{if(a.version!=companion_artifact_version_v1)return companion_status_v1::unsupported_version;if(!valid(a.semantic_summary)||!valid(a.planning_summary)||!valid(a.profile_reference)||!valid(a.toolchain))return companion_status_v1::invalid_identity;if(std::all_of(a.content_hash.begin(),a.content_hash.end(),[](auto b){return b==0;}))return companion_status_v1::missing_hash;if(a.placement.empty())return companion_status_v1::invalid_placement;std::set<std::string>s;for(const auto&f:a.fields)if(!valid(f.field)||f.symbol.empty()||!s.insert(f.symbol).second)return companion_status_v1::duplicate_export;return companion_status_v1::valid;}
}
