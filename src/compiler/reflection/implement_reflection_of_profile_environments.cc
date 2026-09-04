#include <Cellerator/compiler/reflection/implement_reflection_of_profile_environments_v1.hh>
#include <set>
namespace cellerator::compiler::reflection::v1 {
std::optional<reflected_profile_state_v1> query_profile_state_v1(const reflected_profile_environment_v1&e,const std::string&n,const std::string&b){for(const auto&s:e.states)if(s.name==n&&(b.empty()?s.branch_condition.empty():s.branch_condition==b))return s;return{};}
bool validate_profile_environment_v1(const reflected_profile_environment_v1&e)noexcept{std::set<std::string> names;bool selected=false;for(const auto&s:e.states){if(s.name.empty()||s.handle.kind!=handle_kind_v1::profile_state||s.evidence.confidence<0||s.evidence.confidence>1||!names.insert(s.name+"\n"+s.branch_condition).second)return false;if(s.name==e.selected)selected=true;}return selected;}
}
