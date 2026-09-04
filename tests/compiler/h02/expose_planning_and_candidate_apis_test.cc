#include <Cellerator/sdk/expose_planning_and_candidate_apis_v1.hh>
#include <cassert>
namespace ca=cellerator::compiler::api::v1;namespace{bool provider(std::vector<ca::candidate_v1>&v,void*)noexcept{v.push_back({7,"external","tiles",1.5,"measured","external-v1"});return true;}std::uint64_t planner(const std::vector<ca::candidate_v1>&v,void*)noexcept{return v.back().id;}}
int main(){ca::planning_report_v1 r;assert(ca::plan_candidates_v1({{provider},planner,nullptr,0},r));assert(r.selected==7&&r.selected_ruleset=="external-v1"&&!r.forced);ca::planning_report_v1 f;assert(ca::plan_candidates_v1({{provider},planner,nullptr,7},f)&&f.forced);}
