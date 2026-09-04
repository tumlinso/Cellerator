#include <Cellerator/compiler/api/expose_planning_and_candidate_apis_v1.hh>
namespace cellerator::compiler::api::v1 {
bool plan_candidates_v1(const planning_request_v1&r,planning_report_v1&o) noexcept{try{planning_report_v1 x;for(auto p:r.providers)if(!p||!p(x.discovered,r.user_data))return false;if(x.discovered.empty())return false;x.selected=r.forced_candidate?r.forced_candidate:(r.planner?r.planner(x.discovered,r.user_data):x.discovered.front().id);x.forced=r.forced_candidate!=0;for(auto&c:x.discovered)if(c.id==x.selected){x.selected_ruleset=c.ruleset;o=std::move(x);return true;}return false;}catch(...){return false;}}
}
