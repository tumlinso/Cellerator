#include <Cellerator/compiler/diagnostics/implement_candidate_decision_reports_v1.hh>
#include <cassert>
int main(){using namespace cellerator::compiler::diagnostics::v1;auto r=build_candidate_report({{1,100,2,5,candidate_source::compiler,true,false,0},{2,80,1,8,candidate_source::user_edit,true,true,0},{3,60,9,20,candidate_source::fallback,false,false,7}});assert(r.valid&&r.legal_total_ns==180&&r.selected_cost_ns==80&&r.selected_id==2);auto bad=r.candidates;bad[2].rejection_reason=0;assert(!build_candidate_report(bad).valid);}
