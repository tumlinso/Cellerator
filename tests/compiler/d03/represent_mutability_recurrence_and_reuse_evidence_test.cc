#include <Cellerator/compiler/profile/represent_mutability_recurrence_and_reuse_evidence_v1.hh>
#include <cassert>
#include <cmath>
int main() {
 using namespace cellerator::compiler::profile::v1;
 const profile_trace_observation_v1 trace[]={{1,1,1,1,0,0},{1,2,1,1,4,8},{2,3,1,1,4,4},{2,4,2,1,2,6},{2,5,2,2,6,10}};
 reuse_profile_evidence_v1 e{};
 assert(infer_reuse_profile_evidence_v1(trace,5,{1,2},{3,4},&e)==reuse_profile_status_v1::ok);
 assert(e.structure_change.rate==0.25 && e.value_change.rate==1.0);
 assert(e.support_change.rate==0.25 && e.order_change.rate==0.25);
 assert(e.reuse_horizon==1.0 && e.field_frequency==4.0 && e.mean_loop_count==7.0);
 assert(std::abs(e.structure_mutation_half_life-std::log(2.0)/0.25)<1e-12);
}
