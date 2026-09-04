#include <Cellerator/compiler/lto/implement_cross_tu_inlining_and_connected_planning_v1.hh>
#include <cassert>
using namespace cellerator::compiler::lto::v1;
int main(){connected_relation_chain_v1 q{"packed.gene","packed.gene","module.v1","module.v1",100,120,80,40,true,true,true};connected_planning_result_v1 r;assert(plan_connected_cross_tu_chain_v1(q,&r)==connected_planning_status_v1::valid&&r.semantic_inlined&&r.persistent_order&&r.shared_decomposition&&r.materialized_total_ns==340&&r.selected_total_ns==220);q.authorized=false;assert(plan_connected_cross_tu_chain_v1(q,&r)==connected_planning_status_v1::unauthorized&&r.selected_total_ns==340);q.authorized=true;q.effects_permit_inline=false;assert(plan_connected_cross_tu_chain_v1(q,&r)==connected_planning_status_v1::effect_boundary);}
