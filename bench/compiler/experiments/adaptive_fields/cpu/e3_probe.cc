#include <Cellerator/compiler/planning/implement_planning_cache_and_invalidation_v1.hh>
#include <Cellerator/compute/support/adaptive_support.hh>
#include <Cellerator/compiler/backend/implement_generic_cpu_relation_apply_v1.hh>
#include <Cellerator/compiler/backend/implement_cpu_segment_gate_update_bundle_and_chain_paths_v1.hh>
#include <Cellerator/compute/operation/relation_algebra.hh>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace p=Cellerator::compiler::planning;
namespace s=cellerator::compute::support;
namespace cb=cellerator::compiler::backend::v1;
namespace ex=cellerator::execution;
namespace co=cellerator::compute::operation;

ex::persistent_axis_identity axis(std::uint64_t seed){
  return {{ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,
    sizeof(ex::persistent_axis_identity)},{seed,1},{seed+1,1},{seed+2,1},{seed+3,1}};
}
std::vector<float> native_segment_result(const std::vector<std::uint64_t>&ids,
    const std::unordered_map<std::uint64_t,float>&values){
  if(ids.empty())return {};
  std::vector<std::uint64_t> offsets{0},columns;
  std::vector<float> edge_values;
  for(auto id:ids){columns.push_back(id==20?0:1);edge_values.push_back(values.at(id));offsets.push_back(columns.size());}
  const float state[]{2.0f,3.0f};std::vector<float> intermediate(ids.size()),output(ids.size());
  std::vector<std::uint64_t> segment_offsets{0,static_cast<std::uint64_t>(ids.size())};
  // The existing primitive consumes projection ordinals here; stable logical
  // identities remain in the adapter workset/cache maps above.
  std::vector<std::uint64_t> logical(ids.size());for(std::size_t i=0;i<ids.size();++i)logical[i]=i;
  cb::cpu_relation_apply_request_v1 q{};
  q.problem.operation_identity={7,11};q.problem.relation.structure={17,19};q.problem.relation.epoch={1};
  q.problem.relation.source_axis=axis(30);q.problem.relation.destination_axis=axis(40);
  q.problem.relation.logical_edge_count=ids.size();
  q.problem.numeric={ex::numeric_type::f32,ex::numeric_type::f32,ex::numeric_type::f32,
    ex::numeric_type::f32,ex::numeric_type::f32,ex::numeric_type::f32,
    co::core::rounding_policy::nearest_even,co::core::saturation_policy::none,co::nan_policy_v1::reject,{}};
  q.problem.semantic_flags=co::alpha_applied_once|co::beta_applied_once;q.problem.dense_width=1;
  q.projection.destination_offsets=offsets.data();q.projection.source_indices=columns.data();
  q.projection.logical_edge_ids=logical.data();q.projection.source_count=2;
  q.projection.destination_count=ids.size();q.projection.logical_edge_count=ids.size();
  q.relation_values=edge_values.data();q.input=state;q.output=intermediate.data();
  assert(cb::apply_cpu_relation_v1(q)==cb::cpu_relation_apply_status_v1::success);
  assert(cb::run_cpu_segment_v1({cb::cpu_segment_kind_v1::softmax,segment_offsets.data(),1,
    intermediate.data(),output.data()})==cb::cpu_fallback_status_v1::success);
  return output;
}

struct projection_artifact { std::uint64_t epoch=0,order=0;std::vector<std::uint64_t> ids; };
struct counters { std::uint64_t compile=0,index=0,projection=0,workset=0,operands=0,pack=0,rebind=0; };
std::vector<std::uint64_t> support_ids(const s::support_snapshot_pair_v1& snap){
  std::vector<std::uint64_t> ids;
  for(const auto*view:{snap.base,snap.delta})if(view)for(std::uint64_t i=0;i<view->edge_count;++i)ids.push_back(view->edges[i].logical_edge_id);
  std::sort(ids.begin(),ids.end());return ids;
}
std::vector<std::uint64_t> make_workset(const std::vector<std::uint64_t>&ids,
   const std::unordered_map<std::uint64_t,bool>&eligible,const std::vector<std::uint64_t>&sampled){
  std::vector<std::uint64_t> out;for(auto id:ids)if((eligible.contains(id)&&eligible.at(id))||
     std::find(sampled.begin(),sampled.end(),id)!=sampled.end())out.push_back(id);return out;
}
std::vector<std::uint64_t> unique_union(const std::vector<std::uint64_t>&a,const std::vector<std::uint64_t>&b){
  auto out=a;out.insert(out.end(),b.begin(),b.end());std::sort(out.begin(),out.end());
  out.erase(std::unique(out.begin(),out.end()),out.end());return out;
}

int main(){
  p::planning_cache_key_v1 key{1,2,3,4,5,6,7,8,9,10};
  auto same=key;
  auto value_generation=1u, active_generation=1u;
  // Existing planning cache: values and gates are not its key dimensions.
  ++value_generation;
  assert(p::validate_planning_cache_key_v1(key,same));
  ++active_generation;
  assert(p::validate_planning_cache_key_v1(key,same));
  auto changed_structure=key;changed_structure.structure_epoch++;
  auto structure_result=p::validate_planning_cache_key_v1(key,changed_structure);
  assert(structure_result.code==p::planning_cache_validation_code_v1::invalidated &&
         structure_result.resume_at==p::planning_resumption_point_v1::structure_planning);
  auto changed_order=key;changed_order.order_identity++;
  assert(p::validate_planning_cache_key_v1(key,changed_order).resume_at==
         p::planning_resumption_point_v1::order_transitions);
  auto changed_toolchain=key;changed_toolchain.toolchain_identity++;
  assert(p::validate_planning_cache_key_v1(key,changed_toolchain).resume_at==
         p::planning_resumption_point_v1::target_candidates);

  // Exercise real stable logical IDs and manager publication/restore APIs.
  const s::logical_support_edge_v1 edges[]={{10,0,2,0},{20,2,2,0}};
  const s::logical_support_participant_v1 participants[]={{0,0,101},{1,0,201},{0,0,102},{1,0,201}};
  const s::logical_support_view_v1 base{1,0,7,99,3,edges,2,participants,4};
  s::adaptive_support_manager_v1 manager;
  assert(manager.initialize(base,8)==s::adaptive_support_status_v1::success);
  counters count{};count.compile=1;
  s::support_snapshot_pair_v1 initial_snapshot{};
  assert(manager.snapshot(&initial_snapshot)==s::adaptive_support_status_v1::success);
  std::uint64_t order=1;
  projection_artifact projection{initial_snapshot.snapshot_structure_epoch,order,support_ids(initial_snapshot)};
  ++count.projection;++count.index;
  auto projection_current=[&](const projection_artifact&a,std::uint64_t epoch,std::uint64_t ord){return a.epoch==epoch&&a.order==ord;};
  assert(projection.ids==std::vector<std::uint64_t>({10,20}));
  const s::logical_support_participant_v1 next_parts[]={{0,0,103},{1,0,203}};
  assert(manager.stage_promotion({30,next_parts,2,0,1,4,0.9f})==s::adaptive_support_status_v1::success);
  assert(manager.stage_revocation(10)==s::adaptive_support_status_v1::success);
  assert(manager.publish_pending(5)==s::adaptive_support_status_v1::success);
  s::support_snapshot_pair_v1 snapshot{};
  assert(manager.snapshot(&snapshot)==s::adaptive_support_status_v1::success);
  assert(snapshot.snapshot_structure_epoch==5 && snapshot.base->edge_count==1 &&
         snapshot.base->edges[0].logical_edge_id==20 && snapshot.delta->edge_count==1 &&
         snapshot.delta->edges[0].logical_edge_id==30);
  s::adaptive_support_manager_v1 restored;
  assert(restored.initialize(base,8)==s::adaptive_support_status_v1::success);
  assert(restored.restore_snapshot(snapshot,manager.revoked_edge_ids().data(),
      manager.revoked_edge_ids().size(),manager.promotion_records().data(),
      manager.promotion_records().size())==s::adaptive_support_status_v1::success);
  s::support_snapshot_pair_v1 restored_snapshot{};
  assert(restored.snapshot(&restored_snapshot)==s::adaptive_support_status_v1::success);
  assert(restored_snapshot.snapshot_structure_epoch==5 &&
         restored_snapshot.base->edges[0].logical_edge_id==20 &&
         restored_snapshot.delta->edges[0].logical_edge_id==30);

  // Materialize a deliberately small dependency graph against the actual
  // manager snapshot; these adapter counters are not production cache hooks.
  assert(!projection_current(projection,snapshot.snapshot_structure_epoch,order));
  projection={snapshot.snapshot_structure_epoch,order,support_ids(snapshot)};
  ++count.projection;++count.index;
  assert(projection.ids==std::vector<std::uint64_t>({20,30}));
  std::unordered_map<std::uint64_t,bool> eligible{{20,true},{30,true}};
  const std::vector<std::uint64_t> sampled{30};
  auto primal=make_workset(projection.ids,eligible,{});
  auto derivative=make_workset(projection.ids,eligible,sampled);++count.workset;
  assert(primal==std::vector<std::uint64_t>({20,30})&&derivative==std::vector<std::uint64_t>({20,30}));
  std::unordered_map<std::uint64_t,float> value_store{{20,2.0f},{30,3.0f}};
  std::unordered_map<std::uint64_t,float> rebound_store{{20,4.0f},{30,6.0f}};
  const auto* value_binding=&value_store;
  std::vector<std::uint64_t> operand_ids;
  std::unordered_map<std::uint64_t,float> operand_values;
  auto materialize_operands=[&](){operand_ids=unique_union(primal,derivative);operand_values.clear();
    for(auto id:operand_ids)operand_values[id]=value_binding->at(id);++count.operands;};
  materialize_operands();const auto before=native_segment_result(primal,operand_values);
  value_binding=&rebound_store;++count.rebind;materialize_operands();
  const auto after=native_segment_result(primal,operand_values);
  assert(after!=before&&count.compile==1&&count.index==2&&count.projection==2);
  eligible[30]=false;primal=make_workset(projection.ids,eligible,{});
  derivative=make_workset(projection.ids,eligible,sampled);++count.workset;
  materialize_operands();
  assert(primal==std::vector<std::uint64_t>({20})&&derivative==std::vector<std::uint64_t>({20,30}));
  eligible[30]=true;primal=make_workset(projection.ids,eligible,{});
  derivative=make_workset(projection.ids,eligible,sampled);++count.workset;
  materialize_operands();
  assert(primal==std::vector<std::uint64_t>({20,30})&&derivative.size()==2);
  assert(projection_current(projection,snapshot.snapshot_structure_epoch,order));
  ++order;assert(!projection_current(projection,snapshot.snapshot_structure_epoch,order));
  auto reordered=projection.ids;std::reverse(reordered.begin(),reordered.end());
  projection={snapshot.snapshot_structure_epoch,order,reordered};++count.projection;++count.index;
  assert(projection.ids==std::vector<std::uint64_t>({30,20}));

  std::cout<<"cache: value/gate changes reusable; structure/order/toolchain invalidated at declared resume points\n"
           <<"support: logical IDs 10 revoked, 30 promoted, epoch 5 published and checkpoint restored\n"
           <<"local artifact counters: compile="<<count.compile<<" index="<<count.index
           <<" projection="<<count.projection<<" workset="<<count.workset
           <<" value_operand="<<count.operands<<" rebind="<<count.rebind<<" pack="<<count.pack<<"\n"
           <<"limitation: dependency graph and stale-handle check are experiment-local; no physical packing materialized\n";
}
