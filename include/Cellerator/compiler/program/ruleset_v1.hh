#pragma once

#include <Cellerator/compiler/composition/define_the_concrete_cellshard_materialization_request_se_v1.hh>
#include <Cellerator/compiler/composition/deliver_the_profile_to_portable_ruleset_slice_v1.hh>
#include <Cellerator/compiler/composition/import_cross_operation_rewrite_and_fusion_search_v1.hh>
#include <Cellerator/compiler/composition/import_global_operation_graph_ir_v1.hh>
#include <Cellerator/compiler/composition/import_portable_schedule_ruleset_representation_v1.hh>

#include <cstdint>

namespace Cellerator::compiler::program {

inline constexpr std::uint32_t ruleset_contract_version_v1 = 1;

using composition::cellshard_materialization_request_v1;
using composition::compile_profile_to_ruleset_v1;
using composition::connected_rewrite_v1;
using composition::make_cellshard_materialization_request_v1;
using composition::planning_operation_graph_v1;
using composition::portable_schedule_identity_v1;
using composition::portable_schedule_v1;
using composition::profile_ruleset_metrics_v1;
using composition::profile_ruleset_request_v1;
using composition::profile_ruleset_result_v1;
using composition::replay_mode_v1;
using composition::select_connected_rewrites_v1;
using composition::validate_planning_operation_graph_v1;
using composition::validate_portable_schedule_v1;

} // namespace Cellerator::compiler::program
