#pragma once

// Stable Part One driver contract. Semantic analysis remains independent of
// the selected downstream backend; ordinary C++ takes the exact passthrough.
#include <Cellerator/compiler/driver/define_compilation_database_and_dependency_file_behavior_v1.hh>
#include <Cellerator/compiler/driver/define_temporary_artifact_and_cache_policy_v1.hh>
#include <Cellerator/compiler/driver/define_the_compiler_invocation_and_action_graph_v1.hh>
#include <Cellerator/compiler/driver/deliver_the_driver_passthrough_milestone_v1.hh>
#include <Cellerator/compiler/driver/forward_and_remap_downstream_diagnostics_v1.hh>
#include <Cellerator/compiler/driver/implement_plain_c_passthrough_planning_v1.hh>
#include <Cellerator/compiler/driver/implement_response_file_and_argv_normalization_contracts_v1.hh>
#include <Cellerator/compiler/driver/track_downstream_c_language_and_abi_mode_v1.hh>
