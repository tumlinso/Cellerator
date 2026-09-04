#pragma once

#include <Cellerator/compiler/tooling/benchmark_baseline_editor_latency_v1.hh>
#include <Cellerator/compiler/tooling/expose_reusable_tooling_snapshot_apis_v1.hh>
#include <Cellerator/compiler/tooling/forward_completion_hover_navigation_and_rename_v1.hh>
#include <Cellerator/compiler/tooling/freeze_the_celleratord_architecture_v1.hh>
#include <Cellerator/compiler/tooling/implement_clangd_worker_discovery_and_lifecycle_v1.hh>
#include <Cellerator/compiler/tooling/implement_compile_command_and_project_configuration_v1.hh>
#include <Cellerator/compiler/tooling/implement_document_scheduling_and_cancellation_v1.hh>
#include <Cellerator/compiler/tooling/implement_host_only_no_profile_editing_behavior_v1.hh>
#include <Cellerator/compiler/tooling/implement_incremental_source_and_ast_snapshots_v1.hh>
#include <Cellerator/compiler/tooling/implement_json_rpc_and_lsp_transport_v1.hh>
#include <Cellerator/compiler/tooling/implement_virtual_shadow_document_mapping_v1.hh>
#include <Cellerator/compiler/tooling/implement_workspace_symbol_and_indexing_foundations_v1.hh>
#include <Cellerator/compiler/tooling/merge_ordinary_c_and_cellerator_diagnostics_v1.hh>

#include <cstdint>

namespace Cellerator::compiler::tooling {
inline constexpr std::uint32_t language_server_contract_version_v1 = 1;
}
