#pragma once

#include <Cellerator/compiler/tooling/benchmark_baseline_editor_latency_v1.hh>
#include <Cellerator/compiler/tooling/tooling_snapshots_v1.hh>
#include <Cellerator/compiler/tooling/forward_completion_hover_navigation_and_rename_v1.hh>
#include <Cellerator/compiler/tooling/celleratord_architecture_v1.hh>
#include <Cellerator/compiler/tooling/clangd_worker_lifecycle_v1.hh>
#include <Cellerator/compiler/tooling/compile_command_configuration_v1.hh>
#include <Cellerator/compiler/tooling/document_scheduling_cancellation_v1.hh>
#include <Cellerator/compiler/tooling/host_only_profile_behavior_v1.hh>
#include <Cellerator/compiler/tooling/incremental_source_ast_snapshots_v1.hh>
#include <Cellerator/compiler/tooling/json_rpc_lsp_transport_v1.hh>
#include <Cellerator/compiler/tooling/virtual_shadow_document_mapping_v1.hh>
#include <Cellerator/compiler/tooling/workspace_symbol_indexing_v1.hh>
#include <Cellerator/compiler/tooling/merge_ordinary_c_and_cellerator_diagnostics_v1.hh>

#include <cstdint>

namespace Cellerator::compiler::tooling {
inline constexpr std::uint32_t language_server_contract_version_v1 = 1;
}
