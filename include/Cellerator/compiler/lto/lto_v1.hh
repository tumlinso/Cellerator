#pragma once

#include <Cellerator/compiler/lto/assign_cross_tu_field_and_symbol_identities_v1.hh>
#include <Cellerator/compiler/lto/implement_cellerator_link_driver_mode_v1.hh>
#include <Cellerator/compiler/lto/implement_cross_tu_inlining_and_connected_planning_v1.hh>
#include <Cellerator/compiler/lto/implement_cross_tu_semantic_imports_v1.hh>
#include <Cellerator/compiler/lto/implement_explicit_program_planning_authorization_v1.hh>
#include <Cellerator/compiler/lto/implement_incremental_and_thin_summary_lto_v1.hh>
#include <Cellerator/compiler/lto/implement_mixed_backend_re_emission_v1.hh>
#include <Cellerator/compiler/lto/implement_profile_environment_merge_v1.hh>
#include <Cellerator/compiler/lto/implement_program_level_semantic_planning_ir_v1.hh>
#include <Cellerator/compiler/lto/implement_template_instantiation_deduplication_v1.hh>
#include <Cellerator/compiler/lto/object_ceir_v1.hh>

namespace cellerator::compiler::lto::v1 {
inline constexpr std::uint32_t cellerator_lto_contract_version_v1 = 1;
}
