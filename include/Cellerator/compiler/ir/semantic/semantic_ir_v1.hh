#pragma once

#include <Cellerator/compiler/ir/semantic/deliver_source_to_semantic_ir_vertical_slice_v1.hh>
#include <Cellerator/compiler/ir/semantic/freeze_semantic_ir_module_and_symbol_scopes_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_bundle_chain_moments_hierarchy_and_exchange_op_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_contraction_segment_and_normalization_operatio_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_control_flow_and_loop_semantics_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_domain_and_axis_ir_types_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_edge_map_gate_support_mask_and_sparse_update_o_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_execution_field_operations_and_regions_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_generation_and_epoch_transition_operations_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_gradient_and_publication_operations_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_native_and_opaque_c_call_operations_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_relation_apply_and_transpose_operations_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_relation_ir_types_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_semantic_canonicalization_and_equivalence_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_semantic_ir_inlining_and_composition_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_state_and_value_plane_ir_types_v1.hh>

namespace Cellerator::compiler::ir::semantic {

inline constexpr std::uint32_t semantic_ir_schema_version_v1 = 1;

}  // namespace Cellerator::compiler::ir::semantic
