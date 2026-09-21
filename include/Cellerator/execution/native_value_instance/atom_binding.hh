#pragma once
#include <Cellerator/compute/operation/relation_update.hh>
#include <Cellerator/execution/atom_plane/gradient_plane_v1.hh>
namespace cellerator::compute::relation {
// Explicit association between public persistent semantics and process-local
// atom handles. Callers construct this once from their authoritative mapping.
struct native_atom_association {
    operation_descriptor native_relation{};
    execution::structure_handle atom_structure{};
    execution::axis_identity atom_source{},atom_destination{};
};
// Logical-primary, compact f16 values and f32 gradients. Host-resident immutable
// slot maps are checked; caller explicitly associates the native persistent ID
// with the validated atom structure handle (these identities are never inferred); physical composites require their own supported route.
status publish_atom_values(prepared_relation_pair&,
    const execution::atom_plane::relation_value_atom_plane_v1&,const native_atom_association& association,cudaStream_t) noexcept;
// Caller-owned physical scratch lives through stream completion. The retained
// gradient provider produces physical gradients; the existing structural map
// scatters them to the logical atom plane without admitting padding parameters.
status enqueue_atom_gradient(prepared_relation_pair&,const relation_calculus_descriptor&,
    const device_state_view&,const device_state_view&,operand_version,operand_version,
    const execution::atom_plane::gradient_atom_plane_v1&,const native_atom_association& association,const edge_plane_view& physical_scratch,
    gradient_stamp*,cudaStream_t) noexcept;
}
