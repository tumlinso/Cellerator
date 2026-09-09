#pragma once
#include <Cellerator/compute/operation/relation_update.hh>
#include <Cellerator/execution/atom_plane/gradient_plane_v1.hh>
namespace cellerator::compute::relation {
// Logical-primary, compact f16 values and f32 gradients. Host-resident immutable
// slot maps are checked; caller explicitly associates the native persistent ID
// with the validated atom structure handle (these identities are never inferred); physical composites require their own supported route.
status publish_atom_values(prepared_relation_pair&,
    const execution::atom_plane::relation_value_atom_plane_v1&,execution::structure_id expected_structure,cudaStream_t) noexcept;
// Caller-owned physical scratch lives through stream completion. The retained
// gradient provider produces physical gradients; the existing structural map
// scatters them to the logical atom plane without admitting padding parameters.
status enqueue_atom_gradient(prepared_relation_pair&,const relation_calculus_descriptor&,
    const device_state_view&,const device_state_view&,operand_version,operand_version,
    const execution::atom_plane::gradient_atom_plane_v1&,execution::structure_id expected_structure,const edge_plane_view& physical_scratch,
    gradient_stamp*,cudaStream_t) noexcept;
}
