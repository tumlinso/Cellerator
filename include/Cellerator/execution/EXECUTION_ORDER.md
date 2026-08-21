# Execution order and lifetime contract v1

This contract separates immutable relation structure, mutable numerical values,
prepared binding requirements, and per-launch state. It extends biological ABI
v1 without changing its frozen headers or the CPK1 v1 payload.

## Lifetime state machine

1. A `relation_structure` fixes source and destination axes, structure handle,
   structure epoch, projection catalog, and logical edge count.
2. One or more `value_plane` records bind values, precision, quantization,
   value layout, and generation to that structure epoch.
3. A `prepared_binding_contract` fixes a bounded set of required structure
   handles and epochs, operand axes, output-order and output-effect behavior,
   and transient-workspace requirement. It owns no launch pointers or stream.
4. `launch_bindings` supplies current operands, value bindings, scalars,
   stream, and transient workspace. Each value binding states the generation
   expected by that launch.

A value-generation change does not invalidate structural preparation. A
structure-epoch change invalidates value planes, value-position maps, and
prepared contracts. Pointer equality never substitutes for either generation.
Launch validation matches every value plane to its own bound relation; an
unknown relation, missing or duplicate structure, stale epoch, or stale value
generation fails before dispatch.

## Output effect contract

Each output declares whether it overwrites, accumulates, affine-accumulates, or
partially writes its destination, whether initialized destination contents are
required, whether detectable input/output aliasing is legal, and any affine
scalar bindings. This is mutation metadata, not a generic epilogue engine.

## Order contract

Every output declares the affected axis, its input and output identities, and
whether order is preserved, transformed, or explicitly canonicalized. Preserve
contracts require exact axis equality and no transform. Transform and
canonicalize contracts require a generation-checked transform handle. A
producer and consumer may remain packed when the producer output axis exactly
matches the consumer input axis.

`order_transform_view` keeps both directions of entity permutation.
`value_position_map_view` independently maps stable logical edges to projection
value positions for forward or transpose execution. These maps are reusable,
cacheable prepared state; they are not inferred from pointers during launch.

## CP-BP and persistence compatibility

CP-BP v1 feature and inverse permutations, local row order, and canonical
recovery maps remain authoritative compatibility inputs. CE-ARCH-20 will expose
them through read-only adapters. Existing canonical-output wrappers remain
valid. Native internal paths may preserve execution order and canonicalize only
through an explicit transform contract.

CPK1 v1 bytes and meaning are unchanged. It remains a combined v1 artifact.
Structure/value separation in Execution Image v2 does not reinterpret or
rewrite CPK1.

## Cost and memory accounting

Order planning records elements moved, bytes read and written, persistent map
bytes, and transient workspace bytes. Value-plane accounting records payload
bytes separately from immutable structure and projection bytes. A planner must
charge canonicalization, transpose remapping, dynamic packing, and workspace in
the relevant CE-ARCH-30 phases. No transform is justified merely because it is
available.

The contract adds no allocation, hashing, descriptor discovery,
synchronization, or device selection to a run path. Runtime ownership and
stream-ordered workspace implementation belong to CE-ARCH-12.
