# Native modeling composition

`Cellerator::modeling` is an optional interface target composing the native
owners and operations. Reusable data structures live in their canonical
namespaces rather than under a model-specific storage hierarchy. The
[native data guide](../../docs/native_data.qmd) describes these boundaries;
[native model authoring](../../docs/native_nn.qmd) describes ordinary C++ models.

## Models, instances and optional dynamics

`<Cellerator/cell.hh>` supplies the typed `Cell<Model, Instance, ModelHandle>`
binding. A model is an ordinary callable; an instance can own data, retain
shared resources, or select values in a pool. Typed views adapt the instance.
The optional runtime capability bridge is in `<Cellerator/cells/access.hh>`.
No data structure or operation requires Cell inheritance or a lifecycle.

`<Cellerator/modeling/dynamical.hh>` supplies explicit `StepContext`,
`StepResult`, `DynamicalModel` and `DynamicalCell` composition when state
transitions, clocks and retained snapshots are useful. Storage owners remain
responsible for nested resource lifetime, asynchronous completion and value
publication. Calling a Cell or obtaining an edit view does not automatically
publish a generation to prepared execution.

## Data and execution owners

| Capability | Canonical home |
|---|---|
| Persistent identities and logical axes | `structure` |
| Immutable ordered/repeated support and publication | `geometry::dependency::ExplicitPattern`, `PatternOwner` |
| Physical entity slots and permutations | `layout::EntityLayout` |
| Activity and worklist generations | `execution::dependency` |
| Prepared selected incidence | `execution::PreparedIncidence` |
| Changed-source/read indexes and incremental results | `execution::dependency::PreparedReadIndex`, `IncrementalInteractions` |
| Numerical gather/reduce/scatter | `ops::incidence` |
| Specialist numerical values and borrowed views | `tensor` |

The immutable pattern owns one participant-posting index. Read preparation can
retain that owner instead of rebuilding the same semantic index. A compact
ordinal projection for device consumers is a separate explicit physical
preparation product with its own memory and transfer cost.

Incidence arithmetic preserves role/slot order, repeats, explicit extents and
output effects. `joint_reduce` reads entity storage without an expanded gather;
`joint_reduce_scatter` currently uses an edge-sized intermediate. Its name does
not establish fusion or a measured performance advantage.

`ValueField<T>` remains an optional owned field with explicit write publication.
Caller-owned arrays and other providers can be used directly. The optional
Python binding adapts the canonical native types; Python/NumPy conversions and
native CPU operations are separate from the GPU execution contract.

## Validation

Use the focused host `cellerator_native_data` build and `native_data_*` CTest
route in the developer reference. Existing framework/native tests and the
`cell_relation_example.cc` exercise dynamic composition, support publication,
ordered incidence and incremental invalidation. Their source and target
registrations consume the canonical headers; the removed modeling storage
headers and `sparse_relation.cc` are no longer build inputs.

CUDA kernels and the concrete connected two-device example have separate
build targets and require Scheduler admission for execution. See the dated
implementation record for actual evidence and limits.
