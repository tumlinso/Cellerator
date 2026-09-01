# Privileged compiler-component charter

## Authority cursor

This charter freezes the `CE-JBC-B03` component boundary. At observation time,
the Cellerator Git cursor was
`3eb14d28b6896b09f9f866c26337d185b20e6df1`; its committed CellShard gitlink
was `5f6a502b4355732c4ed3cc873a25b8aec66d8338`, while the independently managed
registered CellShard checkout was at
`6ab8932704ac5988ac64853b3cf43e41e991ee98`. The separate Cellerator Todo cursor
was revision `3608`. These cursors are not normalized: Project Control and the
two Git authorities are independently observed and are not globally atomic.

## Frozen categories

Every repository entry under `components/` is classified as exactly one of the
following.

### Framework adapter

A framework adapter translates foreign tensors, streams, registrations, and
lifetime conventions at an explicit boundary. It may expose non-owning views,
register custom operations, and deliberately convert at that boundary. It may
not own or reimplement Cellerator-native mathematics, biological identities,
geometry, projections, planning, allocation, parameters, execution order, or
runtime sessions.

`components/CelleraTorch/` is a framework adapter.

### Privileged compiler component

A privileged compiler component is allowed to orchestrate a system above
libCellerator. It may own higher-level compilation and runtime decisions whose
scope crosses local Cellerator operations, artifacts, storage locations,
devices, or workers. Privilege is explicit, narrow, and directional: it permits
composition above Cellerator but does not transfer Cellerator's local semantic
or numerical ownership into the component.

`components/CellShard/` is the registered privileged compiler component.

## CellShard authority above Cellerator

CellShard may own:

- global atom, artifact, graph, basis, and composition planning;
- canonical persistence and immutable artifact generations;
- global placement, sharding, transport, residency, and leasing;
- distributed scheduling, collection, recovery, and cross-worker runtime;
- complete global cost decisions that include storage, transfer,
  communication, reuse, and failure policy;
- opaque publication and delivery of Cellerator-defined compiler or execution
  artifacts.

The higher-level planner may choose *where*, *when*, and *as part of which
global program* a Cellerator operation runs. It consumes Cellerator affordance,
decomposition, cost, capability, and profiling contracts. It does not silently
invent the biological meaning or local execution contract of an operation.

## Cellerator authority retained below CellShard

Cellerator retains sole ownership of:

- typed biological domains, axes, orders, relations, exact logical coverage,
  structure epochs, and value generations for its operations;
- semantic geometry and the biological meaning of physical projections;
- reusable relation mathematics, partial-result algebra, local decompositions,
  and numerical correctness contracts;
- local provider capabilities, candidate catalogs, complete local phase costs,
  preparation, native kernels, and prepared execution programs;
- local structure/value lifetimes, execution sessions, launch bindings,
  canonical recovery, and exact contribution ownership;
- the decision whether a local experimental mechanism is promoted, retained as
  evidence, or rejected in a measured regime.

CellShard may select among Cellerator-published alternatives in a global cost
problem. It may not manufacture an unvalidated local projection, bypass exact
coverage, reinterpret a statistical cluster as a causal module, or move a
Cellerator numerical primitive into storage/runtime code.

## Independent libCellerator contract

libCellerator has no required CellShard dependency. This is a source, build,
link, and runtime invariant:

- the default root configuration leaves `CELLERATOR_ENABLE_CELLSHARD` off;
- public native Cellerator contracts do not require CellShard headers;
- native Cellerator operation preparation and execution require no CellShard
  callback, scheduler, storage object, runtime instance, or global planner;
- standalone callers may supply identities, values, capacities, policies,
  artifacts, external bindings, and caller-owned memory directly;
- the registered submodule and optional combined build are integration routes,
  not libCellerator's semantic or link authority.

A privileged component may depend on libCellerator. libCellerator does not
depend on the privileged component merely because the combined system can be
more capable.

## Boundary data flow

The permitted joint direction is:

```text
Cellerator operation semantics, alternatives, and measured local costs
  -> explicit versioned provider/export contracts
  -> CellShard global compiler, persistence, placement, and runtime
  -> opaque artifact or explicit external launch binding
  -> independent libCellerator validation, preparation, and execution
```

Persistent identity and exact coverage cross the boundary. Storage location,
runtime pointer, GPU ordinal, topology route, stream, and residency lease are
operational state and never substitute for portable biological identity.
Immutable structure, mutable values, transient execution state, residency, and
cost freshness retain independent generations and invalidation.

## Cost, allocation, and scale rules

Cold global or local builders may own temporary storage only when allocation,
peak memory, and complexity are declared. Public execution views are non-owning
pointer-plus-count records or pointer-free artifacts. Candidate production uses
bounded top-L structures, streaming, sparse maps, count/scan/fill, radix/sort,
or caller-owned workspaces; unbounded all-pairs and unrestricted subgraph
enumeration are rejected except in an explicitly small exact oracle.

Every cross-boundary decision exposes acquisition, validation, packing,
assembly, transfer, communication, canonicalization, synchronization,
preparation, execution, and expected-reuse cost where applicable. Steady-state
Cellerator execution performs no discovery, catalog parsing, hidden allocation,
global sorting, topology search, or distributed-policy decision.

## Compatibility and change rule

CSG1, CPE2, CPK1, CSPACK, and CSH5 remain frozen at their current wire
versions. A changed semantic contract requires an adjacent version and an
explicit compatibility route. Existing CellShard matrix adapters are
compatibility surfaces until a named JBC gate proves replacement coverage and
records their disposition.

Central registries, umbrella headers, package exports, and root CMake remain
integration-task-only changes. This charter itself changes no runtime or build
behavior, activates no CE-AMP work, and neither reopens nor absorbs CE-GEO,
CE-EXOP, CE-PTR, or CE-AMP.
