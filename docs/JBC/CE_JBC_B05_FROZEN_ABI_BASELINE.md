# Frozen ABI and compatibility baseline

## Authority cursor and purpose

This is the `CE-JBC-B05` compatibility snapshot at Cellerator Git commit
`db5dcfc56a09498cd5a214c30054bd112da77218`, registered CellShard source
commit `96a691e4a271fabd738ff5819eef6349ac3621a0`, and separately observed
Cellerator Todo revision `3612`. Git and Todo observations are independent and
are not a globally atomic cursor.

The machine-readable primary-header snapshot is
[`planning/jbc/cellerator_frozen_abi_v1.csv`](../../planning/jbc/cellerator_frozen_abi_v1.csv).
Its hashes identify the exact live source read for this baseline; they are not a
promise that every future compatible implementation has identical source text.
This task changes no ABI, wire record, validator, runtime behavior, or build
input.

## Frozen interface families

### Operation core v2

`operation_core_v2` is the current typed operation schema. Version 2 keeps
domain/order identity, relation identity and orientation, numerical policy,
output effects, determinism, structure epoch, value generation, and launch
state explicit. `validate_operation_problem` and its subordinate validators are
the admission boundary. Existing enum values, record layout, and v1 adapter
meaning are frozen.

Adjacent extension: register a new versioned record or operation-core v3 when
new semantics cannot be represented. Do not smuggle new meaning into reserved
values or maintain a second execution engine.

### Relation algebra v2

Relation algebra v2 binds typed relations, logical and physical value
components, segments, gates, gradients, normalization, and composition to the
operation core. `validate_relation_algebra_problem`,
`validate_relation_value_binding`, and the composition validators fail closed.

Adjacent extension: add explicitly versioned operation or composition records.
Do not renumber v2 enums, infer biological equivalence from shape, or move
model/loss/optimizer policy into the algebra.

### CSG1

CSG1 is the pointer-free portable semantic-geometry image. Schema v1 uses a
320-byte header and carries exact logical cover, biological identities,
portable order/work layout, recovery, checksums, and optional evidence. It
contains no GPU ordinal, topology route, stream, runtime pointer, provider tile,
or mutable generation. `validate_semantic_geometry_image_v1` is independent of
the producer.

Adjacent extension: use an optional section when the directory contract can
represent the information. Use a new top-level version only when a source audit
proves that extension sections cannot preserve the required semantics.

### CPE2

CPE2 is the architecture-specific pointer-free execution image. Schema v2
retains its 256-byte header, 64-byte section entries, and 64-byte projection
entries. It embeds exact CSG1 bytes and separates physical projection meaning
from mutable values. `validate_execution_image_v2_host` and projection-specific
validators remain mandatory.

Adjacent extension: add typed optional sections or adjacent projection records,
including already-supported prebound projection evolution. CPE3 requires an
explicit insufficiency audit; larger payloads alone use chunk manifests rather
than forcing a new top-level format.

### CPK1

CPK1/persistent packing payload schema v1 is compatibility-only. Its combined
CP-BP geometry/value ordering, exact bytes, feature recovery, row order, tiles,
and validator remain unchanged. It is useful evidence and a supported direct
execution route, not the universal compiler IR.

Adjacent extension: adapt validated CPK1 without thawing or reinterpreting it
as a CPE2 compatibility projection, or create a separately versioned format.
Never mutate CPK1 bytes or turn width-16 blocks into universal semantic groups.

### Projection value plane v1

Projection-value-plane v1 preserves logical-primary values, explicit physical
components and maps, mutable generation publication, composite ownership, and
validation independently from immutable geometry. The public views are
non-owning and trivially copyable.

Adjacent extension: register a measured value-pack candidate or add a sibling
v2 contract. A new value pointer or generation must not rebuild structure or
prepared geometry.

### Candidate catalog v3

Candidate-catalog v3 freezes candidate/provider/device/projection/capability/
operation identity, width and numerical bounds, production versus experimental
classification, resource description, and stage inventory.
`validate_candidate_catalog_v3`, inventory validation, and catalog/program
cross-validation remain the cold admission path.

Adjacent extension: register a descriptor representable by v3. Change the
record only through catalog v4. No candidate self-promotes; complete-cost
planner evidence remains authoritative.

### Program v2 and training-program v2

Program v2 freezes prepared stages, dependency indexing, caller launch
bindings, stream, and workspace separation. Training-program v2 freezes its
native stage graph, value-mode and readiness contracts, graph-capture/rebind
checks, and caller-owned optimizer/model policy boundary. Their validators
reject malformed stage graphs and insufficient bindings before execution.

Adjacent extension: compose additional representable stages. A record or
lifetime change requires program v3 or training-program v3; changing pointers,
streams, or mutable generations does not.

### Geometry acquisition v2

Acquisition v2 freezes request, requirements, acquired-geometry, external
payload, projection-set, and assembly validation for CSG1, CPE2, and CPK1
routes. It reports acquisition observations to the existing planner and does
not become a second cost authority.

Adjacent extension: add a versioned external-payload or projection adapter. A
schema-layout change requires acquisition v3. Allocation, validation, transfer,
assembly, fallback, and canonicalization remain explicit costs.

### Hierarchical index space v1

Hierarchical-index v1 freezes global 64-bit extent, stable component identity,
aggregate offset, explicit local width, local-to-global recovery, optional
global identity sidecar, and caller-owned component arrays. Compile-time
trivial-copy/standard-layout assertions protect the records;
`make_component_grid_v1` diagnoses invalid bounds for prepared component
dispatch.

There is no single public aggregate host validator in the v1 header. The only
permitted extension is a separate validator over the frozen v1 view or an
adjacent v2 record family. Callers must not infer identity from extent, truncate
aggregate counts into one component, or hide component allocation.

## Cross-interface compatibility law

- Canonical biological identity and exact coverage are explicit; matching
  shapes or ordinals never establish equivalence.
- Proposal overlap, physical representation overlap, and exact execution
  contribution remain distinct. Every logical contribution has one exact owner
  unless a versioned partial-result algebra proves reconstruction.
- Immutable structure, mutable values, transient launch state, residency, and
  preference/cost evidence have separate generations and invalidation.
- Cold builders declare allocation, capacity, complexity, and peak memory.
  Execution views remain pointer-first/non-owning or pointer-free.
- Steady state performs no discovery, catalog parsing, global sorting, hidden
  allocation, topology search, structure hashing, or implicit synchronization.
- CSG1, CPE2, CPK1, CSPACK, and CSH5 wire semantics change only through an
  adjacent version and explicit compatibility route.
- Central registries, umbrella headers, package exports, and root CMake are
  integration-task-only surfaces.

The frozen interfaces are therefore **preserve** except CPK1, which is
**compatibility-only**. New capability enters only at the adjacent extension
points named above; no baseline item authorizes mutation or retirement.
