# Native data migration: reviewed scope

Historical planning/review record. See [the realized implementation](IMPLEMENTATION.md) and [actual type map](implemented-type-map.csv) for the realized cutover and its evidence.

Status: planning revision, 2026-11-01. No library implementation or cleanup has
been performed. This scope incorporates the user's review of the original
proposal and the current uncommitted Cell redesign. Read it with
[DESIGN.md](DESIGN.md), [MIGRATION.md](MIGRATION.md), and the
[type map](type-map.csv). The protected
[cellular geometry definition](../../cellular_geometry.md) remains authoritative
and unchanged.

## 1. Decisions established with the user

- Prioritize a clean native API. Existing C++ names, include paths, class
  layouts and exported symbols are not compatibility requirements. Update
  in-tree consumers together; do not make sibling-project repair or hypothetical
  downstream consumers a gate for this change.
- Include multi-device forward and backward execution in the first completed
  milestone. A local-only container migration is an intermediate checkpoint.
- Allow explicit persisted-format migration. Preserve declared logical meaning
  and conversion evidence, rather than freezing old C++ records or maintaining
  every old writer. A format change must have an explicit version/conversion
  decision; silently reinterpreting old bytes is not acceptable.
- Keep the namespace responsibilities in DESIGN. Reuse conventional tensors,
  sparse algorithms and the existing execution machinery. Own specialist data
  representations and the mechanisms needed to consume them directly.
- Review and plan now. Source implementation, actual deletion/moves, format
  conversion and GPU execution remain subsequent work. No permission to modify
  `cellular_geometry.md` is implied.

The first multi-device target is two CUDA devices in one host, with explicit
placement and communication. Multi-host transport, cluster scheduling and a
general expert-residency system are separate work. Hardware topology is a
qualification input, not part of biological identity.

## 2. Review of the current Cell and model contracts

The inspected redesign fits this data architecture. The typed
[`Cell<Model, Instance, ModelHandle>`](../../include/Cellerator/cell.hh) binds a
borrowed or shared model handle to a decayed instance value. `model()` returns
the bound ordinary C++ object; `forward()` invokes it with the continuing
instance. There is no required generic `Model` base or representation. The
optional [`CellViews`](../../include/Cellerator/cells/access.hh) bridge supplies
runtime capabilities separately. Dynamics remains opt-in through
[`DynamicalModel` / `DynamicalCell`](../../include/Cellerator/modeling/dynamical.hh).

Independent review found no confirmed defect in this inspected contract. Five
focused host checks passed; [VALIDATION.md](VALIDATION.md) records their scope.
This review does not qualify distributed execution or the entire NN package.

Data integration must respect three specific boundaries:

1. A tensor, dependency foundation, operation or prepared binding remains usable
   without Cell. An application instance may expose a tensor view or retain a
   slot/selection into pooled storage. Tensors do not inherit Cell interfaces.
   Batch execution resolves those selections in bulk; it need not call
   `Cell::forward()` or perform runtime capability discovery for each item.
2. `instance()`, writable views and `forward()` can mutate data. Cell supplies no
   automatic generation tracker. Storage/parameter/context owners must publish
   revisions and readiness consumed by prepared execution. If freshness cannot
   be established, recompute through a valid path or reject the cached path.
3. Instance views do not universally contain a domain, semantic order, epoch or
   occurrence identity. Source preparation obtains and checks required metadata
   separately. `CellIdentityView` borrows a reference: retain the real owner or
   copy the persistent ID into preparation, rather than retaining a view into a
   relocated instance. An expression row number is not semantic identity.

The new data types therefore require no further Cell/model redesign. The
Cell-specific integration consists of ordinary instance adapters and tests.
Current dirty Cell/model files belong to the other workstream and must be
integrated deliberately, not overwritten by this migration.

## 3. First complete implementation milestone

Deliver one clean native data surface and one connected multi-device example.
The example uses overlapping ordered groups, including a 32-feature case,
small condition inputs, shared parameters/context, and overlapping output
contributions. A group is a joint mechanism, not a compulsory list of pairs.

The bounded numerical operation starts with a native joint mechanism supported
by an independent reference, plus a model-defined gate with a certified flat
inactive region. It must exercise first-order input, condition and parameter
VJPs; arbitrary higher-order autodiff is not part of this milestone. The
foundation remains bindable to a second compatible operation family so that
it does not encode one evaluator's arithmetic.

Deliverables:

- Canonical semantic IDs/axes, explicit and implicit dependency patterns,
  applicability/knowledge scope, condition reads and checked selections.
- Reusable segmented/posting storage, physical maps and explicit-device
  allocation owners, extracted from current implementations rather than rebuilt.
- Concrete specialist format views and bulk ownership, beginning with the
  block-masked grammar; migrate the existing low-bit and state-code surfaces
  without pretending they share one encoding. Conventional dense/CSR/COO use
  ATen or caller-owned/library representations.
- Access/effect and exact omission contracts, condition/participant read indexes,
  generation-aware prepared bindings, and direct specialist-view consumption.
- Two-device preparation, forward communication, result accumulation and backward
  communication for one logical computation, with explicit owner/readiness rules.
- Canonical headers, implementation ownership, installed targets, examples,
  bindings and affected tests updated together; obsolete paths removed.
- Explicit migration of any persisted representation actually changed by the
  cutover. Untouched formats need no gratuitous rewrite.

Default example precision is FP16 value storage with FP32 accumulation and
gradient/output buffers where the operation requires them. Declare multiply
precision independently, retain existing explicit caller precision, and check
against the native/conventional computation and an independent reference.

## 4. Required multi-device behavior

Extend the existing source/atom/program contracts with one canonical checked
extent association containing actual placement/device, logical association,
structure/order/placement revisions, value generation, and readiness/lease
information. The current `physical_extent_binding_v1` describes residency but
has no device ordinal. Reusing that descriptor alone is insufficient. Its
internal C++ API may change; a `v1` suffix is not a compatibility exemption.

Use per-device bulk owners and local prepared bindings; reuse
[`fleet_context`](../../include/Cellerator/runtime/multi_gpu/fleet.cuh), existing
streams, [`copy_async`](../../include/Cellerator/memory/copy.cuh), reductions and
readiness where their actual contracts suffice. Add the missing associations
and dependency coordination rather than a second allocator, global graph or
population scheduler. Sharing a logical foundation across devices does not
make device pointers mutually accessible. Device index arrays require valid
replicas/bindings on each consuming device.

The example must contain a real cross-device dependency, not two disconnected
local models. Include a remotely supplied condition input, overlapping
contributions/participants requiring gradient return or reduction, and a shared
parameter used on both devices with one canonical update/publication. Transfer
and staging buffers remain owned through all consumers and backward. Prepared
transfer requirements are distinct from per-invocation completion and values.
The same distributed operation must use the prepared dependency/condition/
omission bindings and consume the specialist view directly in its local kernels
on both devices. Explicitly packed transfer messages are allowed; separate
local-specialist and distributed-dense demonstrations do not establish this
integration, nor does a whole-state decode before the actual distributed work.

Always exercise an explicit staged transfer path. Exercise a direct device
path when the selected pair supports it, and force staging when that direct
path is deliberately disabled. An unsupported direct path is recorded as
unqualified; it neither fails an otherwise supported topology nor counts as a
passed path. If current copy/fence code cannot provide required staged or
supported direct semantics, close that implementation gap explicitly.

Compare distributed placements against the same single-device mathematical
reference, including output/gradient association, tolerance and failure behavior.
Require an active case with nonzero gradients through the remote condition,
shared participant and shared parameter, plus an inactive case checking scoped
omissions and an independently nonzero baseline output. A fixture entirely in
the flat region cannot establish conditioned backward correctness.
Perform one deterministic parameter update using the reduced gradient; compare
both that gradient and resulting parameter values with the reference. Publish
and refresh replica generations, then compare the next forward. A no-op,
duplicate update or stale parameter replica must fail these checks.

Acceptance requires a run on two actual admitted devices. Host tests or
simulated extents can validate records but cannot close that requirement.
Resource availability is not assumed by this planning document.

## 5. Concrete cleanup and extraction scope

The cleanup follows the data plane and its direct consumers. Compiler-wide
renaming, unrelated operation families, historical evidence trees, generated
indexes, dependencies and sibling repositories are outside the cut.

| Current source | Canonical destination/responsibility | Cleanup |
|---|---|---|
| `include/Cellerator/modeling/entity_layout.hh` | `layout/entity_layout.hh`, `layout::EntityLayout` | Move the definition and in-tree users directly; remove the old include. |
| `modeling/sparse_relation.{hh,cc}` | `geometry/dependency/{explicit_pattern,pattern_owner}` and `execution/dependency/workset` | Split immutable pattern/index construction, snapshot publication and mutable activity. Preserve ordered/repeated incidence and epoch meaning. |
| `modeling/incremental.hh` | `geometry/dependency/reads.hh`, `execution/dependency/{read_index,incremental}.hh` | Reuse one prepared participant-posting index instead of rebuilding it from the same relation. Separate additional condition/read indexes. |
| `modeling/incidence_plan.hh` | `containers/segmented.hh`, `execution/dependency/incidence.hh`, `ops/incidence.hh` | Separate offsets/roles, prepared addressing and numerical gather/reduce/scatter. Keep templates header-defined where appropriate. |
| `modeling/cuda/runtime.{hh,cu}` | `memory/cuda/{device_buffer,pinned_buffer,stream,event,error}` | Extract explicit-device resources from model kernels. Consolidate lifetime policy with existing session resources; do not create a second allocation implementation. |
| `runtime/device_buffer.cuh` | Explicit borrowing/shared-retention adapters over canonical memory owners | Retire the second public owner contract after converting internal users; preserve useful shared ownership without implicit device selection. |
| `modeling/cuda/plans.cu` | Dispatch preparation plus `tensor/state_codes/owner.cu` | Move `QuantizedState` allocation/code/override ownership out of the dispatch-plan TU; remove its old model facade once callers move. |
| `compute/operation/indexed_mechanism/training.cu` | `ops/mechanism/{kernels.cu,parameter_owner.cu}` and `execution/dependency/{mechanism_plan.cc,mechanism_run.cu}` | Separate numerical kernels, parameter lifecycle, host preparation and launch/tape/VJP orchestration. Use a private shared header only for real common state. |
| `compute/support`, `compute/indexing`, `compute/gating` public headers | `geometry/dependency`, `containers`, `ops/condition`, `execution/dependency` according to responsibility | Consolidate with existing implementations and targets; remove forwarding-only duplicates after the coordinated cutover. |
| CellPack records/projections and low-bit/codebook declarations | `tensor` for format contracts; `layout/projection` for maps; explicit `interop/formats` codecs where retained | Unify ownership of a representation, not the bytes of different encodings. Keep pack/repack algorithms with their actual operation owners. |
| `modeling/cell.hh`, `modeling/model.hh`, duplicate forwarders in affected data families | Canonical `cell.hh`, `modeling/dynamical.hh`, and the new public headers | Remove obsolete include-only compatibility paths after updating in-tree callers; do not alter Cell behavior. |

`SparseRelation` currently stores reverse postings and `DependencyIncidence`
rebuilds an index from the same relation. Reuse applies to that participant
index; genuinely different condition/read indexes remain separate.
`DynamicRelation` also reconstructs relation/posting state at publication. Give
one snapshot owner clear publication and optional prepared-index ownership.

Do not merge non-equivalent formats: row records and warp tiles are successive
layouts; physical CP-BP geometry is not dependency knowledge; affine low-bit
sparse values and per-feature state codebooks are different encodings. Compare
legacy and templated BELL packing/metadata before deciding conversion/removal.

`joint_reduce_scatter` currently constructs an edge-sized intermediate through
`joint_reduce` followed by `scatter_add`. The move preserves its numerical
meaning and documents that cost. A genuinely fused implementation is a
separate candidate with its own correctness and complete-cost comparison.

Use short responsibility-based filenames, canonical includes and existing
`.hh`/`.cc`/`.cu` conventions. A namespace does not require a new compiled
library. Add targets only for independently useful compiled ownership; keep
host semantic headers independent of Torch/CUDA and modules optional.

## 6. Completion and remaining implementation choices

[MIGRATION.md](MIGRATION.md) supplies ordered checkpoints and acceptance checks.
All checkpoints, including multi-device execution, belong to this first
milestone; later formats and a general distributed runtime do not.

The user's milestone and format-policy questions are resolved above. Before
execution, choose the exact existing mechanism/reference, conversion formats
actually touched, device pair/topology, complete-cost workloads and tolerance
from inspected source and admitted resources. These are bounded implementation
choices, not permission to reopen the Cell contract or broaden the cleanup.

No GPU speedup, integrated distributed capability or source migration is
claimed by this plan. [VALIDATION.md](VALIDATION.md) distinguishes the contract
review and document checks from future implementation acceptance.
