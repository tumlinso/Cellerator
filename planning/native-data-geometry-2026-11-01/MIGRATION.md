# Native data structures: migration and implementation plan

Historical planning/review record. See [the realized implementation](IMPLEMENTATION.md) and [actual type map](implemented-type-map.csv) for the realized cutover and its evidence.

Status: proposal only, revised 2026-11-01. No library migration is executed by
this document. Read [DESIGN.md](DESIGN.md), the reviewed [scope](SCOPE.md), the
[type map](type-map.csv), and the protected
[cellular geometry definition](../../cellular_geometry.md) together.

Inspected base: `b6f1a812c369c9d60fa45eb3c3df1c12f2d5651e`, plus the current
uncommitted typed Cell redesign. The inventory is bounded to native data
families and their direct execution/operation consumers. Refresh source and
target ownership before implementation. The contract checks in VALIDATION do
not qualify the proposed migration.

## 1. Delivery boundary

The first complete milestone includes the clean native data surface, a
specialist representation consumed directly by prepared computation, and a
connected two-device forward/backward example. Containers-only and local-only
checkpoints do not close this milestone. Two CUDA devices on one host are the
initial distribution scope; multi-host scheduling is separate.

The user permits C++ API breaks and explicit file-format migration. Do not
spend a phase introducing old-name aliases or preserving exported manglings.
Move definitions and update in-tree consumers together, then remove redundant
paths. Sibling-project repair is not a gate. Meaning, ownership, exact omission,
generation publication and numerical correctness remain requirements.

This turn reviews and scopes the work. Implementation uses a separately scoped
authorized task through the repaired Project Control front door. No workflow
authority, generated projection or SQLite state is edited by this plan.

The checkpoints below express technical dependencies. A coherent source
cutover may combine them; they are not mandatory management stages. M0 through
M7, including M6, belong to the first milestone.

## 2. Cutover and format rules

1. Source compatibility is free to change. No permanent forwarding header,
   duplicate implementation or legacy facade is required for external users.
   In-tree tests, examples, native modules, build/install exports and optional
   bindings move with the declarations they consume.
2. Distinguish native structs from serialized fields. A version suffix does
   not freeze an internal C++ type. When persisted bytes change, record the
   old/new versions, logical identity/order association, numeric interpretation,
   conversion requirements and unsupported cases. Never reinterpret bytes
   under their old version name.
3. Prefer one current writer and bounded explicit import/conversion where
   needed. Keep old readers only when they serve a demonstrated migration or
   useful input. Untouched formats do not need gratuitous version changes.
4. Consolidate ordinary dense/CSR/COO uses onto ATen or caller/library storage
   and descriptors. Specialist encodings retain their actual semantics. Similar
   names do not establish equivalent BELL, row-record, tile or codebook bytes.
5. Preserve canonical identity, domain/order association, repeated occurrences,
   ordered incidence, masks, output effects, gradient mapping and indirect
   effects. Changing native names does not authorize numerical behavior drift.
6. Preserve the other agent's Cell/model work and all unknown dirty edits.
   Reconcile overlapping ownership before a source cutover. No phase grants
   permission to change `cellular_geometry.md`.
7. Preserve historical evidence and supplied data. Regenerate source maps using
   their maintained tool after actual moves; never edit generated maps by hand.

## 3. Canonical public and build ownership

| Family | Proposed canonical headers | Build responsibility |
|---|---|---|
| Semantic identity/axes | `structure/{identity,coordinates}.hh` | Move existing persistent-ID implementation here; runtime handles remain execution-owned. |
| Dependency foundation | `geometry/dependency/{applicability,selection,reads,patterns,binding}.hh` | Host semantic declarations/validators, independent of CUDA/Torch. |
| Generic storage/indexes | `containers/{segmented,index_set,bitset,posting_index}.hh` | Extract only primitives with real migrated users; reuse current posting implementation. |
| Specialist formats | `tensor/block_masked/{layout,view,owner}.hh`, `tensor/quantized/{csr,blocked_ell}.hh`, `tensor/state_codes/{view,owner}.hh` | Concrete views/owners; numerical kernels remain with their actual operation. |
| Physical maps | `layout/{entity_layout,permutation}.hh`, `layout/projection/{feature_major,transpose,mma_hybrid}.hh` | Reuse/reorganize existing layout/projection sources. |
| Native resources/interop | `memory/cuda/{device_buffer,pinned_buffer,stream,event}.hh`, `interop/aten/`, `interop/formats/` | Canonical explicit-device owners; optional external retention and versioned codecs. |
| Mechanism computation | `ops/mechanism/{incidence,access,omission,training}.hh`, `ops/incidence.hh` | Reuse indexed-mechanism evaluators/VJPs; separate arithmetic from prepared state. |
| Prepared/live execution | `execution/dependency/{binding,incidence,condition_index,workset,distributed}.hh` | Source associations, indexes, placed extents and transfer requirements over existing planes/programs/runtime. |

A namespace does not require a library target. Use existing compiled owners
where cohesive; add a lightweight header target or separately linkable native
memory owner only for actual independent consumption. Split the existing CUDA
framework target's resources from model kernels so borrowing a buffer does not
link unrelated attention/reaction code. Host modules wrap canonical headers;
NVCC and ordinary consumers do not require module tooling.

## 4. M0 — Fix the source cut and contract inventory

Capture base/dirty work, protected hash, canonical source destinations, actual
in-tree users and current build/export ownership. SCOPE section 5 identifies
the concrete cleanup; the type map supplies symbol-level destinations. Do not
launch an exhaustive sibling-consumer preservation project.

Resolve actual overlapping edits with the Cell workstream. Record any affected
persisted formats and decide retain, convert or retire per representation.
Identify the two-device resources needed for qualification without starting a
GPU job during planning.

Exit: attributable source ownership, explicit schema decisions and bounded
checks are established. No legacy ABI or old include is assumed untouchable.

## 5. M1 — Canonical data ownership and cleanup

Move semantic IDs, entity/occurrence maps, explicit patterns, generic posting
storage, native allocation resources and specialist views to their canonical
homes. Update internal includes/definitions/callers directly. Remove old
forwarders and redundant model-specific facades in the affected cut.

Split mixed translation units along SCOPE's named mechanisms: pattern versus
publication/worklist; parameter lifecycle versus kernels/preparation/tapes;
state-code ownership versus dispatch plans. Share one participant-posting index
where relation and incremental execution currently rebuild equivalent data.
Retain genuinely different condition/source indexes separately.

Checks: host-only/header consumers, owner move/borrow/retention rules, explicit
extents/counts, mask/offset mapping, installed targets and dependency closure.
Test the changed public surface, rather than retaining tests of obsolete names.
No Torch/CUDA dependency may leak into host semantic headers.

## 6. M2 — Dependency foundation and implicit patterns

Add applicability, knowledge/coverage scope, typed selections/sources,
domain-qualified participants and explicit occurrence distinctions. Provide
explicit incidence plus at least one genuinely implicit joint-group pattern.
Adapt exact execution coverage only where its actual requirements apply.

Checks: partial knowledge cannot remove admitted work; overlapping groups,
repeated participants, equal endpoints with distinct mechanism IDs, empty
support and input-free operations retain explicit meanings. Implicit
preparation must not first enumerate all pairs. Bind the same foundation to
two compatible operation families and reject incompatible reads/coordinates.

## 7. M3 — Access, freshness and exact omission

Add operation access/effects, condition-source declarations and source-to-
condition/mechanism indexes. Keep model-defined conditions as ordinary
callables; reuse supplied-predicate condition programs where suitable.
Omission rules bind operation/access revision, outcome, scope, numeric semantics
and certified derivative kinds/order. Approximate sampling stays explicit.

Integrate with typed Cell instances through ordinary views/selections. Obtain
domain/order/epochs/revisions from their providers. Cover mutation through
`instance()`, `edit()` and a model forward; none automatically invalidates a
prepared cache. Pin/copy identity metadata safely across instance relocation.

Checks: wrong same-size domain/order, condition-only updates, inferred-context
cost/invalidation, parameter changes, temporal expiration without writes,
pending dirty inactive work, `u*v` at `u=0`, a certified flat gate, whole-output
base terms, nonfinite semantics, and unsupported higher derivative requests.
Preserve finite-time indirect influence through composed local mechanisms.

## 8. M4 — Direct specialist execution and local reference

Connect the prepared source association to existing atom requirements/planes,
fragments and program v2. Keep current values, stream and workspace at launch.
Run the chosen joint operation directly over specialist views with bulk native
ownership, not immediate decoding into another full representation.

Checks: two physical layouts for the same logical operation, canonical recovery,
packed producer/consumer composition, overlapping accumulation, shared immutable
structure/private values, native versus optional ATen ownership, tape leases,
stale values and failure-before-publication. Declare storage/multiply/
accumulation/output precision separately. Preserve a local reference for M6.

## 9. M5 — Persistence, install and source closure

Implement explicit conversions for representations actually changed. Validate
both byte/schema boundaries and decoded identity/order/value meaning. Retire
redundant owners/codecs/old writers after their in-tree use and migration
purpose is resolved. Conversion is boundary work, not per-launch parsing.

Update CMake/install exports, optional module exports, examples, binding source
and developer routing for the new API. Do not expand the Python surface as part
of this native design. Regenerate maintained source maps once the actual cut
is coherent. The first milestone retains no accidental old include tree solely
for compatibility.

## 10. M6 — Connected multi-device execution, required

Provide a canonical placed-extent binding with device/order/structure/value/
placement association and readiness/leases. Build per-device prepared access
and explicit forward/backward transfer requirements over existing runtime
resources. Retain staging, replicas, parameter versions and tapes through their
last consumer. SCOPE section 4 specifies the example and transfer paths.
This is the same specialist-view operation and dependency/condition/omission
binding exercised locally in M4, now consumed directly on both devices. Two
independent demonstrations or a full dense decode do not meet this integration.

Checks on two actual admitted devices: cross-device condition input, joint
participant access, overlapping output/gradient reduction, shared parameter
gradient plus one update/publication, staged transfer and capability-supported
direct copy, delayed transfer, wrong stream/device, stale replica, placement
change and in-flight backward. Record unsupported direct paths as unqualified.
Require active nonzero remote-condition/participant/parameter gradients and an
inactive case preserving baseline output. Compare outputs and first-order VJPs
with the same single-device reference. Compare one deterministic update's
reduced gradients and resulting parameter values, refresh replica generations,
then compare the next forward. A scalar record test, simulated device or two
independent models does not close this checkpoint.

## 11. M7 — Qualification and acceptance

Run the affected host, native/module/install, optional-adapter and CUDA tests.
For numerical/device changes use independent output/derivative oracles,
appropriate sanitizers and the established GPU admission controls. Complete
multi-device acceptance is required; report unavailable resources honestly.

Compare the strongest compatible conventional/library baseline and account for
preparation, context inference, conditions, packing, maintenance, allocation,
communication, fences, execution and observable output conversion over a stated
reuse interval. Use KDA's task-workspace workflow and experiment engine as
relevant; registered benchmark evidence goes through Autobench. A speedup is
not assumed or a substitute for usable, correct containers. Keep alternatives
for regimes where the specialist realization loses.

## 12. Existing checks to adapt

| Family | Current source pointers |
|---|---|
| Typed Cell integration | `tests/cell/{cell_core,cell_views_bridge,cell_adapters,typed_core_no_rtti}_test.cc`; framework/native and NN Cell tests |
| Identity and model bindings | `tests/execution_identity/identity_registry_test.cc`; native NN biological binding/installed consumers |
| Support/indexing/gating | `tests/sparse_support/{indexing_support_gating_test.cc,cuda_worksets_test.cu,derivative_sampling_test.cc}` |
| Layout/incidence/incremental | `tests/framework/native/{sparse_modeling,entity_layout,incremental}_test.cc`; `tests/native_foundation/nary/ordered_incidence_gather_test.cc` |
| Execution coverage/planes/ports | `tests/jbc/interfaces`, `tests/jbc/atom_plane`, `tests/jbc/multi_extent`; gate input and fragment suites |
| Specialist formats and projections | CellPack persistent/image tests, framework quantization/profile tests, `tests/math_core` and `tests/geometry/projection` |
| Runtime/device ownership | Framework CUDA runtime/memory smoke, native-value/readiness/parameter-transfer tests; extend with the connected two-device test |
| Install/modules/bindings | `cmake/CelleratorNN.cmake`, `modules/native/CMakeLists.txt`, native Cell installed consumers, affected native binding tests |
| Structured state | `tests/substrate/state/structured_state_test.cc` |

These are source pointers, not executed migration checks or a claim that each
test is registered in default CTest. Determine exact commands from the affected
build targets and developer reference. New tests must cover the new contracts,
not merely mirror moved implementation code.
