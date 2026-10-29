# Cellerator consolidation and execution development

Preserved implementation plan, approved on 2026-10-28. This document records
scope and acceptance; registered Project Control todos own current progress.
Baseline: `2f851ddb9719a85064f7140864c614245424f4cc`.

## Purpose and fixed decisions

Cellerator is a hardware-aware, high-performance PyTorch extension for cellular
model building. Biological structure guides the mathematical operation and its
physical execution. Shared structure across cells and repeated evaluations can
improve interpretability, eliminate unnecessary computation, and expose better
memory layouts and execution geometry. Cellerator enables scientific questions;
it does not prescribe the scientific model.

The central path is:

`logical structure -> indexing and routing -> conditional work -> physical layouts -> native forward/backward execution -> safe reuse and publication`.

Build on existing native arithmetic, numerical contracts, experiments and real
consumers. Torch supplies tensors, modules, autograd and optimizers. Keep direct
native control available. Execution fields and layout semantics remain central;
field composition, compilation and live code replacement retain experimental
status. Classic single-cell analysis remains a testbed and consumer, with
Scanpy/RAPIDS upstream and AnnData the standard Python scientific interface.

Execution geometry is the organizing objective across indexing, routing,
gating, layout and hardware execution. Natural extensions can generalize
existing prerequisites, conditional work and persistent event-aware state;
they must follow existing machinery, without inventing biological laws or
starting separate regulation/memory subsystems. Do not start new cell-port,
communication or lineage frameworks. Preserve and route existing implementations.

## Registered sequence

0. Back up first. Preserve source, all refs/reflogs/index/worktree metadata,
   submodules, endangered untracked and ignored payloads and full Todo state.
   Use private local block storage; retain manifests, hashes, original paths
   and restore instructions. Verify recovery before disruptive changes.
1. Register dependent outcomes in the existing `cellerator` authority using
   supported preview/apply routes. Reconcile outstanding IS1 obligations as
   complete with evidence, carried forward, or superseded. Ancestry is not
   completion evidence. Do not create another authority or hand-edit generated
   state. Record unavailable host operations accurately and retain protected
   workspaces until their supported lifecycle can be completed.
2. Recheck live branches and content/patch equivalence. Integrate genuine missing
   work; retain newer fixes. Archive tips and bundles before closing local and
   remote working branches. Preserve backup refs and authority integration refs.
   Close the IS1 worktree through supported workflow when available.
   The completed ordinary remote-ref closure and retained history are recorded
   in [branch reconciliation](branches.md); remaining archive and workflow holds
   are listed in [archive disposition](archive-disposition.md).
3. Give every source/artifact a role, owner, destination and disposition. Create
   an agent routing guide and machine-readable map; make gaps and maturity visible.
4. Consolidate source and packaging in complete units: implementation, headers,
   targets, callers, tests and documentation. Establish a coherent installed
   Torch-facing package while preserving native builds and real sibling users.
5. Deepen device-resident structural execution, layouts, derivatives and reuse.
   Use measured strategies with fallbacks and explicit preparation/conversion
   costs. Develop one geometry/execution direction rather than many new products.
6. Reuse the same contracts for experimental fields. Build a bounded non-JIT
   prepared-field witness using current numerical operations; do not promise a
   general compiler or require live recompilation for rebinding/workset changes.
7. Extract linked reusable code before archiving experiments. Retire verified
   redundant scaffolding only after recoverable archival. Qualify native and
   installed package behavior, docs and restoration; publish validated changes
   and set the GitHub description with `gh`.

## Ownership and source layout

- `core` / `structure`: stable identity, axes, relations and operand contracts.
- `support`, `indexing`, `gating`: distinct logical eligibility, lookup and
  conditional-work owners. Generic helpers currently in `src/support` move to
  `common`; avoid a second ambiguous support meaning.
- `routing`: model entities/events to mechanism work. Hardware dispatch remains
  part of execution planning.
- `ops/<family>`: native arithmetic, derivatives and host numerical operations;
  families stay open to future model operations.
- `layout`: physical projections, packing, quantization and logical/physical
  value and gradient maps.
- `geometry`: discovery, grouping and organization strategies, separate from
  physical realization and actual runtime state.
- `execution`: planning, prepared bindings, dependency guards and reuse.
- `runtime`: allocations, readiness, streams, executable lifetime and native
  resources; integrate with Torch's ownership instead of duplicating it.
- internal Torch/bindings: registered operations, tensor adaptation, meta/fake
  behavior and backward integration.
- compiler/frontends/codegen: optional experiments with honest bounded contracts.
- models and scientific fixtures: examples/experiments; extract reusable
  mechanisms without embedding the consumer's biological hypotheses.

Python surface: `ops`, `nn`, `structure`, `support`, `indexing`, `routing`,
`gating`, `layout`, `execution`, `tokenization`, `interop`; private `_torch`;
experimental `fields` and `compilation`. `nn` uses real `torch.nn.Module`
subclasses. Do not introduce replacement tensors, optimizers or autograd.

The disposition map records current owner, role, intended destination,
disposition, maturity, consumers, tests/evidence and archive reference. It
covers retained, moved, experimental, archived, managed/generated, and explicitly
held/unclassified paths. Never erase unique capability because it is unfinished
or native-only. Immutable evidence keeps original bytes; indices describe its
source snapshot and historical paths.

Rename translation units, paired headers, tests and build targets by purpose,
not task provenance. Examples: `named_field_parser`, `field_regions`,
`support_signatures`, `source_plan_slice`, `prepared_relation_bridge`,
`host_runtime_bindings`, `feature_major_sparse_sgd_step`,
`developmental_time_losses`. Keep narrow implementations narrow. Preserve actual
ABI/schema versions, serialized IDs and frozen interfaces unless reconciled
explicitly. Keep already meaningful names.

## Native extensions and parameter semantics

Extend device-resident lookup, routing, condition evaluation and bounded workset
construction with explicit capacity/count/overflow. Reuse indexes, projections,
maps and reserved storage across cells and steps. Invalidation distinguishes
structure, activity, values/parameters and bindings; an optimizer step must not
force an unrelated structural rebuild. Preserve canonical identity and map
physical values and gradients back exactly, including repeated parameter roles
and disjoint residual ownership. Reduce launches, materialization, conversion
and synchronization where the operation permits.

Compare scans, postings, masks, shared-condition reuse, SIMT and packed strategies
as appropriate, with strong conventional/library alternatives and exact fallback.
Use KDA and source-bound experiments on available hardware (initially the
existing V100/SM70 environment). Separate storage, multiply, accumulation and
output precision. Support FP16 where justified, retaining explicit FP32/FP64 and
existing caller contracts. No universal sparse format or Tensor Core mandate.
No performance claim follows from a file move.

The user approved both explicit mechanism transition modes:

- `shared`: the same A/B/bias across mechanisms;
- `per_mechanism`: independent corresponding matrices/bias per mechanism.

Per-cell state remains independent in both modes. Validate shapes and complete
state/parameter gradients. Legacy checkpoints preserve their original shared
block-zero calculation, values and optimizer state; do not activate previously
unused blocks silently. Keep one authoritative learned parameter owner.

Experimental fields reuse parser/composition/IR/effects/lifetime work and the
same numerical contracts. Their bounded witness must account for layouts,
dependencies and reuse. No implicit host synchronization at field exit.

## Validation, documentation and acceptance

Verify archives by hashes and restoration, including managed state and submodule
identity. Record branch equivalence/supersession rather than manufacturing merge
commits. Use independent numerical references, precision checks, JVP/VJP,
repeated-parameter gradients and continuation/checkpoints. Include empty,
duplicate, unsorted, capacity, temporal identity, dormant derivatives and
inactive nonfinite cases. Verify canonical layout recovery, value/gradient maps,
coverage, generations, tapes, publication and stream ownership.

Check registered Torch operations/meta behavior and only claim tested compiler
compatibility. Test installed packages outside the checkout and stock optimizers.
Changed CUDA paths require source/binary-bound memcheck, racecheck and synccheck.
GPU execution uses finite admitted Device Scheduler jobs, exclusive/quiescent
timing where required and verified cleanup receipts. Measure preparation,
routing/scoring/worksets, conversions/transfers, forward/backward and actual
reuse horizons. Unsupported alternatives remain explicit; promote defaults
only within a demonstrated regime.

Create `docs/development/routing.md`, link from AGENTS, and maintain the source
map separately from live Todo state. Rework README, architecture, developer and
user documentation around biological/hardware motivation, Torch model building,
structure/layout/execution/learning semantics, real operation capabilities,
native extension workflow, experimental fields/compiler and evidence/history.
Preserve source-bound reports and archival bytes. Clearly distinguish current
implementation, intended contracts and research questions. Repair links and
render affected pages.

Set repository description with `gh` to:

> Hardware-aware PyTorch extensions for cellular modeling, organizing sparse routing, gating, and GPU execution around shared biological structure.

Acceptance requires every file routed, unique work usable, purpose-based names,
a coherent native/Torch path, tested and measured strategy boundaries with
fallbacks, visible gaps, and agreement between docs, registered state, installed
package and GitHub metadata. A move and a tidy tree alone do not finish this plan.

## Final adoption provenance

The [adoption record](workflow-adoption/README.md) preserves the original
step breakdown and explains the final registered source-adoption contract.
Current execution and review state remain in Project Control.
