# Source and instruction review

These observations were made through the connected, read-only Project Control observatory on 2 October 2026, not inferred solely from the earlier conversation. Paths are relative to the named repository. Tool results did not expose file-citation markers; precise paths, ranges and source revisions provide the provenance here.

## Observation scope

| Repository | Observed HEAD | Todo revision | Focus run |
|---|---|---:|---|
| Cellerator | `47bd95a21d5143d2293dedbdaa35b6f1d97dda3c` | 7474 | CE-ML2-RUN-V1 |
| GlassHelix | `34782c48d3d89465af0b635eb78b56d872c212ac` | 260 | GH-ML2-RUN-V1 |

Both were clean at the observations used here. Program-context observations were independent, not a cross-repository atomic snapshot. Frontier/export results were bounded/partial; this is a review of the relevant current work, not a claim to enumerate every historical task. No source/build/benchmark was executed remotely.

## Cellerator: inspected source and consequences

| Path / range | Observed implementation or instruction | Consequence for this bootstrap |
|---|---|---|
| `AGENTS.md:1–39` | Native ownership, managed Todo front door, separate identities/epochs, existing consumers and owner reconciliation | New experimental paths first; no direct database/generated-ledger edits |
| `docs/design/overview.md:1–33` | Discovery, compilation and runtime separated; dynamic gates intended but not universally complete | Extend physical choices without pretending masks already eliminate work |
| `docs/development/source-map.md:1–15` | Source/compiler/geometry/execution/CT entry points | Reuse the actual library boundaries |
| `planning/learning-v2/README.md:1–61` | Separate schema-v3 plans; documented plan validate/apply front door | Follow observed syntax; no guessed workflow mutation commands |
| `planning/learning-v2/DESIGN.md:1–66` | GH science; CE numerical execution; CT adapter; no duplicate runtime | New model families in GH experiments, reusable operations in CE |
| `docs/learning/TRAIN.md:1–43` | Product mechanism training, canonical FP32 coefficients, derived half plane, guarded optimizer publication, one backward per tape | Preserve completed CE-ML2-TRAIN rather than rebuilding an old missing-gradient diagnosis |
| `include/Cellerator/compute/operation/indexed_mechanism/training.hh:1–104` | Parameter owner, forward/reverse incidence, tape, input/coefficient gradient API | Strong starting point for process packets and refactoring lifetimes |
| `components/CelleraTorch/src/mechanism.cc:1–115` | Actual Torch/native bridge with CUDA rank-two inputs, axes and tensor version checks | Adapter work is real; do not place a second set of math kernels in CT |
| `include/Cellerator/compute/operation/native_foundation_contract.hh:1–195,221–356` | Operand roles, capabilities, snapshots, primal generations, support distinctions and compiled launch hooks | New operations must admit capabilities honestly and preserve derivative support |
| `include/Cellerator/execution/program/program_v2.h:1–61` | Prepared dependency stages with nonowning launch bindings and admission | Compile into this executor; do not build a GH actor interpreter |
| `include/Cellerator/execution/atom_plane/mutable_state_plane_v1.hh:1–75` | Nonowning mutable dense state with persistent order, generation, capacity and optional dirty extents | Actor-local packet can adapt this, not replace ownership with a new allocator |
| `include/Cellerator/memory/view.hh:1–57` | Nonowning typed array/matrix views with placement | Translate prototype pointer views into this established substrate |
| `include/Cellerator/geometry/optimizer.hh:1–133` | Candidate normalization, coarsening/refinement, exact oracle and frozen plan | Adapt the search pattern to computational footprints; prototype score is not native oracle equivalence |
| `src/compute/architecture/providers/nvidia/sm70/relation_apply/apply_wmma_shapes_v1.cu:1–124` | Destination-owned accumulation over source tiles, three documented WMMA shapes | Keep this path and add distinct micro-MMA/matrix-program candidates |
| `planning/learning-v2/machine/cellerator.todo-plan.json:199–229` | CE-ML2-BIO depends on CE-ML2-TRAIN; original acceptance includes same-model checks and full-cost timing | Prototype sprint must not silently claim those older qualification obligations completed |

The new state type is not already fully present. Low-level state/axis/lifetime mechanisms exist; a unified actor-local state packet with support, regrowth lifecycle and multiple physical views is a proposed adapter/extension. The supplied struct is a seed, not a currently exported CE ABI.

The current product mechanism is not just pairwise matmul: its computation can be evaluated once and assembled into multiple destinations. That existing capability should survive the matrix-native experiments.

## GlassHelix: inspected source and consequences

| Path / range | Observation | Consequence |
|---|---|---|
| `AGENTS.md:1–25` | Distinct measurements/state/model/evidence; preserve whole candidates; no parallel runtime | Engineering hypotheses must stay visibly distinct from biological interpretation |
| `docs/design/overview.md:1–33` | Predictive state and identified mechanism differ; promotion is scoped and revisable | Local hidden bases may stay anonymous; supplied refactors are a valid engineering starting point |
| `docs/status/current.md:1–23` | Dated source snapshot, ML2 open; activity labels not compact execution | Do not treat old documentation as live completion evidence |
| `docs/development/source-map.md:1–15` | Installed CE dependency, native foundation and replay paths | No copying sibling CE source into GH |
| `src/system/assembly.cc:1–41` | Scientific ordered roles bind to CE operation contracts and compiled/indexed stages | Consumer integration can use the existing numerical seam |
| `include/GlassHelix/inference/inference.hh:1–134` | Supplied joint candidate states and CE-backed propagation; ordinary normalized weights | Preserve GH-ML2-REPAIR and avoid claiming that a new kernel repairs inference semantics |
| `planning/learning-v2/machine/glasshelix.todo-plan.json:1–196,530–576` | Separate scientific/data/repair/learning/pilot/acceptance outcomes with native lanes | Add research outcomes without erasing data or scientific qualification obligations |

## Current relevant work

CE frontier showed **CE-ML2-BIO ready** and CE-ML2-TRAIN completed in the active lane, followed by CE-ML2-0000 closure. A direct task inspection confirmed BIO has no active claim in that observation and depends on TRAIN. Ampere work remains permission/dependency blocked.

GH frontier showed DESIGN, DATA and REPAIR available; DESIGN is in progress in the project-context view. LEARN depends on DESIGN, PILOT on DATA/LEARN, and ACCEPT on the earlier outcomes. No active dispatch was present in the returned focus-lane observation. This does not grant permission to overwrite future claims.

GH-PREPRINT and GH-SCIENCE-FOUNDATIONS are not part of this bootstrap. The old NF1 and completed NF1A histories remain preserved.

## Explicitly not established

No current code read proves general automatic mechanism discovery, universal hidden-state promotion, compact active-only execution, single-cell Tensor-Core superiority or a production-ready actor-state API. The new hypotheses and delivered source are separated from the observed implementation throughout this package.
