# Cross-project dependency, integration, and parallelism map

This is a proposed dependency structure, not a Todo mutation. The later planner should map workstream tasks into serial queues within lanes, use interfaces/checkpoints as the few genuine barriers, and leave provider lanes independent. Cross-authority dependencies must be represented by mirrored receipts if native edges are unavailable.

# Workstream and task counts

| Code | Workstream | Repository | Lane | Todos | Entry barrier |
| --- | --- | --- | --- | --- | --- |
| CEBOOT | Cellerator baseline, charter, and source ownership | Cellerator | CE-JBC-L-BOOTSTRAP | 6 | JBC-G0-LIVE-BASELINE |
| CEIF | Cellerator-owned joint-compiler thin-waist interfaces | Cellerator | CE-JBC-L-INTERFACES | 12 | JBC-G0-LIVE-BASELINE |
| CEDEC | Biological operation decomposition and partial algebra | Cellerator | CE-JBC-L-DECOMPOSITION | 18 | JBC-G1-ATOM-THIN-WAIST |
| CEFRAG | Atom-aware Cellerator fragment compiler | Cellerator | CE-JBC-L-FRAGMENT | 14 | JBC-G1-ATOM-THIN-WAIST |
| CEMULTI | Multi-atom and multi-extent Cellerator operands | Cellerator | CE-JBC-L-MULTIATOM | 10 | JBC-G3-CELLERATOR-FRAGMENT |
| CEPLANE | Cellerator atom planes, mutable values, gradients, and atom outputs | Cellerator | CE-JBC-L-PLANES | 10 | JBC-G1-ATOM-THIN-WAIST |
| CERESUME | Cellerator lowering-resumption contracts | Cellerator | CE-JBC-L-RESUMPTION | 10 | JBC-G3-CELLERATOR-FRAGMENT |
| CEXOP | Cross-operation Cellerator projection families | Cellerator | CE-JBC-L-CROSSOP | 8 | JBC-G3-CELLERATOR-FRAGMENT |
| CECOST | External global costs and bounded joint compiler exchange | Cellerator | CE-JBC-L-EXTERNAL-COST | 6 | JBC-G3-CELLERATOR-FRAGMENT |
| CEVAL | Cellerator exact verification, profiling, packaging, and integration | Cellerator | CE-JBC-L-VERIFY-INTEGRATE | 6 | JBC-G5-PARTIAL-ARTIFACT |
| CSBOOT | CellShard baseline, successor charter, and compatibility map | CellShard | CS-JBC-L-BOOTSTRAP | 6 | JBC-G0-LIVE-BASELINE |
| CSATOM | CellShard biological execution atom core | CellShard | CS-JBC-L-ATOM-CORE | 20 | JBC-G0-LIVE-BASELINE |
| CSEVID | CellShard atom evidence atlas | CellShard | CS-JBC-L-EVIDENCE-CORE | 16 | JBC-G1-ATOM-THIN-WAIST |
| CSCERT | Independent exact atom certification | CellShard | CS-JBC-L-CERTIFICATION | 16 | JBC-G1-ATOM-THIN-WAIST |
| CSSS | Support-signature atom discovery | CellShard | CS-JBC-L-DISC-SIGNATURE | 10 | JBC-G2-EVIDENCE-CERTIFICATION |
| CSCO | Co-support and affinity atom discovery | CellShard | CS-JBC-L-DISC-COSUPPORT | 10 | JBC-G2-EVIDENCE-CERTIFICATION |
| CSBC | Biclustering and co-clustering atom discovery | CellShard | CS-JBC-L-DISC-BICLUSTER | 8 | JBC-G2-EVIDENCE-CERTIFICATION |
| CSOC | Bounded overlapping-community atom discovery | CellShard | CS-JBC-L-DISC-OVERLAP | 6 | JBC-G2-EVIDENCE-CERTIFICATION |
| CSMF | Typed motif and frequent-fragment atom discovery | CellShard | CS-JBC-L-DISC-MOTIF | 8 | JBC-G2-EVIDENCE-CERTIFICATION |
| CSFT | Factor/topic program atom proposals | CellShard | CS-JBC-L-DISC-FACTOR | 6 | JBC-G2-EVIDENCE-CERTIFICATION |
| CSTR | Trajectory and lineage atom discovery | CellShard | CS-JBC-L-DISC-TRAJECTORY | 12 | JBC-G2-EVIDENCE-CERTIFICATION |
| CSMM | Multimodal identity-spine and atom discovery | CellShard | CS-JBC-L-DISC-MULTIMODAL | 10 | JBC-G2-EVIDENCE-CERTIFICATION |
| CSSQ | Future sequence/Baseplane-compatible atom interfaces | CellShard | CS-JBC-L-DISC-SEQUENCE | 6 | JBC-G1-ATOM-THIN-WAIST |
| CSOT | Operation-trace and graph-family atom discovery | CellShard | CS-JBC-L-DISC-OPTRACE | 8 | JBC-G2-EVIDENCE-CERTIFICATION |
| CSCOMP | Typed atom composition DAG | CellShard | CS-JBC-L-COMPOSITION | 24 | JBC-G2-EVIDENCE-CERTIFICATION |
| CSGRAM | Explicit typed biological execution grammar | CellShard | CS-JBC-L-EXPLICIT-GRAMMAR | 10 | JBC-G2-EVIDENCE-CERTIFICATION |
| CSIGRAM | Induced execution grammar and MDL experiment | CellShard | CS-JBC-L-INDUCED-GRAMMAR | 10 | JBC-G4-COMPOSITION-BASIS |
| CSBASIS | Biological execution basis selection | CellShard | CS-JBC-L-BASIS | 18 | JBC-G2-EVIDENCE-CERTIFICATION |
| CSSUPER | Superatom promotion, demotion, and evolution | CellShard | CS-JBC-L-SUPERATOM | 8 | JBC-G4-COMPOSITION-BASIS |
| CSPART | Persistent partial computation atoms | CellShard | CS-JBC-L-PARTIALS | 18 | JBC-G2-EVIDENCE-CERTIFICATION |
| CSSTORE | Atom-native immutable persistence and lowering artifacts | CellShard | CS-JBC-L-PERSISTENCE | 29 | JBC-G1-ATOM-THIN-WAIST |
| CSGLOBAL | CellShard global operation graph and portable schedule compiler | CellShard | CS-JBC-L-GLOBAL-IR | 14 | JBC-G3-CELLERATOR-FRAGMENT |
| CSRUNTIME | CellShard topology, I/O, transport, residency, and runtime lowering | CellShard | CS-JBC-L-RUNTIME | 22 | JBC-G5-PARTIAL-ARTIFACT |
| CSVAL | Exact validation, biological evidence, vertical slices, and final integration | CellShard | CS-JBC-L-VALIDATION-INTEGRATION | 20 | JBC-G6-RUNTIME-SUBSTRATE |

# Major barriers

## JBC-G0-LIVE-BASELINE

**Requires:** CE-JBC-B01..B06, CS-JBC-B01..B06

**Unlocks:** CE interfaces, CellShard atom core, format/source transition work

**Why this is a real barrier:** Freeze exact source and authority without mutating prior campaigns.

## JBC-G1-ATOM-THIN-WAIST

**Requires:** JBC-I01..I06, CS-JBC-A01..A20

**Unlocks:** evidence, certification, Cellerator planes, most provider fixtures

**Why this is a real barrier:** Identity, coverage, atom envelope, planes, requirements, affordances and partial algebra are stable enough for parallel providers.

## JBC-G2-EVIDENCE-CERTIFICATION

**Requires:** CS-JBC-E01..E16, CS-JBC-C01..C16

**Unlocks:** all discovery providers, composition, basis candidate flow

**Why this is a real barrier:** Approximate proposals and exact certification are operationally separate.

## JBC-G3-CELLERATOR-FRAGMENT

**Requires:** CE-JBC-D01..D18, CE-JBC-F01..F14, JBC-I07..I11

**Unlocks:** global graph, multi-atom candidates, resumption, cross-operation views, external costs

**Why this is a real barrier:** CellShard can request exact atom-bound local alternatives without inventing Cellerator semantics.

## JBC-G4-COMPOSITION-BASIS

**Requires:** CS-JBC-O01..O24, CS-JBC-G01..G10, CS-JBC-BS01..BS18

**Unlocks:** superatoms, induced grammar, graph-family materialization, basis-backed vertical slices

**Why this is a real barrier:** Atoms have real typed composition and a complete no-basis fallback.

## JBC-G5-PARTIAL-ARTIFACT

**Requires:** CS-JBC-PP01..PP18, CS-JBC-ST01..ST25, CE-JBC-R01..R10

**Unlocks:** persistent partial verticals, topology/runtime linking, late-stage Cellerator resumption

**Why this is a real barrier:** Persistent products and lowering stages are exact, versioned and recoverable.

## JBC-G6-RUNTIME-SUBSTRATE

**Requires:** CS-JBC-Q01..Q14, CS-JBC-RT01..RT21, JBC-I18..I20

**Unlocks:** dual-node and transport slices, full recovery, final integration

**Why this is a real barrier:** Portable schedule, exact distributed certificate, topology, transport, residency and command runtime are available.

## JBC-G7-VERTICAL-SLICES

**Requires:** CS-JBC-V08..V17, CE-JBC-V01..V05

**Unlocks:** biological novelty readiness audit, final acceptance

**Why this is a real barrier:** Independent vertical slices prove the atom hierarchy reaches real Cellerator execution and distributed runtime.

# Suggested lanes

| Lane | Repository | Scope | Activation |
| --- | --- | --- | --- |
| CE-JBC-L-BOOTSTRAP | Cellerator | baseline/charter | serial until G0 |
| CE-JBC-L-INTERFACES | Cellerator | thin waist | parallel contract owner after G0 |
| CE-JBC-L-DECOMPOSITION | Cellerator | operation decomposition | parallel after G1 |
| CE-JBC-L-FRAGMENT | Cellerator | fragment compiler | after decomposition interfaces |
| CE-JBC-L-MULTIATOM | Cellerator | multi-atom/extent | after fragment fixture |
| CE-JBC-L-PLANES | Cellerator | planes/outputs | parallel after G1 |
| CE-JBC-L-RESUMPTION | Cellerator | lowering stages | after fragment fixture |
| CE-JBC-L-CROSSOP | Cellerator | cross-operation views | after fragment + planes |
| CE-JBC-L-EXTERNAL-COST | Cellerator | external costs/exchange | after fragment fixture |
| CE-JBC-L-VERIFY-INTEGRATE | Cellerator | verification/integration | integration-only central files |
| CS-JBC-L-BOOTSTRAP | CellShard | baseline/charter | serial until G0 |
| CS-JBC-L-ATOM-CORE | CellShard | atom model | parallel with Cellerator interfaces after G0 |
| CS-JBC-L-EVIDENCE-CORE | CellShard | evidence atlas | after atom envelope |
| CS-JBC-L-CERTIFICATION | CellShard | independent exact certification | after coverage/atom contracts |
| CS-JBC-L-DISC-SIGNATURE | CellShard | support signatures | parallel discovery provider |
| CS-JBC-L-DISC-COSUPPORT | CellShard | co-support | parallel discovery provider |
| CS-JBC-L-DISC-BICLUSTER | CellShard | biclustering | parallel discovery provider |
| CS-JBC-L-DISC-OVERLAP | CellShard | overlapping communities | parallel discovery provider |
| CS-JBC-L-DISC-MOTIF | CellShard | typed motifs | parallel discovery provider |
| CS-JBC-L-DISC-FACTOR | CellShard | factor/topic | parallel discovery provider |
| CS-JBC-L-DISC-TRAJECTORY | CellShard | trajectory/lineage | parallel discovery provider |
| CS-JBC-L-DISC-MULTIMODAL | CellShard | multimodal | parallel discovery provider |
| CS-JBC-L-DISC-SEQUENCE | CellShard | sequence compatibility | parallel interface/provider |
| CS-JBC-L-DISC-OPTRACE | CellShard | operation traces | parallel discovery provider |
| CS-JBC-L-COMPOSITION | CellShard | composition DAG | parallel after certification |
| CS-JBC-L-EXPLICIT-GRAMMAR | CellShard | explicit grammar | after composition contract |
| CS-JBC-L-INDUCED-GRAMMAR | CellShard | experimental grammar | after explicit grammar + traces |
| CS-JBC-L-BASIS | CellShard | basis solvers | after composition interfaces |
| CS-JBC-L-SUPERATOM | CellShard | superatoms | after basis + grammar |
| CS-JBC-L-PARTIALS | CellShard | persistent partials | after partial algebra + atom planes |
| CS-JBC-L-PERSISTENCE | CellShard | atom store | parallel after atom envelope |
| CS-JBC-L-GLOBAL-IR | CellShard | global graph/schedule | after fragment interface |
| CS-JBC-L-RUNTIME | CellShard | topology/I/O/transport/residency | after global IR + store source |
| CS-JBC-L-VALIDATION-INTEGRATION | CellShard | verticals/ablation/integration | central files integration-only |

## Maximal safe fan-out

- Before `JBC-G0`: two bootstrap lanes only.
- After `JBC-G1`: Cellerator decomposition/planes, CellShard evidence/certification/persistence-schema, fixtures, and sequence compatibility can run concurrently.
- After `JBC-G2`: all ten discovery-provider lanes, composition, basis groundwork, Cellerator fragment work, persistence, and validation fixtures can run concurrently.
- Expected immediate fan-out after the thin waist: **24 first-class lanes**.
- Expected peak safe fan-out after evidence/certification and fragment interfaces: **27 first-class lanes**.
- Total named lanes including bootstrap and integration: **34**.

Do not activate all central integration lanes as implementers. They remain idle until provider fragments are ready.

# Cross-repository integration order

1. Implement and integrate Cellerator interface/provider fragments in the Cellerator repository.
2. Freeze/publish exact interface version/hash/source receipts.
3. CellShard consumer tasks inspect and record mirrored receipts.
4. Implement CellShard work in its own repository/worktrees.
5. Integrate and push CellShard first.
6. A parent Cellerator integration task advances the submodule pointer once per checkpoint bundle, not for every leaf commit.
7. Re-run standalone Cellerator and embedded joint builds.
8. Never mutate both repositories from an ordinary provider task.

# Integration-only path policy

Root CMake, umbrella headers, package exports, central registries, shared program documents, and the parent submodule pointer are integration-only. Provider tasks emit isolated source-linked fragments. Downstream work may use mocks/fixtures as soon as an interface is frozen rather than waiting for all providers.

# Proposed explicit task edges

The machine-readable file `dependency_edges.csv` contains 434 explicit task-to-task edges. `external_dependency_receipts.csv` contains 8 checkpoint/interface/receipt dependencies.

# Critical path

The intended critical path is:

```text
G0 live baseline
→ G1 identity/coverage/atom envelope/partial algebra
→ G2 evidence + independent certification
→ G3 Cellerator atom-fragment interface
→ G4 composition + explicit grammar + basis fallback
→ G5 persistent partials + atom store + lowering resumption
→ G6 global schedule + runtime substrate
→ G7 required vertical slices
→ biological novelty audit + final integration
```

Discovery alternatives, induced grammar, advanced basis solvers, direct multi-extent execution, column generation, compression, and process-model variants remain side branches with promotion gates.

# Coordination anomalies the later bootstrap must handle

- Cellerator and CellShard authority observations are not globally atomic.
- Several historical runs remain recorded active despite closed/completed lanes; do not hijack them.
- CellShard has stale heartbeat records.
- `cellshard-cpp-access-adapter-refactor` is raw-closed but can appear effective-ready; record a receipt or maintenance issue rather than duplicate work.
- No current native cross-authority interface relationship exists; use mirrored receipts.
- CE-AMP remains permission-gated and outside this program.
