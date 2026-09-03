# Dependency, barriers, lanes, and integration

## Workstreams

| Code | Workstream | Phase | Prerequisite milestones | Integration milestone | Lane | Leaf tasks |
| --- | --- | --- | --- | --- | --- | --- |
| `A01` | Live authority, specification, and supersession baseline | `P00` | none | `M00` | `CE-CCP1-L-A01` | 9 |
| `A02` | JBC implementation inventory and provenance preservation | `P00` | none | `M00` | `CE-CCP1-L-A02` | 12 |
| `A03` | Cellerator and CellShard compiler-ownership rehoming design | `P00` | none | `M00` | `CE-CCP1-L-A03` | 14 |
| `A04` | Proposed compiler directory and central-file ownership structure | `P00` | none | `M00` | `CE-CCP1-L-A04` | 10 |
| `B01` | Host-only build partition and compiler target graph | `P10` | `M00` | `M10` | `CE-CCP1-L-B01` | 12 |
| `B02` | Compiler driver and NVCC-like downstream toolchain selection | `P10` | `M00` | `M10` | `CE-CCP1-L-B02` | 14 |
| `B03` | Source manager, preprocessing, file-local pragma, and shadow C++ | `P10` | `M00` | `M10` | `CE-CCP1-L-B03` | 15 |
| `B04` | Upstream Clang semantic bridge and ordinary C++ fallthrough | `P10` | `M00` | `M10` | `CE-CCP1-L-B04` | 14 |
| `C01` | Cellerator lexical grammar and parser | `P20` | `M10` | `M20` | `CE-CCP1-L-C01` | 16 |
| `C02` | Cellerator AST, symbols, source mapping, and frontend diagnostics | `P20` | `M10` | `M20` | `CE-CCP1-L-C02` | 12 |
| `C03` | Biological semantic types, relations, and typed operation analysis | `P20` | `M10` | `M20` | `CE-CCP1-L-C03` | 16 |
| `C04` | Execution fields, effects, lifetimes, profiles, and compiler-control hierarchy | `P20` | `M10` | `M20` | `CE-CCP1-L-C04` | 16 |
| `D01` | Common CEIR object model, textual language, and standalone artifacts | `P30` | `M10` | `M30` | `CE-CCP1-L-D01` | 14 |
| `D02` | Semantic IR | `P30` | `M20` | `M30` | `CE-CCP1-L-D02` | 16 |
| `D03` | Representative-data profile artifacts, inference, and multi-state propagation | `P30` | `M20` | `M30` | `CE-CCP1-L-D03` | 15 |
| `E01` | Planning IR search-space model | `P40` | `M30` | `M40` | `CE-CCP1-L-E01` | 16 |
| `E02` | JBC evidence, discovery, certification, and atom compiler rehoming | `P40` | `M00`, `M30` | `M40` | `CE-CCP1-L-E02` | 18 |
| `E03` | JBC composition, grammar, basis, global program, and schedule rehoming | `P40` | `M00`, `M30` | `M40` | `CE-CCP1-L-E03` | 18 |
| `E04` | Decomposition, candidate catalogs, complete costs, and planner integration | `P40` | `M30` | `M40` | `CE-CCP1-L-E04` | 18 |
| `F01` | Realization IR, physical objects, stage graphs, and readiness | `P50` | `M40` | `M50` | `CE-CCP1-L-F01` | 18 |
| `F02` | Backend framework and CPU/native C++ object emission | `P50` | `M10`, `M40` | `M50` | `CE-CCP1-L-F02` | 14 |
| `F03` | CUDA source and NVCC backend | `P50` | `M40` | `M50` | `CE-CCP1-L-F03` | 15 |
| `F04` | Clang CUDA, LLVM/NVPTX, and direct PTX routes | `P50` | `M40` | `M50` | `CE-CCP1-L-F04` | 13 |
| `G01` | Source reflection and inline writable CEIR | `P60` | `M30` | `M60` | `CE-CCP1-L-G01` | 16 |
| `G02` | Open pass pipeline, custom extensions, and same-compilation transforms | `P60` | `M30`, `M40` | `M60` | `CE-CCP1-L-G02` | 18 |
| `G03` | Validation modes, diagnostics, provenance, and compiler explainability | `P60` | `M20`, `M30` | `M60` | `CE-CCP1-L-G03` | 16 |
| `H01` | Cross-translation-unit CEIR, object sections, archives, and Cellerator LTO | `P70` | `M30`, `M50` | `M70` | `CE-CCP1-L-H01` | 16 |
| `H02` | libCellerator compiler APIs and runtime/execution SDK | `P70` | `M30`, `M50` | `M70` | `CE-CCP1-L-H02` | 16 |
| `H03` | Cellerator standard library, reference profiles, installation, and package integration | `P70` | `M20`, `M30`, `M50` | `M70` | `CE-CCP1-L-H03` | 18 |
| `I01` | celleratord core, clangd compatibility, and incremental compiler snapshots | `P80` | `M10`, `M20` | `M80` | `CE-CCP1-L-I01` | 14 |
| `I02` | celleratord biological semantics, profiles, IR, planning, and native navigation | `P80` | `M30`, `M40`, `M60` | `M80` | `CE-CCP1-L-I02` | 14 |
| `J01` | Compiler conformance, fuzzing, differential validation, and resilience | `P90` | `M20`, `M30` | `M90` | `CE-CCP1-L-J01` | 12 |
| `J02` | Complete-cost performance program and benchmarkable vertical slices | `P90` | `M10`, `M30`, `M40`, `M50` | `M90` | `CE-CCP1-L-J02` | 14 |
| `J03` | Integration, specification reconciliation, SDK release, and Part One closure | `P90` | `M60`, `M70`, `M80` | `M90` | `CE-CCP1-L-J03` | 13 |

## Milestones and barriers

| Milestone | Meaning | Workstreams | Integration task | Barrier |
| --- | --- | --- | --- | --- |
| `M00` | Architecture, ownership, and migration authority frozen | `A01`, `A02`, `A03`, `A04` | `CE-CCP1-M00` | `CE-CCP1-B-M00` |
| `M10` | Host-only build, driver, source pipeline, and C++ bridge integrated | `B01`, `B02`, `B03`, `B04` | `CE-CCP1-M10` | `CE-CCP1-B-M10` |
| `M20` | Source language parser, AST, Sema, and execution-field semantics integrated | `C01`, `C02`, `C03`, `C04` | `CE-CCP1-M20` | `CE-CCP1-B-M20` |
| `M30` | Common CEIR, Semantic IR, and representative profile environment integrated | `D01`, `D02`, `D03` | `CE-CCP1-M30` | `CE-CCP1-B-M30` |
| `M40` | Planning IR and Cellerator-owned JBC compiler logic integrated | `E01`, `E02`, `E03`, `E04` | `CE-CCP1-M40` | `CE-CCP1-B-M40` |
| `M50` | Realization IR and CPU/NVIDIA backend foundation integrated | `F01`, `F02`, `F03`, `F04` | `CE-CCP1-M50` | `CE-CCP1-B-M50` |
| `M60` | Reflection, open passes, self-transforms, validation modes, and provenance integrated | `G01`, `G02`, `G03` | `CE-CCP1-M60` | `CE-CCP1-B-M60` |
| `M70` | Cross-TU/LTO, libCellerator, standard library, and installable SDK integrated | `H01`, `H02`, `H03` | `CE-CCP1-M70` | `CE-CCP1-B-M70` |
| `M80` | celleratord core and Cellerator-aware semantic tooling integrated | `I01`, `I02` | `CE-CCP1-M80` | `CE-CCP1-B-M80` |
| `M90` | Part One compiler family final acceptance | `J01`, `J02`, `J03` | `CE-CCP1-M90` | `CE-CCP1-B-M90` |

Milestone integration tasks restore a coherent validated main boundary. Structural work inside isolated lanes may be temporarily incomplete.

## Lanes

| Lane | Role | Workspace mode | Task count | Integration task |
| --- | --- | --- | --- | --- |
| `CE-CCP1-L-COORD` | coordinator | `read_shared` | 45 | `` |
| `CE-CCP1-L-A01` | implementer | `isolated_merge` | 9 | `CE-CCP1-M00` |
| `CE-CCP1-L-A02` | implementer | `isolated_merge` | 12 | `CE-CCP1-M00` |
| `CE-CCP1-L-A03` | implementer | `isolated_merge` | 14 | `CE-CCP1-M00` |
| `CE-CCP1-L-A04` | implementer | `isolated_merge` | 10 | `CE-CCP1-M00` |
| `CE-CCP1-L-B01` | implementer | `isolated_merge` | 12 | `CE-CCP1-M10` |
| `CE-CCP1-L-B02` | implementer | `isolated_merge` | 14 | `CE-CCP1-M10` |
| `CE-CCP1-L-B03` | implementer | `isolated_merge` | 15 | `CE-CCP1-M10` |
| `CE-CCP1-L-B04` | implementer | `isolated_merge` | 14 | `CE-CCP1-M10` |
| `CE-CCP1-L-C01` | implementer | `isolated_merge` | 16 | `CE-CCP1-M20` |
| `CE-CCP1-L-C02` | implementer | `isolated_merge` | 12 | `CE-CCP1-M20` |
| `CE-CCP1-L-C03` | implementer | `isolated_merge` | 16 | `CE-CCP1-M20` |
| `CE-CCP1-L-C04` | implementer | `isolated_merge` | 16 | `CE-CCP1-M20` |
| `CE-CCP1-L-D01` | implementer | `isolated_merge` | 14 | `CE-CCP1-M30` |
| `CE-CCP1-L-D02` | implementer | `isolated_merge` | 16 | `CE-CCP1-M30` |
| `CE-CCP1-L-D03` | implementer | `isolated_merge` | 15 | `CE-CCP1-M30` |
| `CE-CCP1-L-E01` | implementer | `isolated_merge` | 16 | `CE-CCP1-M40` |
| `CE-CCP1-L-E02` | implementer | `isolated_merge` | 18 | `CE-CCP1-M40` |
| `CE-CCP1-L-E03` | implementer | `isolated_merge` | 18 | `CE-CCP1-M40` |
| `CE-CCP1-L-E04` | implementer | `isolated_merge` | 18 | `CE-CCP1-M40` |
| `CE-CCP1-L-F01` | implementer | `isolated_merge` | 18 | `CE-CCP1-M50` |
| `CE-CCP1-L-F02` | implementer | `isolated_merge` | 14 | `CE-CCP1-M50` |
| `CE-CCP1-L-F03` | implementer | `isolated_merge` | 15 | `CE-CCP1-M50` |
| `CE-CCP1-L-F04` | implementer | `isolated_merge` | 13 | `CE-CCP1-M50` |
| `CE-CCP1-L-G01` | implementer | `isolated_merge` | 16 | `CE-CCP1-M60` |
| `CE-CCP1-L-G02` | implementer | `isolated_merge` | 18 | `CE-CCP1-M60` |
| `CE-CCP1-L-G03` | implementer | `isolated_merge` | 16 | `CE-CCP1-M60` |
| `CE-CCP1-L-H01` | implementer | `isolated_merge` | 16 | `CE-CCP1-M70` |
| `CE-CCP1-L-H02` | implementer | `isolated_merge` | 16 | `CE-CCP1-M70` |
| `CE-CCP1-L-H03` | implementer | `isolated_merge` | 18 | `CE-CCP1-M70` |
| `CE-CCP1-L-I01` | implementer | `isolated_merge` | 14 | `CE-CCP1-M80` |
| `CE-CCP1-L-I02` | implementer | `isolated_merge` | 14 | `CE-CCP1-M80` |
| `CE-CCP1-L-J01` | implementer | `isolated_merge` | 12 | `CE-CCP1-M90` |
| `CE-CCP1-L-J02` | implementer | `isolated_merge` | 14 | `CE-CCP1-M90` |
| `CE-CCP1-L-J03` | implementer | `isolated_merge` | 13 | `CE-CCP1-M90` |
| `CE-CCP1-L-INTEGRATE-FOUNDATION` | integrator | `exclusive` | 4 | `` |
| `CE-CCP1-L-INTEGRATE-COMPILER` | integrator | `exclusive` | 4 | `` |
| `CE-CCP1-L-INTEGRATE-FINAL` | validator | `exclusive` | 2 | `` |

## Parallelism principles

- Workstream lanes are serial internally and isolated by worktree.
- Different workstreams can proceed in parallel once their milestone/checkpoint dependencies are satisfied.
- Shared compiler interfaces are thin waists and have one owner.
- Central registries and build/package files are not edited by leaf lanes.
- Integration lanes collect source-linked commits/receipts and run milestone labels.
- GPU resources are requested only for gates that need hardware.
- Historical JBC worktrees are read/migration evidence, not implementation lanes for this program.

## Dependency artifacts

- `machine/dependency_edges.csv`
- `machine/checkpoints.csv`
- `machine/barriers.json`
- `machine/lanes.json`
- `machine/workstreams.json`

The package validator checks task-only and checkpoint-expanded acyclicity, parent resolution, interface publication/consumption, lane membership, and milestone reachability.
