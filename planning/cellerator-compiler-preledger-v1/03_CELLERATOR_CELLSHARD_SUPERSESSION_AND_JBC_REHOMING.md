# Cellerator and CellShard supersession and JBC rehoming

## Superseded ownership

The original joint JBC architecture assigned the following compiler layers to CellShard:

- overlapping evidence discovery;
- atom certification and atom compiler records;
- composition grammar;
- basis and superatom selection;
- global operation graph IR;
- portable/global schedule compilation;
- parts of persistent-partial legality.

That split is now superseded.

Those capabilities discover or compile reusable rules from representative biological structure. They therefore belong in Cellerator.

## New authoritative split

### Cellerator

Cellerator owns:

- representative profile semantics and evidence;
- proposal discovery;
- exact certification;
- atom semantics and exact coverage;
- typed composition and grammar;
- basis and no-basis selection;
- superatom promotion where retained;
- decomposition and partial-result algebra;
- global operation/program IR;
- cross-operation planning;
- portable schedule/ruleset compilation;
- candidate catalogs, costs, selection, realization, native lowering;
- all public CEIR and compiler passes.

### CellShard

CellShard retains concrete application/storage/runtime concerns:

- immutable and mutable concrete artifact storage;
- atom-store containers and encoded replicas;
- sharding of concrete datasets;
- concrete materialization from Cellerator rules;
- staging and assembly of payloads;
- placement and residency;
- transport and delivery;
- leases, recovery, runtime command execution;
- storage-oriented generation and publication.

CellShard may provide external costs and concrete capabilities to Cellerator through generic interfaces. It may not become the semantic planner.

## No-code-loss migration policy

The migration is source-led, not title-led.

Each CellShard JBC file is classified in `inventories/jbc_source_migration.csv` as one of:

- rehome to Cellerator compiler;
- split semantic compiler logic from concrete storage/runtime logic;
- remain in CellShard application/storage/runtime;
- adapt as a versioned bridge;
- migrate test/evidence with its implementation;
- audit required.

Every moved implementation carries:

- source repository;
- source branch and commit;
- original path;
- implementing Todo when recoverable;
- original tests and evidence;
- SHA-256;
- migration disposition;
- target path;
- commit trailer or migration-manifest reference.

A source may be improved while it moves. It must not be silently replaced with a new implementation whose behavior is merely assumed equivalent.

## Migration groups

| Old CellShard compiler area | New Cellerator area | Disposition |
| --- | --- | --- |
| `compiler/evidence`, discovery portfolios | `compiler/profile`, `compiler/discovery`, Planning IR | Move/adapt |
| exact rescan/certification | Planning IR exact coverage and discovery validators | Move/adapt |
| atom core/species/ports/planes | Cellerator discovery and Planning IR | Move/adapt, retain `atom` name |
| composition/production/grammar | `compiler/composition`, Planning IR | Move/adapt |
| basis and no-basis | Cellerator planner/program planning | Move/adapt |
| superatom | Cellerator experimental composed-unit promotion | Move/adapt, preserve non-promotion |
| partial compiler legality/algebra | Cellerator decomposition/planning | Split |
| persistent partial bytes/replicas | CellShard artifact/runtime | Retain |
| global graph | Cellerator program Semantic/Planning IR | Move/adapt |
| portable schedule | Cellerator ruleset Planning IR | Move/adapt |
| concrete placement/residency/transport | CellShard | Retain |
| atom-store persistence | CellShard | Retain |
| Cellerator fragment/resumption interfaces | Cellerator | Preserve/integrate |

## Historical authority

Old JBC Todos and commits remain historical evidence. The new plan adds a supersession record and new interfaces rather than mutating completed results. `CE-JBC-RUN-V1` is not reused.

## Part Two seam

Part One reserves only a narrow compiled-ruleset/materialization-request seam. Deep CellShard application integration is not a dependency of the compiler’s Part One completion.
