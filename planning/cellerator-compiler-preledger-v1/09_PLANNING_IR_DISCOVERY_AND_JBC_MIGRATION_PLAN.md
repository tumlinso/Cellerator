# Planning IR, discovery, and JBC migration plan

## Planning IR is the search space

Planning IR represents alternatives before and after selection. It contains:

- planning problems at operation, field, graph, and profile-family scope;
- candidate families and providers;
- uncertain evidence and independently certified exact coverage;
- atoms, typed ports, planes, requirements, and affordances;
- decomposition alternatives, halos, replicas, and contribution ownership;
- partial-result algebra;
- persistent orders, projections, packing, and conversion routes;
- resource/stage inventories;
- complete costs and evidence;
- rejection, dominance, selected, forced, external, and fallback states.

Programmers can edit all of this directly.

## Rehomed compiler layers

Cellerator absorbs useful old CellShard JBC implementations for:

- evidence atlas and discovery providers;
- exact rescan/certification;
- atom semantics;
- composition/grammar;
- bases and no-basis;
- superatoms;
- persistent-partial compiler legality;
- global operation graph;
- portable schedule/ruleset.

Concrete atom-store persistence, materialization, placement, residency, transport, and delivery remain CellShard.

## Migration method

Movement is governed by `inventories/jbc_source_migration.csv`. Tests and evidence migrate with code. Compatibility aliases point toward Cellerator and name retirement gates. Migrated commits retain source trailers and a manifest.

## Planner integration

Existing geometry strategies, optimizer portfolios, candidate catalogs, external cost exchange, connected-operation planning, and lowering resumption are adapted into public Planning IR. No second hidden planner is created.

## Workstream task catalog

### E01: Planning IR

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-E01-001` | Freeze Planning IR module and decision-state model | Define unresolved, offered, admissible, rejected, dominated, selected, forced, externally selected, and fallback states within one representation rather than separate opaque planner records. |
| `CE-CCP1-E01-002` | Implement planning problems and operation scopes | Represent field, operation, bundle, chain, program, and profile-family planning problems with explicit semantic fingerprints, constraints, objectives, and target classes. |
| `CE-CCP1-E01-003` | Implement candidate-family and provider nodes | Represent provider identity, operation/numeric support, target capabilities, projection contracts, preparation entrypoints, experimental status, and source-linked extension identity. |
| `CE-CCP1-E01-004` | Implement exact logical coverage nodes | Represent member/edge coverage, ownership roles, canonical maps, certification receipts, and exact equations separately from approximate proposal evidence. |
| `CE-CCP1-E01-005` | Implement atom requirement and affordance nodes | Represent required/available planes, coverage, order, numeric/index types, alignment, extent rules, graph stability, generation, target ABI, and transform routes. |
| `CE-CCP1-E01-006` | Implement decomposition-alternative nodes | Represent legal split dimensions, fragments, halos, replicas, exact input/output/contribution coverage, partial algebra, order constraints, and unsplit fallback. |
| `CE-CCP1-E01-007` | Implement partial-result algebra nodes | Represent state schema, neutral element, merge/finalize operations, algebraic laws, determinism/order requirements, numeric policy, and implementation identity. |
| `CE-CCP1-E01-008` | Implement persistent-order, projection, and packing alternatives | Represent logical, canonical, projection-native, and persistent physical orders plus conversion routes, projection schemas, value maps, and packing invalidation. |
| `CE-CCP1-E01-009` | Implement resource and stage-inventory alternatives | Represent persistent/transient memory, workspace alignment, launch count, graph capture, synchronization, libraries, streams, transfers, and target capabilities. |
| `CE-CCP1-E01-010` | Implement complete cost vectors | Represent preparation, conversion, transfer, pack, order, execution, residual, epilogue, synchronization, canonicalization, memory, compile time, and reuse amortization as named dimensions. |
| `CE-CCP1-E01-011` | Implement analytical, measured, and external evidence | Attach distributions, sample counts, uncertainty, contamination, target/toolchain/build/profile identity, revision, validity, and external evidence references. |
| `CE-CCP1-E01-012` | Implement rejection and dominance explanations | Record correctness, capability, resource, numerical, profile, stale-evidence, cost, and user-policy reasons for every removed alternative. |
| `CE-CCP1-E01-013` | Implement user edits and authority hierarchy | Support add/remove candidate, change fact/objective/cost, replace decomposition, force selection, and unsafe assertions while recording who changed the search space. |
| `CE-CCP1-E01-014` | Implement Planning IR parser, printer, and validator | Add compact textual syntax for alternatives, coverage, atoms, costs, evidence, and selections with unknown extension preservation. |
| `CE-CCP1-E01-015` | Implement Semantic-to-Planning lowering | Create planning problems from typed Semantic IR plus profile environments, preserving operation kinds, numeric policies, fields, generations, and explicit constraints. |
| `CE-CCP1-E01-016` | Deliver the first inspectable candidate search space | Lower a profile-aware relation field to Planning IR containing a conventional fallback and at least one structure-dependent candidate, with complete costs and rejection explanations. |

### E02: discovery, certification, and atoms

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-E02-001` | Import the common JBC atom identity adapters | Rehome namespace-qualified persistent identity and atom species/state contracts, adapting CellShard 64-bit strong IDs to Cellerator persistent identity without pointer hashing or content conflation. |
| `CE-CCP1-E02-002` | Import the overlapping evidence-atlas core | Move proposal membership, provenance, confidence, stability, negative evidence, and exact-rescan status into Cellerator discovery. |
| `CE-CCP1-E02-003` | Import support-signature discovery | Rehome repeated support/degree/signature proposal logic with bounded sketches/top-L candidates and biological-stratum provenance. |
| `CE-CCP1-E02-004` | Import co-support and overlap discovery | Rehome co-support source groups, destination convergence/divergence, overlap handling, and null-baseline statistics as proposal generators. |
| `CE-CCP1-E02-005` | Import relation-motif and operation-trace discovery | Move repeated typed relation motifs, operation sequences, field traces, and cross-operation recurrence discovery into compiler profile evidence. |
| `CE-CCP1-E02-006` | Import trajectory and lineage-pattern discovery | Move recurring trajectory prefixes, branch-local deltas, state neighborhoods, and mutation-horizon evidence without importing model or causal interpretation. |
| `CE-CCP1-E02-007` | Import multimodal and identity-spine discovery | Move shared identity spine plus modality-specific overlay and cross-modal relation proposal logic into Cellerator discovery extensions. |
| `CE-CCP1-E02-008` | Import factor, bicluster, and signature proposal strategies | Rehome experimental factorization/bicluster/signature providers as proposal mechanisms with explicit approximation, confidence, bounded work, and no self-promotion. |
| `CE-CCP1-E02-009` | Import exact rescan and proposal certification | Move independent exact scans that turn proposals into certified logical coverage, canonical maps, and omission/duplicate receipts. |
| `CE-CCP1-E02-010` | Import atom envelope and typed ports | Represent candidate/certified atom identity, species, exact coverage, typed inputs/outputs, planes, dependencies, effects, and lineage in Planning IR extensions. |
| `CE-CCP1-E02-011` | Import atom plane separation | Preserve structure, mutable values, active support, gradients, partials, physical views, evidence, and lineage as distinct planes with independent generations. |
| `CE-CCP1-E02-012` | Import atom requirement/affordance matching | Bind migrated atom records to existing Cellerator requirements, affordances, extents, orders, projections, and target capabilities. |
| `CE-CCP1-E02-013` | Import scalable certification indexes | Replace any bounded linear/quadratic duplicate checks unsuitable for compiler-scale atlases with sorted/radix/hash or caller-owned mark strategies while retaining exact results. |
| `CE-CCP1-E02-014` | Preserve migration provenance in source and artifacts | Attach Migrated-From repository/commit/path/Todo metadata in cold manifests and commit instructions, not hot IR records. |
| `CE-CCP1-E02-015` | Create temporary CellShard compiler compatibility adapters | Forward old compiler-facing CellShard includes to Cellerator contracts where required for preserved tests, with deprecation and explicit retirement gates. |
| `CE-CCP1-E02-016` | Port discovery tests and evidence fixtures | Move or adapt all relevant unit/property/malformed/benchmark fixtures, retaining original expected results and documenting intentional semantic changes. |
| `CE-CCP1-E02-017` | Validate no compiler discovery remains authoritative in CellShard | Audit CellShard main and JBC branches after planned migration: concrete storage/runtime may consume rules, but no retained API may select biological proposals, grammar, basis, or schedules. |
| `CE-CCP1-E02-018` | Freeze the migrated discovery and atom compiler slice | Publish Cellerator-owned discovery providers, exact certification, atom Planning IR, compatibility adapters, provenance, and differential evidence. |

### E03: composition, basis, program, schedule

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-E03-001` | Import typed composition production contracts | Move production inputs/outputs, parameters, exact coverage equations, identity/order/generation rules, effects, costs, and verifier references into Planning IR extensions. |
| `CE-CCP1-E03-002` | Import multi-parent derivation DAGs | Represent atom and operation derivations as typed acyclic graphs with lineage, exact reconstruction, alternative parents, and canonical cycle diagnostics. |
| `CE-CCP1-E03-003` | Import explicit grammar compilation | Move hand-specified typed productions and grammar validation into Cellerator compiler passes over discovered/certified atoms and operations. |
| `CE-CCP1-E03-004` | Import induced grammar as experimental search | Preserve induced-production discovery with bounded candidate growth, evidence/confidence, exact verification, complete cost, and valid no-promotion outcome. |
| `CE-CCP1-E03-005` | Import workload-family representation | Map operation/profile recurrence families, mutation horizons, target classes, and objectives into Cellerator program-planning inputs. |
| `CE-CCP1-E03-006` | Import basis manifest semantics | Move selected atoms/productions, redundancy, membership, budgets, validity, evidence freshness, and objective vectors into Planning IR. |
| `CE-CCP1-E03-007` | Implement no-basis and multiple-basis outcomes | Allow empty/no-benefit basis, several profile-specific bases, and externally offered bases without forcing a universal decomposition. |
| `CE-CCP1-E03-008` | Import superatom promotion | Move composed-unit promotion into the Cellerator planner with exact derivation, deconstruction, profile specificity, complete cost, and experimental status. |
| `CE-CCP1-E03-009` | Import persistent-partial compiler semantics | Move dependency closure, contribution coverage, merge/finalize algebra, generation legality, numerical contract, and amortization decision into Cellerator. |
| `CE-CCP1-E03-010` | Import global operation graph IR | Recast provider-neutral typed operations, effects, atom flow, graph families, rewrites, local fragments, and profile variants as Cellerator program Planning IR. |
| `CE-CCP1-E03-011` | Import cross-operation rewrite and fusion search | Move graph rewrites, shared traversals, persistent orders, common outputs, partial trees, and field-authorized fusion into connected-operation planning. |
| `CE-CCP1-E03-012` | Import portable schedule/ruleset representation | Represent machine-independent operation order, atom requirements, partial tree, canonical recovery, and replay modes as Cellerator Planning IR, not concrete placement. |
| `CE-CCP1-E03-013` | Define the concrete CellShard materialization request seam | Map compiled ruleset requirements to a narrow future request that CellShard can use to materialize, shard, place, and deliver concrete instances in Part Two. |
| `CE-CCP1-E03-014` | Port composition, basis, graph, and schedule tests | Move/adapt test suites with provenance, preserving expected derivations, no-basis cases, exact coverage, and performance baselines. |
| `CE-CCP1-E03-015` | Create semantic differential adapters | Run old CellShard compiler implementations and new Cellerator passes on identical fixtures during migration, comparing canonicalized outputs rather than raw struct bytes when schemas improve. |
| `CE-CCP1-E03-016` | Retire CellShard compiler authority in documentation and namespaces | Leave compatibility aliases only where needed; mark global compiler headers non-authoritative and prevent new registration there. |
| `CE-CCP1-E03-017` | Deliver the profile-to-portable-ruleset slice | Compile representative profiles and a multi-operation Semantic IR program through discovery, grammar, basis, global graph, and portable schedule Planning IR. |
| `CE-CCP1-E03-018` | Freeze Cellerator-owned global compiler migration | Publish migrated contracts, provenance, differential receipts, compatibility aliases, and CellShard application seam. |

### E04: decomposition, candidates, costs, planner

| ID | Title | Mechanism focus |
| --- | --- | --- |
| `CE-CCP1-E04-001` | Adapt semantic geometry acquisition to Planning IR | Expose compile-now, precompiled semantic geometry, external exact cover, and conventional fallback as explicit alternatives with acquisition costs and compatibility. |
| `CE-CCP1-E04-002` | Adapt decomposition portfolios to Planning IR | Import greedy, multilevel, oracle, overlap, device-assisted, and user-provided decompositions as bounded providers rather than hidden global choices. |
| `CE-CCP1-E04-003` | Represent split axes, fragments, halos, and replicas | Map decomposition_v1 exact input/output/contribution coverage, replication, halo read roles, orders, and extent bounds into Planning IR. |
| `CE-CCP1-E04-004` | Represent partial-result trees and algebra selection | Choose legal merge/finalize structures under numerical, determinism, order, resource, and reuse constraints. |
| `CE-CCP1-E04-005` | Adapt multi-extent direct binding and assembly fallback | Offer direct multi-extent execution where candidates support it and explicit profiler-visible assembly as the complete fallback. |
| `CE-CCP1-E04-006` | Adapt candidate catalog v3 providers | Translate stable provider/candidate/projection/capability/operation/numeric/resource descriptors into Planning IR with source-linked preparation hooks. |
| `CE-CCP1-E04-007` | Implement custom candidate registration | Allow source, inline IR, external libraries, and migrated providers to add candidates with partial protocol implementations and explicit opaque behavior where necessary. |
| `CE-CCP1-E04-008` | Implement candidate inclusion, exclusion, and forcing | Apply source/pipeline/user edits in the defined authority hierarchy while retaining diagnostics for impossible or dominated choices. |
| `CE-CCP1-E04-009` | Adapt external global cost exchange | Expose storage, movement, replication, invalidation, latency, throughput, and application-supplied costs through generic callbacks/IR evidence without a CellShard dependency. |
| `CE-CCP1-E04-010` | Implement complete-cost normalization | Normalize units, distributions, confidence, recurrence, amortization horizons, and missing phases across analytical, measured, cached, and external evidence. |
| `CE-CCP1-E04-011` | Adapt transition and connected-operation costs | Model order transforms, materialization, shared traversal, fusion, common output ownership, canonicalization, and field-boundary effects between operation alternatives. |
| `CE-CCP1-E04-012` | Implement planner portfolio dispatch | Allow built-in exact/heuristic planners, user replacement planners, externally selected plans, and deterministic fallback under bounded time/memory budgets. |
| `CE-CCP1-E04-013` | Implement profile-family plan variants | Select one or more plans for named profile alternatives, share compatible artifacts, and emit bounded runtime selection requirements without duplicating full semantic programs. |
| `CE-CCP1-E04-014` | Implement planning cache and invalidation | Key plans by semantic fingerprint, profile/evidence revision, structure epoch, order, target class, toolchain, constraints, and planner revision at the earliest reusable stage. |
| `CE-CCP1-E04-015` | Expose complete planning reports | Report considered alternatives, exact coverage, costs, evidence freshness, rejection/dominance, selected source, forced edits, and fallback. |
| `CE-CCP1-E04-016` | Benchmark planning scalability and boundedness | Measure time, peak memory, candidate count, exact certification, search frontier, and quality versus oracle on scalable synthetic and biological fixtures. |
| `CE-CCP1-E04-017` | Deliver source-to-selected-plan vertical slice | Compile a profile-bound relation and a two-operation field from source through Semantic IR, Planning IR, decomposition, candidates, complete cost, and selected portable ruleset. |
| `CE-CCP1-E04-018` | Freeze the public planning compiler interface | Publish provider, planner, cache, report, custom candidate, external cost, and force-control contracts used by Realization IR. |
