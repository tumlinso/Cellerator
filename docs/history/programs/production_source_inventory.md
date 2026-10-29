# CE-PTR live production inventory

## Snapshot

This inventory is the before-migration production snapshot for CE-PTR-01 at
base commit `76711a379f00a3c52a177c58d6acce796ca2a112` on 2026-08-29. It covers
the enforced production roots `include/Cellerator/` and `src/`. The permanent
gate records the exact path, controlled family, occurrence ceiling, rationale,
and responsible migration lane for all remaining debt.

The lexical snapshot contains 236 controlled owning-family spellings in 36
production files:

| Family | Occurrences |
| --- | ---: |
| `std::vector` | 212 |
| `std::map` | 4 |
| `std::unordered_map` | 3 |
| `std::set` | 7 |
| `std::unordered_set` | 2 |
| `std::priority_queue` | 2 |
| `std::shared_ptr` | 6 |

There are no allowlisted production occurrences of `std::deque` or
`std::list`. Tests, benchmarks, compatibility evidence, examples outside the
public Cellerator include tree, and CelleraTorch are classification boundaries,
not proof of production suitability. Public trajectory examples currently
under `include/Cellerator/examples/` are production-enforced because downstream
consumers can import them.

## Classified migration inventory

Cardinality/bounds and identity statements below are requirements to preserve,
not permission to infer missing semantics from container length.

| Subsystem and live paths | Accidental representation | Semantic structure | Lifetime | Cardinality and bounds | Memory domain | Hot/cold | Disposition | Allowlist rationale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Static geometry planner: `geometry/planner.hh`, `geometry/planner.cc` | Parallel vectors | Immutable permutations, signatures, modules, row groups, and regions | Structure epoch / prepared plan | Counts derived from validated axis identities and compiled geometry | Host build, relocatable image, host/device rebound view | Preparation and hot consumers | Image + flat relations | CE-PTR-03 freezes one versioned static-plan image; 18 vector spellings may only decrease. |
| Geometry packing and gating: `geometry/pack.hh`, `geometry/pack.cc`, `geometry/gating.hh`, `geometry/gating.cc`, `geometry/gating_cuda.cuh`, `geometry/gating_cuda.cu` | Coordinate vectors, nested routes, active-region lists | Direct physical-projection compilation, ordered region replay table, compiled route relation | Build / structure epoch | Exact count-scan-fill requirements; region and coordinate counts validated before fill | Host and device build workspaces; relocatable image | Preparation | Eliminate streams + image/relation | CE-PTR-04 replaces 13 vector spellings without changing region order, route identity, or replay semantics. |
| Layout and candidate construction: `geometry/layout_metrics.hh/.cc`, `geometry/layout_selector.hh/.cc`, `geometry/candidate_relation.cc` | Metrics vectors, selection scratch, pair objects | Region metrics table, region selection table, canonical packed feature-pair relation | Build / structure epoch | One record per validated region; exact unique pair count | Host workspace and image, device-capable relation | Preparation | Tables + workspace + relation | CE-PTR-04 retires 10 vector spellings through exact workspaces and packed keys. |
| Packing optimizer state: `src/geometry/optimizer_state.hh`, `src/geometry/optimizer.cc` | Per-block vectors, copied member lists, node maps/sets, deep rollback | Bounded block-member slab, direct feature-slot map, union cache, proposal tables, generation marks, mutation journal | Prepared optimizer state and per-batch workspace | Feature count explicit; widths 8/16/32 evaluated; overflow and proposal capacities queried | Host initially; device-compatible explicit workspaces where promoted | Preparation, algorithmically hot | Tables/workspaces + algorithm migration | CE-PTR-05/06 own 41 vector, 4 map, and 7 set spellings. Exact optimizer result and stable feature identity are mandatory. |
| Statistical validation: `statistical_validation.cc`, `record_statistical_validation.cc` | Nested vectors and hash membership | Sorted group-row relation, split order, row-unit map, packed edge membership, decode workspace | Validation preparation / repeated trials | Exact group, row, split, edge, bootstrap, and null-model counts | Host workspaces | Cold-to-warm validation path | Relations + workspace | CE-PTR-08 retires 14 vector, 1 unordered-map, and 2 unordered-set spellings while preserving statistics, deterministic splits, and provenance. |
| Sampling: `compute/sampling.hh`, `dataset/sampling.cc`, `sampling_materialization.cc` | Result vectors, selection queues, reconstructed CSR | Sample-selection image, stratum table, bounded selector, sampled-CSR image | Prepared sample / structure epoch | Population and requested K explicit, including million and 11-million rows and exact maximum K | Host build, optional device workspace, pointer-free image | Preparation | Images + bounded table/workspace | CE-PTR-07 owns 30 vector and 2 priority-queue spellings; stable row identity, deterministic reproduction, ties, and provenance remain exact. |
| Gene support/candidate pipeline: `candidate_discovery/gene_candidate_discovery.cc` | Host-owned growable candidate staging | Prepared device-resident support-to-unique-pair pipeline | Prepared plan and launch workspace | Support geometry, candidate upper bound, CUB scratch, and terminal result count queried | Device-resident with explicit pinned terminal materialization | Hot GPU pipeline | Device image/workspace | CE-PTR-09 retires 5 vector spellings and prevents intermediate host ownership, extra transfers, and global synchronization. |
| Physical projection builders: `physical_feature_major.cc`, `physical_transpose.cc` | Construction/validation vectors | Exact caller workspace for retained FMP1/CTP1 pointer-free images | Projection build | Exact section counts, bytes, alignment, and capacity queried before fill | Host/device construction workspace and rebound image | Preparation | Workspace; preserve image ABI | CE-PTR-14 owns 4 vector spellings; it must not mutate existing image schemas or direct rebound semantics. |
| Trajectory graph operators: `record_table.cuh`, `slab_index.cuh`, `forward_candidates.cuh`, `forward_prune.cuh`, `branch_detect.cuh`, `incremental_insert.cuh`, `supernode_reduce.cuh` | Many growable/nested vectors and hash DAG aggregation | Trajectory record SoA, embryo spans, bounded K edge table, child CSR, tree/member/Euler images, slab relation, packed DAG edges | Structure epoch plus launch workspace | K=4/8 primary bounded cases; exact node/edge/member counts via two-pass construction | Host/device images and caller workspaces | Preparation and hot GPU graph execution | Images + fixed tables + relations | CE-PTR-10 owns 70 vector and 1 unordered-map spellings; stable node/time/embryo identity and deterministic traversal order remain explicit. |
| Public trajectory examples: `trajectory_build.cuh`, `trajectory_query.cuh` | Materialized vectors | Direct member/subtree spans and caller-provided result path | Caller boundary | Query/build capacities explicit | External/caller host memory | Cold adapter | Views / example migration | CE-PTR-10 owns 7 vector spellings because these headers are publicly importable; no core owner may be reintroduced. |
| Forward-neighbor workflow: `fn_index.hh`, `forward_neighbors.cu`, `cuvs_sharded_knn.cu` | Shared ownership, hash lookup, growable shard staging | Prepared low-level scoring/refinement views separated from downstream index/storage/workflow ownership | Prepared kernel versus downstream durable index | K, shard count, residency, and transfer bounds explicit | Device workspaces and downstream-owned storage | Hot kernels plus cold boundary | Move + redesign | CE-PTR-12 owns 1 vector, 2 unordered-map, and 6 shared-pointer spellings; CellShard/storage and BioPrep policy must not return to core. |
| Runtime buffer ownership: `runtime/device_buffer.cuh` | Shared-owned allocation and blocking helper copies | Execution-session allocation handles, leases, raw typed views, explicit stream copies | Session / prepared / launch, separately | Byte size, alignment, device, generation, and stream binding explicit | Device | Hot runtime | Compatibility keep; prohibit new prepared use | CE-PTR-13 integrates the session substrate but retains 2 shared-pointer spellings for live legacy consumers; no second runtime or generic buffer owner is allowed. |

## Final reconciliation

CE-PTR-15 reconciled the live tree after integration at commit `08ae74f`.
The strict gate now accounts for 169 controlled spellings in 27 production
files: 165 `std::vector`, two `std::priority_queue`, and two
`std::shared_ptr`. This is a reduction of 67 spellings (28.4 percent) and nine
files from the baseline. All production `std::map`, `std::set`,
`std::unordered_map`, and `std::unordered_set` debt is gone. The remaining
path-family ceilings are exact live counts and may only decrease.

Remaining controlled exceptions are transitional and classified by owner:

| Migration owner | Final controlled occurrences | Bounded disposition |
| --- | ---: | --- |
| CE-PTR-03 static planner | 18 vectors | Cold frozen-plan construction compatibility around CPI1/CPK1/CPE2. |
| CE-PTR-04 geometry construction | 15 vectors | Cold count-scan-fill construction tables and compatibility results; no hot runtime ownership. |
| CE-PTR-05/06 optimizer | 22 vectors | Prepared optimizer state/results; repeated proposal tables and mutation journal use explicit slabs/views. |
| CE-PTR-07 sampling | 30 vectors, 2 priority queues | Cold deterministic sample/image construction and compatibility result surfaces. |
| CE-PTR-09 resident pipeline | 5 vectors | Host reference/terminal compatibility around the device-resident pipeline. |
| CE-PTR-10 graph and public examples | 71 vectors | Compatibility image builders, fixed-width relations, and caller-facing examples; sealed native views remain distinct. |
| CE-PTR-12 cuVS adapter | 1 vector | Cold external-library shard staging. |
| CE-PTR-13 runtime compatibility | 2 shared pointers | Legacy `device_buffer` owner only; forbidden as a new prepared-path pattern. |
| CE-PTR-14 projection builders | 3 vectors | Cold FMP1/CTP1 validation/build compatibility around exact workspace contracts. |

Passing the lexical gate does not mean every generic helper is gone.
`host_buffer.hh` remains a grow-by-reallocation compatibility sequence used by
the cuVS boundary. The trajectory compatibility workspace retains allocation
and blocking-copy helpers for batch construction. These are documented
non-lexical debt, not sealed allocation-free execution, and must not be moved
into prepared runtime ownership. Removing live compatibility infrastructure
without migrating its callers was explicitly out of scope for final
convergence.

## Known non-lexical or secondary surfaces

The migration map also classifies the following even when the controlled-token
snapshot is zero or the current owner is not one of the gate families:

- frozen packing plans converge with CPK1/CPE2 image precedents under
  CE-PTR-03 without mechanically changing a sound ABI;
- `host_buffer.hh` remains a generic grow-by-reallocation compatibility
  sequence; its live cuVS consumers prevent deletion without a separately
  scoped caller migration;
- new runtime and prepared paths use session workspaces and explicit stream
  operations; legacy `device_buffer` and trajectory batch helpers retain
  bounded blocking compatibility behavior and are not hot-path evidence;
- raw exact-search public device views remain valid while CE-PTR-11 evaluates
  K-specialized internal register/shared-memory representations on V100;
- CPK1, FMP1, CTP1, and CPE2 remain versioned pointer-free evidence; only their
  generic construction scratch is in scope where measured useful;
- CelleraTorch STL remains at the Torch ABI edge and converts immediately to
  native typed views; native targets remain Torch-independent.

## Enforcement and reconciliation

Run:

```text
python scripts/check_no_inappropriate_core_stl.py
```

The normal gate permits debt removal and rejects new paths, new families, or an
increase above any exact path/family ceiling. `--strict-stale` is the final
convergence mode: it also requires removed debt to be deleted from the
allowlist. CE-PTR-15 completed that reconciliation against live source,
measured exceptions, repository-boundary moves, and all completed lane
evidence; strict mode is now a required final gate.
