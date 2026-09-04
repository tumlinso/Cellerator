# JBC migration order and thin-waist dependencies

This graph turns the source and duplicate inventories into a Part One migration
order. An edge `A -> B` means B cannot become the authoritative implementation
until A is frozen and source-linked. It deliberately reverses the historical
assumption that CellShard owns the compiler: compiler-semantic nodes migrate to
Cellerator, while durable storage, portable command materialization, topology,
transport, and residency remain CellShard responsibilities.

## Nodes

<!-- DAG-NODES-BEGIN -->
| Node | Contract or source group | Authority after Part One | Movement rule / temporary adapter |
|---|---|---|---|
| MIG-F01 | Namespaced persistent identity and typed structure/value generations | Cellerator | Freeze first; `MIG-A01` translates existing CellShard two-u64 and strong IDs fieldwise. |
| MIG-F02 | Exact logical coverage, roles, axes, orders, and structure epoch | Cellerator | Requires F01; `MIG-A02` preserves the current pointer-first CellShard coverage reference. |
| MIG-F03 | Public CEIR extension points for atoms, typed ports, planes, operations, and lowering stages | Cellerator | Requires F01/F02; `MIG-A04` translates legacy graph IDs until graph consumers move. |
| MIG-F04 | Profile, evidence, provenance, confidence, freshness, and negative-result contracts | Cellerator | Requires F01/F02; `MIG-A03` reads the frozen 80-byte CellShard evidence record. |
| MIG-F05 | Complete-cost vector, resource model, Pareto frontier, and bounded planner exchange | Cellerator | Requires F01/F03/F04; `MIG-A05` reports CellShard storage/transport terms without owning selection. |
| MIG-M01 | Common atom envelope, identity classes, semantic planes, lineage, requirements, and affordances | Cellerator | Move after F01-F03; keep only application/materialization references in CellShard. |
| MIG-M02 | Evidence atlas, indexes, merges, statistics, and cross-workload evidence | Cellerator | Move after F04; proposal records remain non-certifying. |
| MIG-M03 | Support, co-support, bicluster, overlap, motif, factor, trajectory, multimodal, sequence, and trace discovery | Cellerator | Move only after CEIR/profile contracts plus atlas and certification inputs exist. |
| MIG-M04 | Independent exact certification and contribution ownership | Cellerator | Move after F01-F03; exact certification is separate from proposal evidence. |
| MIG-M05 | Composition operations, production registry, explicit grammar, and derivation DAG | Cellerator | Move after CEIR, certification, and planner contracts; merge the two current DAG validators. |
| MIG-M06 | Basis selection, fallback, memberships, superatom lifecycle, and promotion evidence | Cellerator | Move after atlas, certification, composition, and cost contracts. |
| MIG-M07 | Partial-result algebra, state families, persistent partial semantics, and compatibility | Cellerator | Move after coverage, CEIR, and planner contracts; `MIG-A06` keeps the storage envelope in CellShard. |
| MIG-M08 | Logical operation graph, provider registry, graph recipe, and global semantic schedule | Cellerator | Move after CEIR, certification, and complete-cost reporting. |
| MIG-M09 | Exact distributed semantic certificate | Cellerator | Move after exact coverage and global semantic scheduling; route realization remains separate. |
| MIG-A01 | Identity/strong-ID compatibility adapter | CellShard compatibility | Temporary fieldwise adapter; no hashing, folding, pointer identity, or new semantic IDs. |
| MIG-A02 | Logical-coverage reference adapter | CellShard compatibility | Temporary pointer-first reference to canonical Cellerator coverage and validation result. |
| MIG-A03 | Evidence-record compatibility reader | CellShard compatibility | Temporary reader for frozen atlas records; converts identities and preserves proposal-only disposition. |
| MIG-A04 | Graph/provider strong-ID adapter | CellShard compatibility | Temporary conversion for graph records until CEIR consumers replace local compiler IDs. |
| MIG-A05 | External-cost reporter | CellShard boundary | Permanent narrow reporter for measured storage, materialization, movement, topology, and transport prices. |
| MIG-A06 | Persistent-partial storage adapter | CellShard boundary | Permanent versioned payload envelope; no independent algebra or coverage policy. |
| MIG-A07 | Schedule/materialization adapter | CellShard boundary | Permanent semantic-schedule-to-command lowering seam; no provider selection or re-planning. |
| MIG-R01 | Atom store, generations, codecs, frames, publication, recovery, and GC | CellShard | Retain; consume canonical identity/coverage/atom metadata through A01/A02. |
| MIG-R02 | Portable launch/copy/barrier/order/publish command artifact and route binding | CellShard | Retain; materialize the Cellerator schedule/certificate through A07. |
| MIG-R03 | Topology, I/O, transport, residency leases, staging, and runtime command execution | CellShard | Retain; depends on store and portable command artifacts, not compiler ownership. |
| MIG-V01 | Cross-boundary independent semantic and compatibility verification | Cellerator integration | Prove exact coverage, costs, generation freshness, old artifact reads, and no CellShard re-planning. |
| MIG-V02 | Adapter retirement/freeze decision | Cellerator integration | After V01 and runtime validation, retire A01-A04 when unused; version and freeze permanent A05-A07. |
<!-- DAG-NODES-END -->

## Directed dependency edges

<!-- DAG-EDGES-BEGIN -->
| Prerequisite | Consumer | Reason |
|---|---|---|
| MIG-F01 | MIG-F02 | Coverage identities and structure epochs need canonical identity/generation types. |
| MIG-F01 | MIG-F03 | Every public CEIR atom, port, plane, and stage needs stable identity. |
| MIG-F01 | MIG-F04 | Evidence subjects, sources, builds, devices, and datasets need stable identity. |
| MIG-F01 | MIG-A01 | The legacy identity adapter targets the frozen canonical representation. |
| MIG-F02 | MIG-F03 | CEIR atoms and operations bind typed exact coverage and order. |
| MIG-F02 | MIG-F04 | Evidence must distinguish proposal membership from certified exact coverage. |
| MIG-F02 | MIG-A02 | The CellShard coverage reference targets the frozen canonical validator. |
| MIG-F01 | MIG-F05 | Cost observations and pricing epochs require stable identities. |
| MIG-F03 | MIG-F05 | Planner candidates and resource records refer to CEIR operations/stages. |
| MIG-F04 | MIG-F05 | Measured costs require provenance and freshness. |
| MIG-F05 | MIG-A05 | CellShard reports external cost terms in the canonical vector. |
| MIG-F01 | MIG-M01 | Atom envelope identities and generations must already be canonical. |
| MIG-F02 | MIG-M01 | Atoms require certified exact coverage and typed order. |
| MIG-F03 | MIG-M01 | Ports, planes, species, and affordances are CEIR extension records. |
| MIG-F04 | MIG-M02 | Atlas records use canonical evidence identity, provenance, and freshness. |
| MIG-F01 | MIG-M04 | Certificates and contribution owners need canonical identities. |
| MIG-F02 | MIG-M04 | Certification proves canonical exact coverage. |
| MIG-F03 | MIG-M04 | Certification binds typed CEIR atom and operation semantics. |
| MIG-F03 | MIG-M03 | Discovery emits typed CEIR atom proposals. |
| MIG-F04 | MIG-M03 | Discovery must emit auditable positive and negative evidence. |
| MIG-M02 | MIG-M03 | Providers query and extend the canonical evidence atlas. |
| MIG-M04 | MIG-M03 | Promotion requires independent exact certification. |
| MIG-F03 | MIG-M05 | Productions and symbols consume/produce typed CEIR interfaces. |
| MIG-F05 | MIG-M05 | Composition choices include complete derivation and realization cost. |
| MIG-M04 | MIG-M05 | Grammar productions preserve certified contribution coverage. |
| MIG-M02 | MIG-M06 | Basis selection consumes measured proposal evidence. |
| MIG-M04 | MIG-M06 | Selected basis atoms must be independently certified. |
| MIG-M05 | MIG-M06 | Basis and superatoms consume explicit productions and derivations. |
| MIG-F05 | MIG-M06 | Basis promotion is decided on complete cost and budgets. |
| MIG-F02 | MIG-M07 | Partial contributions bind exact coverage. |
| MIG-F03 | MIG-M07 | Partial state/algebra identities are typed compiler operations. |
| MIG-F05 | MIG-M07 | Persistence and combination decisions require complete cost. |
| MIG-M07 | MIG-A06 | The CellShard payload envelope adapts canonical partial semantics. |
| MIG-F03 | MIG-A04 | Legacy graph IDs translate to frozen CEIR identities. |
| MIG-F03 | MIG-M08 | Logical graph nodes and providers are CEIR compiler records. |
| MIG-M04 | MIG-M08 | A schedule may select only coverage-certified candidates. |
| MIG-A05 | MIG-M08 | Global selection consumes CellShard external costs through the narrow reporter. |
| MIG-F02 | MIG-M09 | The distributed certificate attests canonical exact coverage. |
| MIG-M08 | MIG-M09 | Certification binds the finalized global semantic schedule. |
| MIG-M01 | MIG-R01 | Stored atoms carry the canonical semantic envelope. |
| MIG-A01 | MIG-R01 | Existing store identity fields are read through the compatibility adapter. |
| MIG-A02 | MIG-R01 | Stored coverage references preserve canonical validation. |
| MIG-M08 | MIG-A07 | Materialization begins from a finalized Cellerator semantic schedule. |
| MIG-M09 | MIG-A07 | Materialization preserves the exact distributed certificate. |
| MIG-A07 | MIG-R02 | The adapter emits the concrete portable command artifact. |
| MIG-R01 | MIG-R03 | Runtime residency and I/O consume durable atom-store records. |
| MIG-R02 | MIG-R03 | Runtime executes concrete portable commands and route bindings. |
| MIG-M03 | MIG-V01 | Verification exercises migrated discovery and proposal evidence. |
| MIG-M06 | MIG-V01 | Verification includes grammar/basis fallback and promotion decisions. |
| MIG-M07 | MIG-V01 | Verification checks partial reconstruction and generation freshness. |
| MIG-M09 | MIG-V01 | Verification checks distributed exact coverage. |
| MIG-R03 | MIG-V01 | Verification reaches concrete storage, transport, residency, and execution. |
| MIG-V01 | MIG-V02 | Adapter retirement requires successful cross-boundary validation. |
| MIG-A01 | MIG-V02 | Temporary identity adapter remains until its consumers are gone. |
| MIG-A02 | MIG-V02 | Temporary coverage adapter remains until its consumers are gone. |
| MIG-A03 | MIG-V02 | Frozen evidence records need a reader until compatibility expires. |
| MIG-A04 | MIG-V02 | Legacy graph IDs need translation until old artifacts expire. |
| MIG-A05 | MIG-V02 | Permanent external-cost seam is frozen after validation. |
| MIG-A06 | MIG-V02 | Permanent partial-storage seam is frozen after validation. |
| MIG-A07 | MIG-V02 | Permanent schedule/materialization seam is frozen after validation. |
<!-- DAG-EDGES-END -->

## Migration waves

1. **Thin waist:** F01 and then F02-F04. These are the only roots for semantic
   migration; no CellShard compiler directory moves before its required root.
2. **Planning waist:** F05 plus A01-A05. Adapters allow old artifacts and
   CellShard measurements to remain usable without granting them compiler
   authority.
3. **Compiler rehome:** M01/M02/M04 first; then M03/M05/M07/M08; then M06/M09.
   Independent provider source can move in parallel once its incoming edges are
   satisfied.
4. **Application boundary:** R01-R03 remain in CellShard and consume finalized
   compiler artifacts through A05-A07. They are not migration targets.
5. **Validation and retirement:** V01 proves both standalone compiler semantics
   and the embedded application path; V02 removes only unused temporary
   adapters and freezes the three permanent narrow seams.

The edge table is acyclic by construction and is mechanically checked by the
A02-008 evidence test. It also prevents direct movement where the required
identity, CEIR extension, profile/evidence, or planner contract is absent.

## Source anchors

The thin waist is implemented today by
`include/Cellerator/execution/joint_compiler/persistent_identity_v1.hh`,
`logical_coverage_v1.hh`, the requirement/affordance and fragment contracts,
`include/Cellerator/planner/external_cost/`, and
`include/Cellerator/profiling/joint_compiler/execution_export_v2.hh`.

The migration groups and retained boundary are grounded in CellShard's
`include/CellShard/compiler/atom/common_atom_v1.hh`,
`compiler/evidence/evidence_atlas_v1.hh`,
`compiler/certification/exact_atom_certificate_v1.hh`,
`compiler/composition/derivation_dag_v1.hh`,
`compiler/grammar/derivation_dag_v1.hh`, `compiler/basis/manifest.hpp`,
`compiler/partial/partial_atom_v1.hh`, `compiler/graph/graph_recipe.hh`,
`compiler/schedule/distributed_certificate.hh`,
`compiler/schedule/portable_artifact.hh`,
`artifact/atom_store/root_manifest_v1.hh`, and
`runtime/v2/residency_lease.hh`.
