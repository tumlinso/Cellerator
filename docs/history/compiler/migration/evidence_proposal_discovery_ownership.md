# Cellerator ownership of evidence and proposal discovery

Todo `CE-CCP1-A03-001` freezes the ownership boundary for the CellShard JBC
evidence atlas and discovery providers. The source review is pinned to
CellShard commit `b9749ad3e5146a04f847533d8c6f1a54146aed20` (the Cellerator
submodule commit at this Todo's start).

## Ownership decision

Cellerator owns evidence semantics, evidence queries, proposal discovery,
algorithm provenance, stability/confidence assessment, negative evidence and
the compiler policies that rank proposals. These APIs move into
`Cellerator::compiler::profile::evidence` and
`Cellerator::compiler::profile::discovery::*`. A proposal is uncertain input
to planning. It never authorizes execution and is not an exact coverage
certificate.

CellShard retains concrete storage and application duties: persistence and
transport of opaque evidence images, durable catalog placement, fetch/cache,
chunk addressing, and application/runtime consumption. Its adapter must expose
the explicit Cellerator semantic identity and generation; it may not synthesize
identity from a pointer, byte offset, file path, chunk address, allocation
handle, process-local ordinal, or payload digest.

Exact execution certification remains a separate Cellerator compiler authority.
Discovery providers may propose members, supports, neighborhoods, alignments,
orders and costs, but only independent exact certification may establish
canonical identity, complete coverage, residuals, halos, contribution owners,
replicas and dependency closure.

## Source-linked API map

| CellShard source family | Cellerator destination | Migration rule |
|---|---|---|
| `compiler/evidence/*` | `compiler::profile::evidence` | Rehome record, atlas, query, provenance, confidence, strata and negative-evidence semantics; split byte-image I/O into a CellShard adapter. |
| `discovery/bicluster/*` | `compiler::profile::discovery::bicluster` | Rehome proposal providers and exact-rescan inputs; certification output stays separate. |
| `discovery/co_support/*` | `compiler::profile::discovery::co_support` | Rehome association, affinity, sampling, stability and rescan evidence; CellShard may store opaque images. |
| `discovery/factor_topic/*` | `compiler::profile::discovery::factor_topic` | Rehome factor/topic proposal evidence; preserve external-provider IDs exactly. |
| `discovery/motif/*` | `compiler::profile::discovery::motif` | Rehome motif mining and recurrence evidence as proposals. |
| `discovery/multimodal/*` | `compiler::profile::discovery::multimodal` | Rehome alignment and cross-modal proposals; promotion still requires independent certification. |
| `discovery/operation_trace/*` | `compiler::profile::discovery::operation_trace` | Rehome trace analysis; retain a narrow field-by-field CellShard compatibility adapter. |
| `discovery/overlap/*` | `compiler::profile::discovery::overlap` | Rehome overlap proposal and stability/cost policy. |
| `discovery/sequence_compat/*` | `compiler::profile::discovery::sequence` | Rehome compiler proposal semantics; retain only the narrow Baseplane/CellShard coordinate adapter. |
| `discovery/support_signature/*` | `compiler::profile::discovery::support_signature` | Rehome MinHash/LSH and exact-rescan proposal machinery; preserve negative evidence. |
| `discovery/trajectory/*` | `compiler::profile::discovery::trajectory` | Rehome lineage/window/transition proposal semantics. |

The machine-readable counterpart is
`include/Cellerator/compiler/migration/define_cellerator_ownership_of_evidence_and_proposal_dis_v1.hh`.
Its thirteen prefix rows cover the eleven public evidence/discovery families and
the two implementation prefixes. The focused gate checks completeness against
the pinned Git tree and proves that the proposal identity contains only five
explicit semantic `uint64_t` fields and cannot authorize execution.

## Compatibility and deliberate differences

The current JBC `evidence_identity_v1` two-field identity, proposal-only
disposition, atlas sorting/query behavior, negative evidence, generations and
provider-specific validation remain behavior to preserve. The target uses
Cellerator namespaces and Cellerator-owned identity vocabulary rather than
freezing `CellShard::compiler` as the public semantic ABI. Pointer-bearing
atlas views remain ephemeral bindings; pointers are not serialized or used as
semantic identity. CellShard image packing and placement remain application
interfaces, not discovery authority.

## Evidence and gate

- Source commit: `b9749ad3e5146a04f847533d8c6f1a54146aed20`
- Source roots reviewed: `include/CellShard/compiler/evidence`,
  `include/CellShard/compiler/discovery`, `src/compiler/evidence`, and the
  (currently empty) `src/compiler/discovery` prefix.
- Gate: `CE-CCP1-A03-001-GATE`
- Command: `ctest --test-dir build --output-on-failure -R '^ce_ccp1_a03_001$'`
- Focused executable: compile and run
  `tests/compiler/a03/define_cellerator_ownership_of_evidence_and_proposal_dis_test.cc`
  with the CellShard Git directory and pinned commit as arguments.

This receipt defines migration ownership only. It neither moves production JBC
code nor changes CellShard persistence, runtime, or wire formats.
