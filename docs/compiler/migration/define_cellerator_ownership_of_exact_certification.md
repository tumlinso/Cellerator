# Cellerator ownership of exact certification

Todo `CE-CCP1-A03-002` assigns all compiler-semantic exact certification to
Cellerator Planning IR and its independent validators. The reviewed CellShard
source is pinned to `b9749ad3e5146a04f847533d8c6f1a54146aed20`.

The sixteen headers under `include/CellShard/compiler/certification` map
one-for-one to the Cellerator contracts recorded in
`define_cellerator_ownership_of_exact_certification_v1.hh`. Cellerator owns
canonical domain and member identity, exact entity and relation-edge coverage,
duplicate and omission proof, contribution ownership, residual accounting,
read-only halos, physical replicas, inverse canonical recovery, multimodal and
trajectory maps, partial-result algebra compatibility, and dependency closure.

Certification is independent of proposal generation. A proposal cannot be
promoted unless every prerequisite is true, and no verifier may reuse a
proposal builder's approximate membership as its proof. Global identities stay
64-bit; compact local widths are permitted only with a checked inverse map.
Scalable validators use count/scan/fill, caller-owned marks, sorting or bounded
hash structures rather than quadratic duplicate scans.

Existing Cellerator `execution/joint_compiler/logical_coverage_v1.hh` and
`coverage_roles_v1.hh` are the transitional exact-coverage ABI; CSG1 geometry
and `geometry/relation_cover.hh` remain compatibility evidence. This decision
does not freeze their current names as the final Planning IR and does not move
CellShard storage, persistence, placement, or runtime ownership.

The focused gate inventories all sixteen CellShard certification headers,
checks every row is uniquely mapped, compares the ownership decision with the
current Cellerator exact-coverage and relation-cover contracts, and exhaustively
proves that omission of any prerequisite prevents certification.

- Gate: `CE-CCP1-A03-002-GATE`
- Source roots: `include/CellShard/compiler/certification`,
  `include/Cellerator/execution/joint_compiler`, and
  `include/Cellerator/geometry/relation_cover.hh`
- Compatibility: existing exact identities, coverage, residual, halo,
  ownership, recovery and dependency assertions are preserved; storage-facing
  serialization remains a CellShard consumer interface.
