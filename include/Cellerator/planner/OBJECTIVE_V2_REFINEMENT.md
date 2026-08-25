# Objective V2 refinement integration

CE-ARCH-87 connects the replaceable Objective V2 calibration to CellPack's
held-out alternating refinement without changing CPK1 or making a predicted
score part of biological identity.

The v2 refinement packet carries raw measured evidence: forward and transpose
elapsed totals with repeat counts, active interactions, partition-cut edges,
and bootstrap median/MAD with a sample count. Profile identity and evidence
revision are explicit. Training and held-out packets must use equivalent repeat
and bootstrap policies; stale or under-sampled profiles are rejected.

`make_objective_v2_refinement_guidance` produces workload weights as follows:

- forward and transpose means use caller-declared workload frequencies;
- persistent preparation is amortized by expected reuse;
- activity uses the measured CE-ARCH-76 useful-interaction coefficient;
- partition cuts are charged only when the caller supplies a separately
  measured per-edge cost and evidence identity;
- bootstrap MAD is an explicit stability penalty.

The alternating controller continues to accept or roll back by held-out total
cost. Predictions do not override measured profiles. Exact-surrogate arithmetic,
held-out rejection, bootstrap stability, evidence invalidation, and legacy
statistical-validation packets are covered by focused regression tests. Future
calibrations or candidate families can replace the guidance record without
changing frozen packing semantics.
