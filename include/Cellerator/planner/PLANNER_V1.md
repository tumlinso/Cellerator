# Cellerator end-to-end planner v1

The planner selects the fastest correct complete execution strategy, including
conventional fallbacks. It is downstream of the operation-core registry and
upstream of preparation; it does not make kernel-only timing authoritative.

Planning identity is deliberately factored into mathematical problem, the
bounded deterministic set of persistent structure identities and epochs,
persistent semantic geometry/order/partition identities, device performance
class, runtime/kernel/driver build, and policy/reuse keys. Runtime identity
slots and generations are process-local aliases and never enter durable
evidence. Each candidate carries its current runtime handles separately.

Candidate selection has seven bounded stages:

1. Reject malformed or correctness-ineligible candidates.
2. Reject candidates violating determinism, graph, persistent-memory, or
   transient-memory policy.
3. Rank legal candidates by a cheap complete-workflow analytical estimate.
4. Keep a bounded top-k shortlist.
5. Reuse only a fresh, sufficiently confident measured cache winner that
   remains legal under every factored key.
6. Otherwise, empirically measure a bounded number of shortlisted candidates
   on the actual structure when reuse and workload size justify tuning.
7. Persist the winner's persistent projection identity/kind/schema/variant,
   confidence, practical tolerance, evidence revision, and explanation; stale
   evidence is rejected rather than silently reused. Lookup matches that
   persistent projection to a legal candidate and uses the candidate's current
   runtime handle.

Connected-plan keys additionally name the persistent partition hierarchy.
Nested membership, ancestry, or placement changes therefore invalidate a
cached connected path without conflating hierarchy with biological geometry.

Confidence depends on sample count, measurement spread, and separation from the
runner-up within the declared tolerance. If every empirical measurement is
failed or contaminated, conservative policy may select the best legal
analytical candidate, but no failed empirical winner is persisted. Cache-store
failure is diagnostic and never falsely reported as persisted evidence.

The total cost charges host preparation, semantic packing divided by structure
reuse, projection construction and backend preparation divided by projection
reuse, H2D, dynamic input packing, kernel, epilogue, order transform,
synchronization, communication, and D2H. Persistent/transient bytes and transfer
bytes are retained beside timings. One-shot and tiny workloads skip empirical
tuning unless policy explicitly enables it. A conventional CSR, SELL, BSR,
valid Blocked-ELL, vendor, or dense candidate receives no negative bias and
must win when its measured end-to-end cost is lower outside the recorded
practical tolerance.

The `objective_v2_*` records define a new operation-aware CP-BP optimization
input. They account for storage, partial occupancy, feature reuse, lane/row
imbalance, module priors, dense width, dtype context, registers, shared memory,
epilogue and order costs, transpose locality, quantization outliers, expected
reuse, and partition cuts. This schema does not modify
`packing_exact_objective_kind`, `row_active_block_references`, frozen v1 plans,
or CPK1 bytes. Biological modules are optional credits, not hard partitions;
future coarsening, sketches, graph partitioning, or a device-resident exact
evaluator activate only after measured total-cost benefit.

The historical `cellpack-packing-plan-cuda-evaluator` remains superseded as an
immediate task. A resident evaluator may later be added as an objective-v2
implementation child after source residency and measured break-even evidence
exist. Existing CP-BP exact oracles, runtime-autotune metrics, and CP-Math
planner tests remain historical evidence rather than the new planner objective.

The interface is validated by deterministic tests for amortization, bounded
measurement, measured conventional fallback selection, one-shot tuning skip,
policy rejection, stale cache detection, explanation, and objective-v2
versioning. Real GPU benchmark execution remains in the CUDA background
controller and must follow the CE-ARCH-30 evidence and resource contracts.
