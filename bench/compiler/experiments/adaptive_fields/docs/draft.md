# Adaptive execution-field experiments

This is the initial experiment contract for the seven probes explicitly
authorized in this conversation on 2026-10-28. It does not authorize production
compiler, API, kernel, default, or application changes.

## Scope and authority

- Experiment-owned source, temporary adapters, generated artifacts, and evidence
  stay below `bench/compiler/experiments/adaptive_fields/`.
- The native Project Control `next_task(project="cellerator")` front door
  returned `no_work` with no handle. No unrelated task is claimed and no Todo
  records, credentials, leases, or generated views are edited. The user's
  explicit campaign instruction supplies the engineering scope.
- Device Scheduler supplies physical GPU admission independently. Use the
  installed Cellerator client, a live UUID-selected exclusive request, finite
  timeout, and supported quiescence precondition. Preserve terminal receipts
  and verified cleanup. Never substitute direct device access for admission.

## Semantic operation and correctness

E1/E4 use `p = segment_softmax(Ax)` with runtime relation values and inputs.
E2 adds same-shape FP32 bias addition with explicit effects and a single supported
ATen overload. E3 separates pointer/value/activity/support/structure/order changes.
E5 preserves the existing native width-64 recurrent sparse-training and declared
surrogate estimator. E6 tests executable pinning, publication, and retirement.
E7 compares existing indexers and numerical realizations plus NVCC/NVRTC loading
of the same small generated region.

Independent FP64 CPU and explicit Torch references are distinct from native
implementations. Work identities, masks, ordering, epochs, and generations must
match exactly. Preserve per-operation tolerances and disclose all local bridges.
Caller-supplied receipts, analytical cost constants, scalar toy referees, old
reports, and mocked event traces cannot establish numerical or performance passes.

## Workloads and resource bounds

- Three deterministic seeds for small correctness cases and frozen manifests
  for existing small, large-event, and large-ranked timing fixtures.
- Existing numerical width and precision contracts; no workload shrinking after
  allocation failure and no newly optimized production kernel.
- At most six indexer/numerical candidates per fixture. Five warmups and 21 paired
  samples in each of two exploration and two confirmation rounds, with identical
  initial states. Measure setup, compile/load, workset construction, forward,
  backward, publication, memory, and reuse horizons 1/16/256 separately.
- One background compilation worker and one pending executable replacement.
- GPU jobs are serialized. CPU build work must finish before clean timing.
  New GPU execution/lifetime paths require separate memcheck/racecheck/synccheck
  evidence before their results are accepted.

## Promotion and stopping

Every probe reports supported, rejected, or unresolved, distinguishing existing
functionality from experimental adapters. Stop a branch if it requires production
redesign and identify the missing capability. Complete unaffected probes.
No winning prototype is made a production default. No synthetic result is
biological validation. A consolidated report selects the smallest justified
production follow-up and explicit deferrals after independent estimator and
executable-lifetime verification.
