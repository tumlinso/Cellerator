# Native delivery milestone and pause, 2026-11-02

Paused after the bounded native correctness/usability milestone, following the
user's instruction: “Or pause. But after the next milestone.” Further GPU
measurement, optimization and route promotion require an explicit resumption.
This checkpoint does not mark the complete performance plan achieved.

## Delivered milestone

- Native Torch-independent C++/CUDA forward and all five VJPs for the agreed
  sigmoid-gated joint law, deterministic preparation, persistent structural
  ownership, FP16/FP32 storage alternatives and a shared accurate fallback.
- A [C++ training example](../../examples/native_cuda/factorized_interaction_training.cu)
  executed 200 updates in an admitted V100 run; focused package installation
  and an external C++ consumer compiled and linked. Optional prepared Torch
  adaptation passed the recorded first-order FP32 runtime checks.
- Independent native qualification: 29 initial comparisons; 32 dtype/mode
  combinations with repeated VJPs; both twelve-case adversarial partitions;
  all 56 cases across memcheck, racecheck, initcheck and synccheck. These are
  bounded correctness checks, not model quality or performance results.
- The [final independent review](validation/native-milestone-independent-review.md)
  found no material blocker for this milestone, matched all 84 recorded
  integration-source hashes, and reran the three host suites successfully.
  Exact source, binary, numerical, tooling and receipt identities are linked
  from [VALIDATION.md](VALIDATION.md).

## Execution is stopped

The `measurement/` and `measurement-performance/` Autobench workspaces are
paused. The performance operator reports zero unresolved attempts. The final
already-admitted timing attempt failed before producing usable observations.
Its [official result and cleanup record](validation/final-timing-retry-cleanup.json)
bind Scheduler job `be22214c-79db-43ad-8fa2-2a899677448f`, attempt
`e2c8acc6-1b7d-4e10-8d55-d69b0379e978`, verified unit/cgroup cleanup and released
lease `c0530449-ec65-4a6c-87c7-8810bf19816e`. No owned GPU resource remains
held according to that receipt. CPU experiment work is stopped as well.

## Remaining work and preserved boundaries

There are **zero accepted timing observations**. No packet advantage, measured
winner or production selection policy has been established. The current native
and optional-adapter default remains the experimental packet request; direct
execution is available explicitly. The minimal-layout and unique-output direct
controls remain private benchmark candidates rather than installed fast routes.
The planned measured selection and later default-route decision are unfinished.
The full SDK install also remains limited by the unrelated matrix-contract
build failure described in the integration record.

The corrected collector SHA256 is
`357d68c432dd23981f9c56c770e8e4c0cda7eea73be9413641ad651a4ccfc214`.
Its [independent cache-ownership review](validation/timing-collector-cache-ownership-independent-review.json)
accepts the CPU fix; the staged wrapper-3 collector has not run a new GPU timing
partition. The CPU recorded-analysis checkpoint is preserved at
`validation/recorded-analysis-checkpoint-20261102T223237Z/`, with manifest SHA256
`8312dd5166c010f5cd9c0fc6b7b271fbc9326ca394b18a5c586a80b423900e83`.
Its 15 focused regressions passed; synthetic replay is not CUDA measurement.

On explicit resumption, inspect current supported Autobench registrations and
Scheduler state before any reconciliation. Preserve the accepted qualification
bundle and immutable native binaries; verify current source and collector
identities. Use a single admitted measurement lane with CPU-heavy validation
outside timing intervals. Follow [PERFORMANCE_PLAN.md](PERFORMANCE_PLAN.md)
against the strongest qualified applicable control, keep shared-stable cases
out of packet-layout promotion, and retain complete costs and original receipts.
Do not assume an `enabled=false` benchmark field stops this installed Autobench;
the supported workspace pause is the established stop control.

All work and evidence remain in the checkout. No commit or push was requested.
Unrelated staged Todo maintenance and concurrent chromatin work were preserved.
