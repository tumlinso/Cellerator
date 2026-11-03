# Shared-product circuit: GPU attempt ledger

Authorized global bound: at most 3 admitted GPU attempts, each Scheduler job
bounded to 180 s; packet kill limit <=170 s because Autobench adds 10 s.
One validation packet includes correctness and four targeted sanitizers; one
measurement packet follows only after accepted validation; at most one retry is
reserved for correctness or plumbing. Stop on first trustworthy result.
Managed worker scratch is capped at 128 MiB. Workspace is initialized with
`--no-activate`; use supported explicit single-step reconciliation, and pause
when finished. No GPU job submitted as of initialization. Count: 0/3.

The Scheduler inventory at 2026-11-03 14:32 UTC listed available Tesla V100
resources; actual allocation and clean timing require a fresh forecast, explicit
UUID selector, and Scheduler `gpu.quiescence` admission. Historical device
identities are not proof of availability.

## Pre-admission forecast, 2026-11-03 14:50 UTC

The explicit GPU-21131915-1488-23af-38dd-1743ae1f5cc8 forecast returned
`waiting_for_forecast_resources`, `no_feasible_now_atomic_placement` and
`capacity_or_protected_claims`: it showed a pre-existing 6 GiB GPU allocation
and 4 CPU / 8 GiB host allocation in `reclaiming`. No submit intent or Scheduler
job was created; admitted count remains 0/3. The separate read-only inventory
is `validation-pre-resources.json` and alone did not reveal that ledger conflict.
Do not force or alter that allocation. The next request selects a different
explicit V100 under the same Autobench principal and fresh quiescence gate.

## Validation attempt, submitted 2026-11-03 14:51 UTC

Autobench attempt `da49f707-e81a-4337-9603-ff2ef5173666`; Scheduler job
`c7490503-49ea-4270-9116-04b1886d9a36`; forecast selected
`GPU-cf22c41f-5b58-77b1-3535-8fadd1ca6505`. Treat as consuming attempt 1/3
until terminal admission and cleanup are inspected. Forecast and submission
result: `validation-alternate-reconcile.json`; worker limit 170 s, Scheduler
limit 180 s, with explicit quiescence precondition. No other new work is active.

Validation execution: Scheduler attempt `64148eb3-09dd-448e-8791-a1be114c2c4c`
was admitted on the forecast UUID; worker result `succeeded`, 13.172 s, exit 0.
The driver reports 28/28 runs passed, including 4 sanitizers × 2 prototype arms.
Original finished Scheduler job/attempt receipt, including verified preconditions,
exact unit/invocation and `cleanup_verified: true`, is retained in
`validation-autobench-results.json`. The import reconciled 1 attempt, submitted 0.
Post-cleanup resource observation is `validation-post-resources.json`.
GPU attempts consumed: **1/3**. Shell-default Scheduler principal could not read
this Autobench-owned job (`permission_denied`); no credential switch was made.
Independent validation acceptance is pending. No timing attempt submitted.

## Measurement attempt, submitted 2026-11-03 15:00 UTC

Autobench attempt `e6e12f3a-bc0c-4b16-a34e-3a0d254d3460`; Scheduler job
`f3b54ee8-316b-476f-b851-9d4a9e3c9ba4`; same V100 UUID as validation.
Count conservatively as 2/3 while awaiting the finished result and receipt.
`measurement-submit-reconcile.json` retains the actual forecast and submission;
`measurement-pre-resources.json` records the clean pre-observation. Driver
references the exact accepted validation summary SHA256
`bb47d128d43b622b26a1e47c6a32404b07d9217b69da6fe620f832bd659a4f54`.

Measurement execution: Scheduler attempt `fde01660-d71e-4f39-9ee0-1156405e6f25`
was admitted on the same V100; worker succeeded in 2.857 s, exit 0. The driver
reports validated `paired_measurement`, 21 raw timing rows, and three independently
checked output artifacts. Original finished Scheduler receipt with verified
quiescence and `cleanup_verified: true` is in `measurement-autobench-results.json`.
Import reconciled 1, submitted 0; the workspace was then paused using
`autobench pause`. Pre/post resource observations are retained in
`measurement-pre-resources.json` and `measurement-post-resources.json`; they
show zero utilization and no visible compute processes, but are not continuous
interference monitoring. Total admitted GPU attempts **2/3**; no retries or
additional GPU work. Full data and raw logs remain under the Autobench evidence
for fingerprint `d1ab1263364a4926ce4ee924dee489b077400e1957f3291efcbf792bc862e31e`.
