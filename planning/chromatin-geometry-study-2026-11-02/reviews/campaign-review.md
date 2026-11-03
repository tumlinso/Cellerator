# Independent campaign-runner review

Initial review on 2026-11-02 against the approved study draft and reviewed
protocol. The runner and analysis script changed concurrently during review;
the identities below define the inspection boundary. No GPU job was run by
this reviewer. These findings require reconciliation with the final immutable
source and its scheduled receipts.

## Findings and acceptance boundaries

1. **Cold total includes independent oracle preparation.** The runner starts
   the cold clock before `materialize_workload` and optional replay preparation.
   Both calculate before/after condition decisions with the CPU oracle
   (`workloads.cc` and `data/replay.cc`); the timed interval also includes
   workload hashing. The runner now labels its cold total as including fixture
   and reference preparation. That is useful diagnostic information, but it
   cannot directly support the required execution-only cold cost or break-even
   comparison. Time legitimate host preparation separately from oracle and
   provenance checks, or report the cold comparison as incomplete.

2. **Combined teacher validation cannot detect a failed teacher update.**
   `teacher_fixture.hh` initializes each table to pass through its first input
   and fits targets with the same sign. The combined runner's host gate oracle
   expects the original eligibility bitmap, and its pre-step snapshot and
   validator cover numerical mechanism weights and STE scores, not teacher
   logits or teacher SGD updates. A teacher that performs no update can retain
   the same threshold decisions and pass this combined check. Component-level
   teacher checks do not establish the combined chain. Validate actual teacher
   fitting/update against an independent oracle and exercise a nontrivial or
   crossing table in an integrated fixture, or record the coverage gap.

3. **Exact work membership and identity need direct evidence.** The runner
   compares gate bytes and primal/derivative planes against host decisions and
   validates numerical outputs and VJPs. It does not compare materialized
   workset entries or cohort cell/group mappings with independent expected IDs.
   Equal outputs can hide identity errors for symmetric or overlapping
   mechanisms. A direct bit-exact descriptor/workset test is needed before
   claiming the approved identity and membership acceptance criterion.

4. **The paired analysis script is a calculator, not an evidence join.** Its
   input trusts caller-supplied times and top-level validation/interference
   booleans; candidate/control UUIDs can be omitted and default to one supplied
   UUID, and source, binary, workload, gate, precision, and admission identities
   are not joined to raw receipts. The current script explicitly states that a
   threshold result is not a performance or eligibility claim. Retain an
   independent receipt-bound matching step for every control/candidate pair
   before drawing conclusions.

The first inspected paired-analysis version used `speedup > 1.05`, which can
accept less than 5% latency reduction. The later version recorded below uses
`speedup > 1/0.95` with a numerical boundary guard and labels its output as a
calculator threshold. The later script's CPU-only test suite passed 3/3; this
does not validate the provenance of future input records.

## Other observations

The runner explicitly identifies its numerical validation as first-and-last
complete steps. It snapshots the actual pre-step state for these checks; this
supports validation of those transitions but is not an independent comparison
of every intermediate update or the entire 16/256-step trajectory. It reports
separate cold, recurrent, recovery and validation fields, and its explanatory
phase probe is outside confirmation timing. Any final conclusion must use a
matched boundary including the applicable transfer, preparation and recovery
costs, and retain the limited validation scope.

The STE primary route constructs all model entries as derivative candidates,
uses native score updates for update-enabled cases, and rejects condition
update modes for STE rather than silently presenting them as measured. User-
required combinations and any missing combinations must still be accounted
for in the coverage report. The teacher's local soft-to-hard change is reported
separately from hard circuit decisions and is not biological validation.

## Source identities

At inspection start the runner SHA256 was
`4ae2245ab67dce710890c85d0eeda2361e5283ab559e51bc65431a7d4e6dd28d`;
it changed during review and the final observed hash was
`1bfca966818ca5e52afbb1c45392b3e8c58d8f0934660c1936806c42f80bb7c2`.
The script changed from
`40d7bc36088fd7e32741d03d53c79cefb87c2d3cc74b578fbbe2c4a09c73e35a` to
`324167b6b476ffbd636ea5c863c80327da966753a6e2d9249b0e3f59dc586344`.
The other final observed SHA256 values under `bench/chromatin_geometry` were:

| File | SHA256 |
| --- | --- |
| `campaign_runner.hh` | `34df6cdda1347374fad0c2d68a57ee330f0101bb33498d05af843a7336b05b2a` |
| `teacher_fixture.hh` | `194b46071472702eafca94681532fe87331478a15828889cd0dc30e1db96c735` |
| `device_fixture.hh` | `2745482669275cb5abce0f909a1d429c17c5b26133f64c06c163000a2842386f` |
| `main.cc` | `2254ab40400adc6713f044c409e5af8afae4ccb42d158cd3abb362dd715e217c` |
| `tests/test_paired_analysis.py` | `5a61ea18c181920a5a3d37e9796b758bd5e7180de239926efaeeb32b0e33a219` |

Later changes and the actual packaged candidate require a fresh source-bound
review and GPU qualification. No device admission or sanitizer evidence was
created by this review.
