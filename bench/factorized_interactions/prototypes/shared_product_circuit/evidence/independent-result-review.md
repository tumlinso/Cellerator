# Independent result review

Reviewed 2026-11-03. The initial assessment was sent to the root before reading
`root-initial-interpretation.md`. The assessment agrees with the root's bounded
conclusion; there is no material disagreement.

## Result

For the fixed M=8, D=32, C=1024 shared-product motif on the admitted Tesla
V100-SXM2, the circuit arm was faster than the matched unfactored arm in all
seven paired host-wall blocks. The medians of block means were 47.7281 µs and
50.8444 µs per complete resident forward plus five-gradient VJP, respectively:
the ratio of medians is 1.0653x, or 6.13% less time. The median of the seven
individual paired control/circuit ratios is 1.0684x; those ratios range from
1.0192x to 1.0718x. These are two different summaries, not contradictory
estimates. Seven blocks within one execution do not characterize run-to-run,
device-to-device, shape, or workload variation. A bootstrap over those blocks
does not establish such external variability.

This supports a modest benefit from the specified source-level factoring in
this specialized comparison, with the usual limitation that other compiler and
hardware effects of the source change are not separately measured. The two
arms use the same owners, shared numerical helpers, tape, scratch, and two
forward/five backward launches; both compute the common FP16-input product once
in FP32. The unfactored arm retains that product within each forward contribution
and separately sums the common-input adjoints, while the circuit moves the
common factors outside those sums. Warmups, 30-step blocks, alternating pair
order, rotating native position, and synchronization match the frozen contract.
Separate event diagnostics are consistent with a backward contribution, but
do not establish a kernel-level cause or replace the primary wall metric.

## Correctness, identity, and limits

The verifier's `validation-acceptance.json` records independent replay against
the frozen unchanged oracle, all six tensors and repeats passing for the two
prototype arms, with maximum observed error/allowance 0.00564. The four
sanitizers reported no findings on the C=64 repeated-tail overwrite fixture for
each prototype arm. Parameter refresh was tested in separate invocations, not
as an in-process prepared-owner lifetime or mutation test. These are bounded
checks, not exhaustive numerical or concurrency proofs.

The timed case's three output and metadata artifacts hash-match the measurement
summary, as does `paired.metrics.json`; the two prototype output wires have
identical hashes for this case. The measurement summary records that all three
outputs passed the tolerance against the timed oracle, with maximum
error/allowance 0.00533. Manifest SHA256
`b759ac045f320f1b97b101cdfba9ad65dd3681ea7ea30c32ddd650b673adc82c`
and executable SHA256
`f948d55a649f6e7c8f78f95a1f4ab324a1b31a88d161e56657362f8022e7f52a`
match the frozen handoff. The receipt reports exclusive GPU admission, two
quiescence observations, successful execution and verified cleanup. Pre/post
observations are not continuous monitoring of other activity.

The requested native direct route actually selected **stable**. Its median
6.7223 ms is a descriptive observation of that general implementation under
this private Release build, not a factorization control. The roughly 140.85x
ratio to the circuit combines differences in specialization, mapping, reduction,
and other implementation details; the evidence does not assign its cause or
establish a production speedup. Both private arms use 3,276,800 bytes of tape
plus scratch, versus 98,304 bytes for native stable, a 33.33x trade in this
accounting. Each private arm's total reported allocation including values and
destinations is 5,518,122 bytes; native's is 2,342,250 bytes. Values are shared
between the simultaneous arms, so these per-arm totals must not be added
without adjustment. Setup and upload are recorded separately and excluded from
the resident measurement; no cold-step, optimizer-step, or end-to-end claim
follows. The allocation/seed and upload observations are single setup timings,
not a controlled amortization study.

Decisive evidence: `docs/draft.md`, frozen source `circuit_kernels.cuh` and
`circuit_runner.cu`, `cpu-build/handoff.json`,
`evidence/validation-acceptance.json`, `evidence/measurement-autobench-results.json`,
and `measurement/.autobench/evidence/shared-product-circuit-measurement/0/`
`d1ab1263364a4926ce4ee924dee489b077400e1957f3291efcbf792bc862e31e/`
`measurement/driver/{measurement_summary.json,paired.metrics.json,paired.*.cfout}`.
No GPU execution, build, or additional experiment was run for this review.
