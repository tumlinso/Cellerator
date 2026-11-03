# Shared-product circuit: first bounded result

2026-11-03. The experiment is complete and stopped at the agreed discussion
checkpoint. Two of three permitted GPU attempts were admitted; no retry or
performance-driven revision was used. The isolated Autobench workspace is
paused, both leases are released, and both original receipts verify cleanup.

## Result

The factored circuit used **6.13% less resident forward/VJP time** than the
matched unfactored control in this one experiment. It was faster in all seven
paired blocks. Both prototype arms produced byte-identical result wires for
the timed values and passed the independent oracle.

| Arm | Actual implementation | Median full forward + five-VJP time | Tape + scratch |
| --- | --- | ---: | ---: |
| Unfactored control | `literal_unfactored` | 50.8444 microseconds | 3,276,800 bytes |
| Circuit | `shared_product_circuit` | 47.7281 microseconds | 3,276,800 bytes |
| Native direct request, descriptive | `stable` | 6.72231 milliseconds | 98,304 bytes |

The ratio of prototype medians is 1.0653x. KDA's experiment-engine paired
bootstrap gives an exploratory 95% interval of [1.0443, 1.0693]x over these
seven blocks (10,000 resamples, seed 20261103). The median individual paired
ratio is 1.0684x; that is a different summary. This is not production selection,
independent-run confirmation, or evidence across shapes/devices.

Separate event diagnostics have equal median forward time (9.216 microseconds)
and backward medians of 44.032 versus 41.984 microseconds. They are consistent
with a backward contribution, but are instrumented separately and do not
establish a kernel-level causal profile or replace the primary host-wall metric.

## What was compared

The [fixed contract](docs/draft.md) has eight arity-three mechanisms sharing
two participants, one output, independently tied alpha/bias/gamma banks, two
conditioning inputs, 32 channels and 1,024 measured cells. Activations and
parameters are runtime FP16; arithmetic, output, cotangents and all five VJPs
are FP32. The circuit pulls `a*b` outside the output and common-input adjoint
sums. The control already reuses that exact FP32 product inside each term.

Both private arms share ownership, numerical helpers, two forward/five backward
launches, tape and scratch. All native and private sources were compiled into
the same standalone Release executable with CUDA 12.9/GCC 12, SM70, `-O3`, and
no fast math. The public implementation and its route defaults were unchanged.

Ten warmups per arm preceded seven blocks of 30 complete evaluations, with
stream synchronization at each block end. Pair order alternated and native's
position rotated. Setup, upload, allocation, downloads and event diagnostics
were outside the primary interval. Values were fixed during timing; this is
neither an optimizer step nor a measured cold-start comparison. Each private
arm reports 5,518,122 total buffer bytes including shared values; native reports
2,342,250. Do not sum those totals without accounting for shared inputs.

## Validation and independent assessment

The frozen float64 oracle and tolerance were unchanged. Independent acceptance
reparsed all retained outputs: 28 validation invocations / 268 tensor checks,
then 18 checks for the three timed output/VJP sets. Largest observed normalized
error was 0.00564 of the allowed error in validation, and 0.00533 after timing.
Memcheck, racecheck, initcheck and synccheck had zero findings on each prototype
arm's C64 repeated-tail fixture. Ordinary seeds, zeros, cancellation, saturated
gates, repeated roles/VJPs, gradient accumulation and changed runtime parameters
were covered. Refresh used separate same-binary invocations, not an in-process
prepared-owner mutation test. This is targeted prototype qualification.

Root [sealed its interpretation](evidence/root-initial-interpretation.md) before
the reviewer received raw timings. The [independent review](evidence/independent-result-review.md)
agrees with the observed direction and bounded interpretation; no material
disagreement remains. Seven serial blocks within one run do not measure
between-run variability. Resource observations before/after timing were idle;
they do not constitute continuous interference monitoring.
The [follow-up discussion](evidence/result-discussion.md) retains the reviewer's
challenge to treating the native gap as a prioritized or causally explained lead.

## What this changes, and what remains open

Factoring the common products/adjoints has a modest demonstrated benefit in
this motif and mapping. The much larger native gap cannot be attributed to
factorization: both private arms specialize ownership and reduction and spend
33.33x more tape/scratch than native stable. It identifies another question,
not an accepted explanation or a production-speedup claim. No packet layout
or default is promoted. No next experiment has been selected.

Decisive records:

- [Validation acceptance](evidence/validation-acceptance.json) and
  [measurement acceptance](evidence/measurement-acceptance.json).
- [Raw-pair KDA analysis](evidence/kda-paired-analysis.json), binding the original
  measurement summary and statistical implementation hashes.
- Original [validation](evidence/validation-autobench-results.json) and
  [measurement](evidence/measurement-autobench-results.json) receipts and the
  [bounded attempt ledger](evidence/attempt-ledger.md).
- [Build/provenance handoff](cpu-build/handoff.json): runner SHA256
  `f948d55a649f6e7c8f78f95a1f4ab324a1b31a88d161e56657362f8022e7f52a`,
  manifest `b759ac045f320f1b97b101cdfba9ad65dd3681ea7ea30c32ddd650b673adc82c`.
- [Workspace preservation](evidence/workspace-preservation.json): no preexisting
  tracked file changed; oracle inputs and protected cellular geometry unchanged.

Raw metrics, output wires and metadata remain under
`measurement/.autobench/evidence/shared-product-circuit-measurement/0/`
`d1ab1263364a4926ce4ee924dee489b077400e1957f3291efcbf792bc862e31e/measurement/driver/`.
No commit or push was requested.
