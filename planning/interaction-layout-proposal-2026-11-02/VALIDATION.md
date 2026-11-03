# Factorized interaction qualification, 2026-11-02

This record qualifies the explicit sigmoid-gated joint-product law in
[CONTRACT.md](CONTRACT.md). It distinguishes host, installed-package, admitted
device and performance evidence. It does not establish biological validity,
arbitrary-operator execution, higher-order differentiation or global novelty.

The bounded native correctness/usability milestone is complete; the
[independent final review](validation/native-milestone-independent-review.md)
found no material blocker and rechecked all 84 integration-source hashes and
three host suites. Performance work is [paused](PAUSED.md) at the user's
request, with no accepted timing observations. The current public default
remains the experimental packet request, with direct execution explicit;
no measured route-selection policy is integrated.

## Host and package evidence

The [integration record](evidence/integration/integration-status.md) retains
commands, logs, compiler dependency paths, source hashes and artifact hashes.
CUDA 12.9.86 targets SM70 with a GCC 12 host compiler.

- Native CUDA component, independent native runner and 200-update C++ example
  compile and link without Torch.
- An [independent CPU replay](evidence/integration/training-example-replay.md)
  checks the example's FP16 SGD trajectory at 200 and 10,000 updates. It reduces
  error while keeping parameters finite and within the initial qualified
  domain. This replay supplements the separately scheduled device execution.
- CTest passes all three affected host suites: structural preparation, checked
  `ExplicitPattern` binding and the CUDA API's host argument/workspace guards.
- The independent Python reference/wire/corpus suite passes 33 tests. It
  includes zero-safe analytic VJPs, finite differences, precision handling,
  repeated roles, tied fields, cancellation and large fan-in fixtures. The
  million-mechanism case is reconstructed byte-for-byte from bounded input
  shards under the worker's actual 64 MiB per-file limit; corruption, reordering
  and missing shards are rejected.
- The later verifier-owned reference cache and its validator adoption add
  thirteen tests; the [independent live suite](validation/reference-cache-independent-review.json)
  passes 46/46 under Autobench's Python interpreter. Cached expected arrays are
  bound to the exact case bytes, oracle and corpus identities. The validator
  loads expected data before the runner and checks the entire dedicated cache
  tree afterward, rejecting cross-case mutation. This later cache work is
  distinct from the initial captured 33-test suite. The same record contains
  seven CPU tests of the 32-invocation precision-qualification driver; these
  driver tests do not establish GPU type coverage.
- The focused install component exposes
  `Cellerator::factorized_interaction_cuda`. An external CXX-only consumer
  configures and links against the installed package, repeats the component
  request and verifies one canonical `Cellerator::memory_cuda` archive. Optional
  NN requests succeed as unavailable, while required NN requests fail clearly
  against this narrow installation. This consumer was not executed locally.
- Optional ATen and Python extension targets compile. Their build success
  alone does not qualify device behavior or autograd lifetimes.

The complete SDK install remains outside this qualification: an unrelated
existing matrix-patch translation unit fails in
`include/Cellerator/math/matrix/contracts.hh`. The integration record identifies
that failure and the consequent limit on mixed full-SDK package loading.
Independent configure-only fixtures nevertheless pass full-first, NN-first and
narrow-first export loading, including repeated mixed requests and canonical
memory-target identity. Those fixtures use placeholder archives and a stub
Torch configuration; they establish export parsing, not full-SDK linking.
See the [review identities](evidence/integration/package-review-identities.json)
and adjacent transcripts.

The staged Quarto HTML book renders all 41 chapters with code execution
disabled. The [render record](validation/docs/book-render.json) binds the
documentation inputs and log; local links in the new guide and this record
were checked against the workspace. Rendering does not execute CUDA examples.
The final milestone edits to the guide and implementation record also
[render successfully](validation/docs/milestone-render.json) in that staged
book. The final local-link scan checks 39 links across the guide, this record
and the pause checkpoint without missing targets.

The protected cellular geometry definition retains SHA256
`d37f504e5c1ab629334f4127dcaa60d008afe94e560d41aa936d8ba161124254`.

## Numerical and execution boundary

The frozen scalar oracle evaluates storage-quantized inputs in float64.
FP32 output/VJP acceptance is
`abs(error) <= 2e-6 + 2e-5 * sum_abs_reference_contributions`; FP16 output adds
one local FP16 ULP against the correctly rounded oracle. The initial domain
and separately declared saturation cases are specified in the
[benchmark guide](../../bench/factorized_interactions/README.md).
The literal FP32 product-residual adversary also uses its explicitly frozen
near-unit values just outside the main range; that result is case-specific.

Both requested routes use the shared compensated backend for unsafe reduction
fan-in. Reports identify the actual backend. Two shared-stable executions are
not evidence of an advantage from packet layout. Low-level callers retain
immutable structure, numerical inputs and per-invocation tape until queued
work completes, and keep every writable buffer disjoint from structural
descriptors/postings and read-only numerical data.

## Device and performance evidence

Only source-bound Scheduler-admitted Autobench runs establish device or timing
qualification. The isolated `measurement/` workspace preserves captured source,
build artifacts, receipts and failed attempts independently of the repository's
existing paused campaigns.

The [initial V100 report](validation/native-correctness-v15.suite.json) passes
29/29 forward/VJP comparisons on Tesla V100-SXM2-16GB UUID
`GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a`. It includes zeros, ordered/repeated
roles, shared fields, overlapping outputs, empty cases, cancellation,
saturation, repeated VJPs, overwrite/accumulate gradients and 100k/1M fan-in.
Five of the eight independent storage-type triples are covered by this first
run. Requested packet execution selected its packet backend in ten cases and
the shared stable backend in five; requested direct execution selected direct
in ten and stable in four.

The same job ran 200 native C++ updates: output moved from `0.2023926` to
`0.230103` toward target `0.25`, with finite in-domain parameters. The prepared
Torch adapter passed all five gradient groups, parameter update/reuse,
cross-stream owner reuse, saved-tensor mutation rejection and first-order-only
guards (Python 3.13.11, Torch 2.7.0+cu126, FP32 primal/output tensors). This
adapter run does not establish all mixed-storage adapter combinations.

Source closure is
`9afe1e5a6e119f5f1565ac02c2971cdf5ad4117453f0aa9f604f69371bb582a6`;
runner SHA256 is
`a208a0f31d2ca8ccde929bf99532faa623f7f99267adb8aa37b5e0b6ce41dd02`.
Scheduler job `ce0899a1-21ef-4a93-b797-d79df8db1d84` binds the selected UUID
and successful cleanup. This run is untimed. Sanitizer completion is established
by the later reports below; numerical correctness alone does not establish
performance selection.

The [subsequent precision gate](validation/dtype32-independent-acceptance.json)
passes 32/32 combinations: all eight activation/parameter/output storage
triples across packet, native direct, minimal direct and unique-output direct.
Every output, five native gradient groups and repeated VJP satisfy the frozen
bounds, with actual backend/layout metadata checked. The independent acceptance
binds the new native-only runner/source/manifest and every case artifact to the
finished Scheduler receipt, selected V100, quiescence and verified cleanup.
This gate is untimed and does not replace sanitizer or performance evidence.

All four new auxiliary partitions pass the twelve frozen adversarial workloads
for both packet and strongest-direct requests: [packet regular](validation/aux-packet-regular-independent-acceptance.json),
[packet 1M](validation/aux-packet-1m-independent-acceptance.json),
[strongest-direct regular](validation/aux-strongest-regular-independent-acceptance.json)
and [strongest-direct 1M](validation/aux-strongest-1m-independent-acceptance.json).
Independent review checks output, all five VJPs, repeated VJPs, byte digests,
dispatch and finished receipt/cleanup identity. Both million-mechanism cases
match the regenerated scalar result exactly and select the shared stable
backend. Reported scratch peaks use sampling and may miss shorter transients.

The [direct sanitizer diagnostic](validation/sanitizer-diagnostic-v8-result.json)
passes one FP16 packet-route `correctness_ordered_tied_zero` case under CUDA
12.9 memcheck with full leak checking: zero errors and zero leaked bytes. Raw
stdout confirms the tool banner preceded the validator JSON and caused the
older suite's parse failure. The [accepted protocol repair](validation/validator-stdout-fix-independent-review.json)
redirects child diagnostics to stderr without changing numerical checks or
failure codes; its independent full CPU suite passes 47/47. The subsequent split
sanitizer reports below establish full tool coverage.

The [full memcheck partition](validation/sanitizer-memcheck-independent-acceptance.json)
subsequently passes all 14 required cases across baseline packet/direct and
candidate minimal/unique-output paths. Independent review checks all 84
output/VJP metrics, 70 retained artifacts and 14 raw tool logs: zero errors and
zero leaked bytes with full leak checking. Source, binary/library, CUDA 12.9
tool tree and the finished V100 receipt/cleanup match. This partition checks
one VJP per case; repeated VJPs are covered separately. The other sanitizer
partitions follow below; this partition is untimed.

The [racecheck](validation/sanitizer-racecheck-independent-acceptance.json)
and [initcheck](validation/sanitizer-initcheck-independent-acceptance.json)
partitions also pass 14/14 cases each with all numerical checks and retained
artifact identities verified. Racecheck reports zero hazards/errors/warnings;
initcheck reports zero errors. The
[synccheck partition](validation/sanitizer-synccheck-independent-acceptance.json)
also passes 14/14 with zero errors and matching artifact identities. Each
finished V100 receipt and cleanup matches the same native binaries. The full
strict matrix therefore passes 56/56 across all four tools.

An isolated reproduction identified an Autobench retention-edge defect: its
submission-time eviction can remove a newly captured snapshot before pinning
it. The [diagnostic record](validation/autobench-snapshot-eviction-diagnosis.md)
retains the reproduction and a proposed patch; no external tool source was
changed. The task uses the supported workspace retention setting, within the
host ceiling, to avoid that eviction condition.

The [predeclared performance protocol](PERFORMANCE_PLAN.md) defines paired
sampling, strong direct controls, complete-cost boundaries and the separate
trusted promotion guard around KDA's provisional rankings.

No timing partition has an accepted measurement. The final admitted attempt
failed on a collector cache-ownership guard before producing usable samples;
the [official cleanup record](validation/final-timing-retry-cleanup.json)
confirms terminal state and a released lease. Both measurement workspaces are
paused. The corrected collector has an
[independent CPU acceptance](validation/timing-collector-cache-ownership-independent-review.json)
but has not executed a new GPU timing partition. The separate recorded-analysis
harness checkpoint includes focused regression evidence; synthetic CPU replay
is not a CUDA performance observation.
