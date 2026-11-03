# Factorized interaction performance comparison

Measurement protocol, 2026-11-02. This records the root-approved comparison
policy and implementation boundaries. It reports no device result or speedup.
The numerical law, fixed corpus and acceptance in [CONTRACT.md](CONTRACT.md)
remain unchanged. First native correctness qualification is independent of
the additional performance-control and evidence work described here.

## Meaning of a measured operation

The initial resident metric is **forward plus explicit VJP**, not a complete
optimizer or model-training step. It includes native dispatch/guards, saved
gate/slope tape writes, output initialization/casts/reductions, gradient
initialization, accurate gate-adjoint evaluation, and all five VJPs/reductions.
Its buffers and structural owner are already allocated and resident.

The primary resident metric is host wall time for an uninstrumented warm block
of `N` forward/VJP calls followed by stream synchronization, divided by `N`.
The timed loop has no per-call GPU event records. It includes the production
caller guards and dispatch. GPU-event forward, VJP and contiguous combined
samples come from a separate instrumented loop, with events allocated before
timing; they are secondary device diagnostics. Do not sum independent phase
medians and call that the combined median. Preserve raw block and event samples.

The existing runner's setup plus warm median is an **estimated setup plus
prepared forward/VJP** cost. It is not a measured first invocation. Likewise,
adding one host parameter upload sample to a warm component median does not
measure an optimizer step. Keep those estimates labelled as estimates until
directly measured. No general optimizer is required for this comparison.

Sample host parameter refresh repeatedly, with raw values and its exact
allocation/upload/synchronization boundary. State whether parameters are the
same or changed. A host refresh is distinct from updating resident parameters
on the device; charge it to a caller only when that caller requires it.
The isolated candidate runner reuploads unchanged parameter values and measures
pinned staging allocation, host copying and three serial upload completions.
Its setup-plus-warm and refresh-plus-warm totals are analytical estimates.

## Preparation, numeric representation and cold ownership

Inputs to this benchmark are already explicit mechanism records and typed
numeric arrays in the declared native axes. Wire decoding, input generation,
biological discovery and general external layout/precision conversion are not
inside the resident operation. Uploading those existing arrays is H2D
materialization, not proof that an arbitrary producer-to-layout conversion is
free. Report the initial input/parameter movement separately.

Measure host preparation, structural upload, numeric allocation and required
initial movement. An actual owner-cold first evaluation must time its first
forward/VJP before warmups, with an initialized CUDA context explicitly stated.
Process/CUDA startup is a different boundary. Keep validation-only buffer
poisoning distinguishable from required production initialization.

Charging both routes for full packet preparation/upload is valid for a common
prepared-owner experiment, but it is not a fair standalone cold-direct control.
The minimal direct control in
[`bench/factorized_interactions/controls/`](../../bench/factorized_interactions/controls/)
prepares canonical records, terms and exact backend maxima without packet
clustering, dictionaries or postings. Its device owner uses the existing CUDA
memory resources and one bulk structural upload. It is applicable only when
the exact invocation selects the ordinary direct backend. Shared stable
fallback cases continue to require the full posting structure and remain
explicitly labelled.

Report preparation plus `H` prepared invocations for declared reuse horizons.
If only setup and prepared samples are combined analytically, call the totals
estimates. Derive break-even only when the recurring measured cost actually
favours the candidate; increasing reuse cannot rescue a slower recurring path.

## Strong controls and backend truth

Compare the packet implementation with the strongest qualified direct control,
including the separate unique-output candidate where it applies. A validated
bijection from mechanisms to output targets permits that direct candidate to
write final typed outputs without FP32 output atomics, sum initialization or
a separate cast pass. It retains the production caller guards, numerical law,
two-plane tape, dtype choices and the existing VJP. The source is isolated in
[`candidates/direct_unique_output/`](../../bench/factorized_interactions/candidates/direct_unique_output/).
It is not qualified by compilation alone.
For packet promotion the strongest qualified applicable control defines the
comparison: the unique-output mode on its verified bijective cases and the
minimal direct mode elsewhere. Baseline direct measurements may be retained as
descriptive results. Sharing timing plumbing does not justify using a weaker
control for selection.

Record requested route and actual implementation/backend for every sample.
Shared stable/stable results remain in the report but are not packet-layout
comparisons. The current frozen corpus predicts nine distinct packet/direct
profiles and four shared-stable profiles; device metadata must confirm this.
The shared-stable cases do not isolate the proposed context-reuse or large-batch
layout benefits. Do not hide them or alter the frozen corpus to create wins.

These native comparisons do not establish a general Torch advantage. An
equivalent optimized Torch/ATen comparison and whole-model evidence remain
separate requirements for that broader claim.

## Paired measurements and selection

After exact-binary correctness and sanitizers pass, use at least **seven
independent paired blocks per case**. Alternate packet/control and
control/packet order within that same case. Each block records its warmup
policy, raw block means and secondary event samples, device UUID, backend identities,
source/binary identities and Scheduler receipt. Inner samples from one block
are not independent paired rounds.
One uninstrumented host-wall block mean supplies each primary observation;
the secondary event samples are not replicated into primary observations.

Bootstrap the paired block comparisons, preserving each pair. The root-approved
production acceptance requires:

- Per-regime point median speedup at least **1.10x**.
- Paired **95% bootstrap confidence lower bound greater than 1.0**.
- At most **5% time regression** against the strongest qualified direct control
  on the complete declared supported workload set.
- Unchanged numerical acceptance and successful mandatory qualification.

Packet losses outside a proposed promotion regime remain visible and use direct
fallback in any selected production route. A favourable subset average does not
authorize a global packet default. The cost of an eventual production guard or
fallback dispatch belongs in its confirming measurement.
At final integration, omitted native and optional-adapter route arguments will
select the strongest qualified direct implementation. Packet execution remains
an explicit request for demonstrated regimes; the captured qualification source
is unchanged while measurements are pending.

## Bounded execution and retained evidence

The installed authority enforces **600 seconds and 128 MiB of scratch per
attempt**. A monolithic engine run with two build trees and all wire files
cannot fit: the 13-case, 14-block, three-arm timing matrix alone produces
1,439,344,284 bytes of input/result wire files. That execution plan is withdrawn.
Neither the workload catalog nor the numerical and selection gates change.

Compile the native source once in the immutable shared build. Later phases use
those exact runner/library bytes or verified byte-identical prebuilt copies;
they record native source, toolchain and binary identity separately from the
collector's source identity. Collectors never rebuild or mutate that build.

Run separate bounded qualification packets for the 32 dtype/mode cases, the
complete 12-case auxiliary corpus for each candidate role, and each of the four
sanitizer tools. Each auxiliary role has an 11-case regular partition and a
separate one-million-mechanism partition; both are mandatory. Each tool
partition executes the unchanged suite's 14 cases;
all four distinct partitions are required to establish the full 56-case gate.
Combine qualification only after matching the original completed receipts,
cleanup, UUID, source and actual runner/library hashes.

Collect timing as one case and one phase per packet: seven exploration blocks
and seven confirmation blocks, each with the paired candidates and descriptive
direct arm. This produces 26 timing partitions while preserving all 13 cases,
14 paired blocks and three arms. AB/BA alternation continues across the phase
boundary. Each CUDA process runs in the sole admitted Autobench/Scheduler lane
and retains its actual inherited job, attempt, cgroup and UUID.

Fresh successful collector wire files may be transient only after the fixed
validator's numerical checks pass and its exact summaries, raw timing vectors,
error summaries, input/output digests and source/binary identity have been
durably written. Failure wires and all previous evidence remain intact.
Collectors account for scratch before cleanup and include meaningful peak-bound
preflights; a small final directory alone is insufficient evidence of compliance.
The 50 ms filesystem sampler reports an observed peak and can miss shorter
transients. Keep that limitation explicit alongside exact wire-size accounting,
CPU calibration of the real reference cache, safety allowance and the worker's
enforced cap; do not describe a sampled maximum as a deterministic bound.

## KDA recorded-evidence analysis

Use KDA's supported task/command interfaces for **CPU-only analysis of recorded
CUDA observations**. Its probe and task describe recorded evidence analysis;
they do not claim fresh GPU execution or fabricate allocation markers. Bind
candidate snapshots and immutable prebuilt artifact copies to the qualified and
timed bytes, and bind every observation to its original raw sample hash and
finished CUDA receipt/cleanup/UUID. Record the distinct CPU analysis receipt.
No engine database or execution state is imported or edited.

Return each original block observation exactly once for its declared
case/phase/round. Secondary event vectors and repeated reads of a file do not
create independent observations. Keep the engine workspace inside its bounded
analysis output and prove its peak scratch usage. Numerical validation still
occurs during the admitted CUDA collection; replay validates the complete,
source-bound evidence rather than manufacturing new numerical or timing data.

The engine's `max_regression` is expressed as lost benefit ratio. A 5% time
cap therefore maps to `1 - 1/1.05`, approximately `0.04761904761904767`.
Use engine `min_improvement=0` for confidence lower bound greater than one,
plus a hashed trusted selection guard for the independent point-median 1.10x
requirement. Do not silently replace this with confidence greater than 1.10x.

KDA evaluation can automatically write a private preferred pointer. Keep its
workspace explicitly isolated and provisional: that pointer records an engine
ranking, not authority to change a production route. Publish a separate accepted
selection manifest only when every exact root policy check passes; retain
rejected attempts and both meanings in the evidence. Root retains native
default/selection acceptance. No KDA source modification is required.

## Evidence boundary

Source-bound qualification and measurement records establish the demonstrated
coverage. Compilation, CPU protocol checks and a provisional engine ranking do
not establish a GPU speedup. Independent oracle review and root acceptance
remain necessary before selecting a native route. The mathematical contract and
protected native inputs remain unchanged by this resource-based execution split.
