# Native chromatin-conditioned execution study

**Paused for strategy discussion on 2026-11-03.** The resumed work added
reviewed process-observation capture and two independently verified full-chain
correctness packets. No new performance result qualified. The matched
contiguous masked arm compiles and passes host tests; GPU qualification is
pending. The [current resource checkpoint](../../../planning/chromatin-geometry-study-2026-11-02/pause-resources-2026-11-03.json)
records 38.869 aggregate GPU-minutes used and verified cleanup.

**Resumed on 2026-11-03.** The pause findings below remain historical evidence;
the resumed campaign is completing the outstanding measurement and acceptance
coverage under the original aggregate budget. See the
[resumption notes](../../../planning/chromatin-geometry-study-2026-11-02/resumption-2026-11-03.md).

The campaign is **paused after the user-requested bounded milestone attempt
on 2026-11-02**. All four v3 sanitizer modes passed. The first paired
staging/update attempt was rejected for an unattributed foreign GPU process;
its timings support no conclusion. The study is not complete.
The [qualified results](qualification-summary.md) establish one
scenario; remaining scenarios, matched staging and dependency-update
comparisons, and final profiler coverage are unfinished. The
[pause handoff](../../../planning/chromatin-geometry-study-2026-11-02/PAUSED.md)
records the resume frontier and resource state.

## Question and scope

Can supplied binary accessibility conditions, learned STE admissions, or a
small hardened teacher reduce the complete cost of ordered products and
grouped-linear blocks? The candidates test condition-plane consumption,
local execution cohorts, and demanded-feature/parameter reuse on V100 GPUs.
They are private native C++/CUDA study implementations. They do not establish
biological gate discovery or introduce installed modeling APIs.

The biological interpretation is deliberately narrow. The public input
supplies observed accessibility patterns. Numerical values, overlapping
regulatory neighborhoods, distal memberships, and parameter sharing are
controlled fixtures. A non-detection is a workload zero, not a certified
biological exclusion. Sequence interpretation remains outside this study.

## Gate and numerical contracts

| Route | Forward multiplier | Learning behavior |
| --- | --- | --- |
| Supplied Boolean | Supplied binary eligibility | Mechanism gradients on admitted work; no gate learning |
| Existing STE | `p*b`, with `p=sigmoid(score/T)` and `b=(p>=threshold)` | Declared `dq/dscore=p*(1-p)/T` over every derivative candidate, including dormant candidates |
| Teacher to Boolean | Thresholded three-input/eight-row Bernoulli teacher, compiled to Boolean rules | Native numerical teacher fitting; hardened decisions stop task gradients |

Hard exclusion skips participant reads in forward. STE recomputes dormant
local values in backward where its estimator requires them. The teacher's
Boolean circuit is checked against its thresholded table for all eight rows;
equivalence to the soft teacher is a separate question. Local rules compose
through condition DAGs rather than enumerating whole-neighborhood states.

Product cases store inputs in FP16 and compute outputs and VJPs in FP32.
Argument order, repeated participants, zeros, shared parameters, and overlapping
outputs are preserved. The existing smooth-gate product passes its own
regression reference; it is not a semantics-matched binary or STE baseline.

Grouped linear retains FP16 input/weight, local output, and local gradient
rounding boundaries with FP32 internal accumulation and canonical reductions.
FP32 is an explicit alternate contract. Candidate/control comparisons keep
these contracts identical. The independent references and frozen compound
bounds are described in [the numerical contract](../reference/CONTRACT.md).

## Workloads and public input

The frozen complete cases cover seven-node and 2,304-node condition banks,
64 or 2,176 numerical groups, low/high participant overlap, sparse/dense
synthetic activity, frozen/every-step parameter updates, reuse horizons
1/16/256, and leaf-input change fractions 0/1/10/100%. Separate producer-only
diagnostics are excluded from complete-consumer performance conclusions.
Actual replay activity and learned gate membership can differ substantially
from a fixture's synthetic label. The [structural analysis](structural-analysis.md)
records those differences and logical demand counts without treating them as
measured HBM traffic.

The source is the [10x PBMC Multiome ARC 1.0.0 dataset](https://www.10xgenomics.com/datasets/pbmc-from-a-healthy-donor-no-cell-sorting-10-k-1-standard-1-0-0).
Native CellShard ingestion preserves the 12,016 cells and 63,751 ATAC peaks.
An independent native HDF5 comparator checks every prepared bit and identity
against the downloaded source; a flipped-bit negative probe fails. The
complete GPU fixtures select the first 256 cells and 4,096 peaks. Controlled
leaf flips form the update cases; they are not a second observed time point.
The seven-node bank reads four accessibility leaves and the 2,304-node bank
reads 128; the study does not exercise every source peak as a condition input.

| Artifact | SHA-256 |
| --- | --- |
| Filtered feature-barcode HDF5 | `0668e2bf9ff521a8e5e7aa37717f3780bad84081f26f888df16a9398d5020e76` |
| Prepared CGATAC01 accessibility payload | `45de0da98a8dc6fa19f33eacc52223bc5c3190e956f769f7ff4b253e1d92bad3` |

The packed replay is expanded to selected host bytes and repacked by the
GPU condition producer. This is not an optimal packed-input transfer path.
The separately measured native HDF5 conversion took 12.389 seconds wall time
with about 1.78 GiB peak RSS, without a GPU or a flushed OS cache; see the
[ingestion receipt](../../../planning/chromatin-geometry-study-2026-11-02/native-ingest-cost.json).

## Controls and execution choices

The matched controls are direct root-map CUDA, existing condition planes
followed by bitmap/workset materialization, and direct condition-plane
consumption. Grouped-linear comparisons also include sequential and batched
cuBLAS, charging packing, activity readback, synchronization and staged
rounding. The strongest applicable measured control is used for each
specialization decision.

Cohorts group equal local `(primal, derivative)` programs and retain supplied
neighborhood boundaries. The original cached arm owns one cell tile per CTA.
An added cached two-tile arm matches shared staging's two successive tiles,
tile order, cohort assignment and inter-tile barrier. Shared staging adds
demanded-feature/weight loads, buffering and synchronization. The paired
comparison therefore isolates the staging strategy under matched tile
ownership; it is not a single-instruction cache comparison.

Tile sizes are 32/64/128 with one/two staging buffers where supported. The
private implementation's 48,000-byte shared-memory cap rejects every frozen
128-cell/two-buffer case before GPU submission. These are capacity
rejections, not timings. All paths use SM70-compatible mechanisms.

Known candidate limits affect interpretation. Bitmap worksets still use a
dense-capacity dispatch grid with empty-warps returning early. Cohort product
VJPs use a scalar quadratic algorithm, whereas the native controls reuse warp
prefix/suffix scans through arity 32. Cohort starts can concentrate work on
one warp. Linear weight gradients reread inputs, and STE recomputes local
values in backward. A cohort win cannot be attributed solely to memory
reuse. These limits are retained for the measured implementation rather than
generalized to all possible encodings or schedules.

## Measurement and acceptance

The primary complete cost is cold execution preparation, plus the recurrent
forward/backward/update horizon, plus output recovery, each counted once.
Applicable replay verification/unpacking, gate construction, fitting,
hardening, rebuilding, routing, packing, transfers and updates are included.
Fixture generation, independent oracle work, trial reset and warmup are
excluded. Recurring and per-phase costs remain separate diagnostic fields.

KDA identifies/evaluates candidates. Autobench supplies immutable captures,
builds and measurement packets; Scheduler grants finite exclusive
UUID-selected GPU jobs. The study has a separate enrollment and one locked
240 aggregate GPU-minute budget. The parent workspace's pause is preserved.
Every result binds source archive, binary, input, toolchain, admission and
cleanup identities. Reconstructed snapshot manifests are checked against
immutable bytes and the official snapshot fingerprint where the original SDK
manifest was unavailable; this limitation is retained in the evidence.

Confirmation jobs are serialized across this study. All-device pre/during/post
samples and retained native-child lifetimes reject observed contamination.
The concurrent timing probe was inconclusive. Sampling cannot prove
host-wide exclusivity or exclude work between observations. The exact
retrospective utilization-attribution checks remain inspectable in the
qualification exporter and raw packets.

Each independent paired round rotates all applicable controls and candidates.
Samples within a round are summarized together. A specialization requires
strictly more than 5% median latency reduction over the strongest matched
control and a paired bootstrap 95% interval supporting improvement. Profiler
replay timings do not enter that decision. Failed, rejected and inconclusive
alternatives remain in the evidence exports.

## Current measured result

One source-bound v2 scenario is qualified: teacher gate, grouped-linear FP32,
public replay, dependency-driven condition updates, one-step reuse, tile 32,
one staging buffer. Six paired rounds compared all seven then-existing arms
on the same V100. Bitmap was the fastest matched control at 465.30 ms median
complete cost.

| Candidate | Median latency reduction vs bitmap | Paired speedup, 95% interval | Outcome |
| --- | ---: | ---: | --- |
| Cached one-tile cohort | -2.55% | 0.9751 [0.9368, 1.0095] | No median improvement |
| Shared two-tile cohort | 0.49% | 1.0050 [0.9718, 1.0553] | Does not meet the practical threshold |

This historical comparison bundles tile ownership with staging. The added
matched cached two-tile arm passed component and complete public-replay
correctness plus four-tool representative sanitizer qualification on the v3
closure, but has no qualified measurement in this table.
Exact measured
binary/source sanitizer joins and retained failure classifications are in
[the qualification export](qualification.json) and
[sanitizer coverage](sanitizer-coverage.json).

Cold preparation dominates this scenario: per-arm medians are roughly
455–495 ms, versus 0.57–1.95 ms for its one-step recurrent horizon. Shared
staging's recurrent median is 1.20 ms versus bitmap's 0.63 ms. The small
complete-cost difference therefore does not establish a faster recurring
numerical path. The [cost and memory breakdown](cost-summary.md) keeps these
components separate and records the teacher's approximately 0.121
workload-weighted mean absolute gate-probability change after hardening.
Circuit equivalence does not imply
zero soft-to-hard change.

## Remaining acceptance work

The current result does not complete the study. The frozen broader matrix,
matched staging/ownership and rebuild/update ablations, reuse break-even
analysis, final HBM/shared-conflict/spill/synchronization profiles, and final
independent acceptance remain unfinished. The v3 attempt retained all 48 base
and 96 update raw records but failed its telemetry guard; none is used for a
performance comparison. No retry was submitted. All study lanes are now
inactive and explicitly paused, all cleanup is verified, and no GPU reservation
is open. The study used 34.212 of 240 aggregate GPU-minutes. Any further campaign
must await explicit user resumption. Available budget is not completion evidence.
The [v3 milestone export](qualification-v3-summary.md) retains the rejected
records and independently qualified sanitizer receipts. It also discloses a
separate ledger-window reconstruction gap for two cancelled automatic
submissions; no missing pre-attempt window was invented.

The [retained diagnostic profile](profile-interpretation.md) supports a
small-grid and metadata/synchronization interpretation for one sparse-after-
toggle product launch. Its low DRAM utilization and eight excessive shared
wavefronts establish neither general HBM savings nor a useful padding
specialization. Final matched v3 counter profiles remain unfinished.

Reproduction starts with the [standalone build and campaign instructions](../README.md).
The [evidence index](../../../planning/chromatin-geometry-study-2026-11-02/EVIDENCE.md)
links the approved protocol, source baseline, immutable contract revisions,
reviews and receipts. The native Todo lifecycle remains unresolved under the
[user-approved workflow exception](../../../planning/chromatin-geometry-study-2026-11-02/workflow-exception.md);
scientific study acceptance does not imply native Todo acceptance.
