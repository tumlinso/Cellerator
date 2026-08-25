# CE-ARCH-30 architecture evidence

This directory defines the reproducible evidence package used to falsify, not
merely confirm, future Cellerator execution strategies. It contains no runtime
format, kernel, benchmark executable, active CUDA watch, or published
biological dataset.

## Scientific boundary

Every committed trace is a structure-only bipartite support fixture. Rows are
the declared observation or source axis; columns are the declared feature or
destination axis. Values, normalization, batch correction, inferred biological
labels, and cross-dataset feature reconciliation are outside this package.

The two available local H5AD files are checksum-pinned in
`../../data/manifests/architecture_evidence/sources.json`. They lack sufficient
donor, sample, batch, chemistry, and species provenance for quantitative or
split-sensitive claims. The GSE147520 `X` values are non-integral and no count
layer is present, so its filename is not treated as evidence of raw counts.
Both sources are therefore support-only inputs. All other biological source
classes are either clearly labeled deterministic structural proxies or blocked
activation recipes with provenance, coordinate, namespace, and checksum gates.

No source is merged, normalized, imputed, harmonized, or silently converted to
a shared feature space. Matched multiome activation must retain distinct RNA
and ATAC feature identities plus the exact barcode linkage. Genomic relations
must pin genome build and coordinate namespace. Learned structures must keep
training-split provenance, immutable topology, and mutable values separate.

## Package contents

- `workloads.json` specifies biological and adversarial structural regimes,
  required dense widths, precision activation rules, reuse horizons, and axis
  semantics.
- `trace_tool.py` deterministically generates compact JSON smoke traces or
  local Matrix Market pattern matrices, and extracts H5AD CSR support without
  treating stored values as biology.
- `benchmark_contract.json` requires complete host-to-consumer phase, memory,
  order, communication, and amortization accounting across native and
  conventional baselines.
- `resource_contracts.json` defines correctness-first smoke, representative,
  throughput, adversarial, and deep-profile tiers.
- `watch_plan.json` preserves historical CUDA watches and leaves all new watch
  families unarmed until real targets and checkpoints exist.
- `validate_evidence.py` and `test_architecture_evidence.py` are CPU-only
  package checks. They do not build or execute Cellerator kernels.
- `real_traces/` contains two checksum-pinned, structure-only support extracts
  from the locally verified PBMC3K and GSE147520 sources. The companion
  `representative_trace_index.json` freezes their source identity, extraction
  provenance, exact payload, planner-ready occupancy features, and the required
  forward/transpose plus runtime-pressure observability for later candidates.

The committed real traces deliberately contain no expression values or inferred
labels. PBMC3K contributes 512 selected rows, 32,738 features, and 433,808
support edges; GSE147520 contributes 256 selected rows, 26,587 features, and
423,731 support edges. Their very different row-degree and 32-feature-block
occupancy distributions prevent the older high/medium/low-sharing synthetic
fixtures from being the only planner feature evidence.

Large generated `.mtx` files remain local under `Cellerator/data/` and are
already ignored. Compact committed traces are capped at one million edges. A
trace checksum identifies exact support bytes; it is not a semantic biological
identity and must not substitute for future DomainId, OrderId, GeometryId, or
StructureId contracts.

## Current validation commands

Run from the Cellerator repository root:

```bash
python3 bench/architecture_evidence/trace_tool.py generate \
  --workloads data/manifests/architecture_evidence/workloads.json \
  --workload scatac-modular-smoke \
  --output /tmp/scatac-modular-smoke.json

python3 bench/architecture_evidence/trace_tool.py validate \
  --trace /tmp/scatac-modular-smoke.json

python3 -m unittest discover \
  -s bench/architecture_evidence \
  -p 'test_*.py'

python3 bench/architecture_evidence/validate_evidence.py \
  --verify-local-sources --json
```

An optional deterministic support extraction, still without quantitative
semantics, is:

```bash
python3 bench/architecture_evidence/trace_tool.py extract-h5ad \
  --sources data/manifests/architecture_evidence/sources.json \
  --source pbmc3k-raw-local \
  --rows 512 --seed 7 \
  --format matrix-market \
  --output data/local/architecture_evidence/pbmc3k-support.mtx
```

## CE-ARCH-76 small-N candidate evidence

`ce_arch_76_v100.jsonl` is the first activated representative result in this
package. It compares the native row-masked, CSR fallback, and feature-major
small-N candidates at N=1,2,4,8,16 over identical 65,536-row, 32,768-feature,
32-nnz/row structures with high, medium, and low feature sharing. Every record
uses the same value generation, row-major dense input, overwrite output effect,
execution-row-major output, f16 sparse values, and f32 multiply/accumulation.

The row-masked and CSR candidates are rank-1 contracts. Their reported
end-to-end steady-state time therefore includes the common row-major-to-column
pack, N native launches, and column-to-row-major output interleave. The
feature-major candidate accepts the common KxN/MxN contract directly. Persistent
projection construction, value packing, and backend preparation are reported
separately and are not repeated inside the 11 timed samples. The checked-in
`ce_arch_76_v100_spec.json` reproduces the mutex-protected controller campaign.

The evidence was captured from clean commit `df9d168` on a Tesla V100-SXM2-16GB
(sm_70), CUDA runtime 12.9, driver API 13.0. Controller evidence id
`46fd716e-5f4e-4d53-950e-05a74f96da7c` records the clean source fingerprint,
binary digest, correctness command, quiescence proof, and resource samples.
Maximum median absolute deviation was 1.32%.

Steady-state winners were:

| feature sharing | N=1 | N=2 | N=4 | N=8 | N=16 |
|---|---|---|---|---|---|
| high | row-masked | feature-major | feature-major | feature-major | feature-major |
| medium | row-masked | row-masked | row-masked | feature-major | feature-major |
| low | CSR | CSR | CSR | feature-major | feature-major |

At the declared eight-use reuse horizon, row-masked won every measured cell
because it reuses CPK1 directly while CSR and feature-major construction costs
were not yet amortized. This is a preparation-horizon result, not evidence that
the feature-major steady-state regime is absent. Objective V2 must retain both
effects and defer uncertain or novel keys to empirical measurement.

## CE-ARCH-77 Objective V2 calibration

`ce_arch_77_objective_v2_v100.json` records the replaceable nonnegative fit
consumed by `objective_v2_calibration.hh`. The fit uses no candidate id or name.
Its predictors are useful interactions, masked row lane slots, linear edge
visits, masked feature lane slots, dense-RHS vector elements, compact feature
value loads, and launch count. Dynamic input packing and output-order bytes have
separate measured phase coefficients. Projection construction, backend prepare,
and value packing stay explicit measured estimates and are amortized through the
existing planner reuse keys.

The current 45-sample fit has 4.80% median relative error and 35.92% worst-case
relative error. Those errors are too large for a 2% final decision, so the model
ranks the analytical shortlist but explicitly requires empirical measurement.
N outside 1,2,4,8,16, shapes outside the measured structural support, and stale
V100/build identities are not extrapolated. Applied predictions revalidate the
complete structure/geometry/device/build/policy key. Thus structure epochs and
all other existing planning invalidators remain authoritative.

For CP-BP alternating refinement, the calibration supplies weights over its
existing held-out measured runtime and preprocessing terms: runtime has unit
weight and preparation has `1 / expected_reuse` weight. It does not alter CPK1,
inject candidate identity, or replace held-out empirical acceptance. The model
is data rather than a frozen planner law, and future candidate families can
populate the same mechanism counts or replace the fit after new evidence.

## CE-ARCH-84 forward plurality evidence

`ce_arch_84_v100.jsonl` extends the equivalent CE-ARCH-76 comparison to
N=17,32,64. The retained CTA schedule consumes FMP1 directly: one 128-thread
block owns a 32-row tile, four warps split dense columns for each row, and each
feature's dense RHS vector is staged once in shared memory. It has a distinct
candidate identity and exact 17..64 capability range, but truthfully retains
the native-feature-major projection kind because it neither constructs another
payload nor converts FMP1 in the launch path.

On the same 65,536-row V100 fixtures, CTA steady-state execution won all three
sharing regimes at N=32 and N=64 plus high sharing at N=17. Row-masked remained
faster for medium-sharing N=17, and CSR remained faster for low-sharing N=17.
Those losing cases are deliberate fallback/low-sharing decisions, not prompts
for another format. At the declared eight-use horizon row-masked still won all
cells because FMP1 construction and value packing had not amortized. Maximum
median absolute deviation was 0.62%, and the independent referee plus the
partial-mask/empty-row boundary test passed at N=17 and N=64. Controller
evidence id `e1703b9d-1675-404b-a721-6ec20c771679` records the mutex-protected
campaign and exact binary/source identities.

## CE-ARCH-92 real and adversarial regime evidence

`ce_arch_92_v100.jsonl` records 36 correct candidate measurements from a
serialized Tesla V100-SXM2-16GB campaign. The inputs are the checksum-pinned
GSE147520 and PBMC3K support traces, the most occupied native 16-feature block
derived deterministically from GSE147520, and the adversarial partial-block
trace. Each trace is measured at N=1,16,32 with three warmups and eleven
repeats. The campaign runner isolates every `(trace, N)` cell in its own process
while the controller holds one benchmark mutex and device lease, preventing
cross-width harness state from affecting comparison. Candidate construction,
value packing, backend preparation, dynamic input packing, kernel, output-order
work, and the eight-use amortized total remain separately visible.

| trace | N=1 | N=16 | N=32 |
|---|---|---|---|
| GSE147520 full support | CSR | feature-major | feature-major CTA |
| PBMC3K full support | CSR | feature-major | feature-major CTA |
| GSE147520 high-sharing block | row-masked | feature-major | feature-major CTA |
| adversarial partial blocks | CSR | feature-major | feature-major CTA |

Every winner clears the runner-up by more than the declared 2% practical
threshold; the narrowest margin is 10.43% and maximum timing MAD is 2.05%.
Every record uses overwrite output semantics, packed-row-major RHS,
execution-row-major output, f16 relation values, f32 multiply/accumulation, and
the independent referee's `1e-5 + 1e-5 * |reference|` tolerance. Controller
evidence id `490d2ba1-99ce-4d3e-ba1c-65db915a42d1` records quiescence,
device/toolchain identity, binary digest, and source fingerprint. Reproduce the
GPU run with `ce_arch_92_v100_spec.json`, then annotate and validate it with
`finalize_ce_arch_92_evidence.py`. The derived high-sharing trace is reproduced
by `trace_tool.py derive-block` and is byte-tested against the committed file.

The evidence justifies retaining all three forward organizations: row-masked
for a preparation-sensitive high-sharing N=1 real regime, conventional CSR for
full real and adversarial N=1 fallbacks, and feature-major warp/CTA execution
when dense-RHS reuse dominates at N=16/32. It is not a universal ordering and
does not measure future candidate families.

## Future measured execution

Candidate families beyond the bounded CE-ARCH-76, CE-ARCH-84, and CE-ARCH-92
activations remain unarmed. After their named checkpoints and targets in
`watch_plan.json` exist,
the CUDA controller must create a clean immutable snapshot, run independent
correctness first, acquire the declared benchmark or profiler and GPU leases,
check memory headroom, and then invoke the future end-to-end benchmark. Deep
profiling is limited to one selected candidate-workload pair per revision.

Each result must record exact repository and toolchain identities, trace
checksum, operation and candidate, numerical and order policies, N, structure
and projection reuse, correctness digest, every phase, persistent and transient
memory, data movement, command, and environment digest. Kernel-only timing is
never sufficient. A conventional CSR, SELL, BSR, valid Blocked-ELL, cuSPARSE,
or dense fallback must win whenever measured end-to-end evidence says it
should.

Stop before timing on any semantic, identity, order, capacity, numeric,
determinism, canary, or sanitizer failure. Report an inconclusive result instead
of a winner when resource contamination or variance exceeds the declared
contract.
