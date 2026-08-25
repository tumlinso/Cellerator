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

## Future measured execution

Apart from the bounded CE-ARCH-76 activation above, the broader evidence
families remain unarmed. After their named checkpoints and targets in
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
