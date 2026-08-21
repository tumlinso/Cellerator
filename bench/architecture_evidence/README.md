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

## Future measured execution

No current executable implements this contract, so no new controller watch is
valid yet. After the named checkpoints and targets in `watch_plan.json` exist,
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
