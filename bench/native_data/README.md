# Native data gated-product benchmark

This directory contains a conventional C++ ATen CUDA baseline and the
deterministic workload generator for the first native-data measurement.
The baseline uses ATen eager `index_select`/`prod`/`index_add` and C++ autograd;
it does not call a Cellerator numerical kernel. It uses the same group ownership
as the specialist runner and the same source homes: FP16 expression and FP32
`theta/phi/tau` on device 0, FP32 regulators on device 1, and cotangents on
device 0. On first use, ATen explicitly copies expression and parameters to
device 1 and regulators to device 0. It reduces outputs, `dx`, and parameter
VJPs to device 0, while canonical `dz` returns to its home on device 1. One
theta SGD update is then replicated to device 1.

The approved logical source-owner order is GPU `cf22…` for slot 0 and GPU
`6c1…` for slot 1. Scheduler forecast returns the same exclusive pair in
resource order `6c1…`, `cf22…`, so the packet maps logical slots to CUDA-visible
ordinals `[1, 0]`. Result JSON records both the UUID by logical slot and the
visible ordinal used by each backend. This preserves the declared data homes
while following the resource order Scheduler admitted.

## Fixed first workload set

`reference_active` and `reference_inactive` are constructed directly from the
independent scalar fixture in `tests/native_data/reference/fixture.hh`. The
exact Fraction oracle in `tests/native_data/reference/exact_oracle.py` remains
the validation authority. For the inactive fixture, outputs, `dx`, zero gate
and parameter VJPs, and post-update outputs are exact; `theta_after_update` must
retain the same FP32 bit pattern as its initial value.

The timing cases are a deliberately small geometry contrast, each with one
logical sample and 32 ordered feature slots per group:

| Case | Groups | Coordinates | Input support | Output support | Active groups |
| --- | ---: | ---: | --- | --- | ---: |
| `low` | 512 | 16,384 | Disjoint blocks, stride 32 | One distinct output per group | 51/512 (9.9609%) |
| `high` | 1,024 | 16,400 | Sliding blocks, stride 16 (50% adjacent overlap) | Adjacent groups share an output, with scales 1 and 1/2 | 922/1,024 (90.0391%) |

Both timing cases reuse all 64 regulator and parameter indices and split group
ownership evenly across the two devices. Per-regulator group frequencies are
balanced within each activity class so 64 shared regulators still produce the
nearest integer 10% and 90% activity fractions. Input values come from a fixed
SplitMix64 stream in `[0.98, 1.02)`. Regulators, theta, phi and tau remain fixed
during a timing packet; theta alone receives the repeated explicit update.
The actual active count and gate-argument range are emitted with each result,
so the packet does not rely on its requested fraction as evidence.

## Cost accounting

The baseline reports host topology/index packing, FP16 conversion, upload,
readiness synchronization, forward, backward, partial reduction, combined
theta update plus replica refresh, and ATen allocator peak allocated/reserved
bytes. `prepared_host_payload_bytes` covers expression, regulator, and
parameter values; topology/index upload bytes are not exposed by both backends,
although preparation time includes their allocation and transfer. Per-cycle
counters record source replication, cotangent, output, VJP, and parameter
refresh traffic. Fixed prepared device bytes and ATen allocator peaks are
labeled as different memory measures. The ATen result stays in canonical
coordinate order, so its device-side canonicalization cost is zero. Copying
arrays to host for oracle validation is reported separately as
`validation_export_ms` and is excluded from timing samples.

The active group index is prepared once because regulators, phi and tau are
fixed for the declared measurement window. The forward and backward still
recompute the gate argument and gate derivative for active groups. A future
workload that updates regulators, phi or tau must charge and measure rebuilding
that index. This baseline uses direct dense-coordinate gathers; it does not
maintain a grouped value projection.

All timing stage boundaries synchronize both selected devices. This reports
complete stage latency, including host dispatch and device completion. Timing
results retain the cold first cycle (including one-time source replication),
warmup cost, and raw recurrent samples plus medians. Result placement metadata
records source owners and canonical result owners by visible device slot, with
the physical UUID assigned to each slot. Autobench receipts establish the
admitted device pair and cleanup; JSON records binary and immutable snapshot
source hashes.

## Execution boundary

The benchmark target is optional and is built only from the CMake integration
owned outside this directory. Compile the CUDA-enabled Torch adapter with
`g++-12` and CUDA SDK 12.9; the installed Torch wheel is 2.7.0+cu126. The
CUDA 12.6 wheel and 12.9 SDK link in the host probe, but GPU runtime
compatibility must be established by the root-coordinated Scheduler job.
Never run the GPU binary directly from a mutable checkout. Registered GPU
commands consume Autobench `{snapshot}` inputs and either a registered
`{build}` artifact or a separately manifested binary built from a verified
Autobench-captured source archive. The scale packet's binaries, archive,
per-file manifest and build log live under the ignored
`.autobench/data/cellerator-native-data-gated-product-scale/` directory. They
are not versioned by Git. Registering `scale-correctness.toml` does not install
or identify a binary by its filename; each execution snapshot carries and
hash-checks the exact staged binary and its external build manifest.

### Rebuilding ignored scale artifacts in a fresh checkout

Set `AUTOBENCH_PYTHON` to a Python interpreter that can import the installed
Autobench package. From the repository root, create the source archive and
detached per-file manifest, then capture the archive with Autobench's ordinary
snapshot implementation:

```sh
"$AUTOBENCH_PYTHON" bench/native_data/freeze_scale_source_archive.py \
  --source-root "$PWD" \
  --archive .autobench/data/cellerator-native-data-gated-product-scale/source-closure-v3.tar \
  --manifest .autobench/data/cellerator-native-data-gated-product-scale/source-closure-v3.manifest.json
"$AUTOBENCH_PYTHON" bench/native_data/capture_scale_source_snapshot.py
```

Use the snapshot path printed by the capture helper to build both binaries
from the captured archive bytes. The script records the source-archive and
manifest hashes, CMake/CUDA/Torch toolchain, and exact build commands; it runs
only a CPU-side `--help` probe and does not launch CUDA kernels.

```sh
BUILD_TAG=$(date -u +%Y%m%dT%H%M%SZ)
"$AUTOBENCH_PYTHON" "$SNAPSHOT/bench/native_data/prepare_scale_artifacts.py" \
  --source-snapshot-root "$SNAPSHOT" \
  --build-root "/tmp/cellerator-native-data-scale-build-$BUILD_TAG" \
  --artifact-root "$PWD/.autobench/data/cellerator-native-data-gated-product-scale"
```

After the binaries and manifests are present, register the correctness packet:

```sh
autobench register --project "$PWD" \
  --file bench/native_data/scale-correctness.toml
```

The source archive and external manifests stay with Autobench's immutable
packet snapshot, while the local build products remain untracked and
reproducible from the recorded recipe.

No performance result is claimed until the native distributed runner and ATen
baseline pass the exact fixture, both scale-case comparison gates pass, and
the parent schedules the shared GPU interval.
