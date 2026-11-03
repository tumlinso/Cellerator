# Chromatin-conditioned CUDA study

This directory contains a private Cellerator execution study. Its fixtures,
CUDA candidates, runner and controls are research artifacts, not installed
APIs, production defaults, a trained biological model, or biological gate
discovery. The reviewed [protocol](../../planning/chromatin-geometry-study-2026-11-02/protocol.md)
and [evidence index](../../planning/chromatin-geometry-study-2026-11-02/EVIDENCE.md)
state the study's contracts, cost boundary, acceptance criteria and evidence.

## Standalone build and host checks

From the Cellerator root, using the pinned CUDA 12.9 and GCC 12 toolchain:

```sh
cmake -S bench/chromatin_geometry -B /tmp/cgs-build \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cgs-build -j2
ctest --test-dir /tmp/cgs-build --output-on-failure
/tmp/cgs-build/cgs_study --help
```

The CTest entries are host-only and do not initialize a GPU. HDF5 C and
OpenSSL Crypto are required for native public-data preparation; CMake reports
if the HDF5-dependent converter is unavailable. The environment's default
`/usr/bin/nvcc` may select an incompatible toolkit, so pass the compiler path
explicitly. For a source-closure build, recreate the Cellerator directory layout
inside the extracted closure and set `CELLERATOR_ROOT` if needed; do not point a
measured build at source outside that closure.

`cgs_contract_smoke`, `cgs_reference_smoke`, `cgs_campaign_layout_test`,
`cgs_replay_timing_test`, `cgs_ste_host_tests`, and `cgs_teacher_host_tests`
are host checks. The replay timing check uses the pinned local payload when
available; CTest marks it skipped otherwise. Set `CGS_REPLAY_PAYLOAD` at
configure time to use another copy of the same verified payload.
`cgs_ste_gpu_tests`,
`cgs_learning_tests`, `cgs_condition_tests`, `cgs_consumer_tests`, and
`cgs_linear_library_tests` are GPU qualification binaries. `cgs_study`
contains diagnostic and campaign runners. None of the GPU binaries should be
launched directly: qualification, sanitizer, profiling and timing use the
study's Autobench/Scheduler controller and the assigned UUID-selected lease.

## Public accessibility replay

The converter uses CellShard's native 10x HDF5 ingestion helpers and does not
create a CUDA context. Given the pinned 10x filtered feature-barcode HDF5 file
and an unused output directory, run:

```sh
/tmp/cgs-build/cgs_prepare_atac "$INPUT_H5" "$OUTPUT_DIR" \
  bbd84ebd43c7bf1f1cc0168fd38c3689b0279730
/tmp/cgs-build/cgs_verify_atac "$OUTPUT_DIR"
```

Set `INPUT_H5` and `OUTPUT_DIR` to the local input file and destination before
running these commands. The output includes `accessibility.cgatac`, peak and
cell identity sidecars, and a provenance manifest. The independently checked
payload SHA-256 is
`45de0da98a8dc6fa19f33eacc52223bc5c3190e956f769f7ff4b253e1d92bad3`;
the source HDF5 SHA-256 is
`0668e2bf9ff521a8e5e7aa37717f3780bad84081f26f888df16a9398d5020e76`.
The runner's replay selects deterministic cell and peak prefixes and verifies
the payload hash. A zero denotes a non-detection in this workload, not a
certified biological exclusion. See the output manifest and evidence index for
the dataset URL, attribution, sidecar hashes and independent checks.

## Running the campaign

`python3 bench/chromatin_geometry/scripts/campaign.py --help` describes the
controller's lane, registration, step, settlement and budget commands. The
controller uses the separate study Autobench enrollment, accounts for aggregate
GPU time and sends work through Scheduler; the parent workspace's pause is not
changed. A source-closure snapshot, input hashes, binary identity, admission
and cleanup receipts are part of an attributable result. Use the controller
and the [protocol](../../planning/chromatin-geometry-study-2026-11-02/protocol.md)
for a campaign rather than running the binary on an unleased device. The
`body-check` mode is a diagnostic, not a complete-cost measurement or an
accepted performance comparison.

The private `cohort-direct-two-tile` candidate is a cached-load control with
two cell tiles per CTA, matching the shared-staging candidate's tile ownership.
The original `cohort-direct` arm retains one tile per CTA. This additional arm
allows staging and tile ownership to be examined separately; its presence does
not imply a performance result or a production specialization.

The private `masked-direct-two-tile` arm preserves contiguous cell order and
uses the same two-tile CTA ownership and numerical body as the cached cohort
arm. It isolates the cost and benefit of forming cohorts; it uses one staging
buffer setting, with cached loads and no shared-memory staging.

The three gate profiles have distinct semantics: supplied and hardened teacher
decisions are Boolean, while STE uses `q = sigmoid(score / T) * admission` and
its declared surrogate includes dormant derivative candidates. Product inputs
are stored in FP16, with FP32 computation, outputs and VJPs. Grouped linear
honors FP16 local input, weight, output and gradient rounding boundaries with
FP32 internal accumulation; FP32 is an explicit validation alternative.
Preparation, gate construction, updates, routing and output recovery must be
included when the corresponding end-to-end comparison requires them. A speed
claim requires the paired, same-device evidence and practical threshold in the
protocol; the existence of a kernel or a compiled target does not establish one.
