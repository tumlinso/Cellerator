# E7 NVCC versus NVRTC probe

This isolated sub-probe compiles the exact same `region.cu` with offline NVCC
and application-lifetime NVRTC. The region computes CSR `Ax`, then segmented
softmax. It takes row offsets, columns, numerical edge values, `x`, scratch,
output, row count, segment offsets, and segment count as runtime bindings.
The fixed fixture includes signed weights, an empty CSR row, an empty output
segment, and singleton segments. It compares both executions with a host FP64
reference and rebinds edge values without recompiling either module.

The kernel intentionally uses a simple single-thread implementation. This
experiment compares compile/load paths and value binding; it does not measure
an optimization candidate or represent a production segment-softmax kernel.
NVCC's fatbinary is prepared off-device, then loaded into the same CUDA driver
context used for the NVRTC PTX. Both paths receive identical device pointers and
scalar arguments on the same stream. The loaded NVCC fatbinary is the
precompiled execution baseline.

## CPU-only preparation

From the repository root:

```sh
python bench/compiler/experiments/adaptive_fields/gpu/runtime/prepare_nvcc.py \
  --output-dir bench/compiler/experiments/adaptive_fields/gpu/runtime/artifacts \
  --arch sm_70
```

This command runs NVCC only to produce the offline fatbinary and the host
launcher executable. It does not initialize a CUDA device or launch a kernel.
The output manifest binds the region source, compiler binary/version/flags,
fatbinary, launcher source, and launcher executable hashes.

## Admitted runtime invocation

The root task owner must submit the exclusive UUID-bound Device Scheduler job.
The job command is:

```sh
python bench/compiler/experiments/adaptive_fields/gpu/runtime/run_nvrtc_compare.py \
  --manifest bench/compiler/experiments/adaptive_fields/gpu/runtime/artifacts/nvcc_artifact_manifest.json \
  --output bench/compiler/experiments/adaptive_fields/gpu/runtime/artifacts/nvrtc_run_result.json
```

The wrapper and executable require the parent-provided experiment marker,
exactly one UUID-routed `CUDA_VISIBLE_DEVICES` value, and the non-empty
`ADAPTIVE_FIELDS_RUN_KEY` correlation key. The marker is an accidental-run
guard only; it is not authorization or isolation. The run key is recorded only
as correlation metadata. The external Device Scheduler JobSpec, submission,
admission, terminal, and cleanup receipts remain the authority. The wrapper
sets `LD_LIBRARY_PATH` to the selected CUDA 12.9 NVRTC library directory,
hashes artifacts immediately before launch, and records cold NVRTC compilation,
module loads, 5 warmups, 21 alternating paired samples in each of four rounds,
correctness, runtime value rebinding, tool versions, and the selected UUID.
The final GPU sanitizer run remains a root-owned check.

This adapter does not compare application-level complete training cost. It
answers only whether runtime compilation/loading adds value beyond selecting
and loading an already prepared module for this small connected region. Results
must be integrated with the larger E7 workload and amortization evidence.
