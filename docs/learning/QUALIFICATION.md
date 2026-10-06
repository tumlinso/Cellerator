# Historical TRAIN qualification receipt

This record describes the component-era TRAIN installation from 2026-09-29. Its paths, package names and loader setup are retained as historical evidence and are not current Cellerator installation instructions. Current package acceptance remains pending the active bindings migration.

Status: **TRAIN completed as implemented on 2026-09-29, revision 7474**.
All six bound Project Control gates passed; no active TRAIN claim remains.

## Installed environments

* C++ producer: `/home/tumlinson/Software/cellerator-ml2-cpp`, using the existing
  `/usr/local` libtorch 2.7 installation.
* Python producer: `/home/tumlinson/Software/cellerator-ml2-python`, separately
  linked against Torch `2.7.0+cu126` in
  `/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126` (Python 3.13).
* Compiler: HPC SDK CUDA 12.9, target `sm_70`, C++ ABI 1. No global Torch or CUDA
  installation was replaced. The Python adapter avoids the newer
  `cudaStreamGetDevice` runtime symbol and uses driver context validation.
* Compute Sanitizer: separately extracted official NVIDIA package
  [cuda-sanitizer-12-9_12.9.79-1_amd64.deb](https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-sanitizer-12-9_12.9.79-1_amd64.deb), SHA-256
  `3b2fdf2c3caf82cddc4c4c0f93b3877a5671de04ab7778f9e7d517e497a05218`.
  Executable: `/home/tumlinson/Software/cuda-sanitizer-12.9/usr/local/cuda-12.9/bin/compute-sanitizer`.
  The system PATH wrapper targets an absent 13.1 installation and is not used.

The historical run used two Torch producers with distinct build/install trees.
It installed the Python producer into the then-separate distribution and set
`CELLERATORCH_NATIVE_LIBRARY` to that producer’s
`lib/libcellera_torch_mechanism.so` before constructing a Python module.

## GPU qualification on 2026-09-27

Source commit: `f42ce55a`. All six configured executable checks passed on a
V100 under one CUDA controller reservation, using the exact commands in
`required-gates.json`. Controller evidence:
`c0704f79-fbbb-48ba-80a5-cc6e1a17b360`.

| Check | Result |
| --- | --- |
| Native FP32/mixed reference, gradients, identity, capacity, lifetime and recovery | Passed |
| Compute Sanitizer memcheck | 0 errors |
| Compute Sanitizer initcheck | 0 errors |
| Installed C++ composition, Adam, checkpoint and next-step parity | Passed |
| Installed Python integration | 22 passed, no skips |
| Retained combined N16 path | Passed |

The first integration runs exposed a composite C++ archive replacing CE-owned
storage, an adapter update left open after failed optimizer recovery, and two
Python fixture setup mistakes. These were fixed and the complete suite rerun.
Review additionally found that restore could supersede an active writer;
both native and adapter paths now reject it, with regression coverage.
Python shared modules retain their single coefficient leaf through an explicit
owner binding, avoiding weak references to Torch tensors.

See `gpu-validation.log` for test output and `capability-receipt.json` for
artifact hashes, build evidence, exact source identity and per-check results.
The installed Python source matches the repository package byte-for-byte.

The small C++ fixture reduced loss from 0.0459375 to 0.000788288 in 40 steps;
it reported 113.458 ms for the training lifecycle and 42.3087 ms total guarded
Adam time. These are fixture observations, not a performance comparison.
The native fixture reserved 1344 bytes in each precision mode; the C++ fixture
reported 1856 bytes. No real dataset or biological fit is claimed.

## Bound workflow acceptance on 2026-09-29

The earlier completion attempt failed GPU idle verification before test launch:
its default 10-second window obtained only two of three required idle samples.
A fresh amendment assembled from the current authority applied at revision
7460 sets all six gates to a 60-second window while retaining three consecutive
idle samples. The supported owner-maintenance plan operation preserved task
semantics, scopes, invariants, queue, historical evidence and unrelated state;
a before/after comparison verified those boundaries. No gate was waived.

Project Control then executed all six bound gates successfully, including
22 Python tests with no skips and zero errors from both CUDA sanitizers.
TRAIN closed as `done` / `implemented` at revision 7474. The client call reached
its 300-second wait limit while the server continued; a subsequent authoritative
inspection confirmed completion, so the operation was not repeated.

See `bound-gates-2026-09-29.log` for the new outputs and the `workflow` section
of `capability-receipt.json` for all six evidence IDs and artifact hashes.
The original numerical source remains `f42ce55a`; the qualification checkout
was `52870f8f` with the recorded gate configuration change. No numerical code
was changed during this repair. The GPUs were already free; no service restart
was required on this date.

No dataset is needed for these checks. No GlassHelix scientific training or
CE-ML2-BIO work has started.
