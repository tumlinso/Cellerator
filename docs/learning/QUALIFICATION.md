# TRAIN qualification receipt

Status: **all six GPU executable checks passed**.
Project Control completion is tracked separately below.

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

The two Torch producers use distinct build/install trees. Load only the Python
producer into the Python distribution. Set `CELLERATORCH_NATIVE_LIBRARY` to
its `lib/libcellera_torch_mechanism.so` before constructing a Python module.

## GPU qualification

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

## Scheduling and workflow acceptance

With the user's authorization, Project Control was restarted to release idle
observer GPU reservations; readiness returned successfully. Tests retained the
CUDA controller's three consecutive idle samples and used a 60-second idle
proof timeout because this host's monitoring queries exceeded the default
10-second window. No competing GPU process was killed manually.

The bound Project Control gates retain their original default idle timeout;
the supported binding operation cannot amend an existing gate. The completion
attempt refused acceptance: C++ gate evidence
`2e852ec7-ecfb-4851-8934-2a8fee89f6c1` reports `gpu_not_quiescent`,
14.616 seconds elapsed and two of three idle samples. Both samples were idle
with no foreign processes; its executable did not run. The same executable
passed in the 60-second-proof suite above. No task state was overridden.

TRAIN remains open; the claim was released through handoff
`ed34da17-9c10-4de2-bfa8-2f8f4368e7e4` at project revision 7435. Completing
the ledger requires a supported amendment of the bound gate idle timeout.

No dataset is needed for these checks. No GlassHelix scientific training or
CE-ML2-BIO work has started.
