# TRAIN qualification receipt

Status: implementation prepared; **CUDA numerical qualification is pending**.
Do not treat compilation or operator registration as training acceptance.

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

## Evidence obtained

The native fixture, both Torch producer variants, the new C++ training test,
and the retained N16 regression compiled. An independent CMake consumer built
against the installed `CelleraTorch::mechanism` package. The installed Python
shared library loaded and registered the custom class and both dispatcher
schemas. The installed Python package's two host declaration/identity tests
passed; those tests do not execute CUDA numerics.

Focused source review corrected tensor ownership, replay admission, writer
stream ordering, failed-writer recovery, optimizer membership, checkpoint
moment permutation and mixed coefficient overflow admission. These fixes still
require runtime qualification.

Build parallelism was capped at 16 per producer because the two producer builds
and the native fixture were compiled concurrently on a 62 GiB host. Native-only
incremental builds used two jobs. No speedup or performance result is claimed.

## Remaining acceptance

The six required executable gates are recorded in `required-gates.json` and
bound to CE-ML2-TRAIN: native reference/lifetime tests, memcheck, initcheck,
installed C++ composition/training/checkpoint tests, installed Python tests,
and the preserved combined N16 regression. The C++ test reports complete
training and optimizer-step timing; the native fixture reports preparation,
forward/VJP timing and reserved memory. These results are pending.

At preflight, all four V100s were reserved by two idle Project Control observer
services. The supported CUDA controller requested preemption, but its test
launches returned `foreground_resource_contention`, including a 30-second
grace attempt. Investigation found the observer process polls preemption after
observer calls rather than on an idle timer; no public per-observer stop API
was available. A Project Control restart was requested from the user to release
the observers. No service was restarted and no competing GPU process was killed.

The claim must not be completed until the required gates pass. No dataset is
needed for these gates; no GlassHelix scientific training or CE-ML2-BIO work
has started.
