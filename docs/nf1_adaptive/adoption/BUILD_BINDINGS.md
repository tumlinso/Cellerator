# NF1A build and gate binding inventory

Read-only inventory collected during CE-NF1A-ADOPT on 2026-09-16. This is not a gate binding, test receipt, build result, or implementation authorization.

## Cellerator

The registered root is `/home/tumlinson/Cellerator` at `ff9d66121de9accde04166c7edd95544c24ca297`. Existing `build`, `build-ccp1-final`, and `build-ss1` all point at that root, use Release and CUDA 12.9 NVCC, and have CellShard disabled. They are not a source-pinned NF1A cross-project build and did not expose materialized RU1 CTest entries in this inspection.

The source registers the bounded relation-update suite only with `CELLERATOR_BUILD_RELATION_UPDATE_SPINE_V1=ON` and `CELLERATOR_BUILD_TESTS=ON`; this also enables the semantic-spine suite. Its inner device CTest names are `ru1_gpu_numerics`, `ru1_gpu_bindings`, `ru1_gpu_lifecycle`, `ru1_readiness`, `ru1_wmma_legality`, `ru1_n1_regression`, `ru1_forward_n16`, `ru1_transpose_n16`, `ru1_sparse_gradient`, `ru1_value_update`, `ru1_update_publication`, `ru1_native_failure`, `ru1_gradient_pack`, `ru1_hybrid_gradient`, `ru1_readiness_component`, and `ru1_read_lease`. These invoke the `ceRU1*` binaries directly and are eligible candidates for later exact-inventory selection; the final selection must follow changed-source review.

The Cellerator CUDA controller at `/home/tumlinson/.agents/skills/cuda/scripts/cuda_controller.py` is the installed foreground lease producer. Its `run --spec` path writes `TODO_GPU_LEASE_RECEIPT`, `CELLERATOR_SS1_GPU_LEASE_RECEIPT`, and `CUDA_VISIBLE_DEVICES`. NF1A's runner additionally requires a bound lease-verifier argv and one shared host lock in both project bindings. Do not put an older outer `run_gate.py` or `run_gpu_gate.py` into the selected CTest inventory: the NF1A runner already owns the lock and rejects nested wrappers.

## GlassHelix

The registered root is `/home/tumlinson/GlassHelix` at `139609465801a0407be056b27255c21d273df09e`. No configured GlassHelix CMake build or CTest tree was present. Its `NF1_HOST_VALIDATION` path requires `tests/native_foundation/independent/CMakeLists.txt`, which is absent on registered `main`; it therefore cannot currently supply `gh_nf1_t01`.

The preserved `gh-nf1-s-v1` workspace at `9d6ac741a175bb284b8f0b28a9b4788f175ceb36` contains `examples/native_foundation_v1`, whose standalone CMake project registers `gh_nf1_demo_cuda` and `gh_nf1_demo_cpu`. It finds an installed `GlassHelix::glasshelix` target and is partial example material, not a registered-root, source-pinned Cellerator consumer build. The historical `extern/Cellerator` gitlink is pinned at `87b68d655ee5c61f1d6b5a12ff8f5f3fcf20759f`; it must not substitute for the qualified sibling Cellerator source root.

## Binding prerequisites

The later root must create or select one source-pinned build whose CMake home and binary/library paths prove actual linkage to the chosen Cellerator and GlassHelix worktrees. It must bind nonempty exact CTest names, the binaries and consumed libraries, current commits, actual lease-verifier argv, matching shared-lock and peer-binding paths, and external evidence/JUnit paths. The current inventory does not provide those cross-project artifacts yet.
