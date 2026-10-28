# Host restart verification — 2026-10-28

The restart checkpoint `8fa6c09e` was clean and synchronized with
`origin/main` before resumed work. Project Control's read-only frontier still
had no ready task for this implementation; no task completion is asserted.

The shared CUDA/SM70 build successfully rebuilt the posting cursor consumers
and the optional Python extensions. Targets were
`cellerator_python_extensions`, `cellerator_sparse_support_indexing_test`,
`cellerator_sparse_support_numeric_reference`,
`cellerator_sparse_support_image_builder`,
`cellerator_sparse_support_cuda_worksets`, and
`cellerator_sparse_support_derivative_sampling`.

Four host CTest cases passed: numerical reference, indexing, image builder,
and derivative sampling. The CUDA workset executable was built here but was
not launched outside Device Scheduler. This record makes no GPU correctness
or sanitizer claim.

Thirteen Python tests passed in 6.54 seconds:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v12-validator/package:/home/tumlinson/Cellerator \
/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/pytest -q \
  tests/sparse_support/test_scaled_fixtures.py \
  tests/sparse_support/test_fixtures_and_gates.py \
  tests/sparse_support/test_synthetic_driver.py
```

The Python run used the retained v12 package with the live fixture and test
sources. The v12 package identity SHA-256 is
`99da528db6f4ba14bec983b6dfe31f12a065dbdc8b39c0dfb224ed0e969fca07`.
Three preceding collection attempts failed because the package path was
absent, pointed at a nonexistent build package directory, or pointed at the
source package without its native extension. They did not execute tests or
indicate a numerical failure. The command above is the corrected package route.
