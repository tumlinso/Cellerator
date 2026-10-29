"""Run only within the finite admitted v13 qualification job."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if os.environ.get("CELLERATOR_RUN_CUDA_TESTS") != "1":
    raise SystemExit("An admitted CUDA qualification job is required")

ROOT = Path("/home/tumlinson/Cellerator")
SNAPSHOT = ROOT / "build-sparse-support/evidence/torch-v13-owner-validation"
EXPECTED_IDENTITY = "4d3b6f0e1a63ea20785975597849bfad0b0f081221c47d9b858d6581b487afd4"
assert hashlib.sha256((SNAPSHOT / "identity.json").read_bytes()).hexdigest() == EXPECTED_IDENTITY

import cellerator.torch._torch as native

extension = Path(native.__file__).resolve()
assert extension.is_relative_to(SNAPSHOT)
print(json.dumps({"extension": str(extension), "extension_sha256":
                  hashlib.sha256(extension.read_bytes()).hexdigest()}), flush=True)

# The scheduler resolves Python symlinks. Use the venv console entry point so
# pytest and its dependencies retain the configured Python environment.
subprocess.run([
    "/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/pytest",
    "-q", str(ROOT / "tests/sparse_support"),
    str(ROOT / "tests/bindings/python/test_mechanism.py"),
    str(ROOT / "tests/bindings/python/test_native_mechanism.py"),
], check=True)
subprocess.run([str(ROOT / "build-sparse-support/tests/sparse_support/"
                          "cellerator_sparse_support_cuda_worksets")], check=True)
print("v13 owner regressions and current posting workset test passed", flush=True)
