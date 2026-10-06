"""Check rebuilt binding consumers after the concurrent native layout changes."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

assert os.environ.get("TODO_GPU_LEASE_RECEIPT"), "CUDA controller lease required"
parser = argparse.ArgumentParser()
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--package-root", type=Path, required=True)
args = parser.parse_args()
env = dict(os.environ, PYTHONPATH=str(args.package_root),
           CELLERATOR_REQUIRE_NATIVE="1", PYTHONDONTWRITEBYTECODE="1")
for executable in (
    args.build / "cellerator_indexed_mechanism_binding_test",
    args.build / "bindings/torch/cellerator_torch_native_views_test",
    args.build / "bindings/torch/cellerator_torch_program_ops_test",
    args.build / "bindings/torch/cellerator_torch_autograd_ops_test",
):
    print(f"Running rebuilt {executable.name}", flush=True)
    subprocess.run([str(executable)], check=True, env=env)
subprocess.run([sys.executable, "-B", "-m", "pytest", "-q",
                "tests/bindings/python"], check=True, env=env)
