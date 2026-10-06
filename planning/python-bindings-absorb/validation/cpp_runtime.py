"""Run adapter correctness executables inside the CUDA controller lease."""
import os
import subprocess
from pathlib import Path
assert os.environ.get("TODO_GPU_LEASE_RECEIPT"), "CUDA controller lease required"
root = Path("/tmp/cellerator-bindings-build/bindings/torch")
for name in ("cellerator_torch_native_views_test", "cellerator_torch_program_ops_test", "cellerator_torch_autograd_ops_test", "celleratorTorchModelCustomOpsTest"):
    print(f"Running {name}", flush=True)
    subprocess.run([str(root / name)], check=True)
