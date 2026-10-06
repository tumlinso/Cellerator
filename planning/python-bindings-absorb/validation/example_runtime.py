"""Execute preserved optional example tests under the CUDA controller lease."""
import os
import subprocess
from pathlib import Path
assert os.environ.get("TODO_GPU_LEASE_RECEIPT"), "CUDA controller lease required"
root = Path("/tmp/cellerator-bindings-cellshard-build/examples/torch")
for name in ("cellerator_torch_dense_reduce_compile_test", "cellerator_torch_cellshard_views_compile_test", "cellerator_torch_quantize_primitive_test"):
    print(name, flush=True)
    subprocess.run([str(root / name)], check=True)
