"""Torch-free native mechanism acceptance inside the CUDA controller lease."""
import os
import importlib.util
import subprocess
import sys
assert os.environ.get("TODO_GPU_LEASE_RECEIPT"), "CUDA controller lease required"
assert importlib.util.find_spec("torch") is None, "must execute without Torch installed"
subprocess.run(["/tmp/cellerator-python-native-no-torch-build/cellerator_indexed_mechanism_binding_test"], check=True)
raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q", "tests/bindings/python/test_native_mechanism.py", "tests/bindings/python/test_product2.py"]))
