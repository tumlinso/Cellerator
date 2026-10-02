"""Small executable acceptance check: CPU reference plus compiled host view."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
subprocess.run([sys.executable, "-B", str(root / "test_state.py")], env=env, check=True)
with tempfile.TemporaryDirectory(prefix="ce-moon-state-") as temp:
    executable = str(Path(temp) / "state_host")
    subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-O2", str(root / "test_state_host.cc"), "-o", executable], check=True, env=env)
    subprocess.run([executable], check=True, env=env)
print("PASS CE-MOON-STATE CPU acceptance; CUDA/native integration unqualified", flush=True)
