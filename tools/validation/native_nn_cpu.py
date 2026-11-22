#!/usr/bin/env python3
"""Run source-package CPU integration with an explicitly selected NN extension."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--extension-dir", type=Path, required=True)
parser.add_argument("--legacy-native-dir", type=Path)
parser.add_argument("--pattern", default="test*.py")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
root = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "python"), str(root)]
import torch
import cellerator

cellerator.__path__.append(str(args.extension_dir.resolve()))
if args.legacy_native_dir:
    cellerator.__path__.append(str(args.legacy_native_dir.resolve()))
import cellerator._nn_native as native

suite = unittest.defaultTestLoader.discover(str(root / "tests/framework"), pattern=args.pattern)
started = time.monotonic()
result = unittest.TextTestRunner(verbosity=1).run(suite)
paths = [*root.glob("include/Cellerator/nn/**/*.hh"), *root.glob("src/nn/*.cc"),
         *root.glob("python/cellerator/torch/*.py"), *root.glob("tests/framework/test_native_nn_bridge.py")]
record = {
    "head_provenance": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "python": sys.version, "torch": torch.__version__, "native_extension": native.__file__,
    "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
    "tests_run": result.testsRun, "passed": result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
    "failures": len(result.failures), "errors": len(result.errors),
    "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
    "elapsed_seconds": time.monotonic()-started,
    "sources": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps({key: record[key] for key in ("tests_run", "passed", "failures", "errors")}))
raise SystemExit(0 if result.wasSuccessful() else 1)
