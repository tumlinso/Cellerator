#!/usr/bin/env python3
"""Run the grouped-linear suite inside an admitted Device Scheduler job."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / "python"))
    import cellerator
    cellerator.__path__.append(str(args.extension_dir.resolve(strict=True)))
    import torch
    import cellerator._modeling_cuda as native
    os.environ["CELLERATOR_TEST_CUDA"] = "1"
    os.environ["CELLERATOR_TEST_CUDA_DEVICE"] = "0"
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.discover(str(root / "tests/framework"), pattern="test_nn_grouped_linear.py")
    )
    torch.cuda.synchronize()
    info = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success": result.wasSuccessful(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": torch.cuda.get_device_capability(0),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "extension": native.__file__,
        "extension_sha256": digest(native.__file__),
        "source_sha256": {
            path: digest(root / path) for path in (
                "python/cellerator/torch/grouped_linear.py",
                "tests/framework/test_nn_grouped_linear.py",
                "include/Cellerator/modeling/cuda/grouped_linear.hh",
                "src/modeling/cuda/grouped_linear.cu",
                "bindings/modeling_cuda/module.cc",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    raise SystemExit(main())
