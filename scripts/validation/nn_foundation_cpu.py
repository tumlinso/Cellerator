#!/usr/bin/env python3
"""Run source-package NN qualification without altering an installed package."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--native-package", type=Path,
                        help="Explicit directory containing compatible compiled extensions")
    parser.add_argument("--require-native", action="store_true")
    parser.add_argument("--pattern", default="test_nn*.py")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.source.resolve(strict=True)
    # This runner never requests or executes GPU work. CUDA qualification has
    # a separate Device Scheduler admission and evidence record.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    package = root / "python/cellerator"
    locations = [str(package)]
    if args.native_package is not None:
        locations.append(str(args.native_package.resolve(strict=True)))
    spec = importlib.util.spec_from_file_location(
        "cellerator", package / "__init__.py", submodule_search_locations=locations)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["cellerator"] = module
    spec.loader.exec_module(module)
    native = (importlib.import_module("cellerator._native")
              if importlib.util.find_spec("cellerator._native") is not None else None)
    if args.require_native and native is None:
        parser.error("CPU native extension is required; pass a compatible --native-package")
    sys.path.insert(0, str(root))
    suite = unittest.defaultTestLoader.discover(
        str(root / "tests/framework"), pattern=args.pattern)
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    elapsed = time.monotonic() - started
    import torch

    candidates = list(package.rglob("*.py"))
    candidates += list((root / "examples/nn_foundation").glob("*.py"))
    candidates += list((root / "tests/framework").glob(args.pattern))
    candidates += [root / "python/cellerator/nn.py", Path(__file__).resolve()]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(set(candidates))}
    metadata = {
        "schema": "cellerator.nn-foundation.cpu-validation.v1",
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "python": sys.version,
        "torch": torch.__version__,
        "pattern": args.pattern,
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": [{"test": str(test), "reason": reason}
                    for test, reason in result.skipped],
        "successful": result.wasSuccessful(),
        "elapsed_seconds": elapsed,
        "native_extension": getattr(native, "__file__", None),
        "native_sha256": (hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
                          if native is not None else None),
        "source_sha256": hashes,
        "limitations": "CPU correctness and training checks; no GPU or performance claim.",
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: metadata[key] for key in
                      ("tests_run", "failures", "errors", "successful", "elapsed_seconds")},
                     sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
