#!/usr/bin/env python3
"""Run prepared ATen/Python lifecycle and VJP checks under Scheduler admission."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests" / "factorized_interactions"
sys.dont_write_bytecode = True
sys.path.insert(0, str(TESTS))

from prepared_torch_runtime import run_prepared_torch_runtime_checks  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True,
                        help="build-local Python package copy with _nn_native installed")
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--expected-python", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    job_id = os.environ.get("AUTOBENCH_SCHEDULER_JOB_ID", "").strip()
    if not job_id:
        parser.error("prepared Torch checks require an Autobench Scheduler job context")
    if not args.expected_uuid.startswith("GPU-"):
        parser.error("--expected-uuid must be the exact Scheduler GPU UUID")

    manifest_path = args.build_manifest.resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "cellerator.factorized_interactions.build.v1":
        parser.error("--build-manifest has an unknown schema")
    expected_python = args.expected_python.resolve(strict=True)
    python_metadata = manifest.get("python", {})
    if python_metadata.get("executable") != str(expected_python) or \
            Path(sys.executable).resolve(strict=True) != expected_python:
        parser.error("prepared Torch suite is not running in the build-selected Python")
    if sha256_file(expected_python) != python_metadata.get("executable_sha256"):
        parser.error("selected Python executable differs from the frozen build manifest")
    extension_entry = manifest.get("binaries", {}).get("python_extension")
    if not extension_entry:
        parser.error("the frozen build manifest omits the NN Python CUDA extension")
    extension = (manifest_path.parent / extension_entry["path"]).resolve(strict=True)
    extension_hash = sha256_file(extension)
    if extension_hash != extension_entry["sha256"]:
        parser.error("NN Python extension differs from its build manifest")

    package_root = args.package_root.resolve(strict=True)
    package = package_root / "cellerator"
    copied_extension = package / extension.name
    if not copied_extension.is_file() or sha256_file(copied_extension) != extension_hash:
        parser.error("build-local Python package does not contain the frozen extension")
    sys.path.insert(0, str(package_root))

    import torch
    if torch.__version__ != python_metadata.get("torch_version"):
        parser.error("imported Torch version differs from the frozen build manifest")
    module_path = Path(importlib.util.find_spec("cellerator").origin).resolve(strict=True)
    if module_path != (package / "__init__.py").resolve(strict=True):
        parser.error("the prepared adapter did not import from the frozen build-local package")

    checks = run_prepared_torch_runtime_checks()
    result = {
        **checks,
        "scheduler_job_id": job_id,
        "expected_device_uuid": args.expected_uuid,
        "build_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_closure_sha256": manifest["source_closure_sha256"],
        "python_extension_sha256": extension_hash,
        "python_package_init": str(module_path),
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "suite": result["schema"],
        "stream_owner_reuse": result["stream_owner_reuse"],
        "parameter_update_reuse": result["parameter_update_reuse"],
        "saved_tensor_mutation_guard": result["saved_tensor_mutation_guard"],
        "first_order_only_guard": result["first_order_only_guard"],
        "timed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
