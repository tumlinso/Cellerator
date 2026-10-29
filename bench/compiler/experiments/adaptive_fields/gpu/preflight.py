#!/usr/bin/env python3
"""CPU-only identity check for the adaptive-fields GPU experiment inputs.

This module deliberately does not import Torch, inspect devices, or invoke a
CUDA utility. The parent task owner must separately grant a scheduler window
before a GPU runner is started.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PLAN = HERE / "experiment_plan.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check() -> dict[str, object]:
    plan = json.loads(PLAN.read_text())
    package_root = Path(plan["frozen_package_root"])
    identity_path = Path(plan["frozen_identity"])
    fixture_root = Path(plan["fixture_root"])
    identity = json.loads(identity_path.read_text())
    frozen_files = identity.get("files", identity)
    if not isinstance(frozen_files, dict):
        raise RuntimeError("frozen package identity has no file map")

    checked: dict[str, str] = {"identity.json": sha256(identity_path)}
    required = (
        "package/cellerator/torch/sparse_support.py",
        "package/cellerator/torch/mechanism.py",
        "package/cellerator/torch/_torch.cpython-313-x86_64-linux-gnu.so",
    )
    for relative in required:
        path = package_root.parent / relative
        expected = frozen_files.get(relative)
        if not isinstance(expected, str) or not expected:
            raise RuntimeError(f"frozen identity lacks {relative}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen package hash mismatch for {relative}: {actual}")
        checked[relative] = actual

    checkout_sparse = ROOT / "python/cellerator/torch/sparse_support.py"
    snapshot_sparse = package_root / "cellerator/torch/sparse_support.py"
    if sha256(checkout_sparse) != sha256(snapshot_sparse):
        raise RuntimeError("frozen sparse-support source does not match current checkout")
    checked["checkout/python/cellerator/torch/sparse_support.py"] = sha256(checkout_sparse)

    expected_fixtures = {
        "small": "c33facc18f330739411c1f293609c237689685bd4003099ca79b331fded144be",
        "large-events": "910a4c742b795a2a827cb21e62c510a3bc5039e6878ae75d4826fc0d33234be8",
        "large-ranked": "674d5d4660befcb6ddbcd618b16b136203d3478ddc577d09a037e116bffd6e32",
    }
    fixtures: dict[str, dict[str, object]] = {}
    for name, expected in expected_fixtures.items():
        path = fixture_root / name / "manifest.json"
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"fixture manifest mismatch for {name}: {actual}")
        manifest = json.loads(path.read_text())
        if (manifest.get("schema") != "cellerator.scaled_sparse_fixture_bundle.v1"
                or manifest.get("metadata", {}).get("schema")
                != "cellerator.scaled_sparse_fixture.v1"):
            raise RuntimeError(f"unsupported fixture schema for {name}")
        fixtures[name] = {
            "manifest_sha256": actual,
            "metadata": manifest.get("metadata", {}),
        }

    source_hashes = {}
    for relative in plan["source_inputs"]:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"missing source input: {relative}")
        source_hashes[relative] = sha256(path)

    return {
        "schema": "cellerator.adaptive_fields.gpu_preflight.v1",
        "gpu_execution_performed": False,
        "torch_imported": False,
        "device_inspected": False,
        "python": sys.version,
        "platform": platform.platform(),
        "plan_sha256": sha256(PLAN),
        "checked_frozen_files": checked,
        "fixture_manifests": fixtures,
        "source_hashes_at_preflight": source_hashes,
        "scheduler_authority": "parent only; no admission requested by this script",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
