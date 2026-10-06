#!/usr/bin/env python3
"""Run prebuilt native and binding integration fixtures inside a CUDA lease."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


GPU_UUID = "GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_lease() -> dict[str, Any]:
    receipt_path = os.environ.get("TODO_GPU_LEASE_RECEIPT")
    if os.environ.get("CUDA_BENCHMARK_FOREGROUND_INTENT_HELD") != "1" or not receipt_path:
        raise RuntimeError("integration fixtures require the foreground CUDA controller lease")
    try:
        receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read CUDA lease receipt: {error}") from error
    resource_ids = receipt.get("resource_ids", []) if isinstance(receipt, dict) else []
    if f"accelerator:{GPU_UUID}" not in resource_ids:
        raise RuntimeError(f"CUDA lease does not cover required UUID {GPU_UUID}")
    return {"path": receipt_path, "sha256": sha256(Path(receipt_path)), "resource_ids": resource_ids}


def run_one(label: str, argv: list[str], cwd: Path, env: dict[str, str], output: Path,
            records: list[dict[str, Any]]) -> None:
    started = time.monotonic()
    result = subprocess.run(argv, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    stdout_path, stderr_path = output / f"{label}.stdout.txt", output / f"{label}.stderr.txt"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    records.append({
        "label": label, "argv": argv, "cwd": str(cwd), "returncode": result.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
    })
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit status {result.returncode}; see {stderr_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build, package_root = args.build.resolve(), args.package_root.resolve()
    native_build, output = args.native_build.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "runtime-result.json"
    binaries = [
        build / "cellerator_indexed_mechanism_binding_test",
        build / "bindings/torch/cellerator_torch_native_views_test",
        build / "bindings/torch/cellerator_torch_program_ops_test",
        build / "bindings/torch/cellerator_torch_autograd_ops_test",
        native_build / "src/compute/operation/native_numeric/ceNativeArithmeticTest",
        native_build / "ceNativeNeighborhoodMoments",
    ]
    missing = [str(path) for path in binaries if not path.is_file()]
    if missing:
        parser.error("missing prebuilt integration binary(s): " + ", ".join(missing))
    native_modules = sorted(package_root.glob("cellerator/_native*.so"))
    torch_modules = sorted(package_root.glob("cellerator/torch/_torch*.so"))
    if not native_modules or not torch_modules:
        parser.error(f"fresh CMake install prefix is missing Python/Torch extension modules: {package_root}")
    installed_modules = native_modules + torch_modules
    lease: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    receipt: dict[str, Any] = {
        "schema": "cellerator.fp64.combined-runtime.v1", "status": "running",
        "source_root": str(root), "gpu_uuid": GPU_UUID,
        "binary_sha256": {str(path): sha256(path) for path in binaries},
        "installed_extension_sha256": {str(path): sha256(path) for path in installed_modules},
        "lease": None, "tests": [], "error": None,
    }
    try:
        lease = require_lease()
        receipt["lease"] = lease
        env = dict(os.environ, PYTHONPATH=str(package_root),
                   CELLERATOR_REQUIRE_NATIVE="1", PYTHONDONTWRITEBYTECODE="1")
        import_check = (
            "import json, cellerator, cellerator._native, cellerator.torch, cellerator.torch._torch; "
            "print(json.dumps({name: getattr(module, '__file__', None) for name, module in "
            "[('cellerator', cellerator), ('cellerator._native', cellerator._native), "
            "('cellerator.torch', cellerator.torch), ('cellerator.torch._torch', cellerator.torch._torch)]}))"
        )
        run_one("fresh_prefix_imports", [sys.executable, "-B", "-c", import_check],
                root, env, output, records)
        try:
            imported = json.loads(Path(records[-1]["stdout_path"]).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"cannot parse fresh-prefix import evidence: {error}") from error
        resolved_imports = {name: Path(value).resolve() for name, value in imported.items() if value}
        required_imports = {"cellerator", "cellerator._native", "cellerator.torch",
                            "cellerator.torch._torch"}
        if set(resolved_imports) != required_imports or any(
                not path.is_relative_to(package_root) for path in resolved_imports.values()):
            raise RuntimeError(f"Cellerator imports did not all resolve under fresh prefix {package_root}: "
                               f"{imported}")
        receipt["package_imports"] = {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in resolved_imports.items()
        }
        labels = ("indexed_mechanism_binding", "torch_native_views", "torch_program_ops",
                  "torch_autograd_ops", "native_arithmetic", "native_neighborhood_moments")
        for label, binary in zip(labels, binaries):
            run_one(label, [str(binary)], root, env, output, records)
        run_one("python_binding_pytest_50", [sys.executable, "-B", "-m", "pytest", "-q",
                 "tests/bindings/python"], root, env, output, records)
        pytest_stdout = Path(records[-1]["stdout_path"]).read_text(encoding="utf-8")
        pass_match = re.search(r"(?:^|\s)(\d+) passed(?:\s|,|$)", pytest_stdout)
        skip_match = re.search(r"(?:^|,\s*)(\d+) skipped(?:\s|,|$)", pytest_stdout)
        passed = int(pass_match.group(1)) if pass_match else 0
        skipped = int(skip_match.group(1)) if skip_match else 0
        receipt["pytest_counts"] = {"expected_passed": 50, "passed": passed, "skipped": skipped}
        if passed != 50 or skipped != 0:
            raise RuntimeError(f"Python binding suite expected 50 passed and 0 skipped; observed "
                               f"{passed} passed and {skipped} skipped")
        receipt["tests"] = records
        receipt["status"] = "passed"
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        receipt["tests"] = records
        receipt["status"] = "failed"
        receipt["error"] = str(error)
        write_json(receipt_path, receipt)
        print(f"Combined integration runtime failed: {error}", file=sys.stderr)
        return 1
    write_json(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path),
                      "tests": len(records), "lease_sha256": lease["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
