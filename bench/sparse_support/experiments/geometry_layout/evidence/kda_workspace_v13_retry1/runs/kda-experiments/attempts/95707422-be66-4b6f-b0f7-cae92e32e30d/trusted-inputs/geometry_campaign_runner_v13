#!/usr/bin/env python3
"""Memcheck hybrid CE-GEO routes, then evaluate the frozen KDA candidate set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PYTHON = Path("/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python")
SANITIZER = Path(
    "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/compute-sanitizer/compute-sanitizer"
)
DRIVER = HERE / "geometry_training_benchmark.py"
KDA_RUNNER = HERE / "run_geometry_kda.py"
PACKAGE_IDENTITY = REPO / "build-sparse-support/evidence/torch-v13-owner-validation/identity.json"
EXPECTED_IDENTITY_SHA256 = "4d3b6f0e1a63ea20785975597849bfad0b0f081221c47d9b858d6581b487afd4"
EXPECTED_EXTENSION_SHA256 = "51b0838ad3466f4c03f693d620aa6b800e8af9ed87a8a3fceb3bd432a05a4f24"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path,
                        default=HERE / "evidence/kda_workspace_v13")
    parser.add_argument("--task", type=Path, default=HERE / "kda_task_v13.json")
    args = parser.parse_args()
    if os.environ.get("CELLERATOR_SPARSE_SUPPORT_BENCH_ADMITTED") != "1":
        parser.error("run only inside the admitted Device Scheduler job")
    package_root = Path(os.environ.get("CELLERATOR_SPARSE_PACKAGE_ROOT", ""))
    if not package_root.is_dir():
        parser.error("CELLERATOR_SPARSE_PACKAGE_ROOT must name the qualified Torch package")
    identity_sha = sha256_file(PACKAGE_IDENTITY)
    identity = json.loads(PACKAGE_IDENTITY.read_text(encoding="utf-8"))
    extension_sha = identity.get("files", {}).get(
        "package/cellerator/torch/_torch.cpython-313-x86_64-linux-gnu.so")
    if identity_sha != EXPECTED_IDENTITY_SHA256 or extension_sha != EXPECTED_EXTENSION_SHA256:
        parser.error("v13 package identity or extension does not match the frozen experiment contract")
    for path in (PYTHON, SANITIZER, DRIVER, KDA_RUNNER):
        if not path.is_file():
            parser.error(f"required campaign executable/source is missing: {path}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, object] = {
        "schema_version": 1,
        "device_scheduler_admitted": True,
        "package_root": str(package_root.resolve()),
        "package_identity_sha256": identity_sha,
        "extension_sha256": extension_sha,
        "compute_sanitizer": str(SANITIZER),
        "compute_sanitizer_sha256": sha256_file(SANITIZER),
        "memcheck": [],
    }
    version = subprocess.run([str(SANITIZER), "--version"], cwd=REPO,
                             text=True, capture_output=True, check=False)
    (output / "compute_sanitizer_version.txt").write_text(
        version.stdout + version.stderr, encoding="utf-8")
    if version.returncode != 0:
        raise RuntimeError(f"compute-sanitizer --version failed: {version.returncode}")

    # Both layouts leave real edges in the explicit residual route at ranked scale.
    # Exercise their complete forward and training VJP paths under memcheck before
    # allowing the broader exact-oracle/timing matrix to start.
    for candidate in ("hybrid8", "hybrid16"):
        result_path = output / "memcheck" / f"large-ranked-{candidate}.json"
        command = [
            str(SANITIZER), "--tool", "memcheck", "--error-exitcode", "1",
            "--target-processes", "all", str(PYTHON), str(DRIVER),
            "--case", "large_ranked", "--candidate", candidate,
            "--output", str(result_path), "--warmup", "1", "--rounds", "1",
            "--iterations", "1", "--setup-pairs", "1", "--seed", "20261027",
        ]
        proc = subprocess.run(command, cwd=REPO, text=True,
                              capture_output=True, env=os.environ.copy(), check=False)
        log_path = output / "memcheck" / f"large-ranked-{candidate}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "COMMAND: " + json.dumps(command) + "\n\nSTDOUT:\n" + proc.stdout
            + "\nSTDERR:\n" + proc.stderr,
            encoding="utf-8")
        row: dict[str, object] = {
            "candidate": candidate,
            "command": command,
            "exit_code": proc.returncode,
            "log": str(log_path),
            "log_sha256": sha256_file(log_path),
            "result": str(result_path),
        }
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            checks = result.get("checks", {})
            row["checks"] = checks
            row["result_sha256"] = sha256_file(result_path)
            row["image"] = result.get("case", {}).get("geometry_image")
            row["correctness_passed"] = bool(
                checks.get("inference_output_allclose_atol_5e-3_rtol_5e-3")
                and checks.get("training_parameter_gradient_allclose_atol_6e-3_rtol_6e-3")
                and checks.get("loss_close_abs_5e-4_rel_5e-3")
                and all(value > 0 for value in
                        checks.get("derivative_only_probability_grad_nonzero", {}).values())
                and checks.get("remaining_read_tickets") == 0)
        else:
            row["correctness_passed"] = False
        evidence["memcheck"].append(row)  # type: ignore[union-attr]
        write_json(output / "campaign_manifest.json", evidence)
        if proc.returncode != 0 or not row["correctness_passed"]:
            print(json.dumps({"status": "error", "stage": "memcheck",
                              "candidate": candidate, "exit_code": proc.returncode},
                             sort_keys=True))
            return proc.returncode or 1

    command = [
        str(PYTHON), str(KDA_RUNNER), "--output-dir", str(output / "kda"),
        "--workspace", str(args.workspace.resolve()), "--task", str(args.task.resolve()),
    ]
    proc = subprocess.run(command, cwd=REPO, text=True,
                          capture_output=True, env=os.environ.copy(), check=False)
    (output / "kda_campaign_stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (output / "kda_campaign_stderr.txt").write_text(proc.stderr, encoding="utf-8")
    evidence["kda_command"] = command
    evidence["kda_exit_code"] = proc.returncode
    evidence["kda_stdout_sha256"] = sha256_file(output / "kda_campaign_stdout.txt")
    evidence["kda_stderr_sha256"] = sha256_file(output / "kda_campaign_stderr.txt")
    write_json(output / "campaign_manifest.json", evidence)
    print(json.dumps({"status": "ok" if proc.returncode == 0 else "error",
                      "stage": "complete", "output_dir": str(output),
                      "kda_exit_code": proc.returncode}, sort_keys=True))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
