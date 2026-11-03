#!/usr/bin/env python3
"""Validate an out-of-process native runner against the frozen CPU oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "tests" / "factorized_interactions"))

from case_io import read_result, verify_case_shards, write_sharded_case  # noqa: E402
from reference import (assert_fp16_contract, assert_fp32_contract,
                       contract_allowance, evaluate, fp16_ulp)  # noqa: E402
from reference_cache import evaluate_cached  # noqa: E402
from workloads import make_workload, workload_ids  # noqa: E402


_REFERENCE_CACHE_NAME = re.compile(
    r"\.factorized-reference-[0-9a-f]{64}-[0-9a-f]{12}-(?:grad|forward)\.npz\Z")


def _contract_evidence(actual: np.ndarray, reference: np.ndarray,
                       sum_abs_contributions: np.ndarray,
                       fp16_output: bool = False) -> dict:
    """Retain the per-element scale behind the frozen comparison bound."""
    actual64 = np.asarray(actual, dtype=np.float64)
    scales = np.asarray(sum_abs_contributions, dtype=np.float64)
    if fp16_output:
        reference16 = np.asarray(reference, dtype=np.float16)
        reference64 = reference16.astype(np.float64)
        allowance = fp16_ulp(reference16) + contract_allowance(scales)
    else:
        reference64 = np.asarray(reference, dtype=np.float64)
        allowance = contract_allowance(scales)
    if actual64.shape != reference64.shape or scales.shape != reference64.shape:
        raise ValueError("contract evidence arrays have inconsistent shapes")
    errors = np.abs(actual64 - reference64)
    if not errors.size:
        return {
            "max_abs_error": 0.0,
            "max_abs_reference_contributions": 0.0,
            "max_allowed_error": 0.0,
            "max_error_to_allowance": 0.0,
            "worst_index": None,
            "worst_actual": None,
            "worst_reference": None,
            "worst_sum_abs_reference_contributions": None,
            "worst_allowed_error": None,
        }
    ratios = errors / allowance
    worst = int(np.argmax(ratios))
    index = np.unravel_index(worst, errors.shape)
    return {
        "max_abs_error": float(np.max(errors)),
        "max_abs_reference_contributions": float(np.max(scales)),
        "max_allowed_error": float(np.max(allowance)),
        "max_error_to_allowance": float(ratios.flat[worst]),
        "worst_index": [int(i) for i in index],
        "worst_actual": float(actual64.flat[worst]),
        "worst_reference": float(reference64.flat[worst]),
        "worst_sum_abs_reference_contributions": float(scales.flat[worst]),
        "worst_allowed_error": float(allowance.flat[worst]),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_cache_inventory(cache_dir: Path) -> dict[str, tuple[int, str]]:
    """Hash the full dedicated cache directory without following symlinks."""
    root = Path(cache_dir).absolute()
    if any(component.is_symlink() for component in (root, *root.parents)):
        raise ValueError("reference cache directory path cannot contain symbolic links")
    if not root.is_dir():
        raise NotADirectoryError(f"reference cache path is not a directory: {root}")
    inventory: dict[str, tuple[int, str]] = {}
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                if entry.is_symlink():
                    raise ValueError("reference cache tree contains a symbolic link")
                if entry.is_dir(follow_symlinks=False):
                    # The cache helper writes flat NPZ archives only. Treat
                    # directories as unexpected and do not recurse into them.
                    raise ValueError("reference cache tree contains an unexpected directory")
                if not entry.is_file(follow_symlinks=False) or \
                        not _REFERENCE_CACHE_NAME.fullmatch(relative):
                    raise ValueError("reference cache tree contains an unexpected entry")
                stat = entry.stat(follow_symlinks=False)
                inventory[relative] = (stat.st_size, _sha256_file(path))
    if not inventory:
        raise ValueError("reference cache tree is empty after oracle preparation")
    return inventory


def _invalidate_reference_cache_dir(cache_dir: Path) -> None:
    """Remove only the explicitly dedicated cache root after candidate edits."""
    root = Path(cache_dir).absolute()
    if any(component.is_symlink() for component in root.parents):
        return
    try:
        if root.is_symlink():
            root.unlink()
        elif root.is_dir():
            shutil.rmtree(root)
        elif root.exists():
            root.unlink()
    except OSError:
        pass


def _check_reference_cache_unchanged(cache_dir: Path,
                                     original_inventory: dict[str, tuple[int, str]]) -> None:
    """Reject and invalidate the tree if the candidate process changed it."""
    try:
        current_inventory = _reference_cache_inventory(cache_dir)
    except (OSError, ValueError) as error:
        _invalidate_reference_cache_dir(cache_dir)
        raise RuntimeError(
            "reference cache changed during native runner execution") from error
    if current_inventory == original_inventory:
        return
    _invalidate_reference_cache_dir(cache_dir)
    raise RuntimeError("reference cache changed during native runner execution")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, choices=workload_ids())
    parser.add_argument("--runner", required=True,
                        help="native runner executable plus fixed arguments")
    parser.add_argument("--route", choices=("packet", "direct"), required=True)
    parser.add_argument("--activation-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--parameter-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--output-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--gradient-mode", choices=("overwrite", "accumulate"),
                        default="overwrite")
    parser.add_argument("--repeat-vjp", action="store_true",
                        help="run a second VJP from the same forward tape and private outputs")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-cache-dir", type=Path,
        help="optional dedicated directory for verified scalar-oracle arrays and bounds")
    parser.add_argument("--expected-uuid")
    parser.add_argument("--runner-binary", type=Path,
                        help="executable under validation, independent of a sanitizer prefix")
    parser.add_argument("--build-manifest", type=Path,
                        help="source/binary identity manifest emitted by build_runner.py")
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--allow-oracle-stub", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    activation_dtype = np.float16 if args.activation_dtype == "fp16" else np.float32
    parameter_dtype = np.float16 if args.parameter_dtype == "fp16" else np.float32
    output_dtype = np.float16 if args.output_dtype == "fp16" else np.float32
    args.artifact_dir = args.artifact_dir.resolve()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    problem, dy = make_workload(args.workload, activation_dtype, parameter_dtype)
    input_path = args.artifact_dir / (
        f"{args.workload}.a{args.activation_dtype}.p{args.parameter_dtype}."
        f"{args.output_dtype}.{args.route}.{args.gradient_mode}.case")
    output_path = input_path.with_suffix(".result")
    identity = write_sharded_case(input_path, problem, dy, output_dtype)
    command = shlex.split(args.runner)
    if not command:
        parser.error("--runner must name a native executable")
    if Path(command[0]).is_file():
        # Keep virtualenv interpreter symlinks intact: resolving them selects
        # the base interpreter and silently discards the environment's packages.
        command[0] = str(Path(command[0]).absolute())
    scheduler_job_id = os.environ.get("AUTOBENCH_SCHEDULER_JOB_ID", "").strip()
    is_oracle_stub = (len(command) >= 2
                      and Path(command[1]).name == "oracle_wire_stub.py")
    if not scheduler_job_id and not (args.allow_oracle_stub and is_oracle_stub):
        parser.error("CUDA execution requires an Autobench worker Scheduler job context")
    if args.allow_oracle_stub and not is_oracle_stub:
        parser.error("--allow-oracle-stub is restricted to the CPU oracle wire stub")
    if not is_oracle_stub and not args.expected_uuid:
        parser.error("CUDA validation requires the explicit Scheduler UUID")
    build_provenance = None
    runner_binary = args.runner_binary
    if not is_oracle_stub and runner_binary is None and Path(command[0]).is_file():
        runner_binary = Path(command[0])
    if args.build_manifest is not None:
        manifest_path = args.build_manifest.resolve(strict=True)
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema") != "cellerator.factorized_interactions.build.v1":
            parser.error("--build-manifest has an unknown schema")
        expected_binary = (manifest_path.parent /
                           manifest["binaries"]["runner"]["path"]).resolve(strict=True)
        if runner_binary is None:
            runner_binary = expected_binary
        runner_binary = runner_binary.resolve(strict=True)
        if runner_binary != expected_binary:
            parser.error("--runner-binary differs from the runner named in the build manifest")
        binary_digest = hashlib.sha256(runner_binary.read_bytes()).hexdigest()
        if binary_digest != manifest["binaries"]["runner"]["sha256"]:
            parser.error("runner binary SHA256 differs from the build manifest")
        build_provenance = {
            "build_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_closure_sha256": manifest["source_closure_sha256"],
            "runner_binary_sha256": binary_digest,
        }
    elif runner_binary is not None:
        runner_binary = runner_binary.resolve(strict=True)
        build_provenance = {
            "runner_binary_sha256": hashlib.sha256(runner_binary.read_bytes()).hexdigest(),
        }
    if args.iterations < 0 or args.iterations > 256:
        parser.error("--iterations must be in the range 0..256")
    if args.iterations and not scheduler_job_id:
        parser.error("timing requires an Autobench worker Scheduler job context")
    command.extend(["--input", identity["parts"][0], "--output", str(output_path),
                    "--route", args.route, "--gradient-mode", args.gradient_mode])
    for part in identity["parts"][1:]:
        command.extend(["--input-shard", part])
    if not is_oracle_stub:
        command.extend(["--expected-uuid", args.expected_uuid])
    if args.repeat_vjp:
        command.append("--repeat-vjp")
    metrics_path = output_path.with_suffix(".metrics.json")
    if args.iterations:
        command.extend(["--iterations", str(args.iterations), "--metrics", str(metrics_path)])
    verify_case_shards(identity)

    # Load the independent reference before launching the runner. The runner
    # shares the packet output tree and could otherwise alter a cache archive
    # before the verifier reads it.
    expected = None
    if args.reference_cache_dir is None:
        reference_cache_inventory = None
    else:
        expected = evaluate_cached(problem, dy, output_dtype, identity,
                                   args.reference_cache_dir)
        reference_cache_inventory = _reference_cache_inventory(
            args.reference_cache_dir)

    try:
        # Keep stdout reserved for the validator's machine-readable summary;
        # tools wrapping the runner may print diagnostic banners there.
        subprocess.run(command, cwd=args.artifact_dir, check=True, stdout=sys.stderr)
        verify_case_shards(identity)
    except BaseException:
        if args.reference_cache_dir is not None and \
                reference_cache_inventory is not None:
            _check_reference_cache_unchanged(
                args.reference_cache_dir, reference_cache_inventory)
        raise

    try:
        actual = read_result(output_path, problem, output_dtype)
        native_execution = None
        if not is_oracle_stub:
            metadata_path = Path(str(output_path) + ".metadata.json")
            if not metadata_path.is_file():
                raise RuntimeError("native runner omitted its backend/dispatch metadata")
            native_execution = json.loads(metadata_path.read_text(encoding="utf-8"))
            if native_execution.get("schema") != \
                    "cellerator.factorized_interactions.runner.v1":
                raise RuntimeError("native runner returned unknown dispatch metadata")
            if native_execution.get("requested_route") != args.route:
                raise RuntimeError("native runner selected a different requested route")
            if native_execution.get("device_uuid") != args.expected_uuid:
                raise RuntimeError("native runner metadata reports a different GPU UUID")
            if native_execution.get("scheduler_job_id") != scheduler_job_id:
                raise RuntimeError("native runner metadata does not match the Scheduler job")
            if native_execution.get("selected_backend") not in ("packet", "direct", "stable"):
                raise RuntimeError("native runner metadata contains an unknown backend")
    except BaseException:
        if args.reference_cache_dir is not None and \
                reference_cache_inventory is not None:
            _check_reference_cache_unchanged(
                args.reference_cache_dir, reference_cache_inventory)
        raise
    # Preserve the original uncached ordering. Cached references are resident
    # before the candidate process starts.
    if expected is None:
        expected = evaluate(problem, dy=dy, output_dtype=output_dtype)
    expected_vjp = {name: getattr(expected.vjp, name).copy()
                    for name in ("dx", "dz", "dalpha", "dbias", "dgamma")}
    if args.gradient_mode == "accumulate":
        for gradient in expected_vjp.values():
            gradient += 0.125
    try:
        if output_dtype == np.float16:
            assert_fp16_contract(actual["y"], expected.y,
                                 expected.sum_abs_y_contributions, "output")
            output_evidence = _contract_evidence(
                actual["y"], expected.y, expected.sum_abs_y_contributions,
                fp16_output=True)
        else:
            assert_fp32_contract(actual["y"], expected.y_float64,
                                 expected.sum_abs_y_contributions, "output")
            output_evidence = _contract_evidence(
                actual["y"], expected.y_float64,
                expected.sum_abs_y_contributions)
        gradient_evidence = {}
        for name in ("dx", "dz", "dalpha", "dbias", "dgamma"):
            assert_fp32_contract(actual[name], expected_vjp[name],
                                 getattr(expected.vjp_scale, name), name)
            gradient_evidence[name] = _contract_evidence(
                actual[name], expected_vjp[name],
                getattr(expected.vjp_scale, name))
            if args.repeat_vjp:
                assert_fp32_contract(actual[f"{name}_repeat"], expected_vjp[name],
                                     getattr(expected.vjp_scale, name),
                                     f"{name} repeated VJP")
                gradient_evidence[f"{name}_repeat"] = _contract_evidence(
                    actual[f"{name}_repeat"], expected_vjp[name],
                    getattr(expected.vjp_scale, name))
    finally:
        if args.reference_cache_dir is not None and \
                reference_cache_inventory is not None:
            _check_reference_cache_unchanged(
                args.reference_cache_dir, reference_cache_inventory)

    metrics = None
    if args.iterations:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("schema") != "cellerator.factorized_interactions.metrics.v1":
            raise RuntimeError("native runner returned unknown timing metrics schema")
        if metrics.get("scheduler_job_id") != scheduler_job_id:
            raise RuntimeError("native timing metrics do not match the Autobench job")
        if not str(metrics.get("device_uuid", "")).startswith("GPU-"):
            raise RuntimeError("native timing metrics omit the CUDA device UUID")
        if args.expected_uuid and metrics.get("device_uuid") != args.expected_uuid:
            raise RuntimeError("native timing device UUID differs from the explicit expected UUID")
        if metrics.get("iterations") != args.iterations:
            raise RuntimeError("native timing metrics report a different iteration count")
        if metrics.get("selected_backend") != native_execution.get("selected_backend"):
            raise RuntimeError("timing metrics and runner metadata disagree on backend")

    summary = {
        "schema": "cellerator.factorized_interactions.validation.v1",
        "workload": args.workload,
        "route": args.route,
        "activation_storage": args.activation_dtype,
        "parameter_storage": args.parameter_dtype,
        "output_storage": args.output_dtype,
        "gradient_mode": args.gradient_mode,
        "input_case_sha256": identity["sha256"],
        "input_case_bytes": identity["bytes"],
        "input_case_shards": len(identity["parts"]),
        "output_bytes": output_path.stat().st_size,
        "contract_evidence": {"y": output_evidence, **gradient_evidence},
        "build_provenance": build_provenance,
        "native_execution": native_execution,
        "validated": True,
        "timed": bool(args.iterations),
        "scheduler_job_id": scheduler_job_id or None,
        "metrics": metrics,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
