#!/usr/bin/env python3
"""Qualify all storage dtypes and candidate modes on one fixed GPU workload.

This performs correctness checks only. It delegates every numeric comparison
to the unchanged validate_runner.py and records source/binary/execution identity
for the resulting coverage matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shlex
import subprocess
import sys
from typing import Any, Callable, Mapping

from in_job_kda import validate_context


SOURCE = Path(__file__).resolve().parents[3]
VALIDATOR = SOURCE / "bench" / "factorized_interactions" / "validate_runner.py"
WORKLOAD = "packet_a4_m32_u16"
VALIDATOR_SHA256 = "466b495ac56fb0df9c248649f7307d871de681200a4a90de43457c0c6447de31"
MODES = ("packet", "native_direct", "minimal_direct", "direct_unique_output")
DTYPES = ("fp16", "fp32")
EXPECTED_CASES = len(DTYPES) ** 3 * len(MODES)


class QualificationError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_closure_digest(entries: Any) -> tuple[str, int]:
    if not isinstance(entries, list) or not entries:
        raise QualificationError("build manifest must enumerate its complete source closure")
    digest = hashlib.sha256()
    prior_path = None
    for row in entries:
        if not isinstance(row, dict):
            raise QualificationError("source manifest entry must be an object")
        name, byte_count, file_hash = row.get("path"), row.get("bytes"), row.get("sha256")
        if not isinstance(name, str) or not name or "\\" in name:
            raise QualificationError("source manifest path is invalid")
        relpath = PurePosixPath(name)
        if relpath.is_absolute() or ".." in relpath.parts or "." in relpath.parts:
            raise QualificationError("source manifest path escapes the candidate closure")
        if prior_path is not None and name <= prior_path:
            raise QualificationError("source manifest paths must be unique and sorted")
        prior_path = name
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise QualificationError("source manifest byte count is invalid")
        if not isinstance(file_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", file_hash):
            raise QualificationError("source manifest file digest is invalid")
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(byte_count.to_bytes(8, "little"))
        digest.update(bytes.fromhex(file_hash))
    return digest.hexdigest(), len(entries)


def mode_route(mode: str) -> str:
    if mode not in MODES:
        raise QualificationError(f"unsupported candidate mode: {mode}")
    return "packet" if mode == "packet" else "direct"


def dtype_matrix() -> list[dict[str, str]]:
    return [
        {"activation": activation, "parameter": parameter, "output": output}
        for activation, parameter, output in itertools.product(DTYPES, repeat=3)
    ]


def expected_identity(mode: str) -> tuple[str, str, str]:
    if mode == "packet":
        return "packet", "packet", "full_packet"
    if mode == "direct_unique_output":
        return "direct", "direct_unique_output", "canonical_direct"
    if mode == "minimal_direct":
        return "direct", "direct", "canonical_direct"
    return "direct", "direct", "full_packet"


def check_build(runner: Path, manifest_path: Path) -> dict[str, Any]:
    if runner.is_symlink() or manifest_path.is_symlink():
        raise QualificationError("runner and build manifest must not be symlinks")
    runner = runner.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "cellerator.factorized_interactions.build.v1":
        raise QualificationError("candidate build manifest has an unknown schema")
    if manifest.get("architecture") != "sm_70" or manifest.get("fast_math") is not False:
        raise QualificationError("build must be the fixed non-fast-math SM70 artifact")
    source_hash = manifest.get("source_closure_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise QualificationError("build manifest omitted its source closure digest")
    recomputed_source_hash, source_file_count = source_closure_digest(
        manifest.get("source_files"))
    if source_hash != recomputed_source_hash:
        raise QualificationError("build source-file manifest does not match its closure digest")
    entry = manifest.get("binaries", {}).get("runner")
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise QualificationError("build manifest omitted the native candidate runner")
    expected_path = manifest_path.parent / entry["path"]
    if expected_path.is_symlink() or not expected_path.is_file():
        raise QualificationError("manifest runner path is missing or not a regular file")
    expected_path = expected_path.resolve(strict=True)
    if not expected_path.is_relative_to(manifest_path.parent):
        raise QualificationError("manifest runner path escapes its immutable build root")
    if runner != expected_path:
        raise QualificationError("requested runner differs from the immutable build manifest")
    runner_hash = sha256_file(runner)
    if runner_hash != entry.get("sha256"):
        raise QualificationError("runner binary SHA256 differs from the build manifest")
    return {
        "runner_path": str(runner),
        "runner_sha256": runner_hash,
        "build_manifest_path": str(manifest_path),
        "build_manifest_sha256": sha256_bytes(manifest_bytes),
        "source_closure_sha256": source_hash,
        "architecture": "sm_70",
        "source_file_count": source_file_count,
    }


def validate_summary(summary: Mapping[str, Any], *, mode: str,
                     dtype: Mapping[str, str], device_uuid: str,
                     scheduler_job_id: str, build: Mapping[str, str]) -> dict[str, Any]:
    route = mode_route(mode)
    expected_backend, implementation, layout = expected_identity(mode)
    if summary.get("schema") != "cellerator.factorized_interactions.validation.v1":
        raise QualificationError("validator returned an unknown result schema")
    if summary.get("validated") is not True or summary.get("timed") is not False:
        raise QualificationError("case was not validated as a correctness-only run")
    expected_fields = {
        "workload": WORKLOAD,
        "route": route,
        "activation_storage": dtype["activation"],
        "parameter_storage": dtype["parameter"],
        "output_storage": dtype["output"],
        "gradient_mode": "overwrite",
        "scheduler_job_id": scheduler_job_id,
    }
    for key, value in expected_fields.items():
        if summary.get(key) != value:
            raise QualificationError(f"validator result mismatch for {key}")
    provenance = summary.get("build_provenance")
    if not isinstance(provenance, dict) or \
            provenance.get("source_closure_sha256") != build["source_closure_sha256"] or \
            provenance.get("runner_binary_sha256") != build["runner_sha256"] or \
            provenance.get("build_manifest_sha256") != build["build_manifest_sha256"]:
        raise QualificationError("validator result is not bound to the supplied build")
    native = summary.get("native_execution")
    if not isinstance(native, dict):
        raise QualificationError("validator result omitted native execution metadata")
    native_expected = {
        "candidate": mode,
        "requested_route": route,
        "selected_backend": expected_backend,
        "actual_implementation": implementation,
        "structural_layout": layout,
        "device_uuid": device_uuid,
        "scheduler_job_id": scheduler_job_id,
    }
    for key, value in native_expected.items():
        if native.get(key) != value:
            raise QualificationError(f"runner metadata mismatch for {key} in {mode}")
    evidence = summary.get("contract_evidence")
    if not isinstance(evidence, dict) or not all(
            name in evidence for name in ("y", "dx", "dz", "dalpha", "dbias", "dgamma",
                                           "dx_repeat", "dz_repeat", "dalpha_repeat",
                                           "dbias_repeat", "dgamma_repeat")):
        raise QualificationError("oracle evidence does not cover output and all five repeated VJPs")
    return {
        "workload": WORKLOAD,
        **dict(dtype),
        "mode": mode,
        "route": route,
        "selected_backend": expected_backend,
        "actual_implementation": implementation,
        "structural_layout": layout,
        "input_case_sha256": summary.get("input_case_sha256"),
        "source_closure_sha256": build["source_closure_sha256"],
        "runner_sha256": build["runner_sha256"],
        "build_manifest_sha256": build["build_manifest_sha256"],
        "contract_evidence": evidence,
        "status": "pass",
    }


def validate_outer_context(expected_uuid: str) -> dict[str, Any]:
    scheduler_job_id = os.environ.get("AUTOBENCH_SCHEDULER_JOB_ID", "").strip()
    scheduler_attempt_id = os.environ.get("AUTOBENCH_SCHEDULER_ATTEMPT_ID", "").strip()
    if not scheduler_job_id or not scheduler_attempt_id:
        raise QualificationError("qualification requires the inherited Autobench Scheduler attempt")
    try:
        context = validate_context(os.environ, Path("/proc/self/cgroup").read_text(encoding="ascii"),
                                   expected_uuid)
    except Exception as exc:
        raise QualificationError(f"outer Scheduler context check failed: {exc}") from exc
    return context


def run_matrix(*, runner: Path, build_manifest: Path, expected_uuid: str,
               artifact_dir: Path, output: Path,
               reference_cache_dir: Path | None = None,
               invoke: Callable[..., subprocess.CompletedProcess] = subprocess.run,
               context_reader: Callable[[str], dict[str, Any]] = validate_outer_context) -> dict[str, Any]:
    context_before = context_reader(expected_uuid)
    scheduler_job_id = context_before["scheduler_job_id"]
    validator_hash = sha256_file(VALIDATOR)
    if validator_hash != VALIDATOR_SHA256:
        raise QualificationError("fixed validator differs from the trusted reference-cache version")
    build = check_build(runner, build_manifest)
    runner = Path(build["runner_path"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if reference_cache_dir is not None:
        reference_cache_dir = reference_cache_dir.expanduser().absolute()
        if any(path.is_symlink() for path in (reference_cache_dir, *reference_cache_dir.parents)):
            raise QualificationError("reference cache path cannot contain symbolic links")
        artifact_root = artifact_dir.resolve(strict=True)
        if not reference_cache_dir.resolve().is_relative_to(artifact_root):
            raise QualificationError("reference cache directory must stay within qualification artifacts")
        reference_cache_dir.mkdir(parents=True, exist_ok=True)
        reference_cache_dir = reference_cache_dir.resolve(strict=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for dtype in dtype_matrix():
        for mode in MODES:
            route = mode_route(mode)
            label = f"{mode}.a{dtype['activation']}.p{dtype['parameter']}.o{dtype['output']}"
            case_dir = artifact_dir / label
            case_dir.mkdir(parents=True, exist_ok=False)
            command = [
                sys.executable, str(VALIDATOR),
                "--workload", WORKLOAD,
                "--runner", shlex.join([str(runner), "--candidate", mode]),
                "--runner-binary", str(runner),
                "--build-manifest", str(build_manifest.resolve(strict=True)),
                "--route", route,
                "--activation-dtype", dtype["activation"],
                "--parameter-dtype", dtype["parameter"],
                "--output-dtype", dtype["output"],
                "--gradient-mode", "overwrite",
                "--expected-uuid", expected_uuid,
                "--repeat-vjp",
                "--iterations", "0",
                "--artifact-dir", str(case_dir),
            ]
            if reference_cache_dir is not None:
                command.extend(["--reference-cache-dir", str(reference_cache_dir.resolve(strict=True))])
            stdout_path = case_dir / "validator.stdout.json"
            stderr_path = case_dir / "validator.stderr.log"
            try:
                result = invoke(command, cwd=case_dir, env=os.environ.copy(),
                                capture_output=True, text=True, check=False)
                stdout_path.write_text(result.stdout or "", encoding="utf-8")
                stderr_path.write_text(result.stderr or "", encoding="utf-8")
                if result.returncode != 0:
                    raise QualificationError(f"validator exited with status {result.returncode}")
                summary = json.loads(result.stdout)
                record = validate_summary(summary, mode=mode, dtype=dtype,
                                          device_uuid=expected_uuid,
                                          scheduler_job_id=scheduler_job_id, build=build)
                record["validator_sha256"] = validator_hash
                records.append(record)
            except Exception as exc:
                failures.append({"case": label, "error": f"{type(exc).__name__}: {exc}",
                                 "stdout": str(stdout_path), "stderr": str(stderr_path)})

    context_after = context_reader(expected_uuid)
    if any(context_before.get(key) != context_after.get(key)
           for key in ("scheduler_job_id", "scheduler_attempt_id", "device_uuid", "cgroup", "unit_name")):
        failures.append({"case": "outer_context", "error": "Scheduler job, attempt, UUID or cgroup changed during suite"})
    runner_hash_after = sha256_file(runner)
    manifest_hash_after = sha256_file(Path(build["build_manifest_path"]))
    if runner_hash_after != build["runner_sha256"] or \
            manifest_hash_after != build["build_manifest_sha256"]:
        failures.append({"case": "build_identity", "error": "runner or build manifest changed during suite"})
    result = {
        "schema": "cellerator.factorized_interactions.dtype_qualification.v1",
        "status": "pass" if not failures and len(records) == EXPECTED_CASES else "fail",
        "scope": "correctness_only_fixed_workload",
        "timing_performed": False,
        "workload": WORKLOAD,
        "modes": list(MODES),
        "dtype_combinations": dtype_matrix(),
        "expected_case_count": EXPECTED_CASES,
        "reference_cache_dir": str(reference_cache_dir)
        if reference_cache_dir is not None else None,
        "passed_case_count": len(records),
        "scheduler_context": context_before,
        "build": build,
        "validator_sha256": validator_hash,
        "records": records,
        "failures": failures,
    }
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--reference-cache-dir", type=Path,
                        help="optional trusted shared reference cache for the unchanged validator")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_matrix(
            runner=args.runner, build_manifest=args.build_manifest,
            expected_uuid=args.expected_uuid, artifact_dir=args.artifact_dir,
            output=args.output, reference_cache_dir=args.reference_cache_dir,
        )
    except Exception as exc:
        print(f"dtype qualification rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "cases": result["passed_case_count"],
                      "expected": result["expected_case_count"], "output": str(args.output)},
                     sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
