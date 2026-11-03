"""Bounded, source-bound invocation of the fixed factorized validator.

Successful case wires are temporary.  Their hashes, the complete validator
summary and both process streams are durably recorded before removal.  Failed
cases retain their wire directory.  This module neither grants nor submits a
GPU allocation.
"""

from __future__ import annotations

from collections import Counter
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import threading
from typing import Any, Callable, Mapping

sys.dont_write_bytecode = True

VALIDATOR_SHA256 = "466b495ac56fb0df9c248649f7307d871de681200a4a90de43457c0c6447de31"
ACCEPTED_SOURCE_SHA256 = "1f8e34ce89594fc3d8fb3d63c46600091c24f3016d6a9ddfd9dc3630eb08543c"
LIMIT_BYTES = 128 * 1024 * 1024
MODES = ("packet", "native_direct", "minimal_direct", "direct_unique_output")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_HEX = re.compile(r"^[0-9a-f]{64}$")


class ValidationFailure(RuntimeError):
    """A case failed; its owned wire directory is intentionally retained."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValidationFailure("manifest path is not a POSIX relative path")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ValidationFailure("manifest path escapes its owner")
    candidate = root.joinpath(*path.parts)
    if any(p.is_symlink() for p in (candidate, *candidate.parents) if p != root.parent):
        raise ValidationFailure("manifest path contains a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)) or not resolved.is_file():
        raise ValidationFailure("manifest path is outside its owner or not a file")
    return resolved


def _source_identity(source: Path, entries: Any) -> str:
    if not isinstance(entries, list) or not entries:
        raise ValidationFailure("native manifest omits its source file inventory")
    digest = hashlib.sha256()
    seen = set()
    prior = ""
    for row in entries:
        if not isinstance(row, dict):
            raise ValidationFailure("invalid source inventory record")
        name, size, expected = row.get("path"), row.get("bytes"), row.get("sha256")
        if not isinstance(name, str) or name <= prior or name in seen:
            raise ValidationFailure("source inventory paths must be sorted and unique")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0 or \
                not isinstance(expected, str) or not _HEX.fullmatch(expected):
            raise ValidationFailure("invalid source inventory size or digest")
        path = _checked_path(source, name)
        if path.stat().st_size != size or sha256_file(path) != expected:
            raise ValidationFailure(f"source input changed: {name}")
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "little"))
        digest.update(bytes.fromhex(expected))
        prior = name
        seen.add(name)
    actual = set()
    for current, dirs, files in os.walk(source, followlinks=False):
        parent = Path(current)
        if any((parent / name).is_symlink() for name in dirs):
            raise ValidationFailure("source inventory contains a symlink directory")
        for name in files:
            path = parent / name
            if path.is_symlink() or not path.is_file():
                raise ValidationFailure("source inventory contains a nonregular file")
            actual.add(path.relative_to(source).as_posix())
    if actual != seen:
        raise ValidationFailure("source file set differs from its manifest")
    return digest.hexdigest()


def verify_build(source: Path, runner: Path, manifest: Path) -> dict[str, Any]:
    source = Path(source).resolve(strict=True)
    manifest = Path(manifest).resolve(strict=True)
    runner = Path(runner).resolve(strict=True)
    if not source.is_dir() or not manifest.is_file() or not runner.is_file():
        raise ValidationFailure("source, manifest, or runner is missing")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema") != "cellerator.factorized_interactions.build.v1" or \
            data.get("architecture") != "sm_70" or data.get("fast_math") is not False:
        raise ValidationFailure("native build is not the approved SM70 non-fast-math build")
    closure = _source_identity(source, data.get("source_files"))
    if closure != data.get("source_closure_sha256") or closure != ACCEPTED_SOURCE_SHA256:
        raise ValidationFailure("source closure differs from the accepted immutable build")
    binaries = data.get("binaries")
    if not isinstance(binaries, dict):
        raise ValidationFailure("native manifest omits binary identities")
    resolved = {}
    for key in ("runner", "factorized_interaction_cuda"):
        item = binaries.get(key)
        if not isinstance(item, dict):
            raise ValidationFailure(f"native manifest omits {key}")
        path = _checked_path(manifest.parent, item.get("path"))
        if path.stat().st_size != item.get("bytes") or sha256_file(path) != item.get("sha256"):
            raise ValidationFailure(f"native {key} differs from its manifest")
        resolved[key] = path
    if resolved["runner"] != runner:
        raise ValidationFailure("requested runner differs from the manifest")
    validator = _checked_path(source, "bench/factorized_interactions/validate_runner.py")
    validator_hash = sha256_file(validator)
    if validator_hash != VALIDATOR_SHA256:
        raise ValidationFailure("fixed validator differs from the independently accepted version")
    return {
        "runner_path": str(runner), "runner_sha256": sha256_file(runner),
        "build_manifest_path": str(manifest), "build_manifest_sha256": sha256_file(manifest),
        "source_closure_sha256": closure, "source_file_count": len(data["source_files"]),
        "library_path": str(resolved["factorized_interaction_cuda"]),
        "library_sha256": sha256_file(resolved["factorized_interaction_cuda"]),
        "validator_sha256": validator_hash,
    }


def _source_module(source: Path, directory: str, name: str):
    base = (Path(source).resolve(strict=True) / directory).resolve(strict=True)
    expected = (base / f"{name}.py").resolve(strict=True)
    existing = sys.modules.get(name)
    if existing is not None and Path(existing.__file__).resolve() != expected:
        raise ValidationFailure(f"{name} was already loaded from a different source closure")
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != expected:
        raise ValidationFailure(f"{name} did not load from the accepted source closure")
    return module


def read_context(source: Path, expected_uuid: str) -> dict[str, Any]:
    guard = _source_module(source, "bench/factorized_interactions/experiments", "in_job_kda")
    return guard.validate_context(os.environ, Path("/proc/self/cgroup").read_text(encoding="ascii"),
                                  expected_uuid)


def _same_context(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in
               ("scheduler_job_id", "scheduler_attempt_id", "device_uuid", "cgroup", "unit_name"))


def workload_facts(source: Path, workload: str, activation: str, parameter: str) -> dict[str, Any]:
    """Derive canonical maxima from the actual ordered records, then release them."""
    import numpy as np

    if activation not in ("fp16", "fp32") or parameter not in ("fp16", "fp32"):
        raise ValidationFailure("unsupported storage dtype")
    module = _source_module(source, "tests/factorized_interactions", "workloads")
    problem, dy = module.make_workload(
        workload, np.float16 if activation == "fp16" else np.float32,
        np.float16 if parameter == "fp16" else np.float32,
    )
    output_counts, input_counts, context_counts = Counter(), Counter(), Counter()
    alpha_counts, bias_counts, gamma_counts = Counter(), Counter(), Counter()
    max_terms = 0
    for mechanism in problem.mechanisms:
        output_counts[mechanism.target] += 1
        input_counts.update(mechanism.participants)
        context_counts.update(mechanism.context_ids)
        alpha_counts[mechanism.alpha_id] += 1
        bias_counts[mechanism.bias_id] += 1
        gamma_counts.update(mechanism.gamma_ids)
        max_terms = max(max_terms, len(mechanism.context_ids))
    maxima = {
        "output_mechanisms": max(output_counts.values(), default=0),
        "input_roles": max(input_counts.values(), default=0),
        "context_terms": max(context_counts.values(), default=0),
        "alpha_mechanisms": max(alpha_counts.values(), default=0),
        "bias_mechanisms": max(bias_counts.values(), default=0),
        "gamma_terms": max(gamma_counts.values(), default=0),
        "mechanism_terms": max_terms,
    }
    cells, channels = int(problem.x.shape[2]), int(problem.x.shape[1])
    mechanisms, outputs = len(problem.mechanisms), int(problem.target_count)
    active = bool(mechanisms and cells and channels)
    stable = active and (
        max_terms > 64 or maxima["output_mechanisms"] > 64 or
        maxima["input_roles"] > 64 or maxima["context_terms"] > 64 or
        maxima["alpha_mechanisms"] * ((cells + 3) // 4) > 64 or
        maxima["bias_mechanisms"] * ((cells + 31) // 32) > 64 or
        maxima["gamma_terms"] * ((cells + 31) // 32) > 64
    )
    unique = bool(active and not stable and outputs == mechanisms and
                  maxima["output_mechanisms"] == 1)
    facts = {"workload": workload, "cells": cells, "channels": channels,
             "mechanisms": mechanisms, "outputs": outputs, "maxima": maxima,
             "stable": bool(stable), "unique": unique,
             "strongest_direct_mode": "direct_unique_output" if unique else "minimal_direct"}
    del problem, dy, output_counts, input_counts, context_counts, alpha_counts, bias_counts, gamma_counts
    gc.collect()
    return facts


def expected_mode_identity(mode: str, facts: Mapping[str, Any]) -> tuple[str, str, str]:
    if mode not in MODES:
        raise ValidationFailure("unknown native candidate mode")
    stable = facts["stable"] is True
    selected = "stable" if stable else ("packet" if mode == "packet" else "direct")
    if mode == "packet":
        return selected, selected, "full_packet"
    if mode == "native_direct":
        return selected, selected, "full_packet"
    if mode == "direct_unique_output" and facts["unique"] is True:
        return "direct", "direct_unique_output", "canonical_direct"
    return selected, selected, "full_packet" if stable else "canonical_direct"


def _tree_bytes(root: Path) -> int:
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = os.scandir(directory)
        except FileNotFoundError:
            continue
        with entries:
            for entry in entries:
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except FileNotFoundError:
                    # Files such as atomic cache temporaries may vanish while
                    # the child runs. Other filesystem errors fail closed.
                    continue
    return total


class _PeakSampler:
    def __init__(self, root: Path):
        self.root = root
        self.stop_event = threading.Event()
        self.peak = _tree_bytes(root)
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(0.05):
            try:
                self.peak = max(self.peak, _tree_bytes(self.root))
            except Exception as exc:
                self.error = exc
                self.stop_event.set()
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, unused_value, unused_traceback) -> None:
        self.stop_event.set()
        self.thread.join()
        if self.error is not None and exc_type is None:
            raise ValidationFailure("cannot measure the complete packet scratch tree") from self.error
        self.peak = max(self.peak, _tree_bytes(self.root))


def _durable_new(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _owned_cache(artifact_root: Path, cache_dir: Path, *,
                 context: Mapping[str, Any], build: Mapping[str, Any], helper_hash: str) -> Path:
    """Use only a cache first created for this exact packet and build.

    The fixed validator may invalidate its dedicated cache after detecting
    tampering. Refusing any pre-existing unowned directory prevents that
    behavior from touching older data supplied by a caller.
    """
    marker = artifact_root / ".bounded-cache-owner.json"
    expected = {"schema": "cellerator.factorized_interactions.bounded_cache_owner.v1",
                "cache_relative_path": cache_dir.relative_to(artifact_root).as_posix(),
                "scheduler_job_id": context["scheduler_job_id"],
                "scheduler_attempt_id": context["scheduler_attempt_id"],
                "source_closure_sha256": build["source_closure_sha256"],
                "runner_sha256": build["runner_sha256"],
                "bounded_validation_sha256": helper_hash}
    if marker.is_symlink() or cache_dir.is_symlink():
        raise ValidationFailure("bounded cache ownership paths cannot be symlinks")
    if marker.exists():
        if not marker.is_file() or not cache_dir.is_dir() or \
                json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValidationFailure("existing reference cache is not owned by this packet")
    else:
        if cache_dir.exists():
            raise ValidationFailure("refusing an existing unowned reference cache")
        cache_dir.mkdir(parents=True, exist_ok=False)
        _durable_new(marker, (json.dumps(expected, sort_keys=True, indent=2) + "\n").encode())
    return marker


def _inventory(root: Path) -> list[dict[str, Any]]:
    result = []
    for current, dirs, files in os.walk(root, followlinks=False):
        base = Path(current)
        if any((base / name).is_symlink() for name in dirs):
            raise ValidationFailure("case output contains a symlink directory")
        for name in files:
            path = base / name
            if path.is_symlink() or not path.is_file():
                raise ValidationFailure("case output contains a nonregular file")
            result.append({"path": path.relative_to(root).as_posix(),
                           "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return sorted(result, key=lambda x: x["path"])


def _check_summary(summary: Mapping[str, Any], *, build: Mapping[str, Any], context: Mapping[str, Any],
                   workload: str, mode: str, activation: str, parameter: str, output_dtype: str,
                   iterations: int, repeat_vjp: bool, facts: Mapping[str, Any] | None,
                   expected_identity: tuple[str, str, str] | None) -> None:
    route = "packet" if mode == "packet" else "direct"
    expected = {"schema": "cellerator.factorized_interactions.validation.v1", "validated": True,
                "timed": bool(iterations), "workload": workload, "route": route,
                "activation_storage": activation, "parameter_storage": parameter,
                "output_storage": output_dtype, "gradient_mode": "overwrite",
                "scheduler_job_id": context["scheduler_job_id"]}
    if any(summary.get(k) != value for k, value in expected.items()):
        raise ValidationFailure("fixed validator summary does not match the requested case")
    provenance = summary.get("build_provenance")
    if not isinstance(provenance, dict) or any(provenance.get(k) != build[k] for k in
            ("source_closure_sha256", "runner_binary_sha256", "build_manifest_sha256")):
        raise ValidationFailure("fixed validator used different build bytes")
    native = summary.get("native_execution")
    if not isinstance(native, dict):
        raise ValidationFailure("fixed validator omitted native execution metadata")
    required_native = {"candidate": mode, "requested_route": route,
                       "device_uuid": context["device_uuid"],
                       "scheduler_job_id": context["scheduler_job_id"]}
    if any(native.get(k) != value for k, value in required_native.items()):
        raise ValidationFailure("native dispatch metadata differs from the requested case")
    identity = (native.get("selected_backend"), native.get("actual_implementation"),
                native.get("structural_layout"))
    if facts is not None:
        if any(native.get(k) != facts[k] for k in ("cells", "channels", "mechanisms", "outputs")) or \
                native.get("structural_maxima") != facts["maxima"]:
            raise ValidationFailure("native structure differs from the fixed workload records")
        if identity != expected_mode_identity(mode, facts):
            raise ValidationFailure("native backend differs from the fixed record-derived predicate")
    elif identity[0] not in {"packet", "direct", "stable"} or \
            identity[1] not in {"packet", "direct", "stable", "direct_unique_output"} or \
            identity[2] not in {"full_packet", "canonical_direct"}:
        raise ValidationFailure("native dispatch identity is incomplete")
    if expected_identity is not None and identity != expected_identity:
        raise ValidationFailure("native backend differs from the expected identity")
    evidence = summary.get("contract_evidence")
    names = ["y", "dx", "dz", "dalpha", "dbias", "dgamma"]
    if repeat_vjp:
        names.extend(f"{name}_repeat" for name in ("dx", "dz", "dalpha", "dbias", "dgamma"))
    if not isinstance(evidence, dict) or any(name not in evidence for name in names):
        raise ValidationFailure("numeric evidence omits output or a required VJP")
    if iterations:
        metrics = summary.get("metrics")
        if not isinstance(metrics, dict) or metrics.get("iterations") != iterations:
            raise ValidationFailure("timed validator result has incomplete metrics")
        for key, value in (("candidate", mode), ("selected_backend", identity[0]),
                           ("actual_implementation", identity[1]), ("structural_layout", identity[2]),
                           ("device_uuid", context["device_uuid"]),
                           ("scheduler_job_id", context["scheduler_job_id"])):
            if metrics.get(key) != value:
                raise ValidationFailure(f"metrics {key} differs from execution metadata")
        samples = metrics.get("samples_ms")
        if not isinstance(samples, dict):
            raise ValidationFailure("raw timing vectors are missing")
        for name in ("forward", "backward", "forward_vjp", "parameter_refresh"):
            values = samples.get(name)
            if not isinstance(values, list) or len(values) != iterations or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or
                    not math.isfinite(float(v)) or float(v) < 0 for v in values):
                raise ValidationFailure(f"raw {name} timing vector is invalid")
        phase = metrics.get("phase_ms")
        if not isinstance(phase, dict):
            raise ValidationFailure("primary timing phase is missing")
        mean, total = phase.get("warm_forward_vjp_batch_host_wall_mean"), phase.get("warm_forward_vjp_batch_host_wall")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or
               float(v) <= 0 for v in (mean, total)) or \
                not math.isclose(float(mean) * iterations, float(total), rel_tol=2e-8, abs_tol=1e-9):
            raise ValidationFailure("primary uninstrumented host-wall block metric is invalid")


def run_validation(*, source: Path, build: Mapping[str, Any], expected_uuid: str,
                   context: Mapping[str, Any], artifact_root: Path, cache_dir: Path,
                   case_id: str, workload: str, mode: str, activation: str,
                   parameter: str, output_dtype: str, iterations: int,
                   repeat_vjp: bool, expected_identity: tuple[str, str, str] | None = None,
                   facts: Mapping[str, Any] | None = None, packet_root: Path | None = None,
                   invoke: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict[str, Any]:
    if not _SAFE_ID.fullmatch(case_id) or mode not in MODES or \
            activation not in ("fp16", "fp32") or parameter not in ("fp16", "fp32") or \
            output_dtype not in ("fp16", "fp32") or iterations not in (0, 30):
        raise ValidationFailure("unsupported case identifier, mode, dtype, or iteration count")
    source = Path(source).resolve(strict=True)
    artifact_root = Path(artifact_root).absolute()
    packet_root = Path(packet_root if packet_root is not None else artifact_root.parent).absolute()
    cache_dir = Path(cache_dir).absolute()
    for path in (artifact_root, packet_root, cache_dir):
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValidationFailure("packet output paths cannot contain symlinks")
    artifact_resolved, cache_resolved = artifact_root.resolve(), cache_dir.resolve()
    if not artifact_resolved.is_relative_to(packet_root.resolve()) or \
            cache_resolved == artifact_resolved or not cache_resolved.is_relative_to(artifact_resolved) or \
            (artifact_resolved / "wire").is_relative_to(cache_resolved) or \
            cache_resolved.is_relative_to(artifact_resolved / "wire") or \
            (artifact_resolved / "evidence").is_relative_to(cache_resolved) or \
            cache_resolved.is_relative_to(artifact_resolved / "evidence") or \
            packet_root.resolve().is_relative_to(source.parent):
        raise ValidationFailure("artifact and cache paths must be within packet output, outside the shared build")
    validator = source / "bench/factorized_interactions/validate_runner.py"
    if sha256_file(validator) != build["validator_sha256"] or build["validator_sha256"] != VALIDATOR_SHA256:
        raise ValidationFailure("fixed validator changed since build verification")
    helper_path = Path(__file__).resolve(strict=True)
    helper_hash = sha256_file(helper_path)
    if context.get("device_uuid") != expected_uuid:
        raise ValidationFailure("outer context UUID differs from the request")
    if not _same_context(context, read_context(source, expected_uuid)):
        raise ValidationFailure("live outer context differs from the case context")
    packet_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    cache_marker = _owned_cache(artifact_root, cache_dir, context=context, build=build,
                                helper_hash=helper_hash)
    cache_marker_hash = sha256_file(cache_marker)
    wire_root, evidence_root = artifact_root / "wire", artifact_root / "evidence"
    wire_root.mkdir(exist_ok=True)
    evidence_root.mkdir(exist_ok=True)
    if cache_dir.resolve().is_relative_to(wire_root.resolve()) or \
            cache_dir.resolve().is_relative_to(evidence_root.resolve()):
        raise ValidationFailure("reference cache must be outside case and evidence directories")
    case_dir = wire_root / case_id
    out_path, err_path = evidence_root / f"{case_id}.stdout", evidence_root / f"{case_id}.stderr"
    evidence_path = evidence_root / f"{case_id}.json"
    if any(path.exists() or path.is_symlink() for path in (out_path, err_path, evidence_path)):
        raise ValidationFailure("case evidence path already exists")
    case_dir.mkdir(exist_ok=False)
    command = [sys.executable, str(validator), "--workload", workload,
               "--runner", shlex.join([build["runner_path"], "--candidate", mode]),
               "--runner-binary", build["runner_path"], "--build-manifest", build["build_manifest_path"],
               "--route", "packet" if mode == "packet" else "direct",
               "--activation-dtype", activation, "--parameter-dtype", parameter,
               "--output-dtype", output_dtype, "--gradient-mode", "overwrite",
               "--expected-uuid", expected_uuid, "--iterations", str(iterations),
               "--artifact-dir", str(case_dir), "--reference-cache-dir", str(cache_dir)]
    if repeat_vjp:
        command.append("--repeat-vjp")
    stdout, stderr = "", ""
    cleanup_started = False
    sampler = _PeakSampler(packet_root)
    try:
        with sampler:
            process = invoke(command, cwd=case_dir, env=os.environ.copy(),
                             capture_output=True, text=True, check=False)
        stdout, stderr = process.stdout or "", process.stderr or ""
        _durable_new(out_path, stdout.encode("utf-8"))
        _durable_new(err_path, stderr.encode("utf-8"))
        if process.returncode != 0:
            raise ValidationFailure(f"fixed validator exited with status {process.returncode}")
        if not _same_context(context, read_context(source, expected_uuid)):
            raise ValidationFailure("outer Scheduler context changed during validation")
        if sha256_file(helper_path) != helper_hash or sha256_file(validator) != build["validator_sha256"]:
            raise ValidationFailure("validator or bounded harness changed during validation")
        if sha256_file(cache_marker) != cache_marker_hash:
            raise ValidationFailure("reference cache ownership marker changed during validation")
        summary = json.loads(stdout)
        if not isinstance(summary, dict):
            raise ValidationFailure("fixed validator returned a non-object summary")
        normalized_build = {**build, "runner_binary_sha256": build["runner_sha256"]}
        _check_summary(summary, build=normalized_build, context=context, workload=workload,
                       mode=mode, activation=activation, parameter=parameter,
                       output_dtype=output_dtype, iterations=iterations,
                       repeat_vjp=repeat_vjp, facts=facts, expected_identity=expected_identity)
        inventory = _inventory(case_dir)
        inputs = [row for row in inventory if row["path"].endswith(".case") or ".case.part" in row["path"]]
        outputs = [row for row in inventory if row["path"].endswith(".result")]
        if not inputs or len(outputs) != 1 or len(inputs) != summary.get("input_case_shards"):
            raise ValidationFailure("fixed validator wire inventory is incomplete")
        stream_hash = hashlib.sha256()
        for row in inputs:
            with (case_dir / row["path"]).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    stream_hash.update(block)
        if stream_hash.hexdigest() != summary.get("input_case_sha256") or \
                sum(row["bytes"] for row in inputs) != summary.get("input_case_bytes") or \
                outputs[0]["bytes"] != summary.get("output_bytes"):
            raise ValidationFailure("wire bytes differ from the fixed validator summary")
        prior = max(sampler.peak, _tree_bytes(packet_root))
        validated_path = evidence_root / f"{case_id}.validated.json"
        evidence = {"schema": "cellerator.factorized_interactions.bounded_case.v1", "status": "pass",
                    "case_id": case_id, "workload": workload, "mode": mode,
                    "summary": summary, "wire_inventory": inventory,
                    "input_case_sha256": stream_hash.hexdigest(),
                    "output_result_sha256": outputs[0]["sha256"], "context": dict(context),
                    "build": dict(build), "validator_sha256": build["validator_sha256"],
                    "bounded_validation_sha256": helper_hash,
                    "cache_owner_relative_path": cache_marker.relative_to(packet_root).as_posix(),
                    "cache_owner_sha256": cache_marker_hash,
                    "stdout_sha256": sha256_file(out_path), "stderr_sha256": sha256_file(err_path),
                    "peak_scratch_bytes": prior, "scratch_limit_bytes": LIMIT_BYTES,
                    "peak_measurement": "50ms sampled whole packet tree plus post-write evidence allowance",
                    "success_wires_removed": False, "cleanup_pending": True,
                    "evidence_path": str(evidence_path),
                    "evidence_relative_path": evidence_path.relative_to(packet_root).as_posix(),
                    "validated_relative_path": validated_path.relative_to(packet_root).as_posix(),
                    "stdout_relative_path": out_path.relative_to(packet_root).as_posix(),
                    "stderr_relative_path": err_path.relative_to(packet_root).as_posix()}
        wire_bytes = sum(row["bytes"] for row in inventory)
        for _ in range(4):
            encoded = (json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            final_record = {**evidence, "success_wires_removed": True,
                            "cleanup_pending": False, "validated_record_sha256": "0" * 64}
            final_size = len((json.dumps(final_record, sort_keys=True, indent=2,
                                         allow_nan=False) + "\n").encode())
            current = _tree_bytes(packet_root)
            predicted = max(sampler.peak, current + len(encoded),
                            current + len(encoded) + final_size - wire_bytes)
            if predicted == evidence["peak_scratch_bytes"]:
                break
            evidence["peak_scratch_bytes"] = predicted
        encoded = (json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        if evidence["peak_scratch_bytes"] >= LIMIT_BYTES:
            raise ValidationFailure("observed packet scratch peak reached the 128 MiB limit")
        _durable_new(validated_path, encoded)
        # No existing case directory is ever reused. Only this newly created,
        # successful case is removed after its durable evidence is complete.
        cleanup_started = True
        shutil.rmtree(case_dir)
        fd = os.open(wire_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        evidence.update({"success_wires_removed": True, "cleanup_pending": False,
                         "validated_record_sha256": sha256_file(validated_path)})
        _durable_new(evidence_path, (json.dumps(evidence, sort_keys=True, indent=2,
                                               allow_nan=False) + "\n").encode())
        return evidence
    except BaseException as exc:
        # Numeric/process failures occur before cleanup and retain every wire.
        # A later filesystem failure may leave a partially removed directory;
        # report that stage honestly alongside the durable pre-cleanup record.
        for path, text in ((out_path, stdout), (err_path, stderr)):
            if not path.exists() and not path.is_symlink():
                _durable_new(path, text.encode("utf-8"))
        failure_path = evidence_root / f"{case_id}.failure.json"
        if not failure_path.exists() and not failure_path.is_symlink():
            wires_exist = case_dir.exists()
            try:
                retained_inventory = _inventory(case_dir) if wires_exist else []
            except (OSError, ValidationFailure):
                retained_inventory = None
            failure = {"schema": "cellerator.factorized_interactions.bounded_case.v1",
                       "status": "fail", "case_id": case_id, "error": f"{type(exc).__name__}: {exc}",
                       "wire_path": str(case_dir), "wire_relative_path": case_dir.relative_to(packet_root).as_posix(),
                       "stdout_relative_path": out_path.relative_to(packet_root).as_posix(),
                       "stderr_relative_path": err_path.relative_to(packet_root).as_posix(),
                       "stdout_sha256": sha256_file(out_path), "stderr_sha256": sha256_file(err_path),
                       "retained_wire_inventory": retained_inventory,
                       "context": dict(context), "build": dict(build),
                       "peak_scratch_bytes": max(sampler.peak, _tree_bytes(packet_root)),
                       "scratch_limit_bytes": LIMIT_BYTES, "cleanup_started": cleanup_started,
                       "wires_exist": wires_exist, "success_wires_removed": cleanup_started and not wires_exist}
            _durable_new(failure_path, (json.dumps(failure, sort_keys=True, indent=2) + "\n").encode())
        disposition = (f"remaining wires at {case_dir}" if case_dir.exists() else
                       "wires were removed after durable validation" if cleanup_started else
                       f"wire directory unavailable at {case_dir}")
        raise ValidationFailure(f"{case_id}: {exc}; {disposition}") from exc
