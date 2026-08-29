#!/usr/bin/env python3
"""Validate CE-GEO command manifests and evidence without running hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


COMMAND_SCHEMA = "CELLERATOR-CE-GEO-COMMAND/1"
EVIDENCE_SCHEMA = "CELLERATOR-CE-GEO-EVIDENCE/1"

COLD_PHASES = {
    "host_preparation",
    "semantic_packing",
    "projection_construction",
    "backend_prepare",
    "static_value_pack",
    "persistent_upload",
}
WARM_PHASES = {
    "dynamic_h2d",
    "dynamic_input_pack",
    "kernel",
    "residual",
    "epilogue",
    "order_transform",
    "synchronization",
    "communication",
    "d2h",
    "consumer_visible_complete",
}
REQUIRED_LEASES = {"benchmark", "gpu"}


class EvidenceError(ValueError):
    """Raised when committed evidence is not decision-safe."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_text(value: Any, field: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{field} is missing")
    return value


def require_digest(value: Any, field: str) -> str:
    text = require_text(value, field)
    require(len(text) == 64 and all(char in "0123456789abcdef" for char in text),
            f"{field} must be a lowercase SHA-256")
    return text


def require_argv(value: Any, field: str) -> list[str]:
    require(isinstance(value, list) and bool(value), f"{field} must be nonempty argv")
    require(all(isinstance(item, str) and item for item in value),
            f"{field} contains an empty or non-string argument")
    return value


def validate_command_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    require(manifest.get("schema") == COMMAND_SCHEMA, "command manifest schema")
    require_text(manifest.get("campaign_id"), "campaign_id")
    require(manifest.get("project_root") == "/home/tumlinson/Cellerator",
            "project_root must identify the Cellerator checkout")

    commands = manifest.get("commands")
    require(isinstance(commands, dict), "commands must be an object")
    for phase in ("build", "correctness", "measure"):
        entries = commands.get(phase)
        require(isinstance(entries, list) and bool(entries),
                f"commands.{phase} must be nonempty")
        for index, argv in enumerate(entries):
            require_argv(argv, f"commands.{phase}[{index}]")

    methodology = manifest.get("methodology")
    require(isinstance(methodology, dict), "methodology must be an object")
    warmups = methodology.get("warmups")
    repeats = methodology.get("repeats")
    require(isinstance(warmups, int) and warmups >= 1, "warmups must be >= 1")
    require(isinstance(repeats, int) and repeats >= 5 and repeats % 2 == 1,
            "repeats must be odd and >= 5")
    spread = methodology.get("maximum_mad_percent")
    require(isinstance(spread, (int, float)) and 0.0 < spread <= 100.0,
            "maximum_mad_percent must be in (0, 100]")
    require(methodology.get("cold_warm_separated") is True,
            "cold and warm accounting must be separated")
    require(methodology.get("correctness_before_measurement") is True,
            "correctness must precede measurement")

    resources = manifest.get("resources")
    require(isinstance(resources, dict), "resources must be an object")
    leases = resources.get("required_leases")
    require(isinstance(leases, list) and REQUIRED_LEASES <= set(leases),
            "benchmark and gpu leases are mandatory")
    require(resources.get("benchmark_mutex") == "bench/benchmark_mutex.hh",
            "repository benchmark mutex is mandatory")
    require(resources.get("run_without_leases") is False,
            "manifest may not authorize unleased execution")

    capture = manifest.get("capture")
    require(isinstance(capture, dict), "capture must be an object")
    for field in ("source", "device", "topology", "toolchain", "build"):
        require(capture.get(field) is True, f"capture.{field} must be true")

    phases = manifest.get("required_phases")
    require(isinstance(phases, dict), "required_phases must be an object")
    require(set(phases.get("cold", [])) == COLD_PHASES,
            "cold phase contract differs")
    require(set(phases.get("warm", [])) == WARM_PHASES,
            "warm phase contract differs")
    return {
        "status": "valid",
        "kind": "command_manifest",
        "campaign_id": manifest["campaign_id"],
        "performance_run": False,
    }


def median_and_mad_percent(samples: list[float]) -> tuple[float, float]:
    median = float(statistics.median(samples))
    mad = float(statistics.median(abs(value - median) for value in samples))
    percent = 0.0 if median == 0.0 and mad == 0.0 else math.inf
    if median != 0.0:
        percent = 100.0 * mad / abs(median)
    return median, percent


def require_identity_block(record: dict[str, Any], field: str,
                           required: tuple[str, ...]) -> dict[str, Any]:
    value = record.get(field)
    require(isinstance(value, dict), f"{field} must be an object")
    for name in required:
        require_text(value.get(name), f"{field}.{name}")
    return value


def validate_phase_samples(evidence: dict[str, Any], repeats: int) -> None:
    phases = evidence.get("phase_samples_ns")
    require(isinstance(phases, dict), "phase_samples_ns must be an object")
    require(set(phases.get("cold", {})) == COLD_PHASES,
            "cold samples do not cover the phase contract")
    require(set(phases.get("warm", {})) == WARM_PHASES,
            "warm samples do not cover the phase contract")
    for temperature, required in (("cold", COLD_PHASES), ("warm", WARM_PHASES)):
        for phase in required:
            samples = phases[temperature][phase]
            require(isinstance(samples, list) and len(samples) == repeats,
                    f"{temperature}.{phase} must have exactly {repeats} samples")
            require(all(isinstance(value, (int, float)) and math.isfinite(value)
                        and value >= 0.0 for value in samples),
                    f"{temperature}.{phase} has an invalid sample")


def validate_evidence(evidence: dict[str, Any], manifest: dict[str, Any],
                      manifest_sha256: str | None = None) -> dict[str, Any]:
    validate_command_manifest(manifest)
    require(evidence.get("schema") == EVIDENCE_SCHEMA, "evidence schema")
    require(evidence.get("campaign_id") == manifest.get("campaign_id"),
            "campaign identity differs from command manifest")
    if manifest_sha256 is not None:
        require(evidence.get("command_manifest_sha256") == manifest_sha256,
                "command manifest digest differs")
    else:
        require_digest(evidence.get("command_manifest_sha256"),
                       "command_manifest_sha256")

    source = require_identity_block(evidence, "source",
        ("revision", "status_digest", "todo_revision"))
    require(source.get("clean") is True, "measured source must be clean")
    require_digest(source.get("status_digest"), "source.status_digest")
    require(isinstance(source.get("submodule_revisions"), dict),
            "source.submodule_revisions must be an object")
    require_identity_block(evidence, "device",
        ("uuid", "name", "pci_bus_id", "performance_class", "driver_version"))
    require_identity_block(evidence, "topology",
        ("capture_command", "capture_sha256"))
    require_digest(evidence["topology"]["capture_sha256"],
                   "topology.capture_sha256")
    require_identity_block(evidence, "toolchain",
        ("cxx", "cuda_toolkit", "nvcc", "cmake"))
    build = require_identity_block(evidence, "build",
        ("mode", "architecture", "binary_sha256", "cmake_cache_sha256"))
    require_digest(build["binary_sha256"], "build.binary_sha256")
    require_digest(build["cmake_cache_sha256"], "build.cmake_cache_sha256")

    command = evidence.get("command")
    require(isinstance(command, dict), "command must be an object")
    require_argv(command.get("argv"), "command.argv")
    require_text(command.get("cwd"), "command.cwd")
    require_digest(command.get("environment_digest"), "command.environment_digest")

    controller = evidence.get("controller")
    require(isinstance(controller, dict), "controller must be an object")
    acquired = controller.get("acquired_leases")
    require(isinstance(acquired, dict) and REQUIRED_LEASES <= set(acquired),
            "controller lacks required leases")
    for lease in REQUIRED_LEASES:
        require_text(acquired.get(lease), f"controller.acquired_leases.{lease}")
    require(controller.get("benchmark_mutex_acquired") is True,
            "benchmark mutex acquisition is not evidenced")

    methodology = evidence.get("methodology")
    require(isinstance(methodology, dict), "evidence methodology must be an object")
    expected = manifest["methodology"]
    require(methodology.get("warmups") == expected["warmups"], "warmup count differs")
    repeats = methodology.get("repeats")
    require(repeats == expected["repeats"], "repeat count differs")
    require(methodology.get("clock") in {"cuda_event", "steady_clock", "controller_wall"},
            "timing clock is unsupported")
    validate_phase_samples(evidence, repeats)

    complete = evidence.get("complete_samples_ns")
    require(isinstance(complete, list) and len(complete) == repeats,
            "complete_samples_ns must match repeats")
    require(all(isinstance(value, (int, float)) and math.isfinite(value)
                and value > 0.0 for value in complete),
            "complete samples must be finite and positive")
    observed_median, observed_mad_percent = median_and_mad_percent(
        [float(value) for value in complete])
    summary = evidence.get("summary")
    require(isinstance(summary, dict), "summary must be an object")
    require(math.isclose(float(summary.get("median_complete_ns", -1.0)),
                         observed_median, rel_tol=1e-12, abs_tol=1e-6),
            "median_complete_ns was not derived from samples")
    require(math.isclose(float(summary.get("mad_percent", -1.0)),
                         observed_mad_percent, rel_tol=1e-12, abs_tol=1e-9),
            "mad_percent was not derived from samples")

    correctness = evidence.get("correctness")
    require(isinstance(correctness, dict) and correctness.get("passed") is True,
            "correctness did not pass before measurement")
    require_digest(correctness.get("digest"), "correctness.digest")
    require(isinstance(correctness.get("numerical_error"), dict),
            "numerical error tuple is missing")

    contamination = evidence.get("contamination")
    require(isinstance(contamination, dict), "contamination must be an object")
    reasons = contamination.get("reasons")
    require(isinstance(reasons, list), "contamination.reasons must be a list")
    contaminated = contamination.get("detected") is True
    require(contaminated == bool(reasons),
            "contamination flag and reasons disagree")
    require(isinstance(contamination.get("attempt"), int)
            and 1 <= contamination["attempt"] <= 3,
            "contamination attempt must be in [1, 3]")
    spread_limit = float(expected["maximum_mad_percent"])
    require(contaminated or observed_mad_percent <= spread_limit,
            "spread exceeds the declared limit without contamination")
    accepted = summary.get("accepted") is True
    require(accepted == (not contaminated and observed_mad_percent <= spread_limit),
            "accepted disposition contradicts contamination/spread")
    return {
        "status": "valid",
        "kind": "evidence",
        "campaign_id": evidence["campaign_id"],
        "accepted": accepted,
        "median_complete_ns": observed_median,
        "mad_percent": observed_mad_percent,
        "performance_run": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", required=True, type=Path)
    result.add_argument("--evidence", type=Path)
    result.add_argument("--json", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    manifest = load_json(args.manifest)
    if args.evidence is None:
        result = validate_command_manifest(manifest)
    else:
        evidence = load_json(args.evidence)
        result = validate_evidence(evidence, manifest, sha256_file(args.manifest))
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"CE-GEO harness {result['kind']} valid; performance_run=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
