#!/usr/bin/env python3
"""Run both exact-oracle ATen fixtures on the Scheduler-visible device pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_uuid(value: object) -> str:
    if not isinstance(value, str):
        raise SystemExit(f"device UUID is not a string: {value!r}")
    normalized = value.strip().casefold()
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    normalized = normalized.replace("-", "")
    if len(normalized) != 32 or any(character not in "0123456789abcdef" for character in normalized):
        raise SystemExit(f"device UUID has an unexpected representation: {value!r}")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-uuid0", required=True)
    parser.add_argument("--expected-uuid1", required=True)
    parser.add_argument("--device0-index", type=int, default=0,
                        help="CUDA-visible ordinal assigned to logical source-owner slot 0")
    parser.add_argument("--device1-index", type=int, default=1,
                        help="CUDA-visible ordinal assigned to logical source-owner slot 1")
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    binary = args.binary.resolve(strict=True)
    artifact_manifest = args.artifact_manifest.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if source not in binary.parents or source not in artifact_manifest.parents:
        raise SystemExit("ATen binary and external build manifest must be included in the Autobench snapshot")
    if not any(
        tuple(source.parts[index:index + 3]) == (".autobench", "state", "snapshots")
        for index in range(len(source.parts) - 2)
    ):
        raise SystemExit("ATen correctness must run from an immutable Autobench snapshot")
    expected_devices = [args.expected_uuid0, args.expected_uuid1]
    expected_normalized = [normalize_uuid(value) for value in expected_devices]
    if len(set(expected_normalized)) != 2:
        raise SystemExit("the approved two-device UUIDs must be distinct")
    expected_visible_indices = [args.device0_index, args.device1_index]
    if (args.device0_index == args.device1_index or
            sorted(expected_visible_indices) != [0, 1]):
        raise SystemExit("logical device slots must map bijectively to visible indices 0 and 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = source / "bench/native_data/run_native_data_bench.py"
    results = {}
    binary_hash = sha256_file(binary)
    for case in ("reference_active", "reference_inactive"):
        result_path = output_dir / f"{case}.aten.json"
        command = [
            sys.executable, str(wrapper),
            "--binary", str(binary), "--artifact-manifest", str(artifact_manifest),
            "--source-root", str(source), "--case", case,
            "--mode", "correctness", "--output", str(result_path),
            "--device0", str(args.device0_index), "--device1", str(args.device1_index),
            "--warmup", "0", "--samples", "1",
            "--atol", "2e-5", "--rtol", "2e-5",
        ]
        completed = subprocess.run(command, cwd=source, text=True, capture_output=True, check=False)
        (output_dir / f"{case}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (output_dir / f"{case}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise SystemExit(f"{case} ATen correctness failed: {completed.stdout} {completed.stderr}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("backend") != "aten_eager_two_device":
            raise SystemExit(f"{case} returned an unexpected backend")
        validation = result.get("correctness_validation", {})
        if validation.get("passed") is not True:
            raise SystemExit(f"{case} did not pass the independent exact Fraction oracle")
        devices = result.get("devices", [])
        observed_devices = [normalize_uuid(device.get("uuid")) for device in devices]
        observed_slots = [device.get("visible_index") for device in devices]
        if observed_devices != expected_normalized or observed_slots != expected_visible_indices:
            raise SystemExit(
                f"{case} Scheduler grant/slot order mismatch: expected {expected_devices} "
                f"at visible indices {expected_visible_indices}, observed "
                f"{observed_devices} at {observed_slots}"
            )
        results[case] = {
            "passed": True,
            "result_file": result_path.name,
            "result_sha256": sha256_file(result_path),
            "binary_sha256": result.get("artifact_identity", {}).get("binary_sha256"),
            "source_snapshot_sha256": result.get("artifact_identity", {}).get("snapshot_source_sha256"),
            "external_build_manifest_sha256": result.get("artifact_identity", {}).get("external_build_manifest_sha256"),
            "compile_source_snapshot_sha256": result.get("artifact_identity", {}).get("compile_source_snapshot_sha256"),
            "devices": result.get("devices"),
            "validation": validation,
        }

    if any(value["binary_sha256"] != binary_hash for value in results.values()):
        raise SystemExit("baseline binary identity changed during fixture validation")
    if any(value["external_build_manifest_sha256"] != sha256_file(artifact_manifest)
           for value in results.values()):
        raise SystemExit("external build manifest identity changed during fixture validation")
    report = {
        "schema": "cellerator.native_data.aten_reference_campaign.v1",
        "backend": "aten_eager_two_device",
        "binary_sha256": binary_hash,
        "snapshot_root": str(source),
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "approved_device_uuids_by_logical_slot": expected_devices,
        "approved_device_uuid_hex_by_logical_slot": expected_normalized,
        "visible_indices_by_logical_slot": expected_visible_indices,
        "scheduler_resource_order_uuids": [args.expected_uuid1, args.expected_uuid0],
        "artifact_kind": "externally_built_autobench_snapshot_artifact",
        "external_build_manifest_sha256": sha256_file(artifact_manifest),
        "compile_source_snapshot_sha256": results["reference_active"]["compile_source_snapshot_sha256"],
        "passed": True,
        "cases": results,
    }
    report_path = output_dir / "campaign.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"passed": True, "binary_sha256": binary_hash,
                      "cases": sorted(results), "devices_by_logical_slot": expected_devices,
                      "visible_indices_by_logical_slot": expected_visible_indices}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
