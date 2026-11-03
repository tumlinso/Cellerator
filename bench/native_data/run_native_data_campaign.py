#!/usr/bin/env python3
"""Run immutable paired correctness/timing packets for the native-data path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ATOL = 2.0e-5
RTOL = 2.0e-5
CASES = ("low", "high")
CASE_RESULT_NAMES = {
    "low": "low_activity_disjoint_32",
    "high": "high_activity_overlap_32",
}
BACKENDS = {
    "aten": ("aten_eager_two_device", "cellerator_native_data_aten_bench"),
    "specialist": ("cellerator_distributed_gated_product",
                   "cellerator_native_data_specialist_bench"),
}
QUIESCENCE_SAMPLES = 3
QUIESCENCE_INTERVAL_SECONDS = 1.0
QUIESCENCE_MAX_UTILIZATION_PERCENT = 0
QUIESCENCE_MINIMUM_FREE_BYTES = 1_610_612_736


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


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


def _query_nvidia_smi(query: str, expected_columns: int) -> tuple[list[list[str]], dict]:
    command = ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    receipt = {"argv": command, "returncode": completed.returncode,
               "stdout": completed.stdout, "stderr": completed.stderr}
    if completed.returncode != 0:
        return [], receipt
    lines = completed.stdout.strip().splitlines()
    if not lines:
        return [], receipt
    if query.startswith("compute-apps=") and all(
            "no running processes found" in line.casefold() for line in lines):
        return [], receipt
    records = list(csv.reader(io.StringIO(completed.stdout)))
    if any(len(record) != expected_columns for record in records):
        receipt["parse_error"] = f"expected {expected_columns} CSV columns"
        return [], receipt
    return [[value.strip() for value in record] for record in records], receipt


def require_timing_quiescence(
    output: Path, expected_uuids: tuple[str, str],
) -> dict:
    """Retain and enforce a strict in-job observation just before each timing run."""
    expected = {normalize_uuid(value) for value in expected_uuids}
    samples = []
    for index in range(QUIESCENCE_SAMPLES):
        gpu_rows, gpu_receipt = _query_nvidia_smi(
            "gpu=uuid,name,utilization.gpu,memory.free,memory.total", 5)
        app_rows, app_receipt = _query_nvidia_smi("compute-apps=gpu_uuid,pid", 2)
        selected = {}
        errors = []
        if gpu_receipt["returncode"] != 0 or gpu_receipt.get("parse_error"):
            errors.append("GPU telemetry query failed or was malformed")
        if app_receipt["returncode"] != 0 or app_receipt.get("parse_error"):
            errors.append("compute-process telemetry query failed or was malformed")
        for row in gpu_rows:
            try:
                uuid = normalize_uuid(row[0])
                if uuid in expected:
                    selected[uuid] = {
                        "uuid": row[0], "name": row[1],
                        "utilization_percent": int(row[2]),
                        "memory_free_bytes": int(float(row[3]) * 1024 * 1024),
                        "memory_total_bytes": int(float(row[4]) * 1024 * 1024),
                    }
            except (IndexError, TypeError, ValueError, SystemExit) as error:
                errors.append(f"invalid GPU telemetry row: {error}")
        processes = []
        for row in app_rows:
            try:
                uuid = normalize_uuid(row[0])
                if uuid in expected:
                    processes.append({"gpu_uuid": row[0], "pid": row[1]})
            except (IndexError, TypeError, ValueError, SystemExit) as error:
                errors.append(f"invalid process telemetry row: {error}")
        if set(selected) != expected:
            errors.append("the expected GPU UUID pair was not observed exactly")
        for uuid in expected:
            item = selected.get(uuid)
            if item is None:
                continue
            if item["utilization_percent"] > QUIESCENCE_MAX_UTILIZATION_PERCENT:
                errors.append(f"GPU {item['uuid']} utilization was nonzero")
            if item["memory_free_bytes"] < QUIESCENCE_MINIMUM_FREE_BYTES:
                errors.append(f"GPU {item['uuid']} free memory was below the required threshold")
        if processes:
            errors.append("one or more compute processes were visible on the selected GPUs")
        samples.append({
            "sample": index,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "passed": not errors,
            "gpus_by_uuid": selected,
            "compute_processes": processes,
            "errors": errors,
            "gpu_query": gpu_receipt,
            "compute_process_query": app_receipt,
        })
        if index + 1 != QUIESCENCE_SAMPLES:
            time.sleep(QUIESCENCE_INTERVAL_SECONDS)
    report = {
        "schema": "cellerator.native_data.timing_quiescence.v1",
        "policy": {
            "samples": QUIESCENCE_SAMPLES,
            "interval_seconds": QUIESCENCE_INTERVAL_SECONDS,
            "max_utilization_percent": QUIESCENCE_MAX_UTILIZATION_PERCENT,
            "minimum_free_bytes_per_gpu": QUIESCENCE_MINIMUM_FREE_BYTES,
            "expected_gpu_uuids": list(expected_uuids),
            "required_compute_process_count": 0,
        },
        "samples": samples,
        "passed": len(samples) == QUIESCENCE_SAMPLES and all(item["passed"] for item in samples),
    }
    write_json(output / "timing-quiescence.json", report)
    if not report["passed"]:
        raise SystemExit(
            f"strict timing quiescence guard failed; observations retained at "
            f"{output / 'timing-quiescence.json'}")
    return report


def run_backend(
    *, source: Path, build: Path | None, binary: Path, artifact_manifest: Path | None,
    output: Path, case: str, mode: str, device_uuids: tuple[str, str],
    device_indices: tuple[int, int],
    warmup: int, samples: int, learning_rate: float,
) -> Path:
    binary = binary.resolve(strict=True)
    if artifact_manifest is None:
        if build is None:
            raise SystemExit("a build root or external artifact manifest is required")
        build = build.resolve(strict=True)
        if build not in binary.parents:
            raise SystemExit(f"benchmark binary must be inside the Autobench build artifact: {binary}")
    else:
        artifact_manifest = artifact_manifest.resolve(strict=True)
        if source not in binary.parents or source not in artifact_manifest.parents:
            raise SystemExit("external binaries and manifests must be included in the immutable packet snapshot")
    matching_backend = next((name for name, (_, executable) in BACKENDS.items()
                             if binary.name == executable), None)
    if matching_backend is None:
        raise SystemExit(f"unrecognized native-data benchmark binary: {binary.name}")
    backend = BACKENDS[matching_backend][0]
    if mode == "correctness":
        warmup, samples = 0, 1
    output.mkdir(parents=True, exist_ok=True)
    if mode == "timing":
        require_timing_quiescence(output, device_uuids)
    result_path = output / f"{case}.{mode}.json"
    wrapper = source / "bench/native_data/run_native_data_bench.py"
    command = [
        sys.executable, str(wrapper), "--binary", str(binary),
        "--source-root", str(source), "--case", case, "--mode", mode,
        "--output", str(result_path), "--device0", str(device_indices[0]),
        "--device1", str(device_indices[1]),
        "--warmup", str(warmup), "--samples", str(samples),
        "--learning-rate", repr(learning_rate),
    ]
    if artifact_manifest is not None:
        command.extend(("--artifact-manifest", str(artifact_manifest)))
    else:
        command.extend(("--build-root", str(build)))
    completed = subprocess.run(command, cwd=source, text=True, capture_output=True,
                               env=os.environ.copy(), check=False)
    (output / f"{case}.{mode}.runner.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / f"{case}.{mode}.runner.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise SystemExit(f"{binary.name} {case} {mode} failed: {completed.stdout} {completed.stderr}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result.get("backend") != backend or
            result.get("case") != CASE_RESULT_NAMES.get(case) or
            result.get("mode") != mode):
        raise SystemExit(f"unexpected backend/workload identity in {result_path}")
    devices = result.get("devices", [])
    observed_uuids = [normalize_uuid(device.get("uuid")) for device in devices]
    expected_uuids = [normalize_uuid(value) for value in device_uuids]
    observed_slots = [device.get("visible_index") for device in devices]
    if observed_uuids != expected_uuids or observed_slots != list(device_indices):
        raise SystemExit(
            f"Scheduler device/slot mismatch: expected logical UUIDs {device_uuids} "
            f"at visible indices {device_indices}, observed {observed_uuids} at {observed_slots}"
        )
    if result.get("placement", {}).get("owner_devices") != {
        "expression": device_indices[0], "regulators": device_indices[1],
        "theta_phi_tau": device_indices[0], "cotangent": device_indices[0]
    }:
        raise SystemExit("backend result does not honor the approved source-owner placement")
    if result.get("placement", {}).get("canonical_devices") != {
        "output": device_indices[0], "dx": device_indices[0],
        "dz": device_indices[1], "dtheta_dphi_dtau": device_indices[0]
    }:
        raise SystemExit("backend result does not honor the approved canonical-VJP placement")
    return result_path


def correctness_pair(paths: dict[str, Path], output: Path) -> dict:
    from compare_backends import compare

    report = compare(paths["aten"], paths["specialist"], ATOL, RTOL)
    report["baseline_result_sha256"] = sha256_file(paths["aten"])
    report["candidate_result_sha256"] = sha256_file(paths["specialist"])
    report["passed"] = report["passed"] is True
    write_json(output, report)
    if not report["passed"]:
        raise SystemExit(f"{report['case']} candidate failed the approved ATen comparison tolerance")
    return report


def preparation_ms(result: dict) -> float:
    timing = result.get("preparation", {})
    return sum(float(timing.get(key, 0.0)) for key in ("host_pack_ms", "upload_ms", "readiness_ms"))


def timing_summary(result_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    preflight = json.loads((result_path.parent / "timing-quiescence.json").read_text(encoding="utf-8"))
    if preflight.get("passed") is not True or len(preflight.get("samples", [])) != QUIESCENCE_SAMPLES:
        raise SystemExit("timing result lacks a passing retained three-sample quiescence guard")
    samples = result.get("summary", {}).get("samples", [])
    full = [float(sample["full_cycle_ms"]) for sample in samples]
    if not full or any(not math.isfinite(value) or value <= 0.0 for value in full):
        raise SystemExit(f"no finite positive full-cycle timing samples in {result_path}")
    if not all(sample.get("update_includes_replica_refresh") is True for sample in samples):
        raise SystemExit("update timing must include completed replica refresh on every backend")
    cold = result["summary"].get("cold_start")
    if not isinstance(cold, dict) or not math.isfinite(float(cold.get("full_cycle_ms", math.nan))):
        raise SystemExit("timing result is missing its synchronized cold-start cycle")
    prep = result.get("preparation", {})
    return {
        "binary_sha256": result.get("artifact_identity", {}).get("binary_sha256"),
        "source_snapshot_sha256": result.get("artifact_identity", {}).get("snapshot_source_sha256"),
        "workload_identity": result.get("workload_identity"),
        "devices": result.get("devices"),
        "placement": result.get("placement"),
        "timing_quiescence": preflight,
        "preparation_ms": preparation_ms(result),
        "preparation_host_pack_ms": prep.get("host_pack_ms"),
        "preparation_upload_ms": prep.get("upload_ms"),
        "preparation_readiness_ms": prep.get("readiness_ms"),
        "prepared_host_payload_bytes": prep.get("prepared_host_payload_bytes"),
        "prepared_device_bytes": prep.get("prepared_device_bytes"),
        "prepared_staging_bytes": prep.get("prepared_staging_bytes"),
        "cold_start": cold,
        "warmup": result["summary"].get("warmup"),
        "recurrent_samples_ms": full,
        "recurrent_mean_ms": statistics.fmean(full),
        "recurrent_population_stddev_ms": statistics.pstdev(full),
        "recurrent_median_ms": statistics.median(full),
        "recurrent_min_ms": min(full),
        "recurrent_max_ms": max(full),
        "first_path_ms": preparation_ms(result) + float(cold["full_cycle_ms"]),
        "median_transfer_bytes": statistics.median(
            float(sample.get("transfer_bytes", 0)) for sample in samples),
        "median_participant_transfer_bytes": statistics.median(
            float(sample.get("participant_transfer_bytes", 0)) for sample in samples),
        "median_condition_transfer_bytes": statistics.median(
            float(sample.get("condition_transfer_bytes", 0)) for sample in samples),
        "median_backward_transfer_bytes": statistics.median(
            float(sample.get("backward_transfer_bytes", 0)) for sample in samples),
        "median_replica_refresh_bytes": statistics.median(
            float(sample.get("replica_refresh_bytes", 0)) for sample in samples),
        "stage_medians_ms": {
            name: statistics.median(float(sample[name]) for sample in samples)
            for name in ("forward_ms", "backward_ms", "reduction_ms", "update_ms", "full_cycle_ms")
        },
        "update_includes_replica_refresh": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("scale-correctness", "scale-timing"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--aten-binary", type=Path, required=True)
    parser.add_argument("--aten-artifact-manifest", type=Path)
    parser.add_argument("--specialist-binary", type=Path, required=True)
    parser.add_argument("--specialist-artifact-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-uuid0", default="GPU-cf22c41f-5b58-77b1-3535-8fadd1ca6505")
    parser.add_argument("--expected-uuid1", default="GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a")
    parser.add_argument("--device0-index", type=int, default=1)
    parser.add_argument("--device1-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_root.resolve(strict=True)
    build = args.build_root.resolve(strict=True) if args.build_root is not None else None
    output = args.output_dir.resolve()
    if not any(
        tuple(source.parts[index:index + 3]) == (".autobench", "state", "snapshots")
        for index in range(len(source.parts) - 2)
    ):
        raise SystemExit("native-data campaigns require an immutable Autobench snapshot")
    device_uuids = (args.expected_uuid0, args.expected_uuid1)
    if device_uuids[0] == device_uuids[1]:
        raise SystemExit("approved GPU UUIDs must be distinct")
    device_indices = (args.device0_index, args.device1_index)
    if device_indices[0] == device_indices[1] or sorted(device_indices) != [0, 1]:
        raise SystemExit("logical device slots must map bijectively to visible indices 0 and 1")
    if (args.aten_artifact_manifest is None) != (args.specialist_artifact_manifest is None):
        raise SystemExit("baseline and specialist must both use immutable external manifests")
    if args.aten_artifact_manifest is None and build is None:
        raise SystemExit("provide --build-root or both external artifact manifests")
    if args.aten_artifact_manifest is not None and build is not None:
        raise SystemExit("choose Autobench build outputs or external snapshot artifacts, not both")
    output.mkdir(parents=True, exist_ok=True)
    correctness_paths: dict[str, dict[str, Path]] = {}
    correctness_reports = {}
    for case in CASES:
        paths = {}
        for backend, binary in (("aten", args.aten_binary), ("specialist", args.specialist_binary)):
            paths[backend] = run_backend(
                source=source, build=build, binary=binary,
                artifact_manifest=(args.aten_artifact_manifest if backend == "aten"
                                   else args.specialist_artifact_manifest),
                output=output / "correctness" / case / backend,
                case=case, mode="correctness", device_uuids=device_uuids,
                device_indices=device_indices,
                warmup=0, samples=1, learning_rate=args.learning_rate,
            )
        correctness_paths[case] = paths
        correctness_reports[case] = correctness_pair(
            paths, output / "correctness" / case / "comparison.json")

    if args.phase == "scale-correctness":
        summary = {
            "schema": "cellerator.native_data.scale_correctness_campaign.v1",
            "phase": args.phase,
            "source_snapshot_sha256": source.name,
            "device_uuids_by_logical_slot": device_uuids,
            "visible_indices_by_logical_slot": device_indices,
            "scheduler_resource_order_uuids": [device_uuids[1], device_uuids[0]],
            "atol": ATOL,
            "rtol": RTOL,
            "passed": all(report["passed"] for report in correctness_reports.values()),
            "cases": correctness_reports,
        }
        write_json(output / "campaign.json", summary)
        print(json.dumps({"phase": args.phase, "passed": summary["passed"],
                          "cases": sorted(correctness_reports)}, sort_keys=True))
        return 0 if summary["passed"] else 1

    timing_reports = {}
    for case in CASES:
        case_paths = {}
        for backend, binary in (("aten", args.aten_binary), ("specialist", args.specialist_binary)):
            case_paths[backend] = run_backend(
                source=source, build=build, binary=binary,
                artifact_manifest=(args.aten_artifact_manifest if backend == "aten"
                                   else args.specialist_artifact_manifest),
                output=output / "timing" / case / backend,
                case=case, mode="timing", device_uuids=device_uuids,
                device_indices=device_indices,
                warmup=args.warmup, samples=args.samples,
                learning_rate=args.learning_rate,
            )
        baseline = timing_summary(case_paths["aten"])
        candidate = timing_summary(case_paths["specialist"])
        if baseline["workload_identity"] != candidate["workload_identity"]:
            raise SystemExit(f"{case} timing binaries did not consume identical workload contents")
        if baseline["devices"] != candidate["devices"]:
            raise SystemExit(f"{case} timing binaries did not use the same physical device pair")
        timing_reports[case] = {
            "correctness_gate": correctness_reports[case],
            "aten": baseline,
            "specialist": candidate,
            "speedup_recurrent_median": baseline["recurrent_median_ms"] / candidate["recurrent_median_ms"],
            "speedup_first_path": baseline["first_path_ms"] / candidate["first_path_ms"],
        }

    summary = {
        "schema": "cellerator.native_data.scale_timing_campaign.v1",
        "phase": args.phase,
        "source_snapshot_sha256": source.name,
        "device_uuids_by_logical_slot": device_uuids,
        "visible_indices_by_logical_slot": device_indices,
        "scheduler_resource_order_uuids": [device_uuids[1], device_uuids[0]],
        "atol": ATOL,
        "rtol": RTOL,
        "correctness_gate_passed": all(report["passed"] for report in correctness_reports.values()),
        "cases": timing_reports,
    }
    if not summary["correctness_gate_passed"]:
        raise SystemExit("timing was not started because a comparison gate failed")
    write_json(output / "campaign.json", summary)
    metrics = {
        "schema": "cellerator.native_data.timing_metrics.v1",
        "cases": {
            case: {
                "correctness_passed": report["correctness_gate"]["passed"],
                "aten_recurrent_median_ms": report["aten"]["recurrent_median_ms"],
                "specialist_recurrent_median_ms": report["specialist"]["recurrent_median_ms"],
                "speedup_recurrent_median": report["speedup_recurrent_median"],
                "speedup_first_path": report["speedup_first_path"],
            }
            for case, report in timing_reports.items()
        },
    }
    write_json(output / "metrics.json", metrics)
    print(json.dumps({"phase": args.phase, "correctness_gate_passed": True,
                      "cases": {case: report["speedup_recurrent_median"]
                                for case, report in timing_reports.items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
