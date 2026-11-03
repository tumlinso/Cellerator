#!/usr/bin/env python3
"""Run a bounded dtype or full auxiliary qualification inside an admitted job."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True

import bounded_validation as bounded


AUXILIARY_IDS = (
    "correctness_ordered_tied_zero",
    "correctness_saturated_gates",
    "correctness_ordered_role_context_variants",
    "correctness_fma_product_residual_cancellation",
    "correctness_context_cancellation_middle",
    "correctness_context_cancellation_tail",
    "correctness_high_fanin_100k",
    "correctness_packet_high_fanin_fp32_100k",
    "correctness_packet_high_fanin_1m",
    "correctness_empty_mechanisms",
    "correctness_empty_cells",
    "correctness_empty_channels",
)
DTYPES = ("fp16", "fp32")
DTYPE_WORKLOAD = "packet_a4_m32_u16"


def _plan(kind: str, role: str | None, partition: str | None = None) -> list[dict[str, str]]:
    if kind == "dtype":
        if role is not None or partition is not None:
            raise bounded.ValidationFailure("dtype qualification covers every mode and takes no role or partition")
        return [{"workload": DTYPE_WORKLOAD, "mode": mode, "activation": activation,
                 "parameter": parameter, "output": output}
                for activation, parameter, output in itertools.product(DTYPES, repeat=3)
                for mode in bounded.MODES]
    if kind != "aux" or role not in ("packet", "strongest-direct"):
        raise bounded.ValidationFailure("auxiliary qualification needs packet or strongest-direct role")
    if partition not in ("regular", "high-fanin-1m"):
        raise bounded.ValidationFailure("auxiliary qualification needs an explicit regular or high-fanin-1m partition")
    rows = []
    for workload in AUXILIARY_IDS:
        is_high = workload == "correctness_packet_high_fanin_1m"
        if is_high != (partition == "high-fanin-1m"):
            continue
        activation = "fp32" if workload == "correctness_fma_product_residual_cancellation" else "fp16"
        parameter = "fp32" if workload in {
            "correctness_fma_product_residual_cancellation",
            "correctness_packet_high_fanin_fp32_100k",
        } else "fp16"
        rows.append({"workload": workload, "mode": "packet" if role == "packet" else "derive",
                     "activation": activation, "parameter": parameter, "output": "fp16"})
    return rows


def run(*, source: Path, runner: Path, build_manifest: Path, expected_uuid: str,
        artifact_dir: Path, output: Path, kind: str, role: str | None,
        partition: str | None = None) -> dict[str, Any]:
    source = source.resolve(strict=True)
    output = output.absolute()
    packet_root = output.parent
    artifact_dir = artifact_dir.absolute()
    for path in (packet_root, artifact_dir, output):
        if any(component.is_symlink() for component in (path, *path.parents)):
            raise bounded.ValidationFailure("qualification paths cannot contain symlinks")
    if not artifact_dir.resolve().is_relative_to(packet_root.resolve()) or \
            packet_root.resolve().is_relative_to(source.parent):
        raise bounded.ValidationFailure("qualification artifacts must remain under packet output, outside the shared build")
    if output.exists():
        raise bounded.ValidationFailure("qualification report already exists")
    plan = _plan(kind, role, partition)
    harness = {"qualification_sha256": bounded.sha256_file(Path(__file__).resolve()),
               "bounded_validation_sha256": bounded.sha256_file(Path(bounded.__file__).resolve())}
    # Check the immutable closure and admitted context before importing its
    # workload code or creating any packet-owned files.
    build = bounded.verify_build(source, runner, build_manifest)
    context = bounded.read_context(source, expected_uuid)
    module = bounded._source_module(source, "tests/factorized_interactions", "workloads")
    if tuple(module.correctness_workload_ids()) != AUXILIARY_IDS:
        raise bounded.ValidationFailure("fixed auxiliary workload corpus has changed")
    packet_root.mkdir(parents=True, exist_ok=True)
    # A fresh caller-owned artifact tree guarantees the cache cannot predate
    # this packet. The validator may invalidate that dedicated cache on error.
    artifact_dir.mkdir(parents=True, exist_ok=False)
    cache = artifact_dir / "reference-cache"
    records = []
    failures = []
    facts_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    peak = bounded._tree_bytes(packet_root)
    if peak >= bounded.LIMIT_BYTES:
        raise bounded.ValidationFailure("packet output is already at the scratch limit")
    for index, row in enumerate(plan):
        label = f"case-{index:03d}"
        try:
            key = row["workload"], row["activation"], row["parameter"]
            if key not in facts_cache:
                facts_cache[key] = bounded.workload_facts(source, *key)
            facts = facts_cache[key]
            mode = facts["strongest_direct_mode"] if row["mode"] == "derive" else row["mode"]
            evidence = bounded.run_validation(
                source=source, build=build, expected_uuid=expected_uuid, context=context,
                artifact_root=artifact_dir, cache_dir=cache, case_id=label,
                workload=row["workload"], mode=mode, activation=row["activation"],
                parameter=row["parameter"], output_dtype=row["output"], iterations=0,
                repeat_vjp=True, facts=facts, packet_root=packet_root,
            )
            path = Path(evidence["evidence_path"])
            records.append({"case_id": label, "workload": row["workload"], "mode": mode,
                            "activation": row["activation"], "parameter": row["parameter"],
                            "output": row["output"],
                            "selected_backend": evidence["summary"]["native_execution"]["selected_backend"],
                            "actual_implementation": evidence["summary"]["native_execution"]["actual_implementation"],
                            "structural_layout": evidence["summary"]["native_execution"]["structural_layout"],
                            "input_case_sha256": evidence["input_case_sha256"],
                            "output_result_sha256": evidence["output_result_sha256"],
                            "evidence_relative_path": evidence["evidence_relative_path"],
                            "evidence_sha256": bounded.sha256_file(path),
                            "peak_scratch_bytes": evidence["peak_scratch_bytes"], "status": "pass"})
            peak = max(peak, evidence["peak_scratch_bytes"])
            print(f"qualified {index + 1}/{len(plan)} {row['workload']} {mode}",
                  file=sys.stderr, flush=True)
        except Exception as exc:
            failures.append({"case_id": label, "workload": row["workload"],
                             "error": f"{type(exc).__name__}: {exc}"})
            break
    try:
        if not bounded._same_context(context, bounded.read_context(source, expected_uuid)):
            raise bounded.ValidationFailure("outer Scheduler context changed during qualification")
        if bounded.verify_build(source, runner, build_manifest) != build:
            raise bounded.ValidationFailure("immutable build changed during qualification")
        if harness != {"qualification_sha256": bounded.sha256_file(Path(__file__).resolve()),
                       "bounded_validation_sha256": bounded.sha256_file(Path(bounded.__file__).resolve())}:
            raise bounded.ValidationFailure("qualification harness changed during execution")
    except Exception as exc:
        failures.append({"case_id": "final_identity", "error": f"{type(exc).__name__}: {exc}"})
    report = {
        "schema": "cellerator.factorized_interactions.bounded_qualification.v1",
        "status": "pass" if not failures and len(records) == len(plan) else "fail",
        "kind": kind, "role": "dtype-all" if kind == "dtype" else role,
        "partition": partition,
        "full_auxiliary_corpus_count": len(AUXILIARY_IDS),
        "full_auxiliary_corpus_qualified_by_this_report": False,
        "partition_workloads": [item["workload"] for item in plan] if kind == "aux" else [],
        "expected_case_count": len(plan), "passed_case_count": len(records),
        "scheduler_context": context, "build": build, "records": records,
        "failures": failures, "scratch_limit_bytes": bounded.LIMIT_BYTES,
        "peak_scratch_bytes": max(peak, bounded._tree_bytes(packet_root)),
        "peak_measurement": "50ms sampled whole packet tree plus post-write evidence allowance; transient peaks may be missed",
        "successful_wire_policy": "digested and removed only after durable fixed-validator evidence",
        "harness": harness,
    }
    for _ in range(4):
        payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        predicted = max(peak, bounded._tree_bytes(packet_root) + len(payload))
        if report["peak_scratch_bytes"] == predicted:
            break
        report["peak_scratch_bytes"] = predicted
    if report["peak_scratch_bytes"] >= bounded.LIMIT_BYTES:
        report["status"] = "fail"
        report["failures"].append({"case_id": "report", "error": "packet reached the 128 MiB scratch limit"})
    bounded._durable_new(output, (json.dumps(report, sort_keys=True, indent=2,
                                             allow_nan=False) + "\n").encode())
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("dtype", "aux"), required=True)
    parser.add_argument("--role", choices=("packet", "strongest-direct"))
    parser.add_argument("--partition", choices=("regular", "high-fanin-1m"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run(source=args.source, runner=args.runner, build_manifest=args.build_manifest,
                     expected_uuid=args.expected_uuid, artifact_dir=args.artifact_dir,
                     output=args.output, kind=args.kind, role=args.role,
                     partition=args.partition)
    except Exception as exc:
        print(json.dumps({"status": "fail", "error": f"{type(exc).__name__}: {exc}"}),
              file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "kind": args.kind,
                      "passed": report["passed_case_count"],
                      "expected": report["expected_case_count"], "output": str(args.output)}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
