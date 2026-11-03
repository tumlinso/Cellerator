#!/usr/bin/env python3
"""Validate an ATen CUDA run against the independent Fraction oracle."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import runpy
import struct
from pathlib import Path


def exact_expectations(case: str, oracle: Path) -> dict[str, list[float]]:
    with contextlib.redirect_stdout(io.StringIO()):
        values = runpy.run_path(str(oracle))
    if case == "reference_active":
        active = values["active"]
        return {
            "output": [float(value) for value in values["outputs"]],
            "dx": [float(value) for value in active["x"]],
            "dz": [float(active["z0"]), float(active["z1"])],
            "dtheta": [float(active["theta"])],
            "dphi": [float(active["phi"])],
            "dtau": [float(active["tau"])],
            "theta_after_update": [float(values["theta_next"])],
            "post_update_output": [float(value) for value in values["next_outputs"]],
        }
    if case == "reference_inactive":
        baseline = values["inactive_y"]
        seeds = values["loss_seed"]
        theta_f32 = struct.unpack("<f", struct.pack("<f", float(values["theta"])))[0]
        return {
            "output": [float(value) for value in baseline],
            "dx": [float(-values["beta"] * seed) for seed in seeds],
            "dz": [0.0, 0.0],
            "dtheta": [0.0],
            "dphi": [0.0],
            "dtau": [0.0],
            "theta_after_update": [theta_f32],
            "post_update_output": [float(value) for value in baseline],
        }
    raise ValueError(f"exact Fraction oracle has no case {case!r}")


def compare(result_path: Path, oracle_path: Path, atol: float, rtol: float) -> dict:
    run = json.loads(result_path.read_text(encoding="utf-8"))
    case = run.get("case")
    if run.get("mode") != "correctness" or "correctness_values" not in run:
        raise ValueError("result is not a correctness run with collected values")
    actual = run["correctness_values"]
    expected = exact_expectations(case, oracle_path)
    # The inactive contract requires exact baseline output/input-VJP values,
    # exact zero mechanism VJPs, and an unchanged FP32 theta bit pattern.
    exact_fields = (
        {"output", "dx", "dz", "dtheta", "dphi", "dtau", "theta_after_update", "post_update_output"}
        if case == "reference_inactive" else set()
    )
    fields: dict[str, dict] = {}
    passed = True
    for name, expected_values in expected.items():
        actual_values = actual.get(name)
        if not isinstance(actual_values, list) or len(actual_values) != len(expected_values):
            fields[name] = {"passed": False, "reason": "extent mismatch"}
            passed = False
            continue
        errors = []
        maxima = 0.0
        worst_relative = 0.0
        field_passed = True
        for observed, reference in zip(actual_values, expected_values):
            observed = float(observed)
            error = abs(observed - reference)
            relative = error / max(abs(reference), 1.0e-30)
            limit = atol + rtol * abs(reference)
            exact_match = observed == reference
            if (name == "theta_after_update" and case == "reference_inactive"
                    and math.isfinite(observed)):
                # JSON emits enough digits to round-trip the CUDA float32 but
                # not necessarily the exact binary32-as-binary64 decimal.
                # Compare the value's reconstructed float32 payload instead.
                exact_match = struct.pack("<f", observed) == struct.pack("<f", reference)
            if (not math.isfinite(observed) or error > limit or
                    (name in exact_fields and not exact_match)):
                field_passed = False
            maxima = max(maxima, error)
            worst_relative = max(worst_relative, relative)
            errors.append(error)
        fields[name] = {
            "passed": field_passed,
            "elements": len(expected_values),
            "max_abs_error": maxima,
            "max_relative_error": worst_relative,
        }
        passed = passed and field_passed
    return {
        "schema": "cellerator.native_data.reference_validation.v1",
        "case": case,
        "oracle": str(oracle_path),
        "oracle_kind": "independent_exact_fraction",
        "atol": atol,
        "rtol": rtol,
        "exact_fields": sorted(exact_fields),
        "passed": passed,
        "fields": fields,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    args = parser.parse_args()
    report = compare(args.result, args.oracle, args.atol, args.rtol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "fields": report["fields"]}, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
