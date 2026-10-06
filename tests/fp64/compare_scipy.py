#!/usr/bin/env python3
"""Run the native CUDA qualification and compare emitted results with SciPy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.sparse import csr_matrix

TUPLES = (("f32", "f32", "f32"), ("f32", "f64", "f64"),
          ("f64", "f32", "f64"), ("f64", "f64", "f64"))
WIDTHS = (1, 3, 15, 16, 17, 33, 65, 257)
DIRECTIONS = ("forward", "transpose")


def compare_case(case):
    if case.get("kind") == "variance":
        matrix = csr_matrix((np.asarray(case["weights"], dtype=np.float32),
                            np.asarray(case["indices"], dtype=np.int64),
                            np.asarray(case["offsets"], dtype=np.int64)),
                           shape=(case["rows"], case["cols"]), dtype=np.float64)
        matrix.sum_duplicates()
        features = np.asarray(case["features"], dtype=np.float64).reshape(case["cols"], 1)
        expected_mean = matrix @ features
        expected_second = matrix @ (features * features)
        expected_identity = expected_second - expected_mean * expected_mean
        mean = np.asarray(case["mean"], dtype=np.float64).reshape(case["rows"], 1)
        second = np.asarray(case["second_moment"], dtype=np.float64).reshape(case["rows"], 1)
        variance = np.asarray(case["variance_identity"], dtype=np.float64).reshape(case["rows"], 1)
        abs_matrix = csr_matrix((np.abs(np.asarray(case["weights"], dtype=np.float64)),
                                 np.asarray(case["indices"], dtype=np.int64),
                                 np.asarray(case["offsets"], dtype=np.int64)),
                                shape=(case["rows"], case["cols"]), dtype=np.float64)
        mass = abs_matrix @ (np.abs(features) * np.abs(features))
        degree = max(1, int(np.max(np.diff(np.asarray(case["offsets"], dtype=np.int64)))))
        eps = np.finfo(np.float64).eps
        mean_error_bound = (4 * degree * eps) * (abs_matrix @ np.abs(features))
        second_error_bound = (4 * degree + 8) * eps * mass
        mean_bound = 3e-13 + 3e-13 * np.abs(expected_mean) + mean_error_bound
        second_bound = 3e-13 + 3e-13 * np.abs(expected_second) + second_error_bound
        # Propagate E[x^2] and E[x] reduction error through E[x^2]-E[x]^2,
        # then include rounding of the final subtract. This widens only with
        # the absolute moment scale in the cancellation case.
        variance_bound = (3e-13 + 3e-13 * np.abs(expected_identity)
                          + second_error_bound
                          + 2 * np.abs(expected_mean) * mean_error_bound
                          + mean_error_bound * mean_error_bound
                          + 4 * eps * (np.abs(expected_second)
                                       + np.abs(expected_mean * expected_mean)))
        if (np.any(np.abs(mean - expected_mean) > mean_bound)
                or np.any(np.abs(second - expected_second) > second_bound)
                or np.any(np.abs(variance - expected_identity) > variance_bound)):
            raise AssertionError("composed E[x^2] - E[x]^2 variance pipeline differs from SciPy within the scale-aware bound")
        centered = np.sum(matrix.toarray() * (features.T - expected_mean) ** 2,
                          axis=1, keepdims=True)
        if np.any(np.abs(variance - centered) > variance_bound):
            raise AssertionError("variance cancellation exceeded the scale-aware reduction bound")
        return {"kind": "variance", "max_centered_error": float(np.max(np.abs(variance - centered))),
                "max_roundoff_bound": float(np.max(variance_bound))}

    rows, cols, width = case["rows"], case["cols"], case["width"]
    offsets = np.asarray(case["offsets"], dtype=np.int64)
    indices = np.asarray(case["indices"], dtype=np.int64)
    weight_dtype = np.float32 if case["weights"] == "f32" else np.float64
    feature_dtype = np.float32 if case["features"] == "f32" else np.float64
    output_dtype = np.float32 if case["output"] == "f32" else np.float64
    weights = np.asarray(case["weight_values"], dtype=weight_dtype).astype(np.float64)
    if "value_indices" in case:
        weights = weights[np.asarray(case["value_indices"], dtype=np.int64)]
    input_extent = cols if case["direction"] == "forward" else rows
    features = np.asarray(case["feature_values"], dtype=feature_dtype)[:input_extent * width]
    features = features.astype(np.float64).reshape(input_extent, width)
    projected_transpose = case.get("kind") == "shared_csr" and case["direction"] == "transpose"
    matrix_shape = (cols, rows) if projected_transpose else (rows, cols)
    matrix = csr_matrix((weights, indices, offsets), shape=matrix_shape, dtype=np.float64)
    matrix.sum_duplicates()
    expected = (matrix @ features) if (case["direction"] == "forward" or projected_transpose) else (matrix.T @ features)
    alpha = output_dtype(case.get("input_scale", 1.0))
    beta = output_dtype(case.get("destination_scale", 0.0))
    expected = (alpha * expected + beta * output_dtype(case.get("initial_output", 0.0))).astype(output_dtype)
    actual = np.asarray(case["actual"], dtype=output_dtype).reshape(expected.shape).astype(np.float64)
    if case["output"] == "f32":
        rtol, atol = 2e-6, 2e-6
        roundoff = np.zeros_like(expected)
    else:
        rtol, atol = 3e-13, 3e-13
        abs_matrix = csr_matrix((np.abs(weights), indices, offsets), shape=matrix_shape, dtype=np.float64)
        abs_matrix.sum_duplicates()
        abs_features = np.abs(features)
        roundoff = (abs_matrix @ abs_features) if (case["direction"] == "forward" or projected_transpose) else (abs_matrix.T @ abs_features)
        # Permit summation-order variation in signed cancellation, scaled by
        # the absolute dot-product mass and the actual number of terms.
        roundoff = np.abs(alpha) * roundoff * (4 * np.finfo(np.float64).eps
                                                 * max(1, int(np.max(np.diff(offsets)))))
    bound = atol + rtol * np.abs(expected) + roundoff
    delta = np.abs(actual - expected)
    if np.any(delta > bound):
        index = np.unravel_index(np.argmax(delta), delta.shape)
        raise AssertionError(
            f"{case['weights']}x{case['features']} {case['direction']} width={width} "
            f"generation={case.get('generation', 'shared-csr')} failed at {index}: actual={actual[index]:.17g} "
            f"expected={expected[index]:.17g} abs_error={delta[index]:.3g} bound={bound[index]:.3g}"
        )


def validate_coverage(cases):
    expected_prepared = Counter(
        ("prepared", weight, feature, output, width, direction, generation)
        for weight, feature, output in TUPLES for width in WIDTHS
        for direction in DIRECTIONS for generation in (1, 2)
    )
    expected_shared = Counter(
        ("shared_csr", weight, feature, output, width, direction)
        for weight, feature, output in TUPLES for width in WIDTHS
        for direction in DIRECTIONS
    )
    expected_variance = Counter(("variance", "well_conditioned") for _ in range(1))
    expected_variance.update(("variance", "cancellation") for _ in range(1))
    actual_prepared, actual_shared, actual_variance = Counter(), Counter(), Counter()
    for case in cases:
        kind = case.get("kind", "prepared")
        if kind == "prepared":
            actual_prepared[(kind, case["weights"], case["features"], case["output"],
                             case["width"], case["direction"], case["generation"])] += 1
        elif kind == "shared_csr":
            actual_shared[(kind, case["weights"], case["features"], case["output"],
                           case["width"], case["direction"])] += 1
        elif kind == "variance":
            actual_variance[(kind, case["regime"])] += 1
        else:
            raise AssertionError(f"unexpected qualification record kind {kind!r}")
    for label, expected, actual in (("prepared", expected_prepared, actual_prepared),
                                    ("shared CSR", expected_shared, actual_shared),
                                    ("variance", expected_variance, actual_variance)):
        if actual != expected:
            missing = list((expected - actual).elements())[:5]
            extra = list((actual - expected).elements())[:5]
            raise AssertionError(f"{label} coverage mismatch; missing={missing}, extra={extra}")
    return {"prepared_cases": sum(actual_prepared.values()),
            "shared_csr_cases": sum(actual_shared.values()),
            "variance_cases": sum(actual_variance.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, help="built fp64_qualification executable")
    parser.add_argument("--receipt", help="optional JSON receipt path")
    parser.add_argument("--source-root", help="Cellerator checkout root; inferred from this script by default")
    args = parser.parse_args()
    source_root = Path(args.source_root).resolve() if args.source_root else Path(__file__).resolve().parents[2]
    binary_path = Path(args.binary).resolve()
    native_result = subprocess.run([args.binary, "--dump-jsonl"], check=False, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if native_result.stderr:
        sys.stderr.write(native_result.stderr)
    if native_result.returncode:
        raise RuntimeError(f"native CUDA qualification exited with status {native_result.returncode}")
    cases = [json.loads(line) for line in native_result.stdout.splitlines() if line.strip()]
    if not cases:
        raise RuntimeError("native qualification emitted no oracle records")
    coverage = validate_coverage(cases)
    variance = []
    relation_cases = 0
    for case in cases:
        comparison_result = compare_case(case)
        if comparison_result:
            variance.append(comparison_result)
        else:
            relation_cases += 1
    source_paths = [
        "tests/fp64/fp64_qualification.cu", "tests/fp64/compare_scipy.py",
        "include/Cellerator/compute/operation/prepared_relation.hh",
        "src/compute/operation/prepared_relation.cu",
        "include/Cellerator/compute/operation/device_elementwise.hh",
        "src/compute/operation/device_elementwise.cuh",
        "include/Cellerator/compute/candidate/sparse/project.hh",
        "src/compute/candidate/sparse/project.cu",
        "include/Cellerator/compute/operation/relation_semantics.hh",
        "src/compute/operation/relation_semantics.cc",
    ]
    source_hashes = {rel: hashlib.sha256((source_root / rel).read_bytes()).hexdigest()
                     for rel in source_paths}
    commit = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    receipt = {"status": "passed", "cases": len(cases), "coverage": coverage,
               "widths": list(WIDTHS),
               "commit": commit, "binary": str(binary_path),
               "binary_sha256": hashlib.sha256(binary_path.read_bytes()).hexdigest(),
               "source_sha256": source_hashes,
               "relation_cases": relation_cases, "variance_cancellation": variance,
               "tuples": [f"{w}x{x}->{y}" for w, x, y in TUPLES],
               "directions": list(DIRECTIONS), "generations": [1, 2],
               "oracle": "SciPy CSR with duplicate-edge coalescing; inputs rounded to stored dtype"}
    encoded = json.dumps(receipt, indent=2) + "\n"
    if args.receipt:
        with open(args.receipt, "w", encoding="utf-8") as stream:
            stream.write(encoded)
        records_path = Path(args.receipt).with_suffix(".jsonl")
        records_path.write_text(native_result.stdout, encoding="utf-8")
        receipt["records_jsonl"] = str(records_path.resolve())
        receipt["records_sha256"] = hashlib.sha256(native_result.stdout.encode()).hexdigest()
        encoded = json.dumps(receipt, indent=2) + "\n"
        Path(args.receipt).write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
