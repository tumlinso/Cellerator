#!/usr/bin/env python3
"""Summarize observed errors from an existing precision qualification receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "tests/artifacts/precision-full-gpu-final.json"
DEFAULT_OUTPUT = ROOT / "tests/results/precision-errors.json"
DTYPES = {1: np.dtype("<f2"), 2: np.dtype("<f4"), 3: np.dtype("<f8")}
BACKEND_CODES = {"scalar": 0, "avx2": 1, "cuda": 2}
MATRIX_FIELDS = ("ms", "mu", "mss", "mus", "velocity", "second_residual")
VECTOR_FIELDS = ("gamma", "offset", "offset2", "first_scale", "second_scale",
                 "r2", "objective", "initial_gamma", "initial_offset")
CATEGORIES = {
    "moments": ("ms", "mu", "mss", "mus"),
    "parameters": ("gamma", "offset", "offset2", "initial_gamma", "initial_offset"),
    "scales": ("first_scale", "second_scale"),
    "velocity_residual": ("velocity", "second_residual"),
    "r2": ("r2",),
    "normalized_objective": ("objective",),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repo_path(path: str | Path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def read_input(path: Path):
    data = path.read_bytes()
    if data[:8] != b"VPRCIN1\0":
        raise ValueError(f"bad input magic: {path}")
    n, g = np.frombuffer(data, dtype="<u8", count=2, offset=8)
    tags = data[24:27]
    arrays = []
    off = 32
    for tag, shape in zip(tags, ((int(n), int(g)), (int(n), int(g)), (int(n), int(n)))):
        dtype = DTYPES[tag]
        count = int(np.prod(shape))
        size = count * dtype.itemsize
        arr = np.frombuffer(data, dtype=dtype, count=count, offset=off).reshape(shape)
        arrays.append(arr)
        off += size
    fit = np.frombuffer(data, dtype="u1", count=int(n), offset=off)
    genes = np.frombuffer(data, dtype="u1", count=int(g), offset=off + int(n))
    if off + int(n) + int(g) != len(data):
        raise ValueError(f"bad input length: {path}")
    return {"n": int(n), "g": int(g), "arrays": arrays, "fit": fit, "genes": genes,
            "sha256": sha256(data)}


def read_output(path: Path):
    data = path.read_bytes()
    if data[:8] != b"VPRCOT1\0":
        raise ValueError(f"bad output magic: {path}")
    tags = np.frombuffer(data, dtype="u1", count=5, offset=8)
    n, g = map(int, np.frombuffer(data, dtype="<u8", count=2, offset=13))
    dtype = np.dtype("<f4" if tags[0] == 1 else "<f8" if tags[0] == 2 else "V0")
    if dtype.itemsize == 0:
        raise ValueError(f"bad output type: {path}")
    fields = {}
    off = 29
    for name in MATRIX_FIELDS:
        count = n * g
        fields[name] = np.frombuffer(data, dtype=dtype, count=count, offset=off).reshape(n, g)
        off += count * dtype.itemsize
    for name in VECTOR_FIELDS:
        fields[name] = np.frombuffer(data, dtype=dtype, count=g, offset=off)
        off += g * dtype.itemsize
    for name in ("selected", "initial_status", "status"):
        fields[name] = np.frombuffer(data, dtype="u1", count=g, offset=off)
        off += g
    if off != len(data):
        raise ValueError(f"bad output length: {path}")
    fields.update(dtype=dtype.name, low_signal=bool(tags[2]), refit_all=bool(tags[3]),
                  backend_code=int(tags[4]), sha256=sha256(data))
    if fields["backend_code"] not in BACKEND_CODES.values():
        raise ValueError(f"invalid backend code: {path}")
    return fields


def ulp_distance(a: np.ndarray, b: np.ndarray):
    if a.dtype != b.dtype or a.dtype not in (np.dtype("float32"), np.dtype("float64")):
        return None
    bits = np.uint32 if a.dtype.itemsize == 4 else np.uint64
    sign = bits(1 << (a.dtype.itemsize * 8 - 1))
    ua, ub = a.view(bits), b.view(bits)
    ka = np.where((ua & sign) != 0, ~ua, ua | sign)
    kb = np.where((ub & sign) != 0, ~ub, ub | sign)
    dist = np.where(ka >= kb, ka - kb, kb - ka)
    return int(np.max(dist, initial=0))


def field_metrics(got: np.ndarray, ref: np.ndarray, include_ulp: bool):
    a, b = np.asarray(got, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} versus {b.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    if not bool(finite.all()):
        return {"compared_values": int(a.size), "nonfinite_values": int(a.size - finite.sum()),
                "max_absolute_error": None, "max_gene_scaled_absolute_error": None,
                "max_gene_relative_l2": None, "max_ulp": None}
    diff = np.abs(a - b)
    if a.ndim == 2:
        ref_max = np.max(np.abs(b), axis=0)
        diff_max = np.max(diff, axis=0)
        ref_l2 = np.linalg.norm(b, axis=0)
        diff_l2 = np.linalg.norm(diff, axis=0)
        flat = int(np.argmax(diff))
        row, flat_gene = np.unravel_index(flat, diff.shape)
        abs_witness = {"gene": int(flat_gene), "row": int(row)}
    else:
        ref_max, diff_max = np.abs(b), diff
        ref_l2, diff_l2 = np.abs(b), diff
        flat_gene = int(np.argmax(diff))
        abs_witness = {"gene": flat_gene}
    scale = np.maximum(1.0, ref_max)
    scaled = diff_max / scale
    scaled_gene = int(np.argmax(scaled))
    nonzero = ref_l2 > 0
    rel = diff_l2[nonzero] / ref_l2[nonzero]
    rel_genes = np.flatnonzero(nonzero)
    if rel.size:
        rel_at = int(np.argmax(rel))
        rel_value = float(rel[rel_at])
        rel_gene = int(rel_genes[rel_at])
    else:
        rel_value, rel_gene = None, None
    return {
        "compared_values": int(a.size),
        "nonfinite_values": 0,
        "zero_reference_genes": int(np.count_nonzero(~nonzero)),
        "max_absolute_error": float(diff.max(initial=0.0)),
        "max_absolute_witness": abs_witness,
        "max_gene_scaled_absolute_error": float(scaled[scaled_gene]),
        "scaled_error_witness": {"gene": scaled_gene},
        "max_gene_relative_l2": rel_value,
        "relative_l2_witness": None if rel_gene is None else {"gene": rel_gene},
        "max_ulp": ulp_distance(np.asarray(got), np.asarray(ref)) if include_ulp else None,
    }


def empty_acc():
    return {"fields": set(), "cases": set(), "compared_values": 0,
            "zero_reference_genes": 0, "nonfinite_values": 0,
            "max_absolute_error": (-1.0, None),
            "max_gene_scaled_absolute_error": (-1.0, None),
            "max_gene_relative_l2": (-1.0, None), "max_ulp": (-1, None)}


def add_field(acc, field, case_id, metrics, include_ulp):
    acc["fields"].add(field)
    acc["cases"].add(case_id)
    acc["compared_values"] += metrics["compared_values"]
    acc["zero_reference_genes"] += metrics.get("zero_reference_genes", 0)
    acc["nonfinite_values"] += metrics.get("nonfinite_values", 0)
    for key, witness_key in (("max_absolute_error", "max_absolute_witness"),
                             ("max_gene_scaled_absolute_error", "scaled_error_witness"),
                             ("max_gene_relative_l2", "relative_l2_witness")):
        value = metrics[key]
        if value is not None and value > acc[key][0]:
            acc[key] = (value, {"case_id": case_id, "field": field,
                                **(metrics[witness_key] or {})})
    if include_ulp and metrics["max_ulp"] is not None and metrics["max_ulp"] > acc["max_ulp"][0]:
        acc["max_ulp"] = (metrics["max_ulp"], {"case_id": case_id, "field": field})


def finish_acc(acc, include_ulp):
    result = {"fields": sorted(acc["fields"]), "case_count": len(acc["cases"]),
              "compared_values": acc["compared_values"],
              "zero_reference_gene_count": acc["zero_reference_genes"],
              "nonfinite_values": acc["nonfinite_values"]}
    for key in ("max_absolute_error", "max_gene_scaled_absolute_error", "max_gene_relative_l2"):
        value, witness = acc[key]
        result[key] = None if value < 0 else float(value)
        result[key + "_witness"] = witness
    if include_ulp:
        value, witness = acc["max_ulp"]
        result["max_ulp"] = None if value < 0 else int(value)
        result["max_ulp_witness"] = witness
    return result


def category_fields(model: str):
    allowed = {
        "moments": ("ms", "mu") if model == "deterministic" else CATEGORIES["moments"],
        "parameters": ("gamma", "offset") if model == "deterministic" else CATEGORIES["parameters"],
        "scales": () if model == "deterministic" else CATEGORIES["scales"],
        "velocity_residual": ("velocity",) if model == "deterministic" else CATEGORIES["velocity_residual"],
        "r2": CATEGORIES["r2"],
        "normalized_objective": () if model == "deterministic" else CATEGORIES["normalized_objective"],
    }
    return allowed


def add_numeric_group(groups, group_key, model, case_id, got, ref, include_ulp):
    group = groups.setdefault(group_key, {name: empty_acc() for name in CATEGORIES})
    for category, fields in category_fields(model).items():
        for field in fields:
            metrics = field_metrics(got[field], ref[field], include_ulp)
            add_field(group[category], field, case_id, metrics, include_ulp)


def group_decisions(acc, case_id, got, ref):
    for name in ("selected", "status", "initial_status"):
        mismatches = int(np.count_nonzero(got[name] != ref[name]))
        record = acc.setdefault(name, {"mismatches": 0, "compared_values": 0,
                                       "exact_case_count": 0, "case_count": 0})
        record["mismatches"] += mismatches
        record["compared_values"] += int(got[name].size)
        record["case_count"] += 1
        record["exact_case_count"] += int(mismatches == 0)
    for name in ("low_signal", "refit_all"):
        record = acc.setdefault(name, {"mismatches": 0, "case_count": 0})
        record["mismatches"] += int(got[name] != ref[name])
        record["case_count"] += 1


def source_hash_report(receipt, receipt_path, raw):
    provenance = receipt["provenance"]
    items = {}
    for case in receipt["cases"]:
        items[case["input_path"]] = case["input_sha256"]
        for item in case["backend_artifacts"].values():
            items[item["path"]] = item["sha256"]
    verified = []
    for path, expected in sorted(items.items()):
        p = Path(path)
        actual = sha256(p.read_bytes())
        if actual != expected:
            raise ValueError(f"raw hash mismatch: {p}")
        verified.append((repo_path(p), actual))
    manifest_bytes = json.dumps(verified, separators=(",", ":")).encode()
    native_matches = {}
    for path, expected in provenance["native_source_sha256"].items():
        native_matches[path] = sha256((ROOT / path).read_bytes()) == expected
    driver_matches = {}
    for name, entry in provenance["drivers"].items():
        driver_matches[name] = sha256(Path(entry["path"]).read_bytes()) == entry["sha256"]
    script_matches = {
        "qualification_driver": sha256((ROOT / "tests/precision_interop_driver.cpp").read_bytes())
        == provenance["qualification_driver_sha256"],
        "qualification_script": sha256((ROOT / "tools/qualify_precision.py").read_bytes())
        == provenance["qualification_script_sha256"],
    }
    report_bytes = receipt_path.read_bytes()
    return {
        "qualification_receipt": repo_path(receipt_path),
        "qualification_receipt_sha256": sha256(report_bytes),
        "run_id": receipt["run_id"], "created_utc": receipt["created_utc"],
        "source_commit": provenance["source_commit"],
        "qualification_driver_sha256": provenance["qualification_driver_sha256"],
        "qualification_script_sha256": provenance["qualification_script_sha256"],
        "native_source_hashes_current_tree_match": {
            "matched": sum(native_matches.values()), "total": len(native_matches),
            "all_match": all(native_matches.values())},
        "driver_hashes_current_tree_match": {
            "matched": sum(driver_matches.values()), "total": len(driver_matches),
            "all_match": all(driver_matches.values())},
        "qualification_source_hashes_current_tree_match": script_matches,
        "raw_artifacts": {"unique_files": len(verified),
                           "sha256_verified": True,
                           "manifest_sha256": sha256(manifest_bytes),
                           "manifest_entries_in_receipt": "tests/artifacts/precision-full-gpu-final.json"},
        "acceptance_receipt_flags": {
            key: receipt[key] for key in ("status", "driver_runs", "backend_complete",
                "pass_all_backend_gates", "pass_all_oracle_gates", "pass_all_precision_gates",
                "pass_all_branch_contracts", "pass_all")},
    }


def build_report(receipt_path: Path):
    receipt_path = receipt_path.resolve()
    receipt = json.loads(receipt_path.read_text())
    cache = {}

    def output(path):
        if path not in cache:
            cache[path] = read_output(Path(path))
        return cache[path]

    # Reproduce each qualification max-absolute field metric from raw outputs.
    reproduced, masked_not_reproduced, discrepancies = 0, 0, []
    backend_groups, backend_decisions = {}, {}
    for case in receipt["cases"]:
        scalar = output(case["backend_artifacts"]["scalar"]["path"])
        for backend in ("avx2", "cuda"):
            got = output(case["backend_artifacts"][backend]["path"])
            if got["backend_code"] != BACKEND_CODES[backend]:
                raise ValueError(f"backend tag mismatch in {case['case_id']}/{backend}")
            key = (case["stored_profile"], tuple(case["stored_dtypes"]),
                   case["compute_precision"], backend)
            add_numeric_group(backend_groups, key, case["model"], case["case_id"], got, scalar, True)
            dec = backend_decisions.setdefault(key, {})
            group_decisions(dec, case["case_id"], got, scalar)
            recorded = case["comparison"]["pairwise"][backend]["checks"]
            for field, record in recorded.items():
                if record["compared"] != got[field].size:
                    masked_not_reproduced += 1
                    continue
                raw = field_metrics(got[field], scalar[field], True)
                reproduced += 1
                delta = abs(raw["max_absolute_error"] - record["max_absolute_error"])
                if delta > 1e-15:
                    discrepancies.append({"case_id": case["case_id"], "backend": backend,
                                          "field": field, "recomputed": raw["max_absolute_error"],
                                          "receipt": record["max_absolute_error"]})

    backend_rows = []
    for (profile, dtypes, compute, backend), categories in sorted(backend_groups.items()):
        backend_rows.append({"stored_profile": profile, "stored_dtypes": list(dtypes),
            "compute_precision": compute, "backend": backend,
            "case_count": backend_decisions[(profile, dtypes, compute, backend)]["selected"]["case_count"],
            "quantities": {name: finish_acc(acc, True) for name, acc in categories.items()
                           if acc["fields"]},
            "decisions": backend_decisions[(profile, dtypes, compute, backend)]})

    # For identical FP32-stored bytes, compare FP32 computation directly with
    # FP64 computation for each existing backend output pair.
    case_by_id = {c["case_id"]: c for c in receipt["cases"]}
    compute_groups, compute_decisions = {}, {}
    precision_check_reproduced = 0
    for item in receipt["precision_comparisons"]:
        low_id = (f"{item['fixture']}_fp32_f32_{item['fit_mask']}_{item['config']}.{item['model']}")
        high_id = (f"{item['fixture']}_fp32_f64_{item['fit_mask']}_{item['config']}.{item['model']}")
        low_case, high_case = case_by_id[low_id], case_by_id[high_id]
        if low_case["input_sha256"] != high_case["input_sha256"] or not item["input_bytes_identical"]:
            raise ValueError(f"FP32/FP64 comparison inputs differ: {low_id}")
        for backend in ("scalar", "avx2", "cuda"):
            low = output(low_case["backend_artifacts"][backend]["path"])
            high = output(high_case["backend_artifacts"][backend]["path"])
            key = (item["stored_profile"], item["compute_pair"], backend)
            add_numeric_group(compute_groups, key, item["model"], low_id, low, high, False)
            dec = compute_decisions.setdefault(key, {})
            group_decisions(dec, low_id, low, high)
            for field, record in item["backends"][backend]["checks"].items():
                if record["compared"] != low[field].size:
                    masked_not_reproduced += 1
                    continue
                raw = field_metrics(low[field], high[field], False)
                precision_check_reproduced += 1
                delta = abs(raw["max_absolute_error"] - record["max_absolute_error"])
                if delta > 1e-15:
                    discrepancies.append({"case_id": low_id, "comparison": "f32_vs_f64",
                        "backend": backend, "field": field,
                        "recomputed": raw["max_absolute_error"],
                        "receipt": record["max_absolute_error"]})
    compute_rows = []
    for (profile, pair, backend), categories in sorted(compute_groups.items()):
        compute_rows.append({"stored_profile": profile, "compute_pair": pair, "backend": backend,
            "case_count": compute_decisions[(profile, pair, backend)]["selected"]["case_count"],
            "quantities": {name: finish_acc(acc, False) for name, acc in categories.items()
                           if acc["fields"]},
            "decisions": compute_decisions[(profile, pair, backend)]})

    # Quantization is compared against canonical FP32 stored inputs and outputs,
    # independently of same-input backend mismatch measurements.
    quant_groups = {}
    input_quant_checks, output_quant_checks = 0, 0
    for item in receipt["storage_quantization"]:
        profile, fixture, model = item["quantized_profile"], item["fixture"], item["model"]
        can_id = f"{fixture}_fp32_f32_all_default.{model}"
        quant_id = f"{fixture}_{profile}_f32_all_default.{model}"
        can_case, quant_case = case_by_id[can_id], case_by_id[quant_id]
        can_in = read_input(Path(can_case["input_path"]))
        quant_in = read_input(Path(quant_case["input_path"]))
        if can_in["n"] != quant_in["n"] or can_in["g"] != quant_in["g"]:
            raise ValueError(f"storage comparison shape mismatch: {fixture}/{profile}")
        qgroup = quant_groups.setdefault(profile, {"cases": 0, "fixtures": set(),
            "output_velocity_max_abs": 0.0, "output_gamma_max_abs": 0.0,
            "selection_change_total": 0, "input": {}, "input_sumsq": {},
            "input_counts": {}, "input_max_abs": {}})
        qgroup["cases"] += 1
        qgroup["fixtures"].add(fixture)
        for i, name in enumerate(("spliced", "unspliced", "weights")):
            a, b = can_in["arrays"][i], quant_in["arrays"][i]
            diff = b.astype(np.float64) - a.astype(np.float64)
            changed = int(np.count_nonzero(diff))
            record = item["profile_input_quantization"][name]
            actual = {"changed_values": changed, "values": int(diff.size),
                      "max_absolute_error": float(np.max(np.abs(diff), initial=0.0)),
                      "rms_error": float(np.sqrt(np.mean(diff * diff)))}
            input_quant_checks += 1
            for k in ("changed_values", "values"):
                if actual[k] != record[k]:
                    discrepancies.append({"fixture": fixture, "profile": profile,
                                          "input_field": name, "metric": k,
                                          "recomputed": actual[k], "receipt": record[k]})
            for k in ("max_absolute_error", "rms_error"):
                if abs(actual[k] - record[k]) > 1e-15:
                    discrepancies.append({"fixture": fixture, "profile": profile,
                                          "input_field": name, "metric": k,
                                          "recomputed": actual[k], "receipt": record[k]})
            qgroup["input_counts"][name] = qgroup["input_counts"].get(name, 0) + changed
            qgroup["input"][name] = qgroup["input"].get(name, 0) + int(diff.size)
            qgroup["input_sumsq"][name] = qgroup["input_sumsq"].get(name, 0.0) + float(np.sum(diff * diff))
            qgroup["input_max_abs"][name] = max(qgroup["input_max_abs"].get(name, 0.0), actual["max_absolute_error"])
        can_out = output(can_case["backend_artifacts"]["scalar"]["path"])
        quant_out = output(quant_case["backend_artifacts"]["scalar"]["path"])
        dvel = float(np.max(np.abs(quant_out["velocity"].astype(np.float64) - can_out["velocity"].astype(np.float64))))
        dgamma = float(np.max(np.abs(quant_out["gamma"].astype(np.float64) - can_out["gamma"].astype(np.float64))))
        changes = int(np.count_nonzero(quant_out["selected"] != can_out["selected"]))
        output_quant_checks += 1
        for name, actual, expected in (("velocity_max_abs_delta", dvel, item["velocity_max_abs_delta"]),
                                       ("gamma_max_abs_delta", dgamma, item["gamma_max_abs_delta"])):
            if abs(actual - expected) > 1e-15:
                discrepancies.append({"fixture": fixture, "profile": profile,
                    "model": model, "metric": name, "recomputed": actual, "receipt": expected})
        if changes != item["changed_selected_genes"]:
            discrepancies.append({"fixture": fixture, "profile": profile, "model": model,
                "metric": "changed_selected_genes", "recomputed": changes,
                "receipt": item["changed_selected_genes"]})
        qgroup["output_velocity_max_abs"] = max(qgroup["output_velocity_max_abs"], dvel)
        qgroup["output_gamma_max_abs"] = max(qgroup["output_gamma_max_abs"], dgamma)
        qgroup["selection_change_total"] += changes

    quant_rows = []
    for profile, q in sorted(quant_groups.items()):
        input_summary = {}
        for name in ("spliced", "unspliced", "weights"):
            count = q["input"][name]
            input_summary[name] = {"changed_values": q["input_counts"][name],
                "values": count,
                "max_absolute_error": q["input_max_abs"][name],
                "rms_error": float(np.sqrt(q["input_sumsq"][name] / count))}
        quant_rows.append({"quantized_profile": profile,
            "canonical_profile": "fp32", "compute_precision": "f32",
            "interpretation": "storage quantization effect; not backend mismatch",
            "case_count": q["cases"], "fixture_count": len(q["fixtures"]),
            "input_quantization": input_summary,
            "output_max_absolute_delta": {"velocity": q["output_velocity_max_abs"],
                                           "gamma": q["output_gamma_max_abs"]},
            "selected_gene_changes_total": q["selection_change_total"]})

    return {
        "schema": "velocity.precision-errors.v1",
        "method": {
            "description": "Read-only postprocessing of existing qualification input/output bytes; no driver, kernel, or qualification run was executed.",
            "same_input_backend_reference": "scalar",
            "gene_scale": "max(1, max(abs(reference)) over rows); vectors use max(1, abs(reference gene))",
            "relative_l2": "maximum per-gene ||candidate-reference||2 / ||reference||2; zero-norm reference genes excluded and counted",
            "ulp": "maximum IEEE-754 ordered-bit distance for same-output-type backend comparisons; omitted for FP32-vs-FP64",
            "grouping": "maximum observed metric per stored profile, compute type, backend, quantity category across all applicable cases",
            "gate_masks": "All-gene summaries are unmasked. Original stochastic parameter/scale/objective gates apply valid status and condition masks; initial coefficients apply initialization masks. The original condition mask is not reconstructed here.",
        },
        "provenance": source_hash_report(receipt, receipt_path, cache),
        "reproduction": {
            "raw_pairwise_field_maxabs_reproduced": reproduced,
            "raw_precision_pair_field_maxabs_reproduced": precision_check_reproduced,
            "receipt_fields_with_condition_or_status_mask_not_reproduced": masked_not_reproduced,
            "raw_input_quantization_fields_reproduced": input_quant_checks,
            "raw_storage_output_deltas_reproduced": output_quant_checks,
            "receipt_metric_discrepancies": discrepancies,
            "receipt_metric_reproduction_pass": not discrepancies,
        },
        "same_stored_input_backend_errors": backend_rows,
        "same_fp32_stored_input_compute_difference": compute_rows,
        "storage_quantization_vs_canonical_fp32": quant_rows,
        "limits": [
            "These maxima summarize the supplied fixtures, masks, configurations, and cases; they are not universal error bounds or a scientific-validity claim.",
            "Same-input backend deviations and storage quantization effects are separate comparisons; FP32-vs-FP64 describes compute sensitivity only for FP32-stored input.",
            "For FP32-vs-FP64, all-gene parameter/scale/objective maxima include genes excluded by the qualification gate masks; only full-coverage receipt fields are directly rechecked, and masked fields are counted separately.",
            "The qualification receipt itself has pass_all=false because the independent source-oracle gate fails; backend, compute-precision, and explicit branch gates pass. This report does not reclassify that source comparison.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": repo_path(args.output),
                      "backend_groups": len(report["same_stored_input_backend_errors"]),
                      "compute_groups": len(report["same_fp32_stored_input_compute_difference"]),
                      "quantization_groups": len(report["storage_quantization_vs_canonical_fp32"]),
                      "receipt_metric_reproduction_pass": report["reproduction"]["receipt_metric_reproduction_pass"]}, indent=2))


if __name__ == "__main__":
    main()
