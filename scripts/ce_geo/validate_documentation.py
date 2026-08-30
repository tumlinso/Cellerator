#!/usr/bin/env python3
"""Validate the CE-GEO documentation spine against accepted evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DOCUMENT_REQUIREMENTS = {
    "README.md": ("CE-GEO Volta implementation status", "22/22"),
    "scope.md": ("CE-GEO boundary after Volta integration", "planner"),
    "docs/architecture.qmd": ("Integrated CE-GEO realization", "CSG1"),
    "docs/current_implementation.qmd": ("CE-GEO integrated implementation", "N=64"),
    "docs/biological_execution_model.qmd": ("Relation-algebra realization", "transpose"),
    "docs/cellpack_cp_bp.qmd": ("CE-GEO geometry outcome", "exact logical covers"),
    "docs/core_execution_cp_math.qmd": ("CE-GEO execution outcome", "planner"),
    "docs/baseplane_integration.qmd": ("CE-GEO boundary confirmation", "narrow seam"),
    "docs/storage_distribution_and_interop.qmd": (
        "CE-GEO persistence and interop outcome",
        "pointer-free",
    ),
    "docs/performance_validation.qmd": ("CE-GEO Volta evidence", "not promoted"),
    "docs/migration_roadmap.qmd": ("CE-GEO Volta convergence", "not_granted"),
    "docs/CE_GEO_PROGRAM.md": ("Implemented Volta record", "22/22"),
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_documents(root: Path) -> None:
    combined = []
    for relative, tokens in DOCUMENT_REQUIREMENTS.items():
        path = root / relative
        require(path.is_file(), f"missing authoritative document: {relative}")
        text = path.read_text(encoding="utf-8")
        for token in tokens:
            require(token in text, f"{relative}: missing required text {token!r}")
        if path.suffix == ".qmd":
            require(text.startswith("---\n"), f"{relative}: frontmatter is not first")
        combined.append(text)

    joined = "\n".join(combined)
    require("CE-AMP" in joined and "not_granted" in joined, "CE-AMP interlock is undocumented")
    require("regime-specific" in joined, "evidence-scoped layout limit is undocumented")


def validate_evidence(evidence: Path) -> None:
    forward = load_json(evidence / "sm70_forward_disposition.json")
    require(forward.get("disposition") == "implemented", "sm70 forward is not implemented")
    require(forward.get("promotion", {}).get("accepted") is True, "sm70 promotion is not accepted")
    measured = forward.get("measured_regime", {})
    require(measured.get("dense_width") == 64, "promoted width is not N=64")
    require(measured.get("logical_edges") == 2176, "unexpected logical edge count")
    require(measured.get("mma_edges") == 2048, "unexpected MMA edge count")
    require(measured.get("residual_edges") == 128, "unexpected residual edge count")
    require(forward.get("evidence", {}).get("max_absolute_error") == 0.0, "forward error is nonzero")

    fusion = load_json(evidence / "fusion_evaluation.json")
    require(fusion.get("disposition") == "evaluated_not_promoted", "fusion disposition changed")
    require(fusion.get("accepted_for_promotion") is False, "fusion was unexpectedly promoted")

    preprint = load_json(evidence / "preprint" / "summary.json")
    require(preprint.get("evidence_valid") == 1, "preprint evidence is invalid")
    require(preprint.get("ablation_family_count") == 10, "unexpected ablation family count")
    require(preprint.get("measurement_count") == 20, "unexpected ablation measurement count")
    require(preprint.get("accepted_for_promotion") is False, "ablations were unexpectedly promoted")
    negative = [
        result
        for decision in preprint.get("decisions", [])
        for result in decision.get("negative_results", [])
        if result.get("correctness_passed") is False
    ]
    require(len(negative) == 1, "expected exactly one retained correctness-negative ablation")

    sanitizer = load_json(evidence / "sanitizer" / "campaign.json")
    sanitizer_summary = sanitizer.get("summary", {})
    require(sanitizer.get("validated") is True, "sanitizer campaign is not validated")
    require(sanitizer_summary.get("sanitizer_run_count") == 22, "unexpected sanitizer run count")
    require(sanitizer_summary.get("passed_run_count") == 22, "sanitizer campaign has failures")

    acceptance = load_json(evidence / "full_volta_validation.json")
    acceptance_summary = acceptance.get("summary", {})
    require(acceptance.get("validated") is True, "full Volta acceptance is not validated")
    require(acceptance_summary.get("validation_command_count") == 22, "unexpected acceptance count")
    require(acceptance_summary.get("passed_validation_command_count") == 22, "acceptance has failures")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    validate_documents(root)
    validate_evidence((root / args.evidence).resolve())
    print("CE-GEO documentation validation passed: 12 documents, 22/22 acceptance, 22/22 sanitizer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
