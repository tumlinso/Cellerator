from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from unittest.mock import patch

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
BOUNDED = ROOT / "bench" / "factorized_interactions" / "bounded"
TESTS = BOUNDED / "tests"
sys.path.insert(0, str(BOUNDED))
sys.path.insert(0, str(TESTS))

import recorded_dataset  # noqa: E402
import recorded_policy_guard as guard  # noqa: E402
import recorded_engine  # noqa: E402
from test_recorded_dataset import _valid as valid_dataset  # noqa: E402


def _dataset(*, direct_ms: float = 10.0, packet_ms: float = 8.0) -> dict:
    data = valid_dataset()
    for case in data["cases"]:
        for phase in guard.recorded_dataset.PHASES:
            for block in case["phases"][phase]:
                for arm_name, timing in ((guard.DIRECT, direct_ms),
                                         (guard.PACKET, packet_ms),
                                         ("native_direct", direct_ms * 1.1)):
                    arm = block["arms"][arm_name]
                    arm["primary_ms"] = timing
                    arm["metrics"]["phase_ms"][recorded_dataset.PRIMARY_METRIC] = timing
                    arm["metrics"]["phase_ms"]["warm_forward_vjp_batch_host_wall"] = timing * 30
                    arm["raw_metrics_sha256"] = recorded_dataset._canonical_sha(arm["metrics"])
                    arm["validated_summary"]["metrics"] = copy.deepcopy(arm["metrics"])
                    native = arm["validated_summary"].setdefault("native_execution", {})
                    native.update({
                        "candidate": arm["candidate_argument"],
                        "requested_route": arm["requested_route"],
                        "selected_backend": arm["selected_backend"],
                        "actual_implementation": arm["actual_implementation"],
                        "structural_layout": arm["structural_layout"],
                    })
                    arm["validation_summary_sha256"] = recorded_dataset._canonical_sha(
                        arm["validated_summary"])
    recorded_dataset.validate_dataset(data)
    return data


def _task(data: dict, dataset_sha: str = "a" * 64) -> dict:
    return {
        "recorded_analysis_schema": guard.TASK_SCHEMA,
        "task_id": "recorded-test-task",
        "target": {"kind": "cpu"},
        "execution": {"mode": "direct"},
        "promotion": dict(guard.PROMOTION),
        "timing": {"metric": "recorded_primary_caller_wall_time",
                   "unit": "ms_per_fwd_vjp_step", "direction": "minimize",
                   "warmup": 0, "repetitions": 1, "rounds": 7,
                   "confirmation_rounds": 7, "seed": 20261102},
        "cases": [recorded_engine._task_case(row["id"], row["workload_sha256"], dataset_sha)
                  for row in data["cases"]],
    }


def _summary(data: dict, *, ci_low: float = 1.01,
             confirmation: bool = True) -> dict:
    ids = [row["id"] for row in data["cases"]]
    validation = {candidate: {"status": "pass", "case_status": {key: "pass" for key in ids}}
                  for candidate in guard.CANDIDATES}
    result = {
        "status": "complete", "trusted_inputs_unchanged": True,
        "tools_unchanged": True,
        "candidate_ids": list(guard.CANDIDATES),
        "requested_candidate_ids": list(guard.CANDIDATES),
        "incumbent_id": "strongest-direct", "benchmark_failures": {},
        "validation": validation,
        "rankings": [{"candidate_id": "packet-candidate", "score": 1.25}],
        "confirmation": None,
    }
    if confirmation:
        result["confirmation"] = {
            "eligible": False,
            "per_case": {
                row["id"]: {"score": 10.0 / 8.0, "ci_low": ci_low,
                            "ci_high": 1.30,
                            "eligible": ci_low >= (1.0 - guard.PROMOTION["max_regression"])}
                for row in data["cases"]
            },
        }
    return result


def test_exact_case_map_requires_point_gain_and_strict_ci() -> None:
    data = _dataset()
    task = _task(data)
    summary = _summary(data)
    output = guard.select_exact_cases(data, task, summary)
    assert output["status"].startswith("conditional_pending")
    assert output["packet_regimes"] == [case["id"] for case in data["cases"]]
    assert all(case["packet_speedup"] == pytest.approx(1.25) for case in output["cases"])
    assert all(case["packet_time_ratio"] == pytest.approx(0.8) for case in output["cases"])

    summary["confirmation"]["per_case"][data["cases"][0]["id"]]["ci_low"] = 1.0
    summary["confirmation"]["per_case"][data["cases"][0]["id"]]["eligible"] = True
    output = guard.select_exact_cases(data, task, summary)
    assert output["cases"][0]["selected_route"] == guard.DIRECT
    assert len(output["packet_regimes"]) == 12


def _set_route(arm: dict, candidate: str, route: str, backend: str,
               implementation: str, layout: str) -> None:
    arm.update({"candidate_argument": candidate, "requested_route": route,
                "selected_backend": backend, "actual_implementation": implementation,
                "structural_layout": layout})
    arm["metrics"]["selected_backend"] = backend
    arm["raw_metrics_sha256"] = recorded_dataset._canonical_sha(arm["metrics"])
    arm["validated_summary"]["metrics"] = copy.deepcopy(arm["metrics"])
    arm["validated_summary"]["native_execution"].update({
        "candidate": candidate, "requested_route": route,
        "selected_backend": backend, "actual_implementation": implementation,
        "structural_layout": layout,
    })
    arm["validation_summary_sha256"] = recorded_dataset._canonical_sha(arm["validated_summary"])


def test_stable_stable_execution_cannot_promote_packet_despite_gain_and_ci() -> None:
    data = _dataset()
    case = data["cases"][0]
    for phase in recorded_dataset.PHASES:
        for block in case["phases"][phase]:
            _set_route(block["arms"][guard.DIRECT], "minimal_direct", "direct",
                       "stable", "stable", "full_packet")
            _set_route(block["arms"][guard.PACKET], "packet", "packet",
                       "stable", "stable", "full_packet")
    output = guard.select_exact_cases(data, _task(data), _summary(data, ci_low=1.20))
    selected = output["cases"][0]
    assert selected["packet_speedup"] == pytest.approx(1.25)
    assert selected["kda_ci_low"] == pytest.approx(1.20)
    assert selected["packet_policy_eligible"] is False
    assert selected["selected_route"] == guard.DIRECT
    assert "stable/shared" in selected["selection_reason"]


def test_non_packet_backend_with_packet_argument_falls_back_to_direct() -> None:
    data = _dataset()
    case = data["cases"][0]
    for phase in recorded_dataset.PHASES:
        for block in case["phases"][phase]:
            _set_route(block["arms"][guard.PACKET], "packet", "packet",
                       "direct", "direct", "canonical_direct")
    selected = guard.select_exact_cases(data, _task(data), _summary(data))["cases"][0]
    assert selected["selected_route"] == guard.DIRECT
    assert selected["packet_execution_contract_eligible"] is False
    assert "did not execute packet/full_packet" in selected["selection_reason"]


def test_missing_or_drifting_packet_route_metadata_falls_back_to_direct() -> None:
    data = _dataset()
    case = data["cases"][0]
    missing = case["phases"]["confirmation"][2]["arms"][guard.PACKET]
    missing.pop("structural_layout")
    selected = guard.select_exact_cases(data, _task(data), _summary(data))["cases"][0]
    assert selected["selected_route"] == guard.DIRECT
    assert "metadata is missing or inconsistent" in selected["selection_reason"]

    data = _dataset()
    case = data["cases"][0]
    drifting = case["phases"]["exploration"][5]["arms"][guard.PACKET]
    _set_route(drifting, "packet", "packet", "direct", "direct", "canonical_direct")
    selected = guard.select_exact_cases(data, _task(data), _summary(data))["cases"][0]
    assert selected["selected_route"] == guard.DIRECT
    assert "identity drifts" in selected["selection_reason"]


def test_missing_or_drifting_direct_identity_falls_back_to_direct() -> None:
    data = _dataset()
    case = data["cases"][0]
    missing = case["phases"]["exploration"][1]["arms"][guard.DIRECT]
    missing["validated_summary"]["native_execution"].pop("actual_implementation")
    missing["validation_summary_sha256"] = recorded_dataset._canonical_sha(
        missing["validated_summary"])
    selected = guard.select_exact_cases(data, _task(data), _summary(data))["cases"][0]
    assert selected["selected_route"] == guard.DIRECT
    assert "strongest direct control metadata is missing or inconsistent" in selected["selection_reason"]

    data = _dataset()
    case = data["cases"][0]
    drifting = case["phases"]["confirmation"][4]["arms"][guard.DIRECT]
    _set_route(drifting, "minimal_direct", "direct", "stable", "stable", "canonical_direct")
    selected = guard.select_exact_cases(data, _task(data), _summary(data))["cases"][0]
    assert selected["selected_route"] == guard.DIRECT
    assert "identity drifts" in selected["selection_reason"]


def test_no_kda_confirmation_returns_direct_for_every_workload() -> None:
    data = _dataset()
    output = guard.select_exact_cases(data, _task(data), _summary(data, confirmation=False))
    assert output["packet_regimes"] == []
    assert output["strongest_direct_regimes"] == [case["id"] for case in data["cases"]]
    assert all(case["kda_ci_low"] is None and case["selected_route"] == guard.DIRECT
               for case in output["cases"])


def test_summary_point_and_case_must_match_recomputed_observations() -> None:
    data = _dataset()
    summary = _summary(data)
    summary["confirmation"]["per_case"][data["cases"][0]["id"]]["score"] = 1.4
    with pytest.raises(guard.PolicyError, match="point score"):
        guard.select_exact_cases(data, _task(data), summary)


def test_missing_case_summary_or_wrong_task_policy_is_rejected() -> None:
    data = _dataset()
    summary = _summary(data)
    del summary["confirmation"]["per_case"][data["cases"][-1]["id"]]
    with pytest.raises(guard.PolicyError, match="confirmation omits"):
        guard.select_exact_cases(data, _task(data), summary)
    task = _task(data)
    task["promotion"]["min_improvement"] = 0.1
    with pytest.raises(guard.PolicyError, match="fixed CPU replay"):
        guard.select_exact_cases(data, task, _summary(data))


def test_nonpositive_times_and_incomplete_validation_are_rejected() -> None:
    data = _dataset()
    data["cases"][0]["phases"]["confirmation"][0]["arms"][guard.PACKET]["primary_ms"] = 0.0
    with pytest.raises(guard.PolicyError, match="recorded dataset failed"):
        guard.select_exact_cases(data, _task(data), _summary(data))
    data = _dataset()
    summary = _summary(data)
    summary["validation"]["packet-candidate"]["case_status"].pop(data["cases"][0]["id"])
    with pytest.raises(guard.PolicyError, match="failed or skipped correctness"):
        guard.select_exact_cases(data, _task(data), summary)


def test_kda_compact_observation_table_matches_all_546_dataset_rows() -> None:
    data = _dataset()
    config = {"recorded_observations": guard._expected_compact_observations(data)}
    guard._verify_compact_observations(data, config)
    config["recorded_observations"][data["cases"][0]["id"]]["phases"]["confirmation"][0]["arms"][guard.PACKET]["primary_ms"] += 1
    with pytest.raises(guard.PolicyError, match="differs from the 546"):
        guard._verify_compact_observations(data, config)


def test_kda_upstream_evidence_sidecar_is_bound_to_dataset_and_task(tmp_path: Path) -> None:
    data = _dataset()
    _materialize_dataset_evidence(data, tmp_path)
    expected = guard._expected_recorded_trusted_inputs(data)
    manifest = tmp_path / "recorded-trusted-inputs.json"
    digest = _write_json(manifest, {
        "schema": "cellerator.factorized_interactions.recorded_trusted_inputs.v1",
        "inputs": expected,
    })
    config = {
        "recorded_trusted_inputs_manifest_path": str(manifest.resolve()),
        "recorded_trusted_inputs_manifest_sha256": digest,
        "recorded_trusted_inputs_digest": guard._digest_json(expected),
        "recorded_trusted_input_count": len(expected),
    }
    task = {"trusted_inputs": [
        {"id": row["id"], "path": row["path"], "role": row["role"]}
        for row in expected
    ] + [{"id": "recorded_trusted_input_manifest",
          "path": str(manifest.resolve()), "role": "harness"}]}
    guard._verify_recorded_trusted_inputs(data, task, config)
    omitted = copy.deepcopy(task)
    omitted["trusted_inputs"] = [row for row in omitted["trusted_inputs"]
                                 if not row["id"].endswith(
                                     hashlib.sha256(expected[0]["dataset_id"].encode()).hexdigest())]
    with pytest.raises(guard.PolicyError, match="does not trust every preserved"):
        guard._verify_recorded_trusted_inputs(data, omitted, config)
    tampered = copy.deepcopy(config)
    tampered["recorded_trusted_inputs_digest"] = "0" * 64
    with pytest.raises(guard.PolicyError, match="sidecar differs"):
        guard._verify_recorded_trusted_inputs(data, task, tampered)


def test_pinned_gzip_archive_and_kda_two_file_snapshot_are_verified(tmp_path: Path) -> None:
    data, task, config = _native_archive_fixture(tmp_path)
    guard._verify_native_source_archive(data, task, config)

    omitted_index = copy.deepcopy(task)
    omitted_index["trusted_inputs"] = [row for row in omitted_index["trusted_inputs"]
                                       if row["id"] != "native_source_archive_member_index"]
    with pytest.raises(guard.PolicyError, match="does not bind the individual"):
        guard._verify_native_source_archive(data, omitted_index, config)

    bad_tree = copy.deepcopy(config)
    bad_tree["candidate_source_tree_sha256"] = "0" * 64
    with pytest.raises(guard.PolicyError, match="snapshot digest"):
        guard._verify_native_source_archive(data, task, bad_tree)


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _native_archive_fixture(tmp_path: Path, data: dict | None = None) -> tuple[dict, dict, dict]:
    archive_root = ROOT / "planning" / "interaction-layout-proposal-2026-11-02" / \
        "measurement-performance" / ".autobench" / "data" / "factorized-interactions" / \
        "native-qualified-source-1"
    tar_path = archive_root / "native-performance-filtered.tar"
    index_path = archive_root / "native-performance-filtered-manifest.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = index["files"]
    closure_sha = guard._NATIVE_SOURCE_CLOSURE_SHA256

    artifacts_root = tmp_path / "native-artifacts"
    artifacts_root.mkdir()

    def write_binary(relative: str, content: bytes) -> dict:
        path = artifacts_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"path": relative, "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content)}

    candidate_runner = write_binary("candidate/runner.bin", b"synthetic accepted-candidate runner")
    baseline_runner = write_binary("baseline/runner.bin", b"synthetic strongest-direct runner")
    library = write_binary("lib/factorized_interaction_cuda.a", b"synthetic shared CUDA archive")
    candidate_manifest = {
        "schema": "cellerator.factorized_interactions.build.v1",
        "architecture": "sm_70", "fast_math": False,
        "source_closure_sha256": closure_sha, "source_files": rows,
        "binaries": {"runner": candidate_runner,
                     "factorized_interaction_cuda": library},
    }
    candidate_manifest_path = artifacts_root / "candidate-manifest.json"
    candidate_manifest_sha = _write_json(candidate_manifest_path, candidate_manifest)
    baseline_manifest = {
        "schema": "cellerator.factorized_interactions.build.v1",
        "architecture": "sm_70", "fast_math": False,
        "source_closure_sha256": closure_sha, "source_files": rows,
        "binaries": {"runner": baseline_runner,
                     "factorized_interaction_cuda": library},
        "candidate_runner": {"manifest_path": "candidate-manifest.json",
                             "binary_path": candidate_runner["path"],
                             "manifest_sha256": candidate_manifest_sha,
                             "binary_sha256": candidate_runner["sha256"]},
    }
    baseline_manifest_path = artifacts_root / "baseline-manifest.json"
    baseline_manifest_sha = _write_json(baseline_manifest_path, baseline_manifest)
    assets = [
        {"role": "candidate_manifest", "path": candidate_manifest_path.name,
         "sha256": candidate_manifest_sha, "bytes": candidate_manifest_path.stat().st_size},
        {"role": "baseline_manifest", "path": baseline_manifest_path.name,
         "sha256": baseline_manifest_sha, "bytes": baseline_manifest_path.stat().st_size},
        {"role": "candidate_runner", **candidate_runner},
        {"role": "baseline_runner", **baseline_runner},
        {"role": "shared_cuda_library", **library},
    ]
    workload_catalog_sha = hashlib.sha256(b"synthetic fixed catalog reference").hexdigest()
    validator_sha = (data["identity"]["validator_sha256"] if data is not None else
                     hashlib.sha256(b"synthetic fixed validator").hexdigest())
    if data is None:
        data = {"qualification_bundle_sha256": hashlib.sha256(b"qualification").hexdigest(),
                "identity": {}}
    identity = data.setdefault("identity", {})
    identity.update({
        "source_closure_sha256": closure_sha,
        "candidate_runner_sha256": candidate_runner["sha256"],
        "candidate_manifest_sha256": candidate_manifest_sha,
        "baseline_runner_sha256": baseline_runner["sha256"],
        "baseline_manifest_sha256": baseline_manifest_sha,
        "library_sha256": library["sha256"],
        "validator_sha256": validator_sha,
        "workload_catalog": {"path": "bench/factorized_interactions/workloads.json",
                              "sha256": workload_catalog_sha},
    })
    for report in data.get("timing_reports", []):
        for key in ("source_closure_sha256", "candidate_runner_sha256",
                    "candidate_manifest_sha256", "baseline_runner_sha256",
                    "baseline_manifest_sha256", "library_sha256", "validator_sha256"):
            report[key] = identity[key]
        report["harness_provenance"]["source_closure_sha256"] = closure_sha
    for case in data.get("cases", []):
        for phases in case.get("phases", {}).values():
            for block in phases:
                for arm in block["arms"].values():
                    arm["binary_identity"] = {key: identity[key] for key in (
                        "source_closure_sha256", "candidate_runner_sha256",
                        "candidate_manifest_sha256", "library_sha256", "validator_sha256")}
                    build = arm["validated_summary"]["build_provenance"]
                    build.update({
                        "source_closure_sha256": closure_sha,
                        "runner_binary_sha256": identity["candidate_runner_sha256"],
                        "build_manifest_sha256": identity["candidate_manifest_sha256"],
                    })
                    arm["validation_summary_sha256"] = recorded_dataset._canonical_sha(
                        arm["validated_summary"])
    dataset = data
    native_identity = {
        "source_closure_sha256": closure_sha,
        "candidate_runner_sha256": candidate_runner["sha256"],
        "candidate_manifest_sha256": candidate_manifest_sha,
        "baseline_runner_sha256": baseline_runner["sha256"],
        "baseline_manifest_sha256": baseline_manifest_sha,
        "library_sha256": library["sha256"],
        "validator_sha256": validator_sha,
        "workload_catalog_path": "bench/factorized_interactions/workloads.json",
        "workload_catalog_sha256": workload_catalog_sha,
    }

    candidate_root = tmp_path / "candidate-source"
    candidate_root.mkdir()
    gzip_path = candidate_root / "native_source.tar.gz"
    import gzip
    with tar_path.open("rb") as source, gzip_path.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output,
                           mtime=0, compresslevel=6) as compressed:
            shutil.copyfileobj(source, compressed, length=1 << 20)
    gzip_sha = hashlib.sha256(gzip_path.read_bytes()).hexdigest()
    native_tree_sha = guard._digest_json([{"path": row["path"], "sha256": row["sha256"]}
                                          for row in rows])
    descriptor = {
        "schema": guard._NATIVE_DESCRIPTOR_SCHEMA,
        "native_source_tree_sha256": native_tree_sha,
        "source_closure_sha256": closure_sha,
        "native_source_archive_tar_path": tar_path.name,
        "native_source_archive_tar_sha256": guard._NATIVE_SOURCE_TAR_SHA256,
        "native_source_archive_tar_bytes": guard._NATIVE_SOURCE_TAR_BYTES,
        "native_source_archive_manifest_path": index_path.name,
        "native_source_archive_manifest_sha256": guard._NATIVE_SOURCE_INDEX_SHA256,
        "native_source_archive_member_count": guard._NATIVE_SOURCE_MEMBER_COUNT,
        "native_source_archive_members_sha256": guard._NATIVE_SOURCE_MEMBERS_SHA256,
        "native_source_archive_source_bytes": guard._NATIVE_SOURCE_BYTES,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "baseline_manifest_sha256": baseline_manifest_sha,
        "candidate_runner_sha256": candidate_runner["sha256"],
        "baseline_runner_sha256": baseline_runner["sha256"],
        "library_sha256": library["sha256"],
        "validator_sha256": validator_sha,
        "workload_catalog_sha256": workload_catalog_sha,
        "qualification_bundle_sha256": dataset["qualification_bundle_sha256"],
        "native_source_transport_path": "native_source.tar.gz",
        "native_source_transport_compression": "gzip",
        "native_source_transport_gzip_mtime": 0,
        "native_source_transport_sha256": gzip_sha,
        "native_source_transport_bytes": gzip_path.stat().st_size,
    }
    descriptor_path = candidate_root / "analysis_descriptor.json"
    descriptor_sha = _write_json(descriptor_path, descriptor)
    candidate_files = [{"path": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size}
                       for path in sorted(candidate_root.iterdir(), key=lambda item: item.name)]
    candidate_tree_sha = guard._digest_json([{"path": row["path"], "sha256": row["sha256"]}
                                             for row in candidate_files])
    source_manifest_path = tmp_path / "kda-source-manifest.json"
    source_manifest = {
        "schema_version": 1,
        "files": [row["path"] for row in candidate_files],
        "dependencies": [],
        "provenance": {
            "candidate_source_tree_sha256": candidate_tree_sha,
            "native_source_tree_sha256": native_tree_sha,
            "source_closure_sha256": closure_sha,
            "native_source_archive_tar_sha256": guard._NATIVE_SOURCE_TAR_SHA256,
            "native_source_archive_manifest_sha256": guard._NATIVE_SOURCE_INDEX_SHA256,
            "native_source_archive_gzip_sha256": gzip_sha,
            "candidate_source_descriptor_sha256": descriptor_sha,
            "qualification_bundle_sha256": dataset["qualification_bundle_sha256"],
        },
    }
    source_manifest_sha = _write_json(source_manifest_path, source_manifest)
    config = {
        "source_closure_sha256": closure_sha,
        "source_tree_sha256": candidate_tree_sha,
        "native_source_tree_sha256": native_tree_sha,
        "native_source_archive_tar_path": str(tar_path.resolve()),
        "native_source_archive_tar_sha256": guard._NATIVE_SOURCE_TAR_SHA256,
        "native_source_archive_tar_bytes": guard._NATIVE_SOURCE_TAR_BYTES,
        "native_source_archive_manifest_path": str(index_path.resolve()),
        "native_source_archive_manifest_sha256": guard._NATIVE_SOURCE_INDEX_SHA256,
        "native_source_archive_member_count": guard._NATIVE_SOURCE_MEMBER_COUNT,
        "native_source_archive_members_sha256": guard._NATIVE_SOURCE_MEMBERS_SHA256,
        "native_source_archive_source_bytes": guard._NATIVE_SOURCE_BYTES,
        "native_source_archive_gzip_path": str(gzip_path.resolve()),
        "native_source_archive_gzip_sha256": gzip_sha,
        "native_source_archive_gzip_bytes": gzip_path.stat().st_size,
        "native_source_archive_gzip_compression": "gzip",
        "native_source_archive_gzip_mtime": 0,
        "native_source_descriptor_path": str(descriptor_path.resolve()),
        "native_source_descriptor_sha256": descriptor_sha,
        "native_source_descriptor": descriptor,
        "candidate_source_root": str(candidate_root.resolve()),
        "candidate_source_tree_sha256": candidate_tree_sha,
        "candidate_source_files": candidate_files,
        "candidate_manifest_path": candidate_manifest_path.name,
        "baseline_manifest_path": baseline_manifest_path.name,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "baseline_manifest_sha256": baseline_manifest_sha,
        "native_artifact_root": str(artifacts_root.resolve()),
        "assets": assets,
        "native_identity": native_identity,
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": source_manifest_sha,
    }
    task = {"trusted_inputs": [
        {"id": "native_source_archive_gzip", "path": str(gzip_path.resolve()), "role": "toolchain"},
        {"id": "native_source_archive_member_index", "path": str(index_path.resolve()), "role": "reference"},
        {"id": "native_source_descriptor", "path": str(descriptor_path.resolve()), "role": "harness"},
        {"id": "kda_source_manifest", "path": str(source_manifest_path.resolve()), "role": "harness"},
    ]}
    return dataset, task, config


def _materialize_dataset_evidence(data: dict, root: Path) -> None:
    evidence = root / "upstream"
    evidence.mkdir(parents=True, exist_ok=True)

    def write(name: str, payload: bytes) -> tuple[str, str]:
        path = evidence / name
        path.write_bytes(payload)
        return str(path.resolve()), hashlib.sha256(payload).hexdigest()

    qpath, qsha = write("qualification.json", b"synthetic accepted qualification reference")
    cpath, csha = write("catalog.json", b"synthetic fixed catalog reference")
    sanitizer_source = ROOT / "planning" / "interaction-layout-proposal-2026-11-02" / \
        "validation" / "sanitizer_tool_partition.py"
    sanitizer_path = evidence / sanitizer_source.name
    shutil.copyfile(sanitizer_source, sanitizer_path)
    spath, ssha = str(sanitizer_path.resolve()), hashlib.sha256(sanitizer_path.read_bytes()).hexdigest()
    data["qualification_bundle"]["path"] = qpath
    data["qualification_bundle"]["sha256"] = qsha
    data["qualification_bundle_sha256"] = qsha
    data["identity"]["workload_catalog"] = {"path": cpath, "sha256": csha}
    data["sanitizer_wrapper"] = {"path": spath, "sha256": ssha}

    shared_path, shared_sha = write("retained-case-evidence.json", b"synthetic retained case evidence")
    for report_index, report in enumerate(data["timing_reports"]):
        for prefix in ("report", "receipt", "pointer"):
            path, digest = write(f"timing-{report_index:02d}-{prefix}.json",
                                 f"synthetic {report_index} {prefix}".encode())
            report[f"{prefix}_path"] = path
            report[f"{prefix}_sha256"] = digest
        case = next(row for row in data["cases"] if row["id"] == report["workload"])
        for block_index, block in enumerate(case["phases"][report["phase"]]):
            for arm_name, arm in block["arms"].items():
                arm["original_report_sha256"] = report["report_sha256"]
                helper_path, helper_sha = write(
                    f"timed-helper-{report['workload']}-{report['phase']}-{block_index}-{arm_name}.json",
                    f"synthetic helper {report['workload']} {report['phase']} {block_index} {arm_name}".encode())
                for path_key, sha_key in (
                    ("validated_record_path", "validated_record_sha256"),
                    ("validator_stdout_path", "validator_stdout_sha256"),
                    ("validator_stderr_path", "validator_stderr_sha256"),
                ):
                    arm[path_key] = shared_path
                    arm[sha_key] = shared_sha
                arm["helper_evidence_path"] = helper_path
                arm["helper_evidence_sha256"] = helper_sha

    references: dict[tuple[str, str, str], dict] = {}

    def retain(path: str, digest: str, role: str) -> None:
        key = path, digest, role
        identifier = "synthetic-" + hashlib.sha256((role + "\0" + path).encode()).hexdigest()[:24]
        references[key] = {"id": identifier, "path": path, "sha256": digest, "role": role}

    retain(qpath, qsha, "qualification_bundle")
    retain(cpath, csha, "fixed_workload_catalog")
    retain(spath, ssha, "fixed_sanitizer_partition_validator")
    for key, filename in recorded_dataset._HARNESS_SIDECARS.items():
        source = BOUNDED / filename
        destination = evidence / filename
        shutil.copyfile(source, destination)
        retain(str(destination.resolve()), hashlib.sha256(destination.read_bytes()).hexdigest(),
               f"timing_harness_{key.removesuffix('_sha256')}")
    for report in data["timing_reports"]:
        for prefix, role in (("report", "timing_partition_report"),
                             ("receipt", "finished_autobench_scheduler_receipt"),
                             ("pointer", "finished_autobench_pointer")):
            retain(report[f"{prefix}_path"], report[f"{prefix}_sha256"], role)
    for case in data["cases"]:
        for phases in case["phases"].values():
            for block in phases:
                for arm in block["arms"].values():
                    for path_key, sha_key, role in (
                        ("helper_evidence_path", "helper_evidence_sha256", "timed_helper_evidence"),
                        ("validated_record_path", "validated_record_sha256", "pre_cleanup_validation_record"),
                        ("validator_stdout_path", "validator_stdout_sha256", "validator_stdout"),
                        ("validator_stderr_path", "validator_stderr_sha256", "validator_stderr"),
                    ):
                        retain(arm[path_key], arm[sha_key], role)
    data["trusted_inputs"] = sorted(references.values(), key=lambda row: (row["path"], row["role"]))
    recorded_dataset.validate_dataset(data)


def _public_kda_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Run the public KDA CLI against synthetic CPU-only observation commands."""
    data = _dataset()
    data["qualification_bundle_sha256"] = hashlib.sha256(
        b"synthetic accepted qualification reference").hexdigest()
    data, archive_task, archive_config = _native_archive_fixture(tmp_path, data)
    _materialize_dataset_evidence(data, tmp_path)
    dataset_path = tmp_path / "recorded-dataset.json"
    _write_json(dataset_path, data)
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()

    artifacts_root = Path(archive_config["native_artifact_root"])
    assets = archive_config["assets"]

    evidence_inputs = guard._expected_recorded_trusted_inputs(data)
    evidence_manifest = tmp_path / "recorded-trusted-inputs.json"
    evidence_manifest_sha = _write_json(evidence_manifest, {
        "schema": "cellerator.factorized_interactions.recorded_trusted_inputs.v1",
        "inputs": evidence_inputs,
    })
    config_path = tmp_path / "recorded-config.json"
    config = {
        **archive_config,
        "schema": guard.TASK_SCHEMA,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha,
        "qualification_status": data["qualification_status"],
        "qualification_bundle_sha256": data["qualification_bundle_sha256"],
        "recorded_device_uuid": data["identity"]["device_uuid"],
        "recorded_observations": guard._expected_compact_observations(data),
        "recorded_trusted_inputs_manifest_path": str(evidence_manifest.resolve()),
        "recorded_trusted_inputs_manifest_sha256": evidence_manifest_sha,
        "recorded_trusted_inputs_digest": guard._digest_json(evidence_inputs),
        "recorded_trusted_input_count": len(evidence_inputs),
        "kda_root": "/home/tumlinson/kda",
        "candidate_ids": dict(guard.CANDIDATES),
    }
    _write_json(config_path, config)

    worker = tmp_path / "synthetic_kda_worker.py"
    worker.write_text('''#!/usr/bin/env python3
import json, shutil, sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
operation = sys.argv[2]
request_path, result_path = Path(sys.argv[3]), Path(sys.argv[4])
request = json.loads(request_path.read_text())
request_id = request["request_id"]
result = {"schema_version": 1, "request_id": request_id, "status": "ok"}

if operation == "probe":
    import platform
    result["environment"] = {"kind": "cpu", "architecture": platform.machine(),
                             "tools": {"python": platform.python_version()}}
elif operation == "build":
    build_dir = Path(request["build"]["directory"])
    copied = []
    for asset in config["assets"]:
        rel = Path("prebuilt-native") / asset["path"]
        destination = build_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(config["native_artifact_root"]) / asset["path"], destination)
        copied.append({"path": rel.as_posix(), "sha256": asset["sha256"]})
    result.update({"candidate_id": request["candidate"]["id"], "artifacts": copied})
elif operation == "validate":
    result["cases"] = [{"case_id": case["id"], "workload_sha256": case["workload_sha256"],
                         "status": "pass"} for case in request["cases"]]
elif operation == "benchmark":
    case_id = request["case"]["id"]
    candidate_id = request["candidate"]["id"]
    arm_name = {"strongest-direct": "strongest_direct",
                "packet-candidate": "packet"}[candidate_id]
    source = config["recorded_observations"][case_id]
    phase = request["phase"]
    block = source["phases"][phase][request["round"]]
    arm = block["arms"][arm_name]
    result.update({
        "case_id": case_id,
        "workload_sha256": request["case"]["workload_sha256"],
        "metric": request["timing"]["metric"], "unit": request["timing"]["unit"],
        "timing_scope": request["timing"]["scope"],
        "warmup": request["timing"]["warmup"], "repetitions": 1,
        "samples": [arm["primary_ms"]],
        "recorded_observation": {
            "dataset_sha256": config["dataset_sha256"],
            "qualification_status": config["qualification_status"],
            "qualification_bundle_sha256": config["qualification_bundle_sha256"],
            "recorded_workload_sha256": source["recorded_workload_sha256"],
            "recorded_observation_sha256": arm["recorded_observation_sha256"],
            "raw_metrics_sha256": arm["raw_metrics_sha256"],
            "validation_summary_sha256": arm["validation_summary_sha256"],
            "original_report_sha256": arm["original_report_sha256"],
            "recorded_cuda_device_uuid": arm["device_uuid"],
            "original_receipt_identity": arm["original_receipt_identity"],
            "scored_arm": arm_name,
            "native_direct_is_descriptive_only": True,
        },
    })
else:
    raise RuntimeError("unsupported operation: " + operation)

result_path.parent.mkdir(parents=True, exist_ok=True)
result_path.write_text(json.dumps(result, sort_keys=True) + "\\n")
''', encoding="utf-8")

    task_path = tmp_path / "recorded-task.json"
    task = _task(data, dataset_sha)
    task.update({
        "schema_version": 1,
        "target": {"kind": "cpu", "architecture": platform.machine()},
        "execution": {"mode": "direct"},
        "numerical_requirements": {"mode": "replay exact immutable observations"},
        "toolchain": {},
        "budgets": {"max_seconds": 600, "max_candidates": 2, "max_build_jobs": 1},
    })
    task["timing"]["scope"] = (
        "one originally measured resident caller-wall block mean for one forward/VJP step; replayed as data on CPU")
    task["commands"] = {
        name: {"argv": [sys.executable, str(worker.resolve()), str(config_path.resolve()),
                         name, "{request}", "{result}"], "timeout_seconds": 30,
               **({"cpu_only": True} if name == "build" else {})}
        for name in ("build", "validate", "benchmark")
    }
    task["probe"] = {"argv": [sys.executable, str(worker.resolve()), str(config_path.resolve()),
                                  "probe", "{request}", "{result}"], "timeout_seconds": 30}

    source = Path(archive_config["candidate_source_root"])
    candidate_manifest = Path(archive_config["source_manifest_path"])
    task["trusted_inputs"] = [
        {"id": "recorded_engine", "path": str(Path(recorded_engine.__file__).resolve()), "role": "harness"},
        {"id": "recorded_dataset", "path": str(dataset_path.resolve()), "role": "reference"},
        {"id": "recorded_configuration", "path": str(config_path.resolve()), "role": "harness"},
        {"id": "recorded_trusted_input_manifest", "path": str(evidence_manifest.resolve()), "role": "harness"},
    ]
    task["trusted_inputs"].extend(archive_task["trusted_inputs"])
    task["trusted_inputs"].extend(
        {"id": f"native_artifact_{index}_{asset['role']}",
         "path": str((artifacts_root / asset["path"]).resolve()), "role": "toolchain"}
        for index, asset in enumerate(assets))
    task["trusted_inputs"].extend(
        {"id": row["id"], "path": row["path"], "role": row["role"]}
        for row in evidence_inputs)
    _write_json(task_path, task)

    workspace = tmp_path / "kda-workspace"
    kda_root = Path(config["kda_root"]).resolve(strict=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(kda_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    base = [sys.executable, "-m", "kda_experiments", "--task", str(task_path.resolve()),
            "--workspace", str(workspace.resolve())]
    command_rows = []

    def run(*args: str) -> dict:
        argv = base + list(args)
        process = subprocess.run(argv, cwd=kda_root, env=env, text=True,
                                 capture_output=True, check=False, timeout=120)
        assert process.returncode == 0, process.stderr or process.stdout
        response = json.loads(process.stdout)
        assert response["status"] == "ok", response
        command_rows.append({"argv": argv, "returncode": process.returncode,
                             "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
                             "stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest(),
                             "response": response})
        return response

    group = "recorded-test-group"
    run("register", "strongest-direct", str(source.resolve()), "--manifest", str(candidate_manifest.resolve()),
        "--hypothesis", "Synthetic direct baseline", "--change", "CPU-only integration fixture",
        "--baseline", "--experiment-group", group)
    run("register", "packet-candidate", str(source.resolve()), "--manifest", str(candidate_manifest.resolve()),
        "--hypothesis", "Synthetic packet candidate", "--change", "CPU-only integration fixture",
        "--experiment-group", group)
    run("prepare", "strongest-direct", "packet-candidate", "--jobs", "1")
    evaluated = run("evaluate", "--candidates", "strongest-direct", "packet-candidate",
                    "--incumbent", "strongest-direct")
    summary_path = Path(evaluated["summary_path"]).resolve(strict=True)
    summary = _read_json_for_test(summary_path)
    receipt_path = tmp_path / "analysis-receipt.json"
    receipt = {
        "schema": guard.ANALYSIS_SCHEMA,
        "status": "provisional_cpu_analysis_completed",
        "outer_receipt_verification": "pending", "cleanup_verification": "pending",
        "cuda_executed": False, "scheduler_admission_requested": False,
        "task_path": str(task_path.resolve()),
        "task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "workspace": str(workspace.resolve()),
        "recorded_cuda_observation": {"device_uuid": data["identity"]["device_uuid"],
            "qualification_status": data["qualification_status"],
            "qualification_bundle_sha256": data["qualification_bundle_sha256"],
            "dataset_sha256": dataset_sha},
        "commands": command_rows,
        "provisional_result": {"run_id": evaluated["run_id"],
            "summary_path": str(summary_path),
            "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "decision": summary["decision"], "rankings": summary["rankings"]},
    }
    _write_json(receipt_path, receipt)
    return dataset_path, task_path, config_path, receipt_path, summary_path


def _read_json_for_test(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_replay_matrix(root: Path, data: dict, task: dict, dataset_sha: str = "a" * 64,
                         *, tamper_response: bool = False) -> dict:
    run_id = "run-policy-test"
    task_cases = {case["id"]: case for case in task["cases"]}
    records = []
    request_id_int = 1
    for phase in ("exploration", "confirmation"):
        for case in data["cases"]:
            for round_index, block in enumerate(case["phases"][phase]):
                for candidate_id, arm_name in guard.CANDIDATES.items():
                    request_id = f"00000000-0000-4000-8000-{request_id_int:012x}"
                    request_id_int += 1
                    component = hashlib.sha256(case["id"].encode()).hexdigest()[:20]
                    stage = (root / "stages" / candidate_id / phase /
                             f"round-{round_index:03d}" / component)
                    request = {"schema_version": 1, "request_id": request_id,
                               "operation": "benchmark", "task_id": task["task_id"],
                               "target": task["target"], "candidate": {"id": candidate_id},
                               "case": task_cases[case["id"]], "phase": phase,
                               "round": round_index, "timing": task["timing"]}
                    arm = block["arms"][arm_name]
                    response = {
                        "schema_version": 1, "request_id": request_id, "status": "ok",
                        "case_id": case["id"],
                        "workload_sha256": task_cases[case["id"]]["workload_sha256"],
                        "metric": task["timing"]["metric"], "unit": task["timing"]["unit"],
                        "timing_scope": "one originally measured resident caller-wall block mean for one forward/VJP step; replayed as data on CPU",
                        "warmup": 0, "repetitions": 1, "samples": [arm["primary_ms"]],
                        "recorded_observation": {
                            "dataset_sha256": dataset_sha,
                            "qualification_status": data["qualification_status"],
                            "qualification_bundle_sha256": data["qualification_bundle_sha256"],
                            "recorded_workload_sha256": case["workload_sha256"],
                            "recorded_observation_sha256": recorded_engine.digest_json(arm),
                            "raw_metrics_sha256": arm["raw_metrics_sha256"],
                            "validation_summary_sha256": arm["validation_summary_sha256"],
                            "original_report_sha256": arm["original_report_sha256"],
                            "recorded_cuda_device_uuid": arm["device_uuid"],
                            "original_receipt_identity": {key: arm[key] for key in (
                                "scheduler_job_id", "scheduler_attempt_id", "autobench_attempt_id")},
                            "scored_arm": arm_name,
                            "native_direct_is_descriptive_only": True,
                        },
                    }
                    request_sha = _write_json(stage / "requests" / f"{request_id}.json", request)
                    response_sha = _write_json(stage / "responses" / f"{request_id}.json", response)
                    records.append({"candidate_id": candidate_id, "case_id": case["id"],
                                    "round": round_index, "phase": phase,
                                    "response_sha256": response_sha, "request_id": request_id})
    if tamper_response:
        first = records[0]
        component = hashlib.sha256(first["case_id"].encode()).hexdigest()[:20]
        path = (root / "stages" / first["candidate_id"] / first["phase"] /
                f"round-{first['round']:03d}" / component / "responses" /
                f"{first['request_id']}.json")
        response = json.loads(path.read_text())
        response["samples"] = [response["samples"][0] + 0.1]
        _write_json(path, response)
    return {"status": "complete", "run_id": run_id, "confirmation": {},
            "benchmark_records": records}


def test_replay_matrix_verifies_each_saved_cpu_response_against_original_blocks(tmp_path: Path) -> None:
    data = _dataset()
    task = _task(data)
    summary = _write_replay_matrix(tmp_path, data, task)
    guard._validate_observed_replays(data, task, {"dataset_sha256": "a" * 64}, summary, tmp_path)
    changed = _write_replay_matrix(tmp_path / "changed", data, task, tamper_response=True)
    with pytest.raises(guard.PolicyError, match="response differs from the summary digest"):
        guard._validate_observed_replays(data, task, {"dataset_sha256": "a" * 64},
                                         changed, tmp_path / "changed")


def test_full_guard_accepts_public_kda_summary_with_synthetic_cpu_observations(tmp_path: Path) -> None:
    dataset_path, task_path, config_path, receipt_path, summary_path = _public_kda_fixture(tmp_path)
    with patch.object(guard, "_verify_rebuilt_dataset", autospec=True):
        result = guard._verify_and_create(dataset_path, task_path, config_path,
                                          receipt_path, summary_path)
    assert result["status"] == "conditional_pending_external_cpu_receipt_and_independent_review"
    assert len(result["cases"]) == 13
    assert result["kda_summary_path"].endswith(
        f"runs/kda-experiments/attempts/{summary_path.parent.name}/summary.json")
    assert result["kda_summary_sha256"] == hashlib.sha256(summary_path.read_bytes()).hexdigest()
    assert result["external_cpu_receipt_verification"] == "pending"
    assert result["cleanup_verification"] == "pending"
    assert result["independent_review_status"] == "pending"

