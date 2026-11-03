from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
BOUNDED = ROOT / "bench" / "factorized_interactions" / "bounded"
sys.path.insert(0, str(BOUNDED))

import recorded_dataset as dataset  # noqa: E402


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _valid() -> dict:
    uuid = "GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a"
    identity = {
        "source_closure_sha256": _sha("source"),
        "candidate_runner_sha256": _sha("candidate-runner"),
        "candidate_manifest_sha256": _sha("candidate-manifest"),
        "baseline_runner_sha256": _sha("baseline-runner"),
        "baseline_manifest_sha256": _sha("baseline-manifest"),
        "library_sha256": _sha("library"),
        "validator_sha256": _sha("validator"),
        "device_uuid": uuid,
        "workload_catalog": {"path": "workloads.json", "sha256": _sha("catalog")},
        "timing_harness": dict(dataset.APPROVED_TIMING_HARNESS_SHA256),
    }
    build_arm = {key: identity[key] for key in (
        "source_closure_sha256", "candidate_runner_sha256", "candidate_manifest_sha256",
        "library_sha256", "validator_sha256")}
    cases = []
    timing_reports = []
    for workload_index in range(13):
        workload = f"case-{workload_index:02d}"
        descriptor = {"id": workload, "extent": workload_index + 1}
        workload_sha = dataset._canonical_sha(descriptor)
        input_digest = _sha(f"input-{workload}")
        phases = {}
        for phase in dataset.PHASES:
            phase_index = dataset.PHASES.index(phase)
            report_sha = _sha(f"report-{workload}-{phase}")
            scheduler_job_id = f"job-{workload_index}-{phase_index}"
            scheduler_attempt_id = f"attempt-{workload_index}-{phase_index}"
            autobench_attempt_id = f"autobench-{workload_index}-{phase_index}"
            timing_reports.append({
                "workload": workload, "phase": phase,
                "report_path": f"/evidence/{workload}-{phase}.json",
                "report_sha256": report_sha,
                "receipt_path": f"/evidence/{workload}-{phase}-receipt.json",
                "receipt_sha256": _sha(f"receipt-{workload}-{phase}"),
                "pointer_path": f"/evidence/{workload}-{phase}-pointer.json",
                "pointer_sha256": _sha(f"pointer-{workload}-{phase}"),
                "scheduler_job_id": scheduler_job_id,
                "scheduler_attempt_id": scheduler_attempt_id,
                "autobench_attempt_id": autobench_attempt_id,
                "device_uuid": uuid,
                "cgroup": f"/system.slice/{scheduler_attempt_id}.service",
                "unit_name": f"{scheduler_attempt_id}.service",
                "source_fingerprint": _sha(f"source-{workload}-{phase}"),
                "build_fingerprint": _sha(f"build-{workload}-{phase}"),
                "build_output_identity": f"build-output-{workload}-{phase}",
                "harness_provenance": {
                    "schema": dataset.HARNESS_SCHEMA,
                    "before": dict(dataset.APPROVED_TIMING_HARNESS_SHA256),
                    "after": dict(dataset.APPROVED_TIMING_HARNESS_SHA256),
                    "unchanged": True,
                    "source_closure_sha256": identity["source_closure_sha256"],
                    "validator_sha256": identity["validator_sha256"],
                },
            })
            blocks = []
            for block_index in range(7):
                global_index = block_index + (0 if phase == "exploration" else 7)
                pair = list(dataset.PAIR_ORDER if global_index % 2 == 0
                            else reversed(dataset.PAIR_ORDER))
                position = global_index % 3
                execution = pair.copy()
                execution.insert(position, "native_direct")
                arms = {}
                for arm_name in dataset.ARMS:
                    mode = {"strongest_direct": "minimal_direct", "packet": "packet",
                            "native_direct": "native_direct"}[arm_name]
                    implementation = {"strongest_direct": ("direct", "direct", "canonical_direct"),
                                      "packet": ("packet", "packet", "full_packet"),
                                      "native_direct": ("direct", "direct", "full_packet")} [arm_name]
                    metrics = {
                        "schema": "cellerator.factorized_interactions.metrics.v1",
                        "device_uuid": uuid, "scheduler_job_id": scheduler_job_id,
                        "selected_backend": implementation[0], "iterations": 30,
                        "phase_ms": {dataset.PRIMARY_METRIC: 1.0 + workload_index / 10,
                                     "warm_forward_vjp_batch_host_wall": 30.0 * (1.0 + workload_index / 10)},
                        "samples_ms": {name: [0.01] * 30 for name in
                                       ("forward", "backward", "forward_vjp", "parameter_refresh")},
                    }
                    contract = {name: {"max_error_to_allowance": 0.1}
                                for name in ("y", "dx", "dz", "dalpha", "dbias", "dgamma")}
                    summary = {
                        "schema": "cellerator.factorized_interactions.validation.v1",
                        "validated": True, "timed": True,
                        "workload": workload, "activation_storage": "fp16",
                        "parameter_storage": "fp16", "output_storage": "fp16",
                        "gradient_mode": "overwrite", "scheduler_job_id": scheduler_job_id,
                        "metrics": metrics, "contract_evidence": contract,
                        "native_execution": {
                            "candidate": mode, "device_uuid": uuid,
                            "scheduler_job_id": scheduler_job_id,
                        },
                        "build_provenance": {
                            "source_closure_sha256": identity["source_closure_sha256"],
                            "runner_binary_sha256": identity["candidate_runner_sha256"],
                            "build_manifest_sha256": identity["candidate_manifest_sha256"],
                        },
                    }
                    arms[arm_name] = {
                        "primary_ms": metrics["phase_ms"][dataset.PRIMARY_METRIC],
                        "metrics": metrics,
                        "raw_metrics_sha256": dataset._canonical_sha(metrics),
                        "validated_summary": summary,
                        "validation_summary_sha256": dataset._canonical_sha(summary),
                        "input_case_sha256": input_digest,
                        "output_result_sha256": _sha(f"output-{workload}-{phase}-{block_index}-{arm_name}"),
                        "candidate_argument": mode,
                        "requested_route": "packet" if mode == "packet" else "direct",
                        "selected_backend": implementation[0],
                        "actual_implementation": implementation[1],
                        "structural_layout": implementation[2],
                        "binary_identity": build_arm,
                        "scheduler_job_id": scheduler_job_id,
                        "scheduler_attempt_id": scheduler_attempt_id,
                        "autobench_attempt_id": autobench_attempt_id,
                        "device_uuid": uuid,
                        "cgroup": f"/system.slice/{scheduler_attempt_id}.service",
                        "unit_name": f"{scheduler_attempt_id}.service",
                        "source_fingerprint": _sha(f"source-{workload}-{phase}"),
                        "build_fingerprint": _sha(f"build-{workload}-{phase}"),
                        "build_output_identity": f"build-output-{workload}-{phase}",
                        "helper_evidence_path": f"/evidence/helper-{workload}-{phase}-{block_index}-{arm_name}.json",
                        "helper_evidence_sha256": _sha(f"helper-{workload}-{phase}-{block_index}-{arm_name}"),
                        "validated_record_path": f"/evidence/validated-{workload}-{phase}-{block_index}-{arm_name}.json",
                        "validated_record_sha256": _sha(f"validated-{workload}-{phase}-{block_index}-{arm_name}"),
                        "validator_stdout_path": f"/evidence/stdout-{workload}-{phase}-{block_index}-{arm_name}.json",
                        "validator_stdout_sha256": _sha(f"stdout-{workload}-{phase}-{block_index}-{arm_name}"),
                        "validator_stderr_path": f"/evidence/stderr-{workload}-{phase}-{block_index}-{arm_name}.log",
                        "validator_stderr_sha256": _sha(f"stderr-{workload}-{phase}-{block_index}-{arm_name}"),
                        "original_report_sha256": report_sha,
                    }
                blocks.append({"block_index": block_index,
                               "global_block_index": global_index,
                               "paired_order": pair,
                               "execution_order": execution,
                               "descriptive_position": position,
                               "arms": arms})
            phases[phase] = blocks
        cases.append({"id": workload, "workload_sha256": workload_sha,
                      "workload_descriptor": descriptor,
                      "input_case_sha256": input_digest, "phases": phases})
    qualification_sha = _sha("qualification")
    result = {
        "schema": dataset.SCHEMA,
        "status": "recorded_cuda_observations_verified_pending_independent_review",
        "qualification_status": "validated_pending_independent_review",
        "qualification_bundle_sha256": qualification_sha,
        "qualification_bundle": {"status": "validated_pending_independent_review",
                                  "path": "/evidence/qualification.json",
                                  "sha256": qualification_sha, "device_uuid": uuid},
        "sanitizer_wrapper": {"path": "/evidence/sanitizer-wrapper.py",
                              "sha256": dataset.APPROVED_TIMING_HARNESS_SHA256["sanitizer_wrapper_sha256"]},
        "identity": identity,
        "timing_reports": timing_reports,
        "cases": cases,
        "coverage": {"workloads": 13, "phases_per_workload": 2,
                     "paired_blocks_per_phase": 7, "arms_per_block": 3,
                     "unique_cuda_observations": 546},
    }
    refs = {}

    def add_ref(path: str, digest: str, role: str) -> None:
        identifier = "recorded-" + hashlib.sha256((role + "\0" + path).encode()).hexdigest()[:20]
        refs[(path, digest, role)] = {"id": identifier, "path": path, "sha256": digest, "role": role}

    add_ref(result["qualification_bundle"]["path"], qualification_sha, "qualification_bundle")
    add_ref(result["identity"]["workload_catalog"]["path"], result["identity"]["workload_catalog"]["sha256"],
            "fixed_workload_catalog")
    add_ref(result["sanitizer_wrapper"]["path"], result["sanitizer_wrapper"]["sha256"], "fixed_sanitizer_partition_validator")
    for key, filename in dataset._HARNESS_SIDECARS.items():
        add_ref(f"/evidence/{filename}", dataset.APPROVED_TIMING_HARNESS_SHA256[key],
                f"timing_harness_{key.removesuffix('_sha256')}")
    for timing_report in timing_reports:
        for prefix, role in (("report", "timing_partition_report"),
                             ("receipt", "finished_autobench_scheduler_receipt"),
                             ("pointer", "finished_autobench_pointer")):
            add_ref(timing_report[f"{prefix}_path"], timing_report[f"{prefix}_sha256"], role)
    for case in cases:
        for phase in dataset.PHASES:
            for block in case["phases"][phase]:
                for arm in block["arms"].values():
                    for path_key, hash_key, role in (
                            ("helper_evidence_path", "helper_evidence_sha256", "timed_helper_evidence"),
                            ("validated_record_path", "validated_record_sha256", "pre_cleanup_validation_record"),
                            ("validator_stdout_path", "validator_stdout_sha256", "validator_stdout"),
                            ("validator_stderr_path", "validator_stderr_sha256", "validator_stderr")):
                        add_ref(arm[path_key], arm[hash_key], role)
    result["trusted_inputs"] = sorted(refs.values(), key=lambda row: (row["path"], row["role"]))
    return result


def test_full_recorded_matrix_has_546_distinct_observations() -> None:
    facts = dataset.validate_dataset(_valid())
    assert facts["unique_cuda_observations"] == 546


def test_rejects_missing_workload_phase_block_or_arm() -> None:
    for mutation in ("workload", "phase", "block", "arm"):
        candidate = _valid()
        if mutation == "workload":
            candidate["cases"].pop()
        elif mutation == "phase":
            candidate["cases"][0]["phases"].pop("confirmation")
        elif mutation == "block":
            candidate["cases"][0]["phases"]["exploration"].pop()
        else:
            candidate["cases"][0]["phases"]["exploration"][0]["arms"].pop("packet")
        with pytest.raises(dataset.DatasetError):
            dataset.validate_dataset(candidate)


def test_rejects_wrong_pair_order_and_duplicate_original_observation() -> None:
    candidate = _valid()
    candidate["cases"][0]["phases"]["exploration"][0]["paired_order"] = ["packet", "strongest_direct"]
    with pytest.raises(dataset.DatasetError):
        dataset.validate_dataset(candidate)

    candidate = _valid()
    first = candidate["cases"][0]["phases"]["exploration"][0]["arms"]["packet"]
    second = candidate["cases"][0]["phases"]["exploration"][1]["arms"]["packet"]
    second["helper_evidence_sha256"] = first["helper_evidence_sha256"]
    with pytest.raises(dataset.DatasetError):
        dataset.validate_dataset(candidate)


def test_rejects_tampered_metric_or_summary_digest() -> None:
    for field in ("metrics", "validated_summary"):
        candidate = _valid()
        arm = candidate["cases"][0]["phases"]["exploration"][0]["arms"]["packet"]
        arm[field]["tampered"] = True
        with pytest.raises(dataset.DatasetError):
            dataset.validate_dataset(candidate)


def test_rejects_numeric_failure_or_wrong_binary_identity() -> None:
    candidate = _valid()
    arm = candidate["cases"][0]["phases"]["exploration"][0]["arms"]["packet"]
    arm["validated_summary"]["contract_evidence"]["dx"]["max_error_to_allowance"] = 1.01
    arm["validation_summary_sha256"] = dataset._canonical_sha(arm["validated_summary"])
    arm["metrics"] = arm["validated_summary"]["metrics"]
    arm["raw_metrics_sha256"] = dataset._canonical_sha(arm["metrics"])
    with pytest.raises(dataset.DatasetError):
        dataset.validate_dataset(candidate)


def test_rejects_changed_timing_sidecar_before_or_after_hash() -> None:
    for when in ("before", "after"):
        candidate = _valid()
        provenance = candidate["timing_reports"][0]["harness_provenance"]
        provenance[when]["bounded_validation_sha256"] = _sha("unapproved bounded validator")
        with pytest.raises(dataset.DatasetError, match="sidecar hashes"):
            dataset.validate_dataset(candidate)


def test_hashes_staged_sidecars_and_detects_changed_file(tmp_path: Path) -> None:
    for index, filename in enumerate(dataset._HARNESS_SIDECARS.values()):
        (tmp_path / filename).write_bytes(f"sidecar-{index}".encode())
    wrapper = tmp_path / "sanitizer_tool_partition.py"
    wrapper.write_bytes(b"approved wrapper copy")
    approved = dataset._local_harness_hashes(tmp_path, wrapper)
    assert dataset._check_local_harness(tmp_path, wrapper, approved) == approved
    (tmp_path / "bounded_validation.py").write_bytes(b"changed validator")
    with pytest.raises(dataset.DatasetError, match="staged sidecar files"):
        dataset._check_local_harness(tmp_path, wrapper, approved)


def test_validates_one_producer_shaped_timed_arm_and_rejects_bad_batch_mean(tmp_path: Path) -> None:
    uuid = "GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a"
    context = {"status": "observed", "scheduler_job_id": "job-1",
               "scheduler_attempt_id": "attempt-1", "device_uuid": uuid,
               "cgroup": "/system.slice/device-scheduler-attempt-a.service",
               "unit_name": "device-scheduler-attempt-a.service"}
    identity = {key: _sha(key) for key in (
        "source_closure_sha256", "candidate_runner_sha256", "candidate_manifest_sha256",
        "library_sha256", "validator_sha256")}
    facts = {"stable": False, "unique": True, "strongest_direct_mode": "direct_unique_output"}
    dispatch = ("direct", "direct_unique_output", "canonical_direct")
    metrics = {
        "schema": "cellerator.factorized_interactions.metrics.v1",
        "device_uuid": uuid, "scheduler_job_id": context["scheduler_job_id"],
        "selected_backend": dispatch[0], "iterations": 30,
        "phase_ms": {dataset.PRIMARY_METRIC: 0.125,
                     "warm_forward_vjp_batch_host_wall": 3.75},
        "samples_ms": {name: [0.125] * 30 for name in
                       ("forward", "backward", "forward_vjp", "parameter_refresh")},
    }
    summary = {
        "schema": "cellerator.factorized_interactions.validation.v1",
        "validated": True, "timed": True, "workload": "fixture-workload",
        "activation_storage": "fp16", "parameter_storage": "fp16", "output_storage": "fp16",
        "gradient_mode": "overwrite", "scheduler_job_id": context["scheduler_job_id"],
        "input_case_sha256": _sha("input"), "metrics": metrics,
        "contract_evidence": {name: {"max_error_to_allowance": 0.1}
                              for name in ("y", "dx", "dz", "dalpha", "dbias", "dgamma")},
        "native_execution": {"candidate": "direct_unique_output", "requested_route": "direct",
                             "selected_backend": dispatch[0], "actual_implementation": dispatch[1],
                             "structural_layout": dispatch[2], "device_uuid": uuid,
                             "scheduler_job_id": context["scheduler_job_id"]},
        "build_provenance": {"source_closure_sha256": identity["source_closure_sha256"],
                             "runner_binary_sha256": identity["candidate_runner_sha256"],
                             "build_manifest_sha256": identity["candidate_manifest_sha256"]},
    }
    root = tmp_path
    (root / "cases").mkdir()
    report_path = root / "timing.json"
    report_path.write_text("{}\n")
    stdout = root / "validator.stdout.json"
    stdout.write_text(json.dumps(summary, sort_keys=True) + "\n")
    stderr = root / "validator.stderr.log"
    stderr.write_bytes(b"")
    inventory = [
        {"path": "input-000.case", "bytes": 5, "sha256": _sha("shard")},
        {"path": "output.result", "bytes": 8, "sha256": _sha("output")},
    ]
    digest_input, digest_output = _sha("input"), _sha("output")
    helper = {
        "schema": "cellerator.factorized_interactions.bounded_case.v1", "status": "pass",
        "case_id": "timed-fixture", "workload": "fixture-workload", "mode": "direct_unique_output",
        "summary": summary, "wire_inventory": inventory,
        "input_case_sha256": digest_input, "output_result_sha256": digest_output,
        "context": context,
        "build": {"source_closure_sha256": identity["source_closure_sha256"],
                  "runner_sha256": identity["candidate_runner_sha256"],
                  "build_manifest_sha256": identity["candidate_manifest_sha256"],
                  "library_sha256": identity["library_sha256"],
                  "validator_sha256": identity["validator_sha256"]},
        "validator_sha256": identity["validator_sha256"],
        "stdout_relative_path": "validator.stdout.json",
        "stderr_relative_path": "validator.stderr.log",
        "stdout_sha256": hashlib.sha256(stdout.read_bytes()).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.read_bytes()).hexdigest(),
        "validated_relative_path": "validated.json",
        "success_wires_removed": True, "cleanup_pending": False,
        "peak_scratch_bytes": 4096,
        "peak_measurement": "50ms sampled whole packet tree plus post-write evidence allowance",
    }
    before = {**helper, "success_wires_removed": False, "cleanup_pending": True}
    validated = root / "validated.json"
    validated.write_text(json.dumps(before, sort_keys=True) + "\n")
    helper["validated_record_sha256"] = hashlib.sha256(validated.read_bytes()).hexdigest()
    evidence_path = root / "evidence.json"
    evidence_path.write_text(json.dumps(helper, sort_keys=True) + "\n")
    facts_report = {"direct_control_facts": facts,
                    "outer_execution_context": {key: value for key, value in context.items()
                                                 if key != "status"}}
    arm = {
        "candidate_argument": "direct_unique_output", "requested_route": "direct",
        "input_case_sha256": digest_input, "success_wires_removed": True,
        "validator_sha256": identity["validator_sha256"], "validated_summary": summary,
        "execution_context": context,
        "binary_identity": {"source_closure_sha256": identity["source_closure_sha256"],
                            "runner_binary_sha256": identity["candidate_runner_sha256"],
                            "build_manifest_sha256": identity["candidate_manifest_sha256"],
                            "library_sha256": identity["library_sha256"],
                            "validator_sha256": identity["validator_sha256"]},
        "metrics": metrics, "primary_ms": 0.125, "output_result_sha256": digest_output,
        "helper_evidence_path": "evidence.json",
        "helper_evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "helper_stdout_path": "validator.stdout.json",
        "helper_stdout_sha256": hashlib.sha256(stdout.read_bytes()).hexdigest(),
        "helper_stderr_path": "validator.stderr.log",
        "helper_stderr_sha256": hashlib.sha256(stderr.read_bytes()).hexdigest(),
        "peak_scratch_bytes": 4096,
        "peak_measurement": helper["peak_measurement"],
    }
    checked = dataset._validated_observation(facts_report, report_path, {}, "strongest_direct", arm,
                                             "fixture-workload", identity, uuid, digest_input)
    assert checked["primary_ms"] == pytest.approx(0.125)
    assert checked["output_result_sha256"] == digest_output
    bad_arm = copy.deepcopy(arm)
    bad_arm["metrics"]["phase_ms"]["warm_forward_vjp_batch_host_wall"] = 3.0
    with pytest.raises(dataset.DatasetError, match="30-call batch total"):
        dataset._validated_observation(facts_report, report_path, {}, "strongest_direct", bad_arm,
                                      "fixture-workload", identity, uuid, digest_input)


def test_resolves_relocated_portable_qualification_members_and_rejects_escape(tmp_path: Path) -> None:
    original = tmp_path / "original"
    evidence = original / "nested"
    evidence.mkdir(parents=True)
    payload = b"immutable staged qualification record"
    (evidence / "report.json").write_bytes(payload)
    original_bundle = {
        "schema": "cellerator.factorized_interactions.recorded_qualification.v1",
        "evidence_root": ".",
        "reports": [{"kind": "dtype", "report_path": "nested/report.json",
                     "report_sha256": hashlib.sha256(payload).hexdigest()}],
    }
    bundle_path = original / "qualification.json"
    bundle_path.write_text(json.dumps(original_bundle))

    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    relocated_bundle = relocated / "qualification.json"
    bundle = json.loads(relocated_bundle.read_text())
    row = bundle["reports"][0]
    resolved = dataset._bundle_evidence_path({**row, "evidence_root": bundle["evidence_root"]},
                                             relocated_bundle, "report_path", "dtype report")
    assert resolved == relocated / "nested/report.json"
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == row["report_sha256"]

    escaped = dict(row, report_path="../outside.json")
    with pytest.raises(dataset.DatasetError, match="escapes"):
        dataset._bundle_evidence_path({**escaped, "evidence_root": "."},
                                      relocated_bundle, "report_path", "dtype report")

    absolute = dict(row, report_path=str(resolved))
    with pytest.raises(dataset.DatasetError, match="escapes"):
        dataset._bundle_evidence_path({**absolute, "evidence_root": "."},
                                      relocated_bundle, "report_path", "dtype report")

    candidate = _valid()
    candidate["cases"][0]["phases"]["exploration"][0]["arms"]["packet"]["binary_identity"]["library_sha256"] = _sha("other")
    with pytest.raises(dataset.DatasetError):
        dataset.validate_dataset(candidate)
