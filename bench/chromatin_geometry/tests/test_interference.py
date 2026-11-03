"""CPU-only tests for the interference packet and receipt analyzer.

No Scheduler API, nvidia-smi command, CUDA binary, or GPU is used here.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "cgs_interference_tested", REPO / "bench/chromatin_geometry/scripts/interference.py")
assert SPEC and SPEC.loader
interference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(interference)

UUID_A = "GPU-aaaa-1111"
UUID_B = "GPU-bbbb-2222"
UUID_C = "GPU-cccc-3333"
UUID_D = "GPU-dddd-4444"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_config(binary_sha: str, source_sha: str, workload_sha: str, repetitions: int = 2) -> dict:
    return {
        "schema": interference.CONFIG_SCHEMA,
        "source_archive_sha256": source_sha,
        "binary_sha256": binary_sha,
        "workload_sha256": workload_sha,
        "case": "dag7-64root-complete-consumer-coherent-dense-high-update-r16-change1",
        "backend": "direct", "gate": "supplied", "body": "product", "precision": "fp16",
        "condition_update": "incremental", "seed": 12601, "tile": 64, "stages": 1,
        "input": "synthetic", "primary_uuid": UUID_A, "concurrent_uuids": [UUID_A, UUID_B],
        "independent_repetitions": repetitions, "warmups": 0,
    }


def iso_time(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat()


def telemetry(active_uuid: str, other_activity: bool = False) -> dict:
    gpus = [{"uuid": u, "utilization_percent": 0}
            for u in (UUID_A, UUID_B, UUID_C, UUID_D)]
    processes = []
    if active_uuid:
        processes.append({"gpu_uuid": active_uuid, "pid": 123, "process_name": "cgs_study"})
    if other_activity:
        gpus[2]["utilization_percent"] = 4
        processes.append({"gpu_uuid": UUID_C, "pid": 456, "process_name": "other"})
    return {"complete": True, "sampled_at_epoch_ns": 1000, "gpus": gpus,
            "compute_processes": processes}


def campaign(cfg: dict, uuid: str, recurrent: float = 10.0) -> dict:
    return {
        "schema": "cgs.campaign.v1", "status": "ok", "accuracy_accepted": True,
        "validation_scope": "first-and-last-complete-steps", "descriptor_validation": "exact-work-membership",
        "workload_sha256": cfg["workload_sha256"], "device_uuid": uuid,
        "case": cfg["case"], "backend": cfg["backend"], "gate": cfg["gate"], "body": cfg["body"],
        "precision": cfg["precision"], "condition_update": cfg["condition_update"],
        "seed": cfg["seed"], "tile": cfg["tile"], "stages": cfg["stages"], "input": cfg["input"],
        "cold_execution_preparation_ms": 2.0,
        "replay_input_preparation_ms": 0.0,
        "host_layout_preparation_ms": 1.0, "cold_device_preparation_ms": 1.0,
        "recurrent_complete_horizon_ms": [recurrent], "output_recovery_ms": [1.0],
    }


def packet_for(root: Path, cfg: dict, config_sha: str, job: str, attempt: str, uuid: str,
               mode: str, repetition: int, interval: tuple[float, float],
               recurrent: float = 10.0, group: str | None = None,
               isolated_clean: bool = True) -> Path:
    raw_campaign = campaign(cfg, uuid, recurrent)
    start, end = interval
    command = ["/frozen/bin/cgs_study", "--mode", "campaign", "--case", cfg["case"],
               "--backend", cfg["backend"], "--gate", cfg["gate"], "--body", cfg["body"],
               "--precision", cfg["precision"], "--condition-update", cfg["condition_update"],
               "--tile", str(cfg["tile"]), "--stages", str(cfg["stages"]),
               "--warmups", str(cfg["warmups"]), "--repeats", "1", "--output", "/raw.json"]
    row = {"schema": interference.RUN_SCHEMA, "status": "ok", "mode": mode,
           "config_sha256": config_sha, "config": cfg,
           **{key: cfg.get(key) for key in interference.IDENTITY_FIELDS},
           "scheduler_job_id": job, "scheduler_attempt_id": attempt, "device_uuid": uuid,
           "binary_path": "/frozen/bin/cgs_study", "hostname": "study-host",
           "repetition": repetition, "barrier": {"barrier_id": group} if group else None,
           "packet_started_at_epoch_ns": int((start - 0.1) * 1e9),
           "packet_ended_at_epoch_ns": int((end + 0.1) * 1e9),
           "runner_started_epoch_ns": int(start * 1e9), "runner_ended_epoch_ns": int(end * 1e9),
           "command": command, "operational_total_ms": 13.0 if recurrent == 10 else 3.0 + recurrent,
           "campaign": raw_campaign, "campaign_raw_json": json.dumps(raw_campaign),
           "telemetry": {"before": telemetry(""),
                         "during": [telemetry(uuid, other_activity=not isolated_clean)],
                         "after": telemetry("")}}
    path = root / f"{job}.json"
    path.write_text(json.dumps(row))
    return path


def receipt_for(root: Path, job: str, attempt: str, uuid: str,
                started: float, ended: float, *, clean: bool = True) -> None:
    data = {"job": {"id": job, "state": "succeeded", "cleanup_verified": clean,
                    "attempt": {"id": attempt, "created_at": iso_time(started),
                                "finished_at": iso_time(ended),
                                "execution_context": {"allocations": [
                                    {"resource_id": f"gpu:v100:{uuid}"}]}}}}
    (root / f"{job}.json").write_text(json.dumps(data))


class ConfigAndPathTests(unittest.TestCase):
    def test_matched_two_tile_backend_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'config.json'
            cfg = make_config('a' * 64, 'b' * 64, 'c' * 64)
            cfg['backend'] = 'cohort-direct-two-tile'
            path.write_text(json.dumps(cfg))
            self.assertEqual(interference._load_config(path)[0]['backend'], 'cohort-direct-two-tile')

    def test_config_requires_two_distinct_declared_uuid_targets(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.json"
            cfg = make_config("a" * 64, "b" * 64, "c" * 64)
            path.write_text(json.dumps(cfg))
            self.assertEqual(interference._load_config(path)[0], cfg)
            cfg["concurrent_uuids"] = [UUID_A, UUID_A]
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(RuntimeError, "exactly two distinct"):
                interference._load_config(path)

    def test_barrier_releases_only_two_exact_distinct_scheduler_participants(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            root.mkdir()
            cfg = make_config("a" * 64, "b" * 64, "c" * 64)
            config_path = Path(td) / "case.json"
            config_path.write_text(json.dumps(cfg))
            _, config_sha = interference._load_config(config_path)
            values = []
            failures = []

            def wait(job: str, attempt: str, uuid: str) -> None:
                try:
                    values.append(interference._wait_for_pair(root, "pair-r0", 0, cfg, config_sha,
                                                              job, attempt, uuid, 2))
                except Exception as exc:  # surfaced in the test thread
                    failures.append(exc)

            a = threading.Thread(target=wait, args=("job-a", "attempt-a", UUID_A))
            b = threading.Thread(target=wait, args=("job-b", "attempt-b", UUID_B))
            a.start(); time.sleep(0.05); b.start()
            a.join(3); b.join(3)
            self.assertFalse(a.is_alive() or b.is_alive())
            self.assertFalse(failures)
            self.assertEqual(len(values), 2)
            self.assertEqual(values[0]["participant_uuids"], [UUID_A, UUID_B])
            arrivals = list((root / "runs/barriers/pair-r0/rep-000").glob("arrival-*.json"))
            self.assertEqual(len(arrivals), 2)

    def test_barrier_fails_closed_when_peer_does_not_arrive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            root.mkdir()
            cfg = make_config("a" * 64, "b" * 64, "c" * 64)
            config_path = Path(td) / "case.json"
            config_path.write_text(json.dumps(cfg))
            _, config_sha = interference._load_config(config_path)
            with self.assertRaisesRegex(RuntimeError, "peer missing"):
                interference._wait_for_pair(root, "lonely", 0, cfg, config_sha,
                                            "job-a", "attempt-a", UUID_A, 1)

    def test_barrier_rejects_symlink_study_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            actual = root / "actual"
            actual.mkdir()
            link = root / "link"
            link.symlink_to(actual, target_is_directory=True)
            cfg = make_config("a" * 64, "b" * 64, "c" * 64)
            config_path = root / "case.json"
            config_path.write_text(json.dumps(cfg))
            _, config_sha = interference._load_config(config_path)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                interference._wait_for_pair(link, "pair", 0, cfg, config_sha,
                                            "job-a", "attempt-a", UUID_A, 1)


class RunPacketTests(unittest.TestCase):
    def test_runner_executes_only_with_admission_env_and_retains_raw_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / "fake-cgs"
            script = """#!/usr/bin/env python3
import json, sys
args=sys.argv[1:]
opts={args[i]:args[i+1] for i in range(0,len(args),2)}
out={"schema":"cgs.campaign.v1","status":"ok","accuracy_accepted":True,
"validation_scope":"first-and-last-complete-steps","descriptor_validation":"exact",
"workload_sha256":"WORKLOAD","device_uuid":"DEVICE","case":"CASE",
"backend":"direct","gate":"supplied","body":"product","precision":"fp16",
"condition_update":"incremental","seed":12601,"tile":64,"stages":1,"input":"synthetic",
"replay_input_preparation_ms":0.0,"host_layout_preparation_ms":1.0,
"cold_device_preparation_ms":1.0,"cold_execution_preparation_ms":2.0,
"recurrent_complete_horizon_ms":[10.0],"output_recovery_ms":[1.0]}
out["workload_sha256"]="WORKLOAD"
out["device_uuid"]=__import__("os").environ["CUDA_VISIBLE_DEVICES"]
out["case"]=opts["--case"]
with open(opts["--output"],"w") as f: json.dump(out,f)
"""
            binary.write_text(script.replace("WORKLOAD", "workload-placeholder")
                              .replace("CASE", "case-placeholder"))
            binary.chmod(0o700)
            cfg = make_config(digest(binary.read_bytes()), "b" * 64, "c" * 64)
            config_path = root / "case.json"
            config_path.write_text(json.dumps(cfg))
            # The tiny fake runner echoes its actual command fields. Replace
            # placeholders with this frozen test config before hashing it.
            script = script.replace("WORKLOAD", cfg["workload_sha256"])
            script = script.replace("CASE", cfg["case"]).replace("\"direct\"", json.dumps(cfg["backend"]))
            script = script.replace("\"supplied\"", json.dumps(cfg["gate"]))
            script = script.replace("\"product\"", json.dumps(cfg["body"]))
            script = script.replace("\"fp16\"", json.dumps(cfg["precision"]))
            script = script.replace("\"incremental\"", json.dumps(cfg["condition_update"]))
            script = script.replace("12601", str(cfg["seed"]))
            script = script.replace("64", str(cfg["tile"]))
            script = script.replace("\"stages\":1", f"\"stages\":{cfg['stages']}")
            script = script.replace("\"synthetic\"", json.dumps(cfg["input"]))
            binary.write_text(script)
            binary.chmod(0o700)
            cfg["binary_sha256"] = digest(binary.read_bytes())
            config_path.write_text(json.dumps(cfg))
            old_env = {k: os.environ.get(k) for k in (
                "AUTOBENCH_SCHEDULER_JOB_ID", "AUTOBENCH_SCHEDULER_ATTEMPT_ID", "CUDA_VISIBLE_DEVICES")}
            try:
                os.environ["AUTOBENCH_SCHEDULER_JOB_ID"] = "job-run"
                os.environ["AUTOBENCH_SCHEDULER_ATTEMPT_ID"] = "attempt-run"
                os.environ["CUDA_VISIBLE_DEVICES"] = UUID_A
                fake_snapshot = lambda: telemetry("")
                prior_snapshot = interference._nvidia_snapshot
                interference._nvidia_snapshot = fake_snapshot
                args = Namespace(mode="isolated", config=config_path, binary=binary,
                                 output_dir=root / "output", repetition=0, source_root=None,
                                 shared_study_root=None, barrier_id=None, barrier_timeout=45)
                result = interference.run_packet(args)
                self.assertEqual(result["status"], "ok", result.get("error"))
                self.assertEqual(result["operational_total_ms"], 13.0)
                self.assertEqual(json.loads(result["campaign_raw_json"]), result["campaign"])
                self.assertTrue((root / "output/interference-run.json").is_file())
            finally:
                interference._nvidia_snapshot = prior_snapshot
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_runner_refuses_unadmitted_environment(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root / "fake"
            binary.write_text("#!/bin/sh\nexit 0\n")
            binary.chmod(0o700)
            cfg = make_config(digest(binary.read_bytes()), "b" * 64, "c" * 64)
            config = root / "config.json"
            config.write_text(json.dumps(cfg))
            previous = {k: os.environ.get(k) for k in (
                "AUTOBENCH_SCHEDULER_JOB_ID", "AUTOBENCH_SCHEDULER_ATTEMPT_ID", "CUDA_VISIBLE_DEVICES")}
            try:
                for key in previous:
                    os.environ.pop(key, None)
                args = Namespace(mode="isolated", config=config, binary=binary,
                                 output_dir=root / "out", repetition=0, source_root=None,
                                 shared_study_root=None, barrier_id=None, barrier_timeout=45)
                with self.assertRaisesRegex(RuntimeError, "AUTOBENCH_SCHEDULER"):
                    interference.run_packet(args)
            finally:
                for key, value in previous.items():
                    if value is not None:
                        os.environ[key] = value


class AnalysisTests(unittest.TestCase):
    def prepare(self, root: Path, ratios=(1.0, 1.0), telemetry_clean=True, repetitions=2):
        binary_bytes = b"saved campaign executable"
        source_bytes = b"immutable source closure"
        binary_sha, source_sha = digest(binary_bytes), digest(source_bytes)
        workload_sha = "c" * 64
        cfg = make_config(binary_sha, source_sha, workload_sha, repetitions=repetitions)
        config_path = root / "config.json"
        config_path.write_text(json.dumps(cfg))
        _, config_sha = interference._load_config(config_path)
        receipts = root / "receipts"
        receipts.mkdir()
        closures = root / "closures"
        closures.mkdir()
        builds = root / "builds"
        (builds / binary_sha).mkdir(parents=True)
        (closures / f"{source_sha}.tar.gz").write_bytes(source_bytes)
        (builds / binary_sha / "binary").write_bytes(binary_bytes)
        (builds / binary_sha / "build.json").write_text(json.dumps({
            "archive_sha256": source_sha, "binary_sha256": binary_sha}))
        isolated_paths, concurrent_paths = [], []
        base = 1_800_000_000.0
        for rep in range(repetitions):
            iso_start = base + rep * 20
            iso_job, iso_attempt = f"iso-{rep}", f"iso-attempt-{rep}"
            iso = packet_for(root, cfg, config_sha, iso_job, iso_attempt, UUID_A, "isolated", rep,
                             (iso_start, iso_start + 2), recurrent=10.0,
                             isolated_clean=telemetry_clean)
            isolated_paths.append(str(iso))
            receipt_for(receipts, iso_job, iso_attempt, UUID_A, iso_start - 1, iso_start + 3)
            conc_start = base + 100 + rep * 20
            group = f"pair-r{rep}"
            primary_job, primary_attempt = f"conc-a-{rep}", f"conc-a-attempt-{rep}"
            peer_job, peer_attempt = f"conc-b-{rep}", f"conc-b-attempt-{rep}"
            primary = packet_for(root, cfg, config_sha, primary_job, primary_attempt, UUID_A,
                                 "concurrent", rep, (conc_start, conc_start + 2),
                                 recurrent=10.0 * (ratios[rep] if rep < len(ratios) else ratios[-1]), group=group)
            peer = packet_for(root, cfg, config_sha, peer_job, peer_attempt, UUID_B,
                               "concurrent", rep, (conc_start + 0.1, conc_start + 2.1),
                               recurrent=10.0, group=group)
            concurrent_paths.extend([str(primary), str(peer)])
            receipt_for(receipts, primary_job, primary_attempt, UUID_A, conc_start - 1, conc_start + 3)
            receipt_for(receipts, peer_job, peer_attempt, UUID_B, conc_start - 1, conc_start + 3)
        args = Namespace(config=config_path, receipts_root=receipts, closures_root=closures,
                         builds_root=builds, isolated=isolated_paths, concurrent=concurrent_paths,
                         serialized_policy=None, output=None)
        return cfg, args, receipts, root

    def test_analysis_passes_only_with_raw_identity_receipt_overlap_and_isolated_telemetry(self):
        with tempfile.TemporaryDirectory() as td:
            cfg, args, _, _ = self.prepare(Path(td), repetitions=5)
            result = interference.analyze(args)
            self.assertEqual(result["schema"], "cgs.interference.v1")
            self.assertEqual(result["status"], "inconclusive")
            self.assertEqual(result["device_uuid"], UUID_A)
            self.assertEqual(len(result["isolated_receipts"]), 5)
            self.assertEqual(len(result["concurrent_receipts"]), 5)
            self.assertEqual(len(result["concurrent_partner_receipts"]), 5)
            all_ids = (result["isolated_receipts"] + result["concurrent_receipts"] +
                       result["concurrent_partner_receipts"])
            self.assertEqual(len(all_ids), len(set(all_ids)))
            self.assertEqual(result["measurement"]["median_concurrent_over_isolated_ratio"], 1.0)
            self.assertEqual(result["source_archive_sha256"], cfg["source_archive_sha256"])

    def test_cost_boundary_requires_complete_preparation_components_and_exact_sum(self):
        cfg = make_config("a" * 64, "b" * 64, "c" * 64)
        raw = campaign(cfg, UUID_A)
        raw.pop("replay_input_preparation_ms")
        with self.assertRaisesRegex(RuntimeError, "replay_input_preparation_ms"):
            interference._validate_campaign(raw, cfg, UUID_A)
        raw = campaign(cfg, UUID_A)
        raw["cold_execution_preparation_ms"] = 2.25
        with self.assertRaisesRegex(RuntimeError, "does not equal replay input"):
            interference._validate_campaign(raw, cfg, UUID_A)

    def test_concurrent_attempt_windows_and_nvidia_activity_are_checked(self):
        with tempfile.TemporaryDirectory() as td:
            _, args, receipts, root = self.prepare(Path(td), ratios=(1.2, 1.2), telemetry_clean=False)
            result = interference.analyze(args)
            self.assertEqual(result["status"], "inconclusive")
            # Move the partner Scheduler receipt away from the actual pair
            # interval; its run packet overlaps but the admitted jobs do not.
            peer = json.loads((receipts / "conc-b-0.json").read_text())
            future_start, future_end = 1_800_000_200.0, 1_800_000_202.0
            peer["job"]["attempt"]["created_at"] = iso_time(future_start - 1)
            peer["job"]["attempt"]["finished_at"] = iso_time(future_end + 1)
            (receipts / "conc-b-0.json").write_text(json.dumps(peer))
            peer_packet = next(Path(p) for p in args.concurrent if "conc-b-0" in p)
            packet_data = json.loads(peer_packet.read_text())
            packet_data["runner_started_epoch_ns"] = int(future_start * 1e9)
            packet_data["runner_ended_epoch_ns"] = int(future_end * 1e9)
            packet_data["packet_started_at_epoch_ns"] = int((future_start - 0.1) * 1e9)
            packet_data["packet_ended_at_epoch_ns"] = int((future_end + 0.1) * 1e9)
            peer_packet.write_text(json.dumps(packet_data))
            rejected = interference.analyze(args)
            self.assertEqual(rejected["status"], "inconclusive")
            self.assertIn("did not produce overlapping runner windows", rejected["reason"])

    def test_two_identical_pairs_do_not_hide_zero_variance_uncertainty(self):
        with tempfile.TemporaryDirectory() as td:
            _, args, _, _ = self.prepare(Path(td), ratios=(1.0, 1.0), repetitions=2)
            result = interference.analyze(args)
            self.assertEqual(result["status"], "inconclusive")
            self.assertIsNone(result["measurement"]["paired_log_ratio_t_95_percent_interval"])
            self.assertIn("zero observed variance", result["measurement"]["uncertainty_note"])

    def test_failed_cleanup_receipt_invalidates_analysis(self):
        with tempfile.TemporaryDirectory() as td:
            _, args, receipts, _ = self.prepare(Path(td))
            data = json.loads((receipts / "iso-0.json").read_text())
            data["job"]["cleanup_verified"] = False
            (receipts / "iso-0.json").write_text(json.dumps(data))
            result = interference.analyze(args)
            self.assertEqual(result["status"], "inconclusive")
            self.assertIn("verified cleanup", result["reason"])

    def test_material_difference_is_serialized_only_with_policy_and_receipt_proof(self):
        with tempfile.TemporaryDirectory() as td:
            _, args, _, root = self.prepare(Path(td), ratios=(1.2, 1.2))
            policy = root / "serialized-policy.json"
            policy.write_text(json.dumps({
                "schema": "cgs.serialized-policy.v1", "policy_id": "exclusive-primary-jobs",
                "policy_description": "Run primary UUID GPU jobs one at a time, serializing transfers and recovery.",
                "phase_scope": "whole_scheduler_job", "phases": ["transfer", "recovery"],
                "job_ids": ["iso-0", "iso-1"],
                "source_archive_sha256": args.config and json.loads(args.config.read_text())["source_archive_sha256"],
                "binary_sha256": json.loads(args.config.read_text())["binary_sha256"],
                "workload_sha256": json.loads(args.config.read_text())["workload_sha256"],
            }))
            args.serialized_policy = policy
            result = interference.analyze(args)
            self.assertEqual(result["status"], "inconclusive")
            self.assertIn("Scheduler whole-job windows", result["serialized_policy_evidence"]["proof_scope"])

    def test_missing_closure_or_binary_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            _, args, _, root = self.prepare(Path(td))
            binary_sha = json.loads(args.config.read_text())["binary_sha256"]
            (args.builds_root / binary_sha / "binary").unlink()
            result = interference.analyze(args)
            self.assertEqual(result["status"], "inconclusive")
            self.assertIn("does not exist", result["reason"])


if __name__ == "__main__":
    unittest.main()
