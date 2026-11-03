from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from common import candidate_tree_identity, digest_file, digest_json, write_json  # noqa: E402
from run_dtype_qualification import source_closure_digest  # noqa: E402
from run_performance_packet import (  # noqa: E402
    CHALLENGER_ID,
    INCUMBENT_ID,
    PacketError,
    SANITIZER_SUITE_RELATIVE,
    SanitizerConfiguration,
    _REGISTERED_SANITIZER_HOME,
    _build_variant_map,
    _qualification_groups,
    _read_sanitizer_registration,
    run_packet,
)


UUID = "GPU-21131915-1488-23af-38dd-1743ae1f5cc8"
CONTEXT = {
    "schema": "cellerator.factorized_interactions.outer_execution_context.v1",
    "status": "observed",
    "scheduler_job_id": "autobench-job-1",
    "scheduler_attempt_id": "autobench-attempt-1",
    "device_uuid": UUID,
    "cgroup": "/system.slice/device-scheduler-attempt-test.service",
    "unit_name": "device-scheduler-attempt-test.service",
}


def _write_source_fixture(root: Path) -> Path:
    files = {
        "CMakeLists.txt": b"cmake_minimum_required(VERSION 3.26)\n",
        "bench/factorized_interactions/CMakeLists.txt": b"project(test)\n",
        "bench/factorized_interactions/build_runner.py": b"pass\n",
        "bench/factorized_interactions/native_runner.cu": b"// runner\n",
        "include/Cellerator/modeling/cuda/factorized_interaction.hh": b"// API\n",
        "src/modeling/cuda/factorized_interaction.cu": b"// CUDA\n",
        "tests/factorized_interactions/reference.py": b"# oracle\n",
        "tests/factorized_interactions/workloads.py": b"# workloads\n",
        "bench/factorized_interactions/candidate_integration/build_candidate_runner.py": b"# builder\n",
        "bench/factorized_interactions/candidate_integration/PROVENANCE.json": b"{}\n",
        "bench/factorized_interactions/candidate_integration/candidate_runner.cu": b"// candidate\n",
        "bench/factorized_interactions/run_sanitizer_suite.py":
            b'COMPUTE_SANITIZER = "/tmp/fake-compute-sanitizer"\n',
    }
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    byte_entries, source_closure_hash = _build_source_entries(root)
    entries = [{"path": row["path"], "sha256": row["sha256"]}
               for row in byte_entries]
    manifest = root.parent / "source-closure.json"
    write_json(manifest, {
        "source_closure_sha256": source_closure_hash,
        "files": entries,
    }, exclusive=True)
    return manifest


def _build_source_entries(snapshot: Path) -> tuple[list[dict], str]:
    rows = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            rows.append({
                "path": path.relative_to(snapshot).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
    source_hash, _ = source_closure_digest(rows)
    return rows, source_hash


class PerformancePacketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.source_manifest = _write_source_fixture(self.source)
        self.output = self.root / "managed-output"
        self.output.mkdir()
        self.site_policy = self.root / "site-policy.json"
        self.site_policy.write_text("{}\n", encoding="utf-8")
        self.nvcc = self.root / "nvcc"
        self.nvcc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.nvcc.chmod(0o755)
        self.kda = self.root / "kda"
        self.kda.mkdir()
        self.sanitizer_home = self.root / "compute-sanitizer-support"
        self.sanitizer_home.mkdir()
        (self.sanitizer_home / "support.dat").write_bytes(b"registered support")
        self.sanitizer_binary = self.root / "compute-sanitizer"
        self.sanitizer_binary.write_bytes(b"fake sanitizer binary")
        self.sanitizer = SanitizerConfiguration(
            home=self.sanitizer_home,
            binary=self.sanitizer_binary,
            binary_sha256=digest_file(self.sanitizer_binary),
            support_tree_sha256=candidate_tree_identity(self.sanitizer_home)[0],
            expected_version="2025.2.1.0",
        )

    def test_variant_map_declares_packet_and_exact_direct_modes(self) -> None:
        mapping = _build_variant_map(self.source, self.nvcc, "packet-test")
        self.assertEqual(mapping["comparison"]["strongest_direct_control"], INCUMBENT_ID)
        self.assertTrue(mapping["comparison"]["same_source_runner_required"])
        direct = mapping["variants"][INCUMBENT_ID]
        self.assertEqual(direct["route"], "direct")
        self.assertIn("direct_unique_output", direct["allowed_implementations"])
        packet = mapping["variants"][CHALLENGER_ID]
        self.assertEqual(packet["candidate_argument"], "packet")
        self.assertEqual(packet["win_backend"], "packet")

    def test_qualification_groups_require_same_runner_and_library_bytes(self) -> None:
        from run_performance_packet import CandidateBuild

        path = self.root / "unused"
        first = CandidateBuild("a", path, path, "1" * 64, "2" * 64, "s" * 64, {},
                               path, path, "3" * 64, "4" * 64)
        second = CandidateBuild("b", path, path, "1" * 64, "2" * 64, "s" * 64, {},
                                path, path, "3" * 64, "4" * 64)
        third = CandidateBuild("c", path, path, "1" * 64, "3" * 64, "s" * 64, {},
                               path, path, "5" * 64, "6" * 64)
        groups = _qualification_groups({"a": first, "b": second, "c": third})
        self.assertEqual(sorted(len(items) for _, items in groups), [1, 2])

    def test_sanitizer_config_requires_registered_inherited_selector(self) -> None:
        suite = self.source / SANITIZER_SUITE_RELATIVE
        suite.write_text(
            f'COMPUTE_SANITIZER = "{self.sanitizer_binary}"\n', encoding="utf-8",
        )
        with patch("run_performance_packet._REGISTERED_SANITIZER_HOME", self.sanitizer_home), \
                patch.dict(os.environ, {
                    "NVCOMPILER_COMPUTE_SANITIZER_HOME": str(self.sanitizer_home),
                }, clear=False):
            config = _read_sanitizer_registration(suite)
        self.assertEqual(config.home, self.sanitizer_home.resolve())
        self.assertEqual(config.binary, self.sanitizer_binary.resolve())
        self.assertEqual(config.binary_sha256, digest_file(self.sanitizer_binary))
        self.assertEqual(
            config.support_tree_sha256, candidate_tree_identity(self.sanitizer_home)[0],
        )
        with patch("run_performance_packet._REGISTERED_SANITIZER_HOME", self.sanitizer_home), \
                patch.dict(os.environ, {
                    "NVCOMPILER_COMPUTE_SANITIZER_HOME": str(self.root / "other"),
                }, clear=False):
            with self.assertRaisesRegex(PacketError, "differs from the oracle-registered path"):
                _read_sanitizer_registration(suite)

    def test_packet_flow_is_cpu_mockable_and_preserves_outer_markers(self) -> None:
        packet_id = "packet-flow-test"
        packet_dir = self.output / f"factorized-{packet_id}"
        prepared_results: dict[str, dict] = {}
        observed_envs: list[dict[str, str]] = []

        def make_manifest(source, closure, _dependencies):
            entries = []
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    entries.append({
                        "path": path.relative_to(source).as_posix(),
                        "sha256": digest_file(path),
                    })
            return {
                "schema_version": 1, "files": [row["path"] for row in entries],
                "dependencies": [],
                "provenance": {
                    "source_closure_manifest_sha256": digest_file(closure),
                    "source_closure_sha256": json.loads(closure.read_text())[
                        "source_closure_sha256"],
                    "kda_source_tree_sha256": candidate_tree_identity(source)[0],
                    "file_count": len(entries),
                },
            }

        def make_task(**kwargs):
            task = {
                "schema_version": 1, "task_id": kwargs["task_id"],
                "target": {"kind": "cuda", "expected_device_uuid": UUID},
                "factorized_interaction": {"expected_device_uuid": UUID},
                "trusted_inputs": [],
            }
            write_json(kwargs["output"], task, exclusive=True)
            return task

        def register(**kwargs):
            workspace = kwargs["workspace"]
            snapshot_hash, _entries = candidate_tree_identity(kwargs["source_dir"])
            snapshot_dir = workspace / "runs" / "kda-experiments" / "candidates" / kwargs["candidate_id"] / snapshot_hash
            snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(kwargs["source_dir"], snapshot_dir)
            row = {
                "record_type": "candidate", "candidate_id": kwargs["candidate_id"],
                "snapshot_sha256": snapshot_hash, "snapshot_dir": str(snapshot_dir),
                "manifest_sha256": digest_file(kwargs["manifest_path"]),
            }
            with (workspace / "candidates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            return {"candidate_id": kwargs["candidate_id"], "status": "registered"}

        def fake_build(workspace: Path, candidate_id: str) -> dict:
            registry = [json.loads(line) for line in (workspace / "candidates.jsonl").read_text().splitlines()]
            record = next(row for row in registry if row["candidate_id"] == candidate_id)
            snapshot = Path(record["snapshot_dir"])
            source_rows, source_hash = _build_source_entries(snapshot)
            build = workspace / "runs" / "kda-experiments" / "builds" / candidate_id / "fingerprint" / "build"
            native = build / "native_candidate" / "native_build"
            binaries = native / "cmake"
            binaries.mkdir(parents=True, exist_ok=True)
            (binaries / "runner").write_bytes(b"same prepared runner")
            (binaries / "baseline_runner").write_bytes(b"same prepared baseline runner")
            (binaries / "libfactorized.so").write_bytes(b"same native library")
            native_manifest = native / "factorized_interactions_build_manifest.json"
            baseline_manifest = native / "factorized_interactions_baseline_build_manifest.json"
            manifest = {
                "schema": "cellerator.factorized_interactions.build.v1",
                "architecture": "sm_70", "fast_math": False,
                "source_closure_sha256": source_hash,
                "source_files": source_rows,
                "binaries": {
                    "runner": {
                        "path": "cmake/runner",
                        "sha256": digest_file(binaries / "runner"),
                    },
                    "factorized_interaction_cuda": {
                        "path": "cmake/libfactorized.so",
                        "sha256": digest_file(binaries / "libfactorized.so"),
                    },
                },
            }
            write_json(native_manifest, manifest, exclusive=True)
            baseline_metadata = {
                **manifest,
                "binaries": {
                    "runner": {
                        "path": "cmake/baseline_runner",
                        "sha256": digest_file(binaries / "baseline_runner"),
                    },
                    "factorized_interaction_cuda": manifest["binaries"][
                        "factorized_interaction_cuda"],
                },
                "candidate_runner": {
                    "manifest_path": native_manifest.name,
                    "manifest_sha256": digest_file(native_manifest),
                    "binary_path": "cmake/runner",
                    "binary_sha256": digest_file(binaries / "runner"),
                },
            }
            write_json(baseline_manifest, baseline_metadata, exclusive=True)
            artifacts = []
            for path in (native_manifest, baseline_manifest, binaries / "runner",
                         binaries / "baseline_runner", binaries / "libfactorized.so"):
                artifacts.append({
                    "path": path.relative_to(build).as_posix(),
                    "sha256": digest_file(path),
                })
            result = {
                "candidate_id": candidate_id, "status": "ok", "directory": str(build),
                "fingerprint": "f" * 64, "task_contract_sha256": "e" * 64,
                "artifacts": artifacts,
                "manifest_path": str(build / "build.json"),
                "manifest_sha256": "d" * 64,
            }
            prepared_results[candidate_id] = result
            return result

        def invoke(argv, **kwargs):
            observed_envs.append(kwargs["env"])
            if len(argv) > 1 and str(argv[1]).endswith("run_sanitizer_suite.py"):
                events.append("sanitizer")
                if sanitizer_failure == "exit":
                    return subprocess.CompletedProcess(argv, 1, stdout="", stderr="injected strict failure")
                baseline_manifest = Path(argv[argv.index("--build-manifest") + 1])
                baseline = json.loads(baseline_manifest.read_text())
                candidate_link = baseline["candidate_runner"]
                candidate_manifest = baseline_manifest.parent / candidate_link["manifest_path"]
                candidate = json.loads(candidate_manifest.read_text())
                candidate_runner = baseline_manifest.parent / candidate_link["binary_path"]
                report_path = Path(argv[argv.index("--output") + 1])
                report_count = 55 if sanitizer_failure == "bad-report" else 56
                summary_count = 56 if sanitizer_failure == "bad-report" else report_count
                report = {
                    "schema": "cellerator.factorized_interactions.sanitizer_suite.v1",
                    "scheduler_job_id": kwargs["env"]["AUTOBENCH_SCHEDULER_JOB_ID"],
                    "device_uuid": UUID,
                    "compute_sanitizer": str(self.sanitizer.binary),
                    "compute_sanitizer_version": "compute-sanitizer 2025.2.1.0",
                    "build_manifest_sha256": digest_file(baseline_manifest),
                    "source_closure_sha256": baseline["source_closure_sha256"],
                    "runner_binary_sha256": digest_file(
                        baseline_manifest.parent / baseline["binaries"]["runner"]["path"]),
                    "candidate_build_manifest_sha256": digest_file(candidate_manifest),
                    "candidate_runner_binary_sha256": digest_file(candidate_runner),
                    "timed": False, "case_count": report_count, "passed": report_count,
                }
                write_json(report_path, report, exclusive=True)
                return subprocess.CompletedProcess(
                    argv, 0,
                    stdout=json.dumps({
                        "suite": report["schema"], "case_count": summary_count,
                        "passed": summary_count, "timed": False,
                    }), stderr="",
                )
            command_text = " ".join(str(item) for item in argv)
            if " prepare " in f" {command_text} ":
                workspace = Path(argv[argv.index("--workspace") + 1])
                ids = [argv[argv.index("prepare") + 1], argv[argv.index("prepare") + 2]]
                results = [fake_build(workspace, candidate_id) for candidate_id in ids]
                stdout = json.dumps({"status": "ok", "results": results})
                return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
            if str(argv[1]).endswith("in_job_kda.py"):
                events.append("evaluate")
                workspace = Path(argv[argv.index("--workspace") + 1])
                run_id = str(uuid.uuid4())
                summary_path = workspace / "runs" / "kda-experiments" / "attempts" / run_id / "summary.json"
                summary_path.parent.mkdir(parents=True)
                write_json(summary_path, {
                    "status": "complete", "run_id": run_id,
                    "builds": {key: value for key, value in prepared_results.items()},
                }, exclusive=True)
                stdout = json.dumps({"status": "ok", "run_id": run_id,
                                     "summary_path": str(summary_path)})
                return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
            self.fail(f"unexpected subprocess argv: {argv}")

        dtype_calls = []
        events: list[str] = []
        sanitizer_failure: str | None = None

        def dtype_runner(**kwargs):
            events.append("dtype")
            dtype_calls.append(kwargs)
            report = {
                "status": "pass", "passed_case_count": 32,
                "build": {"runner_sha256": digest_file(kwargs["runner"]),
                          "build_manifest_sha256": digest_file(kwargs["build_manifest"])},
            }
            kwargs["output"].parent.mkdir(parents=True, exist_ok=True)
            write_json(kwargs["output"], report, exclusive=True)
            return report

        def selection(task, workspace, run_id, variant_map, output):
            result = {
                "status": "conditional", "run_id": run_id,
                "outer_receipt_verification": "pending",
                "production_route_changed": False,
            }
            write_json(output, result, exclusive=True)
            return result

        with patch.dict(os.environ, {
            "DEVICE_SCHEDULER_JOB_ID": CONTEXT["scheduler_job_id"],
            "DEVICE_SCHEDULER_ATTEMPT_ID": CONTEXT["scheduler_attempt_id"],
            "AUTOBENCH_SCHEDULER_JOB_ID": CONTEXT["scheduler_job_id"],
            "AUTOBENCH_SCHEDULER_ATTEMPT_ID": CONTEXT["scheduler_attempt_id"],
            "CUDA_VISIBLE_DEVICES": UUID,
            "PYTHONPATH": "/original/path",
            "NVCOMPILER_COMPUTE_SANITIZER_HOME": str(_REGISTERED_SANITIZER_HOME),
        }, clear=False):
            result = run_packet(
                source=self.source, source_manifest=self.source_manifest,
                expected_uuid=UUID, output_root=self.output,
                site_policy=self.site_policy, nvcc=self.nvcc, kda_root=self.kda,
                task_id=packet_id, candidate_win_cases=["packet_a4_m32_u16"],
                max_seconds=120.0,
                context_reader=lambda _uuid: dict(CONTEXT),
                invoke=invoke, manifest_factory=make_manifest,
                task_factory=make_task, registration=register,
                dtype_runner=dtype_runner, selection_runner=selection,
                sanitizer_config_reader=lambda _suite: self.sanitizer,
                task_loader=lambda _path, _root: None,
            )

        self.assertEqual(result["status"], "conditional")
        self.assertEqual(result["outer_execution_context"]["scheduler_attempt_id"],
                         CONTEXT["scheduler_attempt_id"])
        self.assertEqual(result["outer_receipt_verification"].split(";")[0], "pending")
        self.assertEqual(len(dtype_calls), 1)
        self.assertEqual(events, ["sanitizer", "dtype", "evaluate"])
        self.assertEqual(dtype_calls[0]["reference_cache_dir"],
                         dtype_calls[0]["artifact_dir"] / "reference-cache")
        self.assertEqual({env["DEVICE_SCHEDULER_JOB_ID"] for env in observed_envs},
                         {CONTEXT["scheduler_job_id"]})
        self.assertEqual({env["AUTOBENCH_SCHEDULER_ATTEMPT_ID"] for env in observed_envs},
                         {CONTEXT["scheduler_attempt_id"]})
        self.assertTrue(all(env["PYTHONPATH"].startswith(str(self.kda) + os.pathsep)
                            for env in observed_envs))
        self.assertEqual({env["NVCOMPILER_COMPUTE_SANITIZER_HOME"] for env in observed_envs},
                         {str(_REGISTERED_SANITIZER_HOME)})
        sanitizer_index = json.loads(
            (packet_dir / "strict-sanitizer" / "index.json").read_text()
        )
        self.assertEqual(sanitizer_index["status"], "pass")
        self.assertEqual(sanitizer_index["groups"][0]["passed"], 56)
        task_inputs = json.loads((packet_dir / "task.json").read_text())["trusted_inputs"]
        task_inputs_by_id = {item["id"]: item for item in task_inputs}
        self.assertEqual(task_inputs_by_id["strict_sanitizer_suite"]["role"], "validator")
        self.assertEqual(
            task_inputs_by_id["strict_sanitizer_suite"]["path"],
            str((self.source / SANITIZER_SUITE_RELATIVE).resolve()),
        )
        self.assertEqual(task_inputs_by_id["compute_sanitizer_binary"]["role"], "toolchain")
        self.assertEqual(task_inputs_by_id["compute_sanitizer_support_files"]["role"], "toolchain")
        evaluate_command = json.loads(
            (packet_dir / "commands" / "guarded-kda-evaluate" / "command.json").read_text()
        )["argv"]
        self.assertEqual(Path(evaluate_command[1]).name, "in_job_kda.py")
        self.assertTrue((packet_dir / "conditional-selection.json").is_file())
        self.assertTrue((packet_dir / "packet.json").is_file())

        before_dtype_calls = len(dtype_calls)
        before_evaluations = events.count("evaluate")
        def run_failing_packet(packet_id: str, failure: str, error_pattern: str) -> Path:
            nonlocal sanitizer_failure
            sanitizer_failure = failure
            with patch.dict(os.environ, {
                "DEVICE_SCHEDULER_JOB_ID": CONTEXT["scheduler_job_id"],
                "DEVICE_SCHEDULER_ATTEMPT_ID": CONTEXT["scheduler_attempt_id"],
                "AUTOBENCH_SCHEDULER_JOB_ID": CONTEXT["scheduler_job_id"],
                "AUTOBENCH_SCHEDULER_ATTEMPT_ID": CONTEXT["scheduler_attempt_id"],
                "CUDA_VISIBLE_DEVICES": UUID,
                "PYTHONPATH": "/original/path",
                "NVCOMPILER_COMPUTE_SANITIZER_HOME": str(_REGISTERED_SANITIZER_HOME),
            }, clear=False):
                with self.assertRaisesRegex(PacketError, error_pattern):
                    run_packet(
                        source=self.source, source_manifest=self.source_manifest,
                        expected_uuid=UUID, output_root=self.output,
                        site_policy=self.site_policy, nvcc=self.nvcc, kda_root=self.kda,
                        task_id=packet_id, candidate_win_cases=["packet_a4_m32_u16"],
                        max_seconds=120.0, context_reader=lambda _uuid: dict(CONTEXT),
                        invoke=invoke, manifest_factory=make_manifest,
                        task_factory=make_task, registration=register,
                        dtype_runner=dtype_runner, selection_runner=selection,
                        sanitizer_config_reader=lambda _suite: self.sanitizer,
                        task_loader=lambda _path, _root: None,
                    )
            return self.output / f"factorized-{packet_id}"

        failed_packet = run_failing_packet(
            "packet-sanitizer-fail", "exit", "strict-sanitizer.*exited with status 1",
        )
        self.assertEqual(len(dtype_calls), before_dtype_calls)
        self.assertEqual(events.count("evaluate"), before_evaluations)
        failure_record = json.loads((failed_packet / "packet.json").read_text())
        self.assertEqual(failure_record["status"], "failed")
        self.assertFalse((failed_packet / "commands" / "guarded-kda-evaluate").exists())

        short_packet = run_failing_packet(
            "packet-sanitizer-short", "bad-report",
            "strict sanitizer report identity/count mismatch for case_count",
        )
        self.assertEqual(len(dtype_calls), before_dtype_calls)
        self.assertEqual(events.count("evaluate"), before_evaluations)
        short_record = json.loads((short_packet / "packet.json").read_text())
        self.assertEqual(short_record["status"], "failed")
        self.assertFalse((short_packet / "commands" / "guarded-kda-evaluate").exists())

    def test_invalid_outer_context_prevents_packet_side_effects(self) -> None:
        packet_dir = self.output / "factorized-not-admitted"
        with self.assertRaisesRegex(PacketError, "allocation"):
            run_packet(
                source=self.source, source_manifest=self.source_manifest,
                expected_uuid=UUID, output_root=self.output,
                site_policy=self.site_policy, nvcc=self.nvcc, kda_root=self.kda,
                task_id="not-admitted", candidate_win_cases=["case"],
                max_seconds=10.0,
                context_reader=lambda _uuid: (_ for _ in ()).throw(PacketError("allocation absent")),
            )
        self.assertFalse(packet_dir.exists())

    def test_packet_output_must_not_overlap_immutable_source(self) -> None:
        with self.assertRaisesRegex(PacketError, "disjoint"):
            run_packet(
                source=self.source, source_manifest=self.source_manifest,
                expected_uuid=UUID, output_root=self.source,
                site_policy=self.site_policy, nvcc=self.nvcc, kda_root=self.kda,
                task_id="overlap", candidate_win_cases=["case"],
                max_seconds=10.0, context_reader=lambda _uuid: dict(CONTEXT),
                sanitizer_config_reader=lambda _suite: self.sanitizer,
            )


if __name__ == "__main__":
    unittest.main()
