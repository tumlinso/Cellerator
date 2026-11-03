from __future__ import annotations

import csv
import hashlib
import gzip
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import tarfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import recorded_engine as replay


UUID = "GPU-6c1cac7f-a360-0aef-ba98-2828bfd1db1a"
KDA_ROOT = Path("/home/tumlinson/kda")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _source_closure(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        name = str(row["path"])
        size = int(row["bytes"])
        expected = str(row["sha256"])
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(size.to_bytes(8, "little"))
        digest.update(bytes.fromhex(expected))
    return digest.hexdigest()


def _dataset(source_root: Path, candidate_manifest: Path,
             baseline_manifest: Path, *, full_inventory: bool = False) -> dict[str, object]:
    catalog_path = source_root / "bench" / "factorized_interactions" / "workloads.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    validator_path = catalog_path.with_name("validate_runner.py")
    evidence_root = source_root.parent / "retained-recorded-evidence"
    if full_inventory:
        evidence_root.mkdir(parents=True, exist_ok=False)
    refs: dict[tuple[str, str, str], dict[str, str]] = {}

    def add_input(name: str, role: str, size: int, *, shared_path: Path | None = None) -> dict[str, str]:
        path = shared_path or (evidence_root / name)
        if shared_path is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = (f"synthetic retained evidence {name}\n".encode("utf-8") or b"x")
            repetitions = (size + len(payload) - 1) // len(payload)
            path.write_bytes((payload * repetitions)[:size])
        digest = replay.sha256_file(path)
        identity = (str(path.resolve()), digest, role)
        if identity not in refs:
            identifier = "fixture-" + hashlib.sha256(
                (role + "\0" + identity[0]).encode("utf-8")).hexdigest()[:24]
            refs[identity] = {"id": identifier, "path": identity[0],
                              "sha256": digest, "role": role}
        return {"path": identity[0], "sha256": digest}

    trusted_inputs = [
        {
            "id": "test-fixed-workload-catalog",
            "path": str(catalog_path.resolve()),
            "sha256": replay.sha256_file(catalog_path),
            "role": "fixed_workload_catalog",
        },
        {
            "id": "test-fixed-sanitizer-validator",
            "path": str(validator_path.resolve()),
            "sha256": replay.sha256_file(validator_path),
            "role": "fixed_sanitizer_partition_validator",
        },
    ]
    timing_reports: list[dict[str, str]] = []
    cases = []
    for case_index, spec in enumerate(catalog["cases"]):
        phases = {}
        for phase_index, phase in enumerate(replay.PHASES):
            blocks = []
            for block_index in range(replay.BLOCKS_PER_PHASE):
                arms = {}
                for arm_index, arm_name in enumerate(replay.ARMS):
                    arms[arm_name] = {
                        "primary_ms": 1.0 + case_index * 0.01 + phase_index * 0.1 +
                                     block_index * 0.001 + arm_index * 0.2,
                        "raw_metrics_sha256": hashlib.sha256(
                            f"raw:{case_index}:{phase_index}:{block_index}:{arm_index}".encode()).hexdigest(),
                        "validation_summary_sha256": hashlib.sha256(
                            f"summary:{arm_name}".encode()).hexdigest(),
                        "original_report_sha256": hashlib.sha256(
                            f"report:{case_index}:{phase_index}:{block_index}:{arm_index}".encode()).hexdigest(),
                        "scheduler_job_id": f"scheduler-job-{case_index}-{phase_index}-{block_index}",
                        "scheduler_attempt_id": f"scheduler-attempt-{case_index}-{phase_index}-{block_index}",
                        "autobench_attempt_id": f"autobench-attempt-{case_index}-{phase_index}-{block_index}",
                        "device_uuid": UUID,
                    }
                    if full_inventory:
                        arms[arm_name]["helper_evidence"] = add_input(
                            f"arm-{case_index}-{phase_index}-{block_index}-{arm_index}-helper.json",
                            "timed_helper_evidence", 4 * 1024)
                        arms[arm_name]["validated_record"] = add_input(
                            f"arm-{case_index}-{phase_index}-{block_index}-{arm_index}-validated.json",
                            "pre_cleanup_validation_record", 16 * 1024)
                        arms[arm_name]["validator_stdout"] = add_input(
                            f"arm-{case_index}-{phase_index}-{block_index}-{arm_index}-stdout.log",
                            "validator_stdout", 1024)
                        # Real collection deduplicates shared empty stderr files.
                        # Alias 26 arms to one immutable empty-log reference.
                        stderr_path = None
                        if case_index == 0 and phase_index == 0 and block_index == 0 and arm_index == 0:
                            stderr_path = evidence_root / "shared-empty-stderr.log"
                            stderr_path.write_bytes(b"")
                        elif arm_index == 0 and case_index * 14 + phase_index * 7 + block_index < 26:
                            stderr_path = evidence_root / "shared-empty-stderr.log"
                        arms[arm_name]["validator_stderr"] = add_input(
                            f"arm-{case_index}-{phase_index}-{block_index}-{arm_index}-stderr.log",
                            "validator_stderr", 1024, shared_path=stderr_path)
                blocks.append({"block_index": block_index, "arms": arms})
            phases[phase] = blocks
            if full_inventory:
                report = add_input(f"timing-{case_index}-{phase_index}-report.json",
                                   "timing_partition_report", 256 * 1024)
                receipt = add_input(f"timing-{case_index}-{phase_index}-receipt.json",
                                     "finished_autobench_scheduler_receipt", 8 * 1024)
                pointer = add_input(f"timing-{case_index}-{phase_index}-pointer.json",
                                    "finished_autobench_pointer", 2 * 1024)
                timing_reports.append({"workload": spec["id"], "phase": phase,
                                       "report": report, "receipt": receipt,
                                       "pointer": pointer})
        cases.append({
            "id": spec["id"],
            "workload_descriptor": spec,
            "workload_sha256": replay.digest_json(spec),
            "phases": phases,
        })
    if full_inventory:
        add_input("qualification-bundle.json", "qualification_bundle", 64 * 1024)
        for index in range(9):
            add_input(f"qualification-{index}-report.json", "qualification_report", 64 * 1024)
            add_input(f"qualification-{index}-receipt.json", "qualification_receipt", 8 * 1024)
            add_input(f"qualification-{index}-pointer.json", "qualification_pointer", 2 * 1024)
        trusted_inputs.extend(refs.values())
    candidate_native = replay.read_json(candidate_manifest)
    baseline_native = replay.read_json(baseline_manifest)
    identity = {
        "device_uuid": UUID,
        "source_closure_sha256": candidate_native["source_closure_sha256"],
        "candidate_runner_sha256": candidate_native["binaries"]["runner"]["sha256"],
        "candidate_manifest_sha256": replay.sha256_file(candidate_manifest),
        "baseline_runner_sha256": baseline_native["binaries"]["runner"]["sha256"],
        "baseline_manifest_sha256": replay.sha256_file(baseline_manifest),
        "library_sha256": candidate_native["binaries"]["factorized_interaction_cuda"]["sha256"],
        "validator_sha256": replay.sha256_file(source_root / "bench" / "factorized_interactions" / "validate_runner.py"),
        "workload_catalog": {
            "path": "bench/factorized_interactions/workloads.json",
            "sha256": replay.sha256_file(catalog_path),
        },
    }
    return {
        "schema": replay.SCHEMA,
        "qualification_status": replay.QUALIFICATION_STATUS,
        "qualification_bundle_sha256": "a" * 64,
        "identity": identity,
        "trusted_inputs": trusted_inputs,
        "timing_reports": timing_reports,
        "coverage": {"workloads": 13, "phases_per_workload": 2,
                     "paired_blocks_per_phase": 7, "arms_per_block": 3,
                     "unique_cuda_observations": 546},
        "cases": cases,
    }


def _write_sparse(path: Path, prefix: bytes, size: int) -> None:
    with path.open("wb") as stream:
        stream.write(prefix)
        if size > len(prefix):
            stream.seek(size - 1)
            stream.write(b"\0")


def _native_files(
    root: Path, *, candidate_bytes: int = 0, baseline_bytes: int = 0,
    library_bytes: int = 0, full_source_inventory: bool = False,
) -> tuple[Path, Path, Path]:
    source = root / "source"
    (source / "src").mkdir(parents=True)
    (source / "src" / "model.cc").write_bytes(b"immutable native source\n")
    workload_path = source / "bench" / "factorized_interactions" / "workloads.json"
    workload_path.parent.mkdir(parents=True)
    catalog_source = Path(__file__).resolve().parents[4] / "bench" / "factorized_interactions" / "workloads.json"
    shutil.copyfile(catalog_source, workload_path)
    validator_path = workload_path.with_name("validate_runner.py")
    validator_source = catalog_source.with_name("validate_runner.py")
    shutil.copyfile(validator_source, validator_path)
    source_file = source / "src" / "model.cc"
    if full_source_inventory:
        existing = [source_file, workload_path, validator_path]
        target_bytes = 20_035_237
        target_files = 3_979
        remaining_files = target_files - len(existing)
        remaining_bytes = target_bytes - sum(path.stat().st_size for path in existing)
        if remaining_files <= 0 or remaining_bytes < remaining_files:
            raise AssertionError("realistic synthetic source inventory target is too small")
        payload_root = source / "src" / "captured_translation_units"
        payload_root.mkdir(parents=True)
        base_size, extra = divmod(remaining_bytes, remaining_files)
        import random
        for index in range(remaining_files):
            size = base_size + (1 if index < extra else 0)
            rng = random.Random(index)
            block = rng.randbytes(min(1024, size))
            payload = (block * ((size + len(block) - 1) // len(block)))[:size]
            (payload_root / f"unit_{index:04d}.cc").write_bytes(payload)
        existing = [path for path in source.rglob("*") if path.is_file()]
        if len(existing) != target_files or sum(path.stat().st_size for path in existing) != target_bytes:
            raise AssertionError("synthetic source inventory differs from requested file/byte totals")
    else:
        existing = [source_file, workload_path, validator_path]
    source_rows = [{"path": path.relative_to(source).as_posix(), "bytes": path.stat().st_size,
                    "sha256": replay.sha256_file(path)} for path in existing]
    source_rows.sort(key=lambda row: row["path"])
    closure_sha = _source_closure(source_rows)

    native = root / "prepared" / "native"
    binaries = native / "cmake"
    binaries.mkdir(parents=True)
    candidate_runner = binaries / "candidate_runner"
    baseline_runner = binaries / "baseline_runner"
    library = binaries / "libfactorized_interaction.a"
    if candidate_bytes:
        _write_sparse(candidate_runner, b"candidate runner bytes\n", candidate_bytes)
    else:
        candidate_runner.write_bytes(b"candidate runner bytes\n")
    if baseline_bytes:
        _write_sparse(baseline_runner, b"baseline runner bytes\n", baseline_bytes)
    else:
        baseline_runner.write_bytes(b"baseline runner bytes\n")
    if library_bytes:
        _write_sparse(library, b"shared native CUDA library bytes\n", library_bytes)
    else:
        library.write_bytes(b"shared native CUDA library bytes\n")
    runner_info = {"path": "cmake/candidate_runner",
                   "bytes": candidate_runner.stat().st_size,
                   "sha256": replay.sha256_file(candidate_runner)}
    baseline_info = {"path": "cmake/baseline_runner",
                     "bytes": baseline_runner.stat().st_size,
                     "sha256": replay.sha256_file(baseline_runner)}
    library_info = {"path": "cmake/libfactorized_interaction.a",
                    "bytes": library.stat().st_size,
                    "sha256": replay.sha256_file(library)}

    candidate_manifest = native / "factorized_interactions_build_manifest.json"
    baseline_manifest = native / "factorized_interactions_baseline_build_manifest.json"
    candidate_data = {
        "schema": replay.NATIVE_BUILD_SCHEMA,
        "architecture": "sm_70",
        "fast_math": False,
        "source_closure_sha256": closure_sha,
        "source_files": source_rows,
        "binaries": {"runner": runner_info,
                      "factorized_interaction_cuda": library_info},
    }
    _write_json(candidate_manifest, candidate_data)
    baseline_data = {
        **candidate_data,
        "binaries": {"runner": baseline_info,
                      "factorized_interaction_cuda": library_info},
        "candidate_runner": {
            "manifest_path": candidate_manifest.name,
            "manifest_sha256": replay.sha256_file(candidate_manifest),
            "binary_path": runner_info["path"],
            "binary_sha256": runner_info["sha256"],
        },
    }
    _write_json(baseline_manifest, baseline_data)
    return source, candidate_manifest, baseline_manifest


def _native_archive(root: Path, source: Path,
                    candidate_manifest: Path, *,
                    archive_subdirectory: str = "accepted-native-source") -> tuple[Path, Path]:
    """Build a deterministic synthetic accepted-source archive for CPU tests."""
    manifest_data = replay.read_json(candidate_manifest)
    rows = manifest_data["source_files"]
    archive_dir = root / archive_subdirectory
    archive_dir.mkdir(parents=True, exist_ok=False)
    archive = archive_dir / "native-performance-filtered.tar"
    with tarfile.open(archive, mode="w", format=tarfile.PAX_FORMAT) as output:
        for row in rows:
            member = tarfile.TarInfo(row["path"])
            member.size = row["bytes"]
            member.mode = 0o444
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = 0
            with (source / row["path"]).open("rb") as payload:
                output.addfile(member, payload)
    archive_identity = {
        "schema": replay.NATIVE_ARCHIVE_SCHEMA,
        "archive_path": archive.name,
        "archive_sha256": replay.sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "file_count": len(rows),
        "source_bytes": sum(row["bytes"] for row in rows),
        "source_closure_sha256": manifest_data["source_closure_sha256"],
        "files": rows,
    }
    index = archive_dir / "native-performance-filtered-manifest.json"
    _write_json(index, archive_identity)
    return archive, index


class RecordedEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="recorded-kda-test-")
        self.root = Path(self.temp.name)
        self.source, self.candidate_manifest, self.baseline_manifest = _native_files(self.root)
        self.archive, self.archive_index = _native_archive(
            self.root, self.source, self.candidate_manifest)
        self.dataset = _dataset(self.source, self.candidate_manifest, self.baseline_manifest)
        self.dataset_path = self.root / "dataset.json"
        _write_json(self.dataset_path, self.dataset)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _task(self, name: str = "analysis"):
        output = self.root / name
        return replay.make_task(
            dataset_path=self.dataset_path,
            source_root=self.source,
            candidate_manifest_path=self.candidate_manifest,
            baseline_manifest_path=self.baseline_manifest,
            native_source_archive_path=self.archive,
            native_source_archive_manifest_path=self.archive_index,
            expected_native_source_archive_sha256=replay.sha256_file(self.archive),
            expected_native_source_archive_manifest_sha256=replay.sha256_file(self.archive_index),
            candidate_source_dir=self.root / f"{name}-candidate-source",
            output_dir=output,
            kda_root=KDA_ROOT,
            python=Path(sys.executable),
        )

    @staticmethod
    def _request(config: dict, operation: str, candidate_id: str | None = None,
                 case_id: str | None = None, phase: str = "exploration", round_index: int = 0):
        request = {"schema_version": 1, "request_id": "test-request", "operation": operation}
        if candidate_id is not None:
            request["candidate"] = {
                "id": candidate_id,
                "snapshot_sha256": config["source_tree_sha256"],
                "source_dir": config["candidate_source_root"],
            }
        if operation == "benchmark":
            case = next(row for row in json.loads(Path(config["task_path"]).read_text())["cases"]
                        if row["id"] == case_id)
            request.update({
                "case": case,
                "phase": phase,
                "round": round_index,
                "timing": {"metric": "recorded_primary_caller_wall_time",
                           "unit": "ms_per_fwd_vjp_step", "repetitions": 1},
            })
        return request

    def test_task_is_cpu_direct_with_frozen_recorded_comparison(self) -> None:
        made = self._task()
        config = replay._load_config(Path(made["config_path"]))
        task = json.loads(Path(made["task_path"]).read_text(encoding="utf-8"))
        self.assertEqual(task["target"], {"kind": "cpu", "architecture": platform.machine()})
        self.assertEqual(task["execution"], {"mode": "direct"})
        self.assertIs(task["commands"]["build"]["cpu_only"], True)
        self.assertEqual(task["timing"]["rounds"], 7)
        self.assertEqual(task["timing"]["confirmation_rounds"], 7)
        self.assertEqual(task["timing"]["repetitions"], 1)
        self.assertEqual(task["promotion"], replay.PROMOTION)
        self.assertEqual(len(task["cases"]), 13)
        recorded_inputs = task["trusted_inputs"]
        self.assertTrue(any(row["role"] == "validator" for row in recorded_inputs))
        self.assertTrue(any(row["role"] == "reference" and
                            Path(row["path"]).name == "workloads.json"
                            for row in recorded_inputs))
        source_index = {row["id"]: row for row in recorded_inputs}
        self.assertEqual(source_index["native_source_archive_gzip"]["path"],
                         config["native_source_archive_gzip_path"])
        self.assertEqual(source_index["native_source_archive_member_index"]["path"],
                         config["native_source_archive_manifest_path"])
        self.assertEqual(source_index["native_source_descriptor"]["path"],
                         config["native_source_descriptor_path"])
        candidate_root = Path(config["candidate_source_root"])
        self.assertEqual(sorted(path.name for path in candidate_root.iterdir()),
                         ["analysis_descriptor.json", "native_source.tar.gz"])
        source_manifest = json.loads(Path(made["source_manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(source_manifest["files"], ["analysis_descriptor.json", "native_source.tar.gz"])
        self.assertEqual([case["id"] for case in task["cases"]],
                         [row["id"] for row in replay._fixed_workload_catalog(self.source)])
        self.assertEqual(made["qualification_status"], replay.QUALIFICATION_STATUS)

    def test_candidate_source_gzip_roundtrips_exact_pinned_tar(self) -> None:
        made = self._task("gzip")
        config = replay._load_config(Path(made["config_path"]))
        compressed = Path(config["native_source_archive_gzip_path"])
        with gzip.open(compressed, "rb") as stream:
            inner = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                inner.update(chunk)
                size += len(chunk)
        self.assertEqual(inner.hexdigest(), config["native_source_archive_tar_sha256"])
        self.assertEqual(size, config["native_source_archive_tar_bytes"])
        self.assertEqual(config["native_source_archive_gzip_mtime"], 0)
        self.assertEqual(config["native_source_archive_gzip_compression"], "gzip")
        self.assertEqual(config["native_source_archive_members_sha256"],
                         replay.digest_json(replay.read_json(self.archive_index)["files"]))

    def test_archive_bytes_must_match_the_pinned_member_index(self) -> None:
        self.archive.write_bytes(self.archive.read_bytes() + b"changed archive tail")
        with self.assertRaisesRegex(replay.ReplayError, "member index"):
            replay.make_task(
                dataset_path=self.dataset_path,
                source_root=self.source,
                candidate_manifest_path=self.candidate_manifest,
                baseline_manifest_path=self.baseline_manifest,
                native_source_archive_path=self.archive,
                native_source_archive_manifest_path=self.archive_index,
                expected_native_source_archive_sha256=replay.sha256_file(self.archive),
                expected_native_source_archive_manifest_sha256=replay.sha256_file(self.archive_index),
                candidate_source_dir=self.root / "tampered-candidate-source",
                output_dir=self.root / "tampered-analysis",
                kda_root=KDA_ROOT,
                python=Path(sys.executable),
            )

    def test_probe_and_benchmark_echo_cpu_and_one_exact_observation(self) -> None:
        made = self._task()
        config_path = Path(made["config_path"])
        config = replay._load_config(config_path)
        task = json.loads(Path(made["task_path"]).read_text(encoding="utf-8"))
        config["task_path"] = made["task_path"]

        probe = replay._dispatch(config, {
            "request_id": "probe-1", "operation": "probe",
        })
        self.assertEqual(probe["environment"]["kind"], "cpu")
        self.assertEqual(probe["environment"]["architecture"], platform.machine())
        self.assertEqual(probe["environment"]["recorded_gpu_observation"]["device_uuid"], UUID)
        self.assertIs(probe["environment"]["cuda_executed"], False)
        self.assertNotIn("scheduler_job_id", probe["environment"])

        case_id = task["cases"][4]["id"]
        response = replay._dispatch(config, self._request(
            config, "benchmark", "packet-candidate", case_id,
            phase="confirmation", round_index=5))
        expected = self.dataset["cases"][4]["phases"]["confirmation"][5]["arms"]["packet"]
        self.assertEqual(response["samples"], [expected["primary_ms"]])
        self.assertEqual(response["recorded_observation"]["raw_metrics_sha256"],
                         expected["raw_metrics_sha256"])
        self.assertEqual(response["recorded_observation"]["original_receipt_identity"]["scheduler_job_id"],
                         expected["scheduler_job_id"])
        self.assertEqual(response["repetitions"], 1)

    def test_runtime_uses_the_captured_catalog_not_the_adapter_directory(self) -> None:
        made = self._task("relocated")
        task = json.loads(Path(made["task_path"]).read_text(encoding="utf-8"))
        config_path = Path(made["config_path"])
        config = replay._load_config(config_path)
        config["task_path"] = made["task_path"]
        relocated_adapter = self.root / "relocated-adapter" / "bench" / "factorized_interactions"
        relocated_adapter.mkdir(parents=True)
        relocated_module = relocated_adapter.parent.parent / "recorded_engine.py"
        shutil.copyfile(replay._MODULE_PATH, relocated_module)
        (relocated_adapter / "workloads.json").write_text('{"cases":[]}', encoding="utf-8")
        previous_module_path = replay._MODULE_PATH
        replay._MODULE_PATH = relocated_module
        try:
            loaded = replay._load_config(config_path)
            probe = replay._dispatch(loaded, {
                "request_id": "relocated-probe", "operation": "probe",
            })
            self.assertEqual(probe["environment"]["kind"], "cpu")
            case_id = task["cases"][2]["id"]
            response = replay._dispatch(loaded, self._request(
                {**loaded, "task_path": made["task_path"]}, "benchmark",
                "strongest-direct", case_id, phase="exploration", round_index=3))
            expected = self.dataset["cases"][2]["phases"]["exploration"][3]["arms"]["strongest_direct"]
            self.assertEqual(response["samples"], [expected["primary_ms"]])
        finally:
            replay._MODULE_PATH = previous_module_path

    def test_rejects_wrong_round_or_descriptor(self) -> None:
        made = self._task()
        config = replay._load_config(Path(made["config_path"]))
        config["task_path"] = made["task_path"]
        case_id = self.dataset["cases"][0]["id"]
        with self.assertRaises(replay.ReplayError):
            replay._dispatch(config, self._request(
                config, "benchmark", "packet-candidate", case_id,
                phase="exploration", round_index=7))
        request = self._request(config, "benchmark", "packet-candidate", case_id)
        request["case"]["input"]["recorded_workload_sha256"] = "f" * 64
        with self.assertRaises(replay.ReplayError):
            replay._dispatch(config, request)

    def test_build_copies_and_rechecks_exact_five_prebuilt_files(self) -> None:
        made = self._task()
        config = replay._load_config(Path(made["config_path"]))
        build_dir = self.root / "candidate-build"
        build_dir.mkdir()
        request = self._request(config, "build", "packet-candidate")
        request["build"] = {"directory": str(build_dir)}
        response = replay._dispatch(config, request)
        self.assertEqual(len(response["artifacts"]), 5)
        self.assertIs(response["provenance"]["executed_any_binary"], False)
        for asset in config["assets"]:
            relative = (Path("prebuilt-native") / asset["path"]).as_posix()
            copied = build_dir / relative
            self.assertEqual(replay.sha256_file(copied), asset["sha256"])
        copied_runner = build_dir / "prebuilt-native" / config["assets"][2]["path"]
        copied_runner.write_bytes(b"tampered")
        candidate = {
            "id": "packet-candidate",
            "snapshot_sha256": config["source_tree_sha256"],
            "source_dir": config["candidate_source_root"],
            "build": {"directory": str(build_dir), "artifacts": response["artifacts"]},
        }
        with self.assertRaises(replay.ReplayError):
            replay._verify_candidate_build(config, candidate)

    def test_dataset_rejects_missing_observation_or_mixed_device(self) -> None:
        invalid = json.loads(json.dumps(self.dataset))
        invalid["cases"][0]["phases"]["exploration"].pop()
        with self.assertRaises(replay.ReplayError):
            replay.validate_dataset(invalid, expected_workloads=replay._fixed_workload_catalog(self.source))
        invalid = json.loads(json.dumps(self.dataset))
        invalid["cases"][0]["phases"]["exploration"][0]["arms"]["packet"]["device_uuid"] = "GPU-other"
        with self.assertRaises(replay.ReplayError):
            replay.validate_dataset(invalid, expected_workloads=replay._fixed_workload_catalog(self.source))

    def test_rejects_changed_retained_trusted_input_before_analysis(self) -> None:
        evidence = self.root / "retained-report.json"
        evidence.write_text('{"accepted":true}\n', encoding="utf-8")
        self.dataset["trusted_inputs"].append({
            "id": "test-retained-report",
            "path": str(evidence.resolve()),
            "sha256": replay.sha256_file(evidence),
            "role": "timing_partition_report",
        })
        _write_json(self.dataset_path, self.dataset)
        made = self._task("retained-input")
        evidence.write_text('{"accepted":false}\n', encoding="utf-8")
        config = replay._load_config(Path(made["config_path"]))
        with self.assertRaisesRegex(replay.ReplayError, "recorded trusted input changed"):
            replay._dispatch(config, {"request_id": "probe-mutated", "operation": "probe"})

    @unittest.skipUnless(KDA_ROOT.is_dir(), "KDA engine checkout is unavailable")
    def test_public_kda_cpu_replay_completes_once_within_scratch_cap(self) -> None:
        source_root, candidate_manifest, baseline_manifest = _native_files(
            self.root / "representative-prebuilt", candidate_bytes=2 * 1024 * 1024,
            baseline_bytes=2 * 1024 * 1024, library_bytes=8 * 1024 * 1024,
            full_source_inventory=True)
        source_files = [path for path in source_root.rglob("*") if path.is_file()]
        self.assertEqual(len(source_files), 3_979)
        self.assertEqual(sum(path.stat().st_size for path in source_files), 20_035_237)
        archive, archive_index = _native_archive(
            self.root, source_root, candidate_manifest,
            archive_subdirectory="representative-accepted-source")
        dataset_path = self.root / "representative-dataset.json"
        _write_json(dataset_path, _dataset(source_root, candidate_manifest, baseline_manifest,
                                           full_inventory=True))
        output_dir = self.root / "e2e"
        made = replay.make_task(
            dataset_path=dataset_path,
            source_root=source_root,
            candidate_manifest_path=candidate_manifest,
            baseline_manifest_path=baseline_manifest,
            native_source_archive_path=archive,
            native_source_archive_manifest_path=archive_index,
            expected_native_source_archive_sha256=replay.sha256_file(archive),
            expected_native_source_archive_manifest_sha256=replay.sha256_file(archive_index),
            candidate_source_dir=self.root / "representative-candidate-source",
            output_dir=output_dir,
            kda_root=KDA_ROOT,
            python=Path(sys.executable),
        )
        config = replay._load_config(Path(made["config_path"]))
        task = json.loads(Path(made["task_path"]).read_text(encoding="utf-8"))
        self.assertEqual(config["recorded_trusted_input_count"], 2267)
        self.assertTrue(Path(config["recorded_trusted_inputs_manifest_path"]).is_file())
        workspace = self.root / "e2e" / "kda-workspace"
        receipt_path = self.root / "e2e" / "recorded_kda_cpu_receipt.json"
        sampler = _DirectoryPeakSampler(output_dir)
        analysis_started = time.monotonic()
        sampler.start()
        try:
            result = replay.orchestrate(
                task_path=Path(made["task_path"]),
                config_path=Path(made["config_path"]),
                workspace=workspace,
                receipt_path=receipt_path,
                kda_root=KDA_ROOT,
            )
        finally:
            sampler.stop()
        self.assertTrue(receipt_path.is_file())
        receipt = result["result"]
        self.assertEqual(receipt["analysis_kind"], "recorded_cuda_observation_replay")
        self.assertEqual(receipt["status"], "provisional_cpu_analysis_completed")
        self.assertEqual(receipt["external_cpu_scheduler_receipt_and_cleanup"], "pending")
        self.assertEqual(receipt["outer_receipt_verification"], "pending")
        self.assertEqual(receipt["cleanup_verification"], "pending")
        self.assertIs(receipt["cuda_executed"], False)
        self.assertEqual(receipt["recorded_cuda_observation"]["device_uuid"], UUID)
        self.assertEqual(receipt["selection_authority"].split(";")[0], "provisional KDA analysis only")
        self.assertEqual(len(receipt["commands"]), 4)
        self.assertEqual(receipt["commands"][-1]["response"]["status"], "ok")
        provisional = receipt["provisional_result"]
        self.assertTrue(provisional["summary_path"])
        summary_path = Path(provisional["summary_path"])
        self.assertTrue(summary_path.is_file())
        self.assertEqual(replay.sha256_file(summary_path), provisional["summary_sha256"])
        summary = replay.read_json(summary_path)
        self.assertEqual(summary["run_id"], provisional["run_id"])
        self.assertEqual(set(summary["confirmation"]["per_case"]),
                         {case["id"] for case in json.loads(Path(made["task_path"]).read_text())["cases"]})
        self.assertIn("trusted_inputs", summary)
        with (workspace / "benchmark.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 13 * 14 * 2)
        keys = {(row["candidate_id"], row["phase"], row["case_id"], row["round"])
                for row in rows}
        self.assertEqual(len(keys), len(rows))
        self.assertEqual({row["candidate_id"] for row in rows}, set(replay.CANDIDATES))
        self.assertGreater(sampler.peak_bytes, 40 * 1024 * 1024,
                           f"representative replay output peak was only {sampler.peak_bytes} bytes")
        self.assertLess(sampler.peak_bytes, 128 * 1024 * 1024,
                        f"CPU analysis output peak exceeded the hard cap: {sampler.peak_bytes} bytes")
        print(f"SYNTHETIC_SOURCE_TREE_FILES={len(source_files)}")
        print(f"SYNTHETIC_SOURCE_TREE_BYTES={sum(path.stat().st_size for path in source_files)}")
        print(f"SYNTHETIC_SOURCE_TAR_BYTES={archive.stat().st_size}")
        print(f"SYNTHETIC_GZIP_TRANSPORT_BYTES={config['native_source_archive_gzip_bytes']}")
        print(f"SYNTHETIC_RECORDED_EVIDENCE_REFERENCES={config['recorded_trusted_input_count']}")
        print(f"SYNTHETIC_KDA_TRUSTED_INPUT_BYTES={sum(Path(row['path']).stat().st_size for row in task['trusted_inputs'])}")
        print(f"SYNTHETIC_TEST_OUTPUT_TREE_PEAK_BYTES={sampler.peak_bytes}")
        print(f"SYNTHETIC_TEST_ANALYSIS_SECONDS={time.monotonic() - analysis_started:.3f}")


class _DirectoryPeakSampler:
    def __init__(self, root: Path):
        self.root = root
        self.peak_bytes = self._size()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _size(self) -> int:
        total = 0
        for current, dirs, files in os.walk(self.root, followlinks=False):
            base = Path(current)
            dirs[:] = [name for name in dirs if not (base / name).is_symlink()]
            for name in files:
                path = base / name
                try:
                    if path.is_file() and not path.is_symlink():
                        total += path.stat().st_size
                except FileNotFoundError:
                    continue
        return total

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            self.peak_bytes = max(self.peak_bytes, self._size())

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self._size())


if __name__ == "__main__":
    unittest.main()
