from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import bounded_validation as bounded  # noqa: E402


UUID = "GPU-12345678-1234-5678-9abc-def012345678"
CONTEXT = {"scheduler_job_id": "job", "scheduler_attempt_id": "attempt",
           "device_uuid": UUID, "cgroup": "/job.service", "unit_name": "job.service"}
FACTS = {"cells": 1, "channels": 1, "mechanisms": 1, "outputs": 1,
         "stable": False, "unique": True,
         "maxima": {"output_mechanisms": 1, "input_roles": 1, "context_terms": 0,
                    "alpha_mechanisms": 1, "bias_mechanisms": 1,
                    "gamma_terms": 0, "mechanism_terms": 0}}


class BoundedValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "build/source"
        validator = self.source / "bench/factorized_interactions/validate_runner.py"
        validator.parent.mkdir(parents=True)
        validator.write_text("fixed validator fixture\n")
        (self.source / "other.txt").write_text("source fixture\n")
        self.runner = self.root / "native-build/runner"
        self.runner.parent.mkdir()
        self.runner.write_bytes(b"candidate binary")
        self.library = self.runner.parent / "library.a"
        self.library.write_bytes(b"static library")
        entries = []
        for path in sorted(p for p in self.source.rglob("*") if p.is_file()):
            entries.append({"path": path.relative_to(self.source).as_posix(),
                            "bytes": path.stat().st_size, "sha256": bounded.sha256_file(path)})
        closure = bounded._source_identity(self.source, entries)
        self.manifest = self.runner.parent / "manifest.json"
        self.manifest.write_text(json.dumps({
            "schema": "cellerator.factorized_interactions.build.v1", "architecture": "sm_70",
            "fast_math": False, "source_closure_sha256": closure, "source_files": entries,
            "binaries": {"runner": {"path": "runner", "bytes": self.runner.stat().st_size,
                                       "sha256": bounded.sha256_file(self.runner)},
                         "factorized_interaction_cuda": {"path": "library.a", "bytes": self.library.stat().st_size,
                                                          "sha256": bounded.sha256_file(self.library)}}}))
        self.validator_patch = mock.patch.object(bounded, "VALIDATOR_SHA256", bounded.sha256_file(validator))
        self.source_patch = mock.patch.object(bounded, "ACCEPTED_SOURCE_SHA256", closure)
        self.context_patch = mock.patch.object(bounded, "read_context", return_value=CONTEXT)
        for patcher in (self.validator_patch, self.source_patch, self.context_patch):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.build = bounded.verify_build(self.source, self.runner, self.manifest)
        self.packet = self.root / "packet"
        self.artifacts = self.packet / "cases"
        self.cache = self.artifacts / "reference-cache"

    def fake_invoke(self, *, sizes=(9,), code=0, iterations=0, transient=0):
        def invoke(command, *, cwd, env, capture_output, text, check):
            self.assertIn("--reference-cache-dir", command)
            self.assertEqual(command[command.index("--iterations") + 1], str(iterations))
            digest = hashlib.sha256()
            for index, size in enumerate(sizes):
                path = cwd / ("input.case" if len(sizes) == 1 else f"input.case.part{index:05d}")
                with path.open("wb") as stream:
                    stream.truncate(size)
                block = b"\0" * (1024 * 1024)
                remaining = size
                while remaining:
                    part = block[:min(len(block), remaining)]
                    digest.update(part)
                    remaining -= len(part)
            result = cwd / "input.result"
            result.write_bytes(b"native output")
            if transient:
                extra = self.cache / "sampled.tmp"
                with extra.open("wb") as stream:
                    stream.truncate(transient)
                time.sleep(0.20)
                extra.unlink()
            if code:
                return subprocess.CompletedProcess(command, code, stdout="bad", stderr="numeric failure")
            native = {"candidate": "direct_unique_output", "requested_route": "direct",
                      "selected_backend": "direct", "actual_implementation": "direct_unique_output",
                      "structural_layout": "canonical_direct", "device_uuid": UUID,
                      "scheduler_job_id": "job", "structural_maxima": FACTS["maxima"],
                      **{key: FACTS[key] for key in ("cells", "channels", "mechanisms", "outputs")}}
            evidence = {key: {} for key in ("y", "dx", "dz", "dalpha", "dbias", "dgamma",
                                            "dx_repeat", "dz_repeat", "dalpha_repeat", "dbias_repeat", "dgamma_repeat")}
            metrics = None
            if iterations:
                metrics = {**{k: native[k] for k in ("candidate", "selected_backend", "actual_implementation",
                                                     "structural_layout", "device_uuid", "scheduler_job_id")},
                           "iterations": iterations,
                           "phase_ms": {"warm_forward_vjp_batch_host_wall_mean": 0.1,
                                        "warm_forward_vjp_batch_host_wall": 0.1 * iterations},
                           "samples_ms": {key: [0.1] * iterations for key in
                                          ("forward", "backward", "forward_vjp", "parameter_refresh")}}
            summary = {"schema": "cellerator.factorized_interactions.validation.v1", "validated": True,
                       "timed": bool(iterations), "workload": "fixture", "route": "direct",
                       "activation_storage": "fp16", "parameter_storage": "fp16", "output_storage": "fp16",
                       "gradient_mode": "overwrite", "scheduler_job_id": "job",
                       "input_case_sha256": digest.hexdigest(), "input_case_bytes": sum(sizes),
                       "input_case_shards": len(sizes), "output_bytes": result.stat().st_size,
                       "native_execution": native, "contract_evidence": evidence, "metrics": metrics,
                       "build_provenance": {"source_closure_sha256": self.build["source_closure_sha256"],
                                            "runner_binary_sha256": self.build["runner_sha256"],
                                            "build_manifest_sha256": self.build["build_manifest_sha256"]}}
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(summary), stderr="native diagnostic")
        return invoke

    def run_case(self, name="case-1", **kwargs):
        return bounded.run_validation(source=self.source, build=self.build, expected_uuid=UUID,
                                      context=CONTEXT, artifact_root=self.artifacts,
                                      cache_dir=self.cache, case_id=name, workload="fixture",
                                      mode="direct_unique_output", activation="fp16", parameter="fp16",
                                      output_dtype="fp16", iterations=kwargs.pop("iterations", 0),
                                      repeat_vjp=True, facts=FACTS, packet_root=self.packet, **kwargs)

    def test_build_binding_rejects_changed_source_and_library(self):
        self.library.write_bytes(b"tampered")
        with self.assertRaisesRegex(bounded.ValidationFailure, "library|factorized"):
            bounded.verify_build(self.source, self.runner, self.manifest)
        self.library.write_bytes(b"static library")
        (self.source / "other.txt").write_text("changed")
        with self.assertRaisesRegex(bounded.ValidationFailure, "source input changed"):
            bounded.verify_build(self.source, self.runner, self.manifest)

    def test_workload_facts_use_the_requested_fp32_storage_for_fma(self):
        mechanism = types.SimpleNamespace(target=0, participants=(0, 1),
                                          context_ids=(0,) * 65, gamma_ids=(0,) * 65,
                                          alpha_id=0, bias_id=0)
        problem = types.SimpleNamespace(x=np.ones((2, 1, 1), dtype=np.float32),
                                        target_count=1, mechanisms=(mechanism,))
        fixture = mock.Mock(return_value=(problem, np.ones((1, 1, 1), dtype=np.float32)))
        with mock.patch.object(bounded, "_source_module", return_value=types.SimpleNamespace(make_workload=fixture)):
            facts = bounded.workload_facts(self.source, "fma", "fp32", "fp32")
        self.assertIs(fixture.call_args.args[1], np.float32)
        self.assertIs(fixture.call_args.args[2], np.float32)
        self.assertTrue(facts["stable"])
        self.assertEqual(facts["strongest_direct_mode"], "minimal_direct")

    def test_success_records_everything_before_removing_owned_wires(self):
        original = shutil.rmtree
        observed = []
        def remove(path, *args, **kwargs):
            path = Path(path)
            validated = self.artifacts / "evidence/case-1.validated.json"
            precommit = json.loads(validated.read_text())
            self.assertTrue(precommit["cleanup_pending"])
            self.assertFalse(precommit["success_wires_removed"])
            self.assertTrue((self.artifacts / "evidence/case-1.stdout").is_file())
            self.assertTrue((self.artifacts / "evidence/case-1.stderr").is_file())
            self.assertTrue((path / "input.result").is_file())
            observed.append(path)
            return original(path, *args, **kwargs)
        with mock.patch.object(bounded.shutil, "rmtree", side_effect=remove):
            result = self.run_case(iterations=30, invoke=self.fake_invoke(iterations=30))
        self.assertEqual(len(observed), 1)
        self.assertFalse(observed[0].exists())
        self.assertTrue(result["success_wires_removed"])
        self.assertEqual(result["summary"]["metrics"]["samples_ms"]["forward"], [0.1] * 30)
        final = self.packet / result["evidence_relative_path"]
        self.assertEqual(json.loads(final.read_text()), result)
        self.assertEqual(result["output_result_sha256"], hashlib.sha256(b"native output").hexdigest())

    def test_numeric_failure_retains_wire_and_failure_record(self):
        with self.assertRaisesRegex(bounded.ValidationFailure, "remaining wires"):
            self.run_case(invoke=self.fake_invoke(code=1))
        self.assertTrue((self.artifacts / "wire/case-1/input.case").exists())
        failure = json.loads((self.artifacts / "evidence/case-1.failure.json").read_text())
        self.assertFalse(failure["cleanup_started"])
        self.assertTrue(failure["wires_exist"])
        self.assertIn("numeric failure", (self.artifacts / "evidence/case-1.stderr").read_text())

    def test_success_exit_with_wrong_dispatch_is_rejected_and_retained(self):
        base = self.fake_invoke()
        def invoke(*args, **kwargs):
            result = base(*args, **kwargs)
            summary = json.loads(result.stdout)
            summary["native_execution"]["actual_implementation"] = "direct"
            return subprocess.CompletedProcess(result.args, 0, json.dumps(summary), result.stderr)
        with self.assertRaisesRegex(bounded.ValidationFailure, "backend differs"):
            self.run_case(invoke=invoke)
        self.assertTrue((self.artifacts / "wire/case-1/input.result").exists())
        self.assertFalse((self.artifacts / "evidence/case-1.json").exists())

    def test_peak_samples_72_million_byte_wire_and_temporary_cache(self):
        sizes = (32 * 1024 * 1024, 32 * 1024 * 1024, 72_000_112 - 64 * 1024 * 1024)
        result = self.run_case(invoke=self.fake_invoke(sizes=sizes, transient=32 * 1024 * 1024))
        self.assertGreaterEqual(result["peak_scratch_bytes"], 72_000_112 + 32 * 1024 * 1024)
        self.assertLess(result["peak_scratch_bytes"], 128 * 1024 * 1024)
        self.assertEqual(sum(item["bytes"] for item in result["wire_inventory"]
                             if ".case.part" in item["path"]), 72_000_112)

    def test_over_limit_fails_and_keeps_fresh_wire(self):
        with self.assertRaisesRegex(bounded.ValidationFailure, "128 MiB"):
            self.run_case(invoke=self.fake_invoke(sizes=(128 * 1024 * 1024,)))
        self.assertTrue((self.artifacts / "wire/case-1/input.case").exists())

    def test_post_cleanup_error_reports_wire_disposition_honestly(self):
        original = bounded._durable_new
        def write(path, value):
            if path.name == "case-1.json":
                raise OSError("final evidence write failed")
            return original(path, value)
        with mock.patch.object(bounded, "_durable_new", side_effect=write):
            with self.assertRaisesRegex(bounded.ValidationFailure, "removed after durable validation"):
                self.run_case(invoke=self.fake_invoke())
        failure = json.loads((self.artifacts / "evidence/case-1.failure.json").read_text())
        self.assertTrue(failure["cleanup_started"])
        self.assertFalse(failure["wires_exist"])
        self.assertTrue((self.artifacts / "evidence/case-1.validated.json").exists())

    def test_context_change_rejects_and_preserves_wire(self):
        changed = {**CONTEXT, "scheduler_attempt_id": "other"}
        with mock.patch.object(bounded, "read_context", side_effect=[CONTEXT, changed]):
            with self.assertRaisesRegex(bounded.ValidationFailure, "context changed"):
                self.run_case(invoke=self.fake_invoke())
        self.assertTrue((self.artifacts / "wire/case-1/input.case").exists())

    def test_refuses_cache_ownership_of_whole_artifact_tree(self):
        self.artifacts.mkdir(parents=True)
        old = self.artifacts / "preserve-existing.txt"
        old.write_text("old data")
        with self.assertRaises(bounded.ValidationFailure):
            bounded.run_validation(
                source=self.source, build=self.build, expected_uuid=UUID, context=CONTEXT,
                artifact_root=self.artifacts, cache_dir=self.artifacts, case_id="case-1",
                workload="fixture", mode="packet", activation="fp16", parameter="fp16",
                output_dtype="fp16", iterations=0, repeat_vjp=True, packet_root=self.packet,
                invoke=self.fake_invoke(),
            )
        self.assertEqual(old.read_text(), "old data")
        self.assertFalse((self.artifacts / "wire").exists())

    def test_refuses_existing_unowned_cache_and_preserves_its_data(self):
        self.cache.mkdir(parents=True)
        old = self.cache / "old-data.bin"
        old.write_bytes(b"old cache data")
        with self.assertRaisesRegex(bounded.ValidationFailure, "unowned reference cache"):
            self.run_case(invoke=self.fake_invoke())
        self.assertEqual(old.read_bytes(), b"old cache data")
        self.assertFalse((self.artifacts / "wire").exists())

    def test_candidate_cannot_silently_change_cache_ownership(self):
        base = self.fake_invoke()
        def invoke(*args, **kwargs):
            result = base(*args, **kwargs)
            (self.artifacts / ".bounded-cache-owner.json").write_text("changed")
            return result
        with self.assertRaisesRegex(bounded.ValidationFailure, "ownership marker changed"):
            self.run_case(invoke=invoke)
        self.assertTrue((self.artifacts / "wire/case-1/input.result").exists())


if __name__ == "__main__":
    unittest.main()
