"""CPU protocol checks for engine-shaped requests borrowing an outer job."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

import adapter  # noqa: E402
from common import AdapterError, digest_file  # noqa: E402
from in_job_kda import CONTEXT_FILE, SCHEMA, scheduler_unit, validate_context  # noqa: E402


class BorrowedContextProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name) / "workspace"
        self.attempt = self.workspace / "runs/kda-experiments/attempts/attempt-1"
        self.request_path = self.attempt / "stages/probe/requests/request-1.json"
        self.request_path.parent.mkdir(parents=True)
        self.request_path.write_text("{}\n", encoding="utf-8")
        (self.workspace / "candidates.jsonl").write_text("", encoding="utf-8")
        self.uuid = "GPU-21131915-1488-23af-38dd-1743ae1f5cc8"
        self.env = {
            "DEVICE_SCHEDULER_JOB_ID": "outer-job-1",
            "AUTOBENCH_SCHEDULER_JOB_ID": "outer-job-1",
            "DEVICE_SCHEDULER_ATTEMPT_ID": "outer-attempt-1",
            "AUTOBENCH_SCHEDULER_ATTEMPT_ID": "outer-attempt-1",
            "CUDA_VISIBLE_DEVICES": self.uuid,
        }
        self.cgroup = f"0::/user.slice/app.slice/{scheduler_unit('outer-attempt-1')}\n"
        self.task_path = self.workspace / "task.json"
        self.entrypoint = EXPERIMENTS / "in_job_kda.py"
        task = {
            "execution": {"mode": "direct"},
            "target": {"kind": "cuda", "expected_device_uuid": self.uuid},
            "factorized_interaction": {
                "expected_device_uuid": self.uuid,
                "outer_execution_context_schema": SCHEMA,
                "outer_execution_context_file": CONTEXT_FILE,
                "in_job_entrypoint_path": str(self.entrypoint),
                "in_job_entrypoint_sha256": digest_file(self.entrypoint),
            },
            "trusted_inputs": [{"id": "in_job_entrypoint", "path": str(self.entrypoint)}],
        }
        self.task_path.write_text(json.dumps(task), encoding="utf-8")
        batch = {"task_path": str(self.task_path), "task_sha256": digest_file(self.task_path)}
        (self.attempt / "batch-request.json").write_text(json.dumps(batch), encoding="utf-8")
        self.context = {
            **validate_context(self.env, self.cgroup, self.uuid),
            "task_sha256": digest_file(self.task_path),
            "entrypoint_sha256": digest_file(self.entrypoint),
            "pid": 1234,
        }
        self.context_path = self.workspace / CONTEXT_FILE
        self.context_path.write_text(json.dumps(self.context), encoding="utf-8")
        original_read = Path.read_text

        def read_text(path: Path, *args, **kwargs):
            return self.cgroup if str(path) == "/proc/self/cgroup" else original_read(path, *args, **kwargs)

        self.read_patch = patch.object(Path, "read_text", read_text)
        self.read_patch.start()
        self.addCleanup(self.read_patch.stop)
        self.env_patch = patch.dict(os.environ, self.env, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def test_probe_and_validation_bind_same_outer_context(self) -> None:
        request = {"declared_target": {"expected_device_uuid": self.uuid}}
        context = adapter._require_admitted(request, self.request_path)
        environment = {
            "scheduler_job_id": context["scheduler_job_id"],
            "scheduler_attempt_id": context["scheduler_attempt_id"],
            "device_uuid": context["device_uuid"],
            "outer_execution_context_sha256": context["context_sha256"],
            "outer_cgroup": context["cgroup"],
            "outer_unit_name": context["unit_name"],
        }
        validation = {"target": request["declared_target"], "environment": environment}
        self.assertEqual(adapter._require_admitted(validation, self.request_path), context)
        environment["scheduler_attempt_id"] = "different-attempt"
        with self.assertRaises(AdapterError):
            adapter._require_admitted(validation, self.request_path)

    def test_changed_task_or_context_cannot_reuse_admission_observation(self) -> None:
        request = {"target": {"expected_device_uuid": self.uuid}}
        self.context["scheduler_job_id"] = "different-job"
        self.context_path.write_text(json.dumps(self.context), encoding="utf-8")
        with self.assertRaises(AdapterError):
            adapter._require_admitted(request, self.request_path)
        self.task_path.write_text(self.task_path.read_text() + "\n", encoding="utf-8")
        with self.assertRaises(AdapterError):
            adapter._require_admitted(request, self.request_path)

    def test_marker_without_matching_cgroup_is_rejected(self) -> None:
        self.cgroup = "0::/user.slice/unrelated.service\n"
        with self.assertRaises(AdapterError):
            adapter._require_admitted({"target": {"expected_device_uuid": self.uuid}}, self.request_path)


if __name__ == "__main__":
    unittest.main()
