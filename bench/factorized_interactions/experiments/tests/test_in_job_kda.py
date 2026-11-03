"""CPU-only contract tests; no Scheduler job or GPU operation is launched."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ENTRYPOINT = Path(__file__).resolve().parents[1] / "in_job_kda.py"
spec = importlib.util.spec_from_file_location("in_job_kda", ENTRYPOINT)
assert spec and spec.loader
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

GPU = "GPU-21131915-1488-23af-38dd-1743ae1f5cc8"
ENV = {"DEVICE_SCHEDULER_JOB_ID": "job-1", "DEVICE_SCHEDULER_ATTEMPT_ID": "attempt-1",
       "AUTOBENCH_SCHEDULER_JOB_ID": "job-1", "AUTOBENCH_SCHEDULER_ATTEMPT_ID": "attempt-1",
       "CUDA_VISIBLE_DEVICES": GPU, "OTHER_INHERITED": "unchanged"}
CGROUP = f"0::/user.slice/user-1000.slice/{guard.scheduler_unit('attempt-1')}\n"


class ContextTests(unittest.TestCase):
    def test_accepts_exact_attempt_unit_in_v2_hierarchy(self):
        result = guard.validate_context(ENV, CGROUP, GPU)
        self.assertEqual(result["scheduler_job_id"], "job-1")
        self.assertEqual(result["scheduler_attempt_id"], "attempt-1")
        self.assertEqual(result["device_uuid"], GPU)
        self.assertEqual(result["status"], "observed")

    def test_rejects_missing_mismatched_or_forged_markers(self):
        for field, value in (("DEVICE_SCHEDULER_JOB_ID", ""),
                             ("AUTOBENCH_SCHEDULER_JOB_ID", "other"),
                             ("AUTOBENCH_SCHEDULER_ATTEMPT_ID", "other"),
                             ("CUDA_VISIBLE_DEVICES", "0")):
            with self.subTest(field=field):
                changed = dict(ENV, **{field: value})
                with self.assertRaises(guard.ContextError):
                    guard.validate_context(changed, CGROUP, GPU)
        for cgroup in ("0::/user.slice\n", CGROUP.replace(".service", ".service-other"),
                       CGROUP + CGROUP, "1:name=systemd:/a\n"):
            with self.subTest(cgroup=cgroup):
                with self.assertRaises(guard.ContextError):
                    guard.validate_context(ENV, cgroup, GPU)

    def test_rejects_invalid_task_bindings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_path = root / "task.json"
            task = {"execution": {"mode": "direct"}, "target": {"kind": "cuda", "expected_device_uuid": GPU},
                    "factorized_interaction": {"expected_device_uuid": GPU},
                    "trusted_inputs": [{"id": "in_job_entrypoint", "path": str(ENTRYPOINT)}]}
            guard.validate_task(task, task_path, GPU, ENTRYPOINT)
            for field, value in (("execution", {"mode": "launcher"}),
                                 ("target", {"kind": "cpu"}),
                                 ("factorized_interaction", {"expected_device_uuid": "other"}),
                                 ("trusted_inputs", [])):
                with self.subTest(field=field), self.assertRaises(guard.ContextError):
                    guard.validate_task(dict(task, **{field: value}), task_path, GPU, ENTRYPOINT)

    def test_entrypoint_records_context_then_execs_without_changing_env(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            workspace.mkdir()
            task_path = root / "task.json"
            task_path.write_text(json.dumps({"execution": {"mode": "direct"},
                "target": {"kind": "cuda", "expected_device_uuid": GPU},
                "factorized_interaction": {"expected_device_uuid": GPU},
                "trusted_inputs": [{"id": "in_job_entrypoint", "path": str(ENTRYPOINT)}]}),
                encoding="utf-8")
            argv = ["--task", str(task_path), "--workspace", str(workspace),
                    "--expected-uuid", GPU, "--candidates", "packet",
                    "--incumbent", "direct"]
            original_read = Path.read_text

            def read_text(path, *args, **kwargs):
                if str(path) == "/proc/self/cgroup":
                    return CGROUP
                return original_read(path, *args, **kwargs)

            with mock.patch.dict(os.environ, ENV, clear=True), \
                 mock.patch.object(Path, "read_text", read_text), \
                 mock.patch.object(guard.os, "execv", side_effect=RuntimeError("intercept")) as execute:
                with self.assertRaisesRegex(RuntimeError, "intercept"):
                    guard.main(argv)
                self.assertEqual(dict(os.environ), ENV)
            record = json.loads((workspace / guard.CONTEXT_FILE).read_text(encoding="utf-8"))
            self.assertEqual(record["task_sha256"], guard.digest_file(task_path))
            self.assertEqual(record["entrypoint_sha256"], guard.digest_file(ENTRYPOINT))
            self.assertEqual(record["cgroup"], CGROUP.strip().split("::", 1)[1])
            self.assertEqual(execute.call_count, 1)
            forwarded = execute.call_args.args[1]
            self.assertIn("kda_experiments", forwarded)
            self.assertIn("evaluate", forwarded)
            self.assertEqual(forwarded[forwarded.index("--candidates"):],
                             ["--candidates", "packet", "--incumbent", "direct"])
            self.assertNotIn("launcher", forwarded)

    def test_outside_job_fails_before_creating_context_or_execing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task = root / "task.json"
            task.write_text(json.dumps({"execution": {"mode": "direct"},
                "target": {"kind": "cuda", "expected_device_uuid": GPU},
                "factorized_interaction": {"expected_device_uuid": GPU},
                "trusted_inputs": [{"id": "in_job_entrypoint", "path": str(ENTRYPOINT)}]}))
            argv = ["--task", str(task), "--workspace", str(root), "--expected-uuid", GPU,
                    "--candidates", "b", "--incumbent", "a"]
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(guard.os, "execv") as execute:
                self.assertEqual(guard.main(argv), 2)
            execute.assert_not_called()
            self.assertFalse((root / guard.CONTEXT_FILE).exists())

    def test_kda_cpu_child_inherits_environment_and_cgroup(self):
        # Exercise KDA's actual subprocess path without a GPU or Scheduler marker.
        kda_root = "/home/tumlinson/kda"
        with mock.patch.object(sys, "path", [kda_root, *sys.path]):
            from kda_experiments import runner
        with tempfile.TemporaryDirectory() as temp, \
             mock.patch.dict(os.environ, {"KDA_INHERITANCE_TEST": "inherited"}):
            root = Path(temp)
            child_code = ("import json,os,pathlib; print(json.dumps({"
                          "'marker':os.environ.get('KDA_INHERITANCE_TEST'),"
                          "'cgroup':pathlib.Path('/proc/self/cgroup').read_text()}))")
            runner._run([sys.executable, "-c", child_code], cwd=root,
                        timeout_seconds=10, output_prefix=root / "child",
                        env=runner.task_environment())
            result = json.loads((root / "child.stdout.txt").read_text())
            self.assertEqual(result["marker"], "inherited")
            self.assertEqual(result["cgroup"], Path("/proc/self/cgroup").read_text())


if __name__ == "__main__":
    unittest.main()
