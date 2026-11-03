"""CPU protocol smoke test; it does not validate a CUDA candidate."""

import json
import os
from pathlib import Path
import shlex
import site
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class NativeValidatorProtocolSmokeTests(unittest.TestCase):
    def test_runner_stdout_is_retained_on_stderr_without_corrupting_json(self):
        validator = ROOT / "bench" / "factorized_interactions" / "validate_runner.py"
        original_stub = ROOT / "tests" / "factorized_interactions" / "oracle_wire_stub.py"
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            stub = temporary / "oracle_wire_stub.py"
            for exit_code in (0, 99):
                with self.subTest(exit_code=exit_code):
                    body = (f"sys.exit({exit_code})\n" if exit_code else
                            f"sys.path.insert(0, {str(original_stub.parent)!r})\n"
                            f"runpy.run_path({str(original_stub)!r}, run_name='__main__')\n")
                    stub.write_text("import runpy, sys\n"
                                    "print('FAKE SANITIZER BANNER', flush=True)\n" + body,
                                    encoding="utf-8")
                    command = [
                        sys.executable, str(validator),
                        "--workload", "correctness_ordered_tied_zero",
                        "--runner", shlex.join([sys.executable, str(stub)]),
                        "--route", "packet", "--allow-oracle-stub",
                        "--artifact-dir", str(temporary / f"case-{exit_code}"),
                    ]
                    env = os.environ.copy()
                    env["PYTHONDONTWRITEBYTECODE"] = "1"
                    result = subprocess.run(command, cwd=temporary, env=env,
                                            check=False, capture_output=True, text=True)
                    self.assertIn("FAKE SANITIZER BANNER", result.stderr)
                    if exit_code:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("exit status 99", result.stderr)
                        self.assertEqual(result.stdout, "")
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertTrue(json.loads(result.stdout)["validated"])

    def test_wrapper_preserves_symlinked_python_executable(self):
        validator = ROOT / "bench" / "factorized_interactions" / "validate_runner.py"
        original_stub = ROOT / "tests" / "factorized_interactions" / "oracle_wire_stub.py"
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            python_link = temporary / "selected-python"
            python_link.symlink_to(sys.executable)
            checked_stub = temporary / "oracle_wire_stub.py"
            checked_stub.write_text(
                "import os, runpy, sys\n"
                "if sys.executable != os.environ['EXPECTED_PYTHON_LINK']:\n"
                "    raise RuntimeError('selected interpreter symlink was resolved')\n"
                "sys.path.insert(0, os.environ['FACTORIZED_TEST_SOURCE'])\n"
                "runpy.run_path(os.environ['FACTORIZED_ORACLE_STUB'], run_name='__main__')\n",
                encoding="utf-8")
            env = os.environ.copy()
            env["EXPECTED_PYTHON_LINK"] = str(python_link)
            env["FACTORIZED_TEST_SOURCE"] = str(original_stub.parent)
            env["FACTORIZED_ORACLE_STUB"] = str(original_stub)
            # The temporary interpreter link intentionally lives outside any
            # parent venv, so explicitly retain its installed NumPy path.
            env["PYTHONPATH"] = os.pathsep.join(
                [*site.getsitepackages(), env.get("PYTHONPATH", "")])
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            command = [
                sys.executable, str(validator),
                "--workload", "correctness_ordered_tied_zero",
                "--runner", shlex.join([str(python_link), str(checked_stub)]),
                "--route", "direct", "--allow-oracle-stub",
                "--artifact-dir", str(temporary / "artifacts"),
            ]
            result = subprocess.run(command, cwd=temporary, env=env, check=False,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["validated"])

    def test_wrapper_serializes_invokes_and_checks_oracle_wire_stub(self):
        validator = ROOT / "bench" / "factorized_interactions" / "validate_runner.py"
        stub = ROOT / "tests" / "factorized_interactions" / "oracle_wire_stub.py"
        command = [
            sys.executable, str(validator),
            "--workload", "correctness_ordered_tied_zero",
            "--runner", f"{sys.executable} {stub}",
            "--route", "direct",
            "--activation-dtype", "fp16",
            "--parameter-dtype", "fp32",
            "--output-dtype", "fp16",
            "--gradient-mode", "accumulate",
            "--repeat-vjp",
            "--allow-oracle-stub",
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(command + ["--artifact-dir", directory],
                                    cwd=ROOT, check=True, capture_output=True, text=True)
            summary = json.loads(result.stdout)
        self.assertTrue(summary["validated"])
        self.assertFalse(summary["timed"])
        self.assertEqual(summary["route"], "direct")
        self.assertEqual(summary["gradient_mode"], "accumulate")
        self.assertEqual(summary["activation_storage"], "fp16")
        self.assertEqual(summary["parameter_storage"], "fp32")
        self.assertTrue(summary["input_case_sha256"])
        self.assertIsNone(summary["build_provenance"])
        dx_evidence = summary["contract_evidence"]["dx"]
        self.assertIn("worst_sum_abs_reference_contributions", dx_evidence)
        self.assertIn("worst_allowed_error", dx_evidence)
        self.assertLessEqual(dx_evidence["max_error_to_allowance"], 1.0)
        self.assertIn("dx_repeat", summary["contract_evidence"])

    def test_native_runner_invocation_fails_closed_without_scheduler_job(self):
        validator = ROOT / "bench" / "factorized_interactions" / "validate_runner.py"
        command = [
            sys.executable, str(validator),
            "--workload", "correctness_empty_cells",
            "--runner", "/bin/true",
            "--route", "packet",
        ]
        env = os.environ.copy()
        env.pop("AUTOBENCH_SCHEDULER_JOB_ID", None)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(command + ["--artifact-dir", directory],
                                    cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Autobench worker Scheduler job context", result.stderr)


if __name__ == "__main__":
    unittest.main()
