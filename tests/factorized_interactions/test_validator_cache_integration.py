"""CPU integration checks for the optional scalar-reference cache."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_SOURCE = ROOT / "tests" / "factorized_interactions"
VALIDATOR = ROOT / "bench" / "factorized_interactions" / "validate_runner.py"
STUB = TEST_SOURCE / "oracle_wire_stub.py"
sys.path.insert(0, str(TEST_SOURCE))

import reference_cache  # noqa: E402


def _validator_arguments(workload: str, output_dir: Path, *,
                         activation_dtype: str, parameter_dtype: str,
                         output_dtype: str, gradient_mode: str,
                         repeat_vjp: bool, cache_dir: Path | None = None,
                         runner: Path = STUB) -> list[str]:
    args = [
        "--workload", workload,
        "--runner", shlex.join([sys.executable, str(runner)]),
        "--route", "direct",
        "--activation-dtype", activation_dtype,
        "--parameter-dtype", parameter_dtype,
        "--output-dtype", output_dtype,
        "--gradient-mode", gradient_mode,
        "--artifact-dir", str(output_dir),
        "--allow-oracle-stub",
    ]
    if repeat_vjp:
        args.append("--repeat-vjp")
    if cache_dir is not None:
        args.extend(["--reference-cache-dir", str(cache_dir)])
    return args


def _run_validator(root: Path, label: str, **kwargs) -> subprocess.CompletedProcess:
    args = _validator_arguments(output_dir=root / f"artifacts-{label}", **kwargs)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, str(VALIDATOR), *args], cwd=ROOT,
                          env=env, check=False, capture_output=True, text=True)


def _load_summary(result: subprocess.CompletedProcess) -> dict:
    if result.returncode != 0:
        raise AssertionError(f"validator failed ({result.returncode}): {result.stderr}")
    return json.loads(result.stdout)


def _invoke_with_reference_evaluation_forbidden(root: Path, label: str,
                                                 **kwargs) -> dict:
    """Run the CLI in-process and fail if a verified cache hit recomputes."""
    spec = importlib.util.spec_from_file_location("factorized_validate_runner", VALIDATOR)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load validate_runner.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    args = _validator_arguments(output_dir=root / f"artifacts-{label}", **kwargs)
    stdout = []
    with mock.patch.object(reference_cache, "evaluate",
                           side_effect=AssertionError("reference cache miss")), \
            mock.patch.object(sys, "argv", [str(VALIDATOR), *args]), \
            contextlib.redirect_stdout(_ListWriter(stdout)):
        status = validator.main()
    if status != 0 or len(stdout) != 1:
        raise AssertionError(f"cached validator returned {status}: {stdout}")
    return json.loads(stdout[0])


class _ListWriter:
    def __init__(self, lines: list[str]):
        self.lines = lines

    def write(self, value: str) -> int:
        if value.strip():
            self.lines.append(value)
        return len(value)

    def flush(self) -> None:
        return None


def _write_faulting_stub(path: Path) -> None:
    path.write_text(
        """import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
sys.path.insert(0, os.environ["FACTOR_INTERACTION_TEST_SOURCE"])
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--route", required=True)
parser.add_argument("--gradient-mode", default="overwrite")
parser.add_argument("--repeat-vjp", action="store_true")
args, _ = parser.parse_known_args()
subprocess.run([sys.executable, os.environ["FACTOR_INTERACTION_ORACLE_STUB"],
                *sys.argv[1:]], check=True)
from case_io import read_case, read_result, write_result
problem, _, output_dtype = read_case(args.input)
actual = read_result(args.output, problem, output_dtype)
names = ("dx", "dz", "dalpha", "dbias", "dgamma")
values = {name: actual[name].copy() for name in names}
repeat_values = {name: actual[name + "_repeat"].copy() for name in names} \\
    if args.repeat_vjp else None
fault = os.environ.get("FACTOR_INTERACTION_FAULT")
if fault == "missing_accumulation":
    for value in values.values():
        value -= 0.125
    if repeat_values is not None:
        for value in repeat_values.values():
            value -= 0.125
elif fault == "wrong_gradient":
    values["dx"].flat[0] += 1.0
vjp = SimpleNamespace(**values)
repeat = SimpleNamespace(**repeat_values) if repeat_values is not None else None
write_result(args.output, problem, actual["y"], vjp, output_dtype, repeat)
if fault == "rewrite_other_cache_entry":
    import reference_cache
    cache_file = Path(os.environ["FACTOR_INTERACTION_CACHE_FILE"])
    with np.load(cache_file, allow_pickle=False) as archive:
        cache_arrays = {name: np.asarray(archive[name]).copy()
                        for name in archive.files}
    cache_arrays["vjp_dx"].flat[0] += 3.0
    manifest = json.loads(cache_arrays["__manifest__"].tobytes().decode("utf-8"))
    manifest["arrays"]["vjp_dx"]["sha256"] = \\
        reference_cache._array_digest(cache_arrays["vjp_dx"])
    cache_arrays["__manifest__"] = np.frombuffer(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8)
    np.savez_compressed(cache_file, **cache_arrays)
    Path(os.environ["FACTOR_INTERACTION_MUTATION_MARKER"]).write_text(
        hashlib.sha256(cache_file.read_bytes()).hexdigest(), encoding="ascii")
""",
        encoding="utf-8")


class ValidatorReferenceCacheIntegrationTests(unittest.TestCase):
    def test_cached_and_uncached_runner_checks_match_for_dtype_and_vjp_modes(self):
        configurations = (
            {
                "workload": "correctness_ordered_role_context_variants",
                "activation_dtype": "fp16",
                "parameter_dtype": "fp32",
                "output_dtype": "fp16",
                "gradient_mode": "overwrite",
                "repeat_vjp": True,
            },
            {
                "workload": "correctness_ordered_tied_zero",
                "activation_dtype": "fp32",
                "parameter_dtype": "fp16",
                "output_dtype": "fp32",
                "gradient_mode": "accumulate",
                "repeat_vjp": True,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "reference-cache"
            for index, config in enumerate(configurations):
                with self.subTest(configuration=config):
                    uncached = _load_summary(_run_validator(
                        root, f"uncached-{index}", **config))
                    _load_summary(_run_validator(
                        root, f"cache-fill-{index}", cache_dir=cache_dir, **config))
                    cached = _invoke_with_reference_evaluation_forbidden(
                        root, f"cache-hit-{index}", cache_dir=cache_dir, **config)
                    self.assertTrue(uncached["validated"])
                    self.assertTrue(cached["validated"])
                    self.assertEqual(cached["input_case_sha256"],
                                     uncached["input_case_sha256"])
                    self.assertEqual(cached["contract_evidence"],
                                     uncached["contract_evidence"])
                    self.assertEqual(cached["contract_evidence"].keys(),
                                     uncached["contract_evidence"].keys())
                    self.assertIn("dx_repeat", cached["contract_evidence"])
                    if config["gradient_mode"] == "accumulate":
                        # The oracle cache contains the base VJP.  The cached
                        # invocation still validates the runner's initialized
                        # accumulation result and its repeated VJP above it.
                        self.assertIn("dgamma_repeat", cached["contract_evidence"])

    def test_damaged_and_stale_entries_fail_closed(self):
        config = {
            "workload": "correctness_ordered_tied_zero",
            "activation_dtype": "fp16",
            "parameter_dtype": "fp32",
            "output_dtype": "fp16",
            "gradient_mode": "accumulate",
            "repeat_vjp": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "reference-cache"
            summary = _load_summary(_run_validator(
                root, "cache-fill", cache_dir=cache_dir, **config))
            cache_file = reference_cache.cache_path(
                summary["input_case_sha256"], np.float16, True, cache_dir)
            valid_archive = cache_file.read_bytes()

            damaged = b"PK\x03\x04truncated reference cache archive"
            cache_file.write_bytes(damaged)
            damaged_result = _run_validator(
                root, "damaged-cache", cache_dir=cache_dir, **config)
            self.assertNotEqual(damaged_result.returncode, 0)
            self.assertIn("reference cache archive is damaged or unreadable",
                          damaged_result.stderr)
            self.assertEqual(cache_file.read_bytes(), damaged)

            cache_file.write_bytes(valid_archive)
            with np.load(cache_file, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]).copy()
                          for name in archive.files}
            manifest = json.loads(arrays["__manifest__"].tobytes().decode("utf-8"))
            manifest["reference_sha256"] = "0" * 64
            arrays["__manifest__"] = np.frombuffer(
                json.dumps(manifest, sort_keys=True,
                           separators=(",", ":")).encode("utf-8"),
                dtype=np.uint8)
            np.savez_compressed(cache_file, **arrays)
            stale = cache_file.read_bytes()
            stale_result = _run_validator(
                root, "stale-cache", cache_dir=cache_dir, **config)
            self.assertNotEqual(stale_result.returncode, 0)
            self.assertIn("reference cache identity differs from the current request",
                          stale_result.stderr)
            self.assertEqual(cache_file.read_bytes(), stale)

    def test_cached_accumulate_reference_still_rejects_wrong_candidate_gradients(self):
        config = {
            "workload": "correctness_ordered_role_context_variants",
            "activation_dtype": "fp16",
            "parameter_dtype": "fp32",
            "output_dtype": "fp16",
            "gradient_mode": "accumulate",
            "repeat_vjp": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "reference-cache"
            _load_summary(_run_validator(
                root, "cache-fill", cache_dir=cache_dir, **config))
            faulting_stub = root / "oracle_wire_stub.py"
            _write_faulting_stub(faulting_stub)

            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["FACTOR_INTERACTION_TEST_SOURCE"] = str(TEST_SOURCE)
            env["FACTOR_INTERACTION_ORACLE_STUB"] = str(STUB)
            env["FACTOR_INTERACTION_FAULT"] = "missing_accumulation"
            args = _validator_arguments(
                output_dir=root / "artifacts-wrong-accumulate",
                runner=faulting_stub, cache_dir=cache_dir, **config)
            result = subprocess.run([sys.executable, str(VALIDATOR), *args], cwd=ROOT,
                                    env=env, check=False, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dx: outside FP32 contract", result.stderr)

    def test_candidate_cannot_rewrite_another_valid_cache_entry(self):
        current = {
            "workload": "correctness_ordered_role_context_variants",
            "activation_dtype": "fp16",
            "parameter_dtype": "fp32",
            "output_dtype": "fp16",
            "gradient_mode": "overwrite",
            "repeat_vjp": True,
        }
        other = {
            "workload": "correctness_ordered_tied_zero",
            "activation_dtype": "fp16",
            "parameter_dtype": "fp32",
            "output_dtype": "fp16",
            "gradient_mode": "overwrite",
            "repeat_vjp": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "reference-cache"
            _load_summary(_run_validator(
                root, "cache-current", cache_dir=cache_dir, **current))
            other_summary = _load_summary(_run_validator(
                root, "cache-other", cache_dir=cache_dir, **other))
            other_cache_file = reference_cache.cache_path(
                other_summary["input_case_sha256"], np.float16, True, cache_dir)
            original_other = other_cache_file.read_bytes()
            faulting_stub = root / "oracle_wire_stub.py"
            _write_faulting_stub(faulting_stub)

            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["FACTOR_INTERACTION_TEST_SOURCE"] = str(TEST_SOURCE)
            env["FACTOR_INTERACTION_ORACLE_STUB"] = str(STUB)
            env["FACTOR_INTERACTION_FAULT"] = "rewrite_other_cache_entry"
            env["FACTOR_INTERACTION_CACHE_FILE"] = str(other_cache_file)
            mutation_marker = root / "mutated-other-cache.sha256"
            env["FACTOR_INTERACTION_MUTATION_MARKER"] = str(mutation_marker)
            args = _validator_arguments(
                output_dir=root / "artifacts-cache-mutation",
                runner=faulting_stub, cache_dir=cache_dir, **current)
            result = subprocess.run([sys.executable, str(VALIDATOR), *args], cwd=ROOT,
                                    env=env, check=False, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("reference cache changed during native runner execution",
                          result.stderr)
            self.assertNotEqual(mutation_marker.read_text(encoding="ascii"),
                                hashlib.sha256(original_other).hexdigest())
            self.assertFalse(cache_dir.exists())


if __name__ == "__main__":
    unittest.main()
