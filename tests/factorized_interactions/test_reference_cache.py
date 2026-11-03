"""Integrity and equivalence tests for the packet-local scalar-oracle cache."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

import reference_cache
from case_io import read_result, write_case
from reference import Evaluation, evaluate
from workloads import make_workload


ROOT = Path(__file__).resolve().parents[2]
STUB = Path(__file__).resolve().with_name("oracle_wire_stub.py")


def _assert_evaluations_equal(test: unittest.TestCase, actual: Evaluation,
                              expected: Evaluation) -> None:
    for name in ("y_float64", "y", "sum_abs_y_contributions"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    test.assertEqual(actual.vjp is None, expected.vjp is None)
    test.assertEqual(actual.vjp_scale is None, expected.vjp_scale is None)
    if expected.vjp is not None:
        for name in ("dx", "dz", "dalpha", "dbias", "dgamma"):
            np.testing.assert_array_equal(getattr(actual.vjp, name),
                                          getattr(expected.vjp, name))
            np.testing.assert_array_equal(getattr(actual.vjp_scale, name),
                                          getattr(expected.vjp_scale, name))


class FactorizedInteractionReferenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.problem, self.dy = make_workload(
            "correctness_ordered_role_context_variants", np.float16, np.float32)

    def _write_case(self, directory: Path, problem=None, output_dtype=np.float16):
        return write_case(directory / "input.case", self.problem if problem is None else problem,
                          self.dy, output_dtype)

    def test_cached_result_matches_uncached_scalar_oracle_and_keeps_all_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_identity = self._write_case(root)
            cached = reference_cache.evaluate_cached(
                self.problem, self.dy, np.float16, case_identity, root)
            uncached = evaluate(self.problem, self.dy, output_dtype=np.float16)
            _assert_evaluations_equal(self, cached, uncached)
            self.assertEqual(cached.y.dtype, np.dtype(np.float16))
            self.assertEqual(cached.y_float64.dtype, np.dtype(np.float64))
            self.assertEqual(cached.vjp_scale.dz.dtype, np.dtype(np.float64))
            self.assertEqual(len(list(root.glob(".factorized-reference-*.npz"))), 1)

    def test_candidate_case_paths_share_cache_by_exact_bytes_and_match_oracle_wire_stub(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packet_output = root / "suite-output"
            first_identity = self._write_case(root / "candidate-a")
            second_identity = self._write_case(root / "candidate-b")
            self.assertEqual(first_identity["sha256"], second_identity["sha256"])
            first = reference_cache.evaluate_cached(
                self.problem, self.dy, np.float16, first_identity, packet_output)

            with mock.patch.object(reference_cache, "evaluate",
                                   side_effect=AssertionError("cache miss")):
                second = reference_cache.evaluate_cached(
                    self.problem, self.dy, np.float16, second_identity, packet_output)
            _assert_evaluations_equal(self, second, first)

            stub_output = root / "oracle.result"
            subprocess.run([sys.executable, str(STUB), "--input",
                            first_identity["path"], "--output", str(stub_output),
                            "--route", "direct"], check=True, capture_output=True,
                            text=True, cwd=ROOT)
            wire = read_result(stub_output, self.problem, np.float16)
            np.testing.assert_array_equal(wire["y"], second.y)
            for name in ("dx", "dz", "dalpha", "dbias", "dgamma"):
                np.testing.assert_array_equal(wire[name],
                                              getattr(second.vjp, name).astype(np.float32))
            self.assertEqual(hashlib.sha256(Path(first_identity["path"]).read_bytes()).hexdigest(),
                             first_identity["sha256"])
            self.assertEqual(len(list(packet_output.glob(".factorized-reference-*.npz"))), 1)

    def test_altered_case_bytes_are_rejected_and_new_exact_bytes_get_a_new_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_identity = self._write_case(root / "case-a")
            reference_cache.evaluate_cached(
                self.problem, self.dy, np.float16, case_identity, root / "cache")
            cache_before = set((root / "cache").glob(".factorized-reference-*.npz"))

            case_path = Path(case_identity["path"])
            payload = bytearray(case_path.read_bytes())
            payload[-1] ^= 0x01
            case_path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                reference_cache.evaluate_cached(
                    self.problem, self.dy, np.float16, case_identity, root / "cache")

            changed_x = self.problem.x.copy()
            changed_x[0, 0, 0] = np.float16(changed_x[0, 0, 0] + np.float16(0.25))
            changed_problem = type(self.problem)(
                changed_x, self.problem.z, self.problem.alpha, self.problem.bias,
                self.problem.gamma, self.problem.mechanisms, self.problem.target_count)
            changed_identity = self._write_case(root / "case-b", changed_problem)
            changed = reference_cache.evaluate_cached(
                changed_problem, self.dy, np.float16, changed_identity, root / "cache")
            expected = evaluate(changed_problem, self.dy, output_dtype=np.float16)
            _assert_evaluations_equal(self, changed, expected)
            cache_after = set((root / "cache").glob(".factorized-reference-*.npz"))
            self.assertEqual(len(cache_before), 1)
            self.assertEqual(len(cache_after), 2)
            self.assertNotEqual(case_identity["sha256"], changed_identity["sha256"])

    def test_changed_oracle_or_workload_identity_rejects_existing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_identity = self._write_case(root)
            reference_cache.evaluate_cached(
                self.problem, self.dy, np.float16, case_identity, root)
            current = reference_cache._source_hashes()
            for name in ("reference_sha256", "workloads_sha256",
                         "workload_manifest_sha256"):
                changed = dict(current)
                changed[name] = hashlib.sha256((name + " changed").encode()).hexdigest()
                with self.subTest(identity=name):
                    with mock.patch.object(reference_cache, "_source_hashes",
                                           return_value=changed):
                        with self.assertRaisesRegex(ValueError, "identity differs"):
                            reference_cache.evaluate_cached(
                                self.problem, self.dy, np.float16, case_identity, root)

    def test_problem_structure_mismatch_cannot_reuse_same_packet_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_identity = self._write_case(root)
            reference_cache.evaluate_cached(
                self.problem, self.dy, np.float16, case_identity, root)
            changed_problem = replace(
                self.problem, mechanisms=(self.problem.mechanisms[0],))
            with self.assertRaisesRegex(ValueError, "identity differs"):
                reference_cache.evaluate_cached(
                    changed_problem, self.dy, np.float16, case_identity, root)

    def test_tampered_array_content_is_rejected_by_cached_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_identity = self._write_case(root)
            reference_cache.evaluate_cached(
                self.problem, self.dy, np.float16, case_identity, root)
            path = reference_cache.cache_path(case_identity["sha256"], np.float16,
                                              True, root)
            with np.load(path, allow_pickle=False) as archive:
                contents = {name: np.asarray(archive[name]).copy()
                            for name in archive.files}
            contents["vjp_dx"].flat[0] += 1.0
            np.savez_compressed(path, **contents)
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                reference_cache.evaluate_cached(
                    self.problem, self.dy, np.float16, case_identity, root)

    def test_tampered_array_shape_and_dtype_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case_identity = self._write_case(root)
            reference_cache.evaluate_cached(
                self.problem, self.dy, np.float16, case_identity, root)
            path = reference_cache.cache_path(case_identity["sha256"], np.float16,
                                              True, root)
            original = path.read_bytes()
            for mode in ("shape", "dtype"):
                with self.subTest(mode=mode):
                    with np.load(path, allow_pickle=False) as archive:
                        contents = {name: np.asarray(archive[name]).copy()
                                    for name in archive.files}
                    manifest = json.loads(contents["__manifest__"].tobytes().decode("utf-8"))
                    if mode == "shape":
                        contents["vjp_dx"] = contents["vjp_dx"].reshape(1, -1, 1)
                        manifest["arrays"]["vjp_dx"]["shape"] = [1, 3, 1]
                    else:
                        contents["y"] = contents["y"].astype(np.float32)
                    contents["__manifest__"] = np.frombuffer(
                        json.dumps(manifest, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8"),
                        dtype=np.uint8)
                    np.savez_compressed(path, **contents)
                    expected_error = "shape mismatch" if mode == "shape" else "dtype metadata mismatch"
                    with self.assertRaisesRegex(ValueError, expected_error):
                        reference_cache.evaluate_cached(
                            self.problem, self.dy, np.float16, case_identity, root)
                    path.write_bytes(original)

    def test_output_dtype_and_gradient_availability_are_part_of_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fp16_identity = self._write_case(root / "fp16", output_dtype=np.float16)
            fp32_identity = self._write_case(root / "fp32", output_dtype=np.float32)
            fp16 = reference_cache.evaluate_cached(
                self.problem, self.dy, np.float16, fp16_identity, root)
            fp32 = reference_cache.evaluate_cached(
                self.problem, self.dy, np.float32, fp32_identity, root)
            self.assertEqual(fp16.y.dtype, np.dtype(np.float16))
            self.assertEqual(fp32.y.dtype, np.dtype(np.float32))

            forward = reference_cache.evaluate_cached(
                self.problem, None, np.float16, fp16_identity, root)
            self.assertIsNone(forward.vjp)
            self.assertIsNone(forward.vjp_scale)
            self.assertNotEqual(
                reference_cache.cache_path(fp16_identity["sha256"], np.float16,
                                           True, root),
                reference_cache.cache_path(fp16_identity["sha256"], np.float16,
                                           False, root))
            self.assertEqual(len(list(root.glob(".factorized-reference-*.npz"))), 3)

    def test_cache_output_path_rejects_symbolic_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "outside"
            destination.mkdir()
            link = root / "output-link"
            link.symlink_to(destination, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic links"):
                reference_cache.cache_path("0" * 64, np.float16, True, link)
            self.assertEqual(list(destination.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

