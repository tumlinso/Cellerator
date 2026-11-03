"""Round-trip and rejection tests for the native runner wire protocol."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from case_io import (HEADER, MAGIC_RESULT, TYPE_CODE, read_case, read_result,
                     write_case)
from reference import evaluate
from workloads import make_workload


def encode_result(problem, output, vjp, output_dtype, repeat_vjp=None):
    term_count = sum(len(m.context_ids) for m in problem.mechanisms)
    counts = (
        problem.x.shape[2], problem.x.shape[1], problem.x.shape[0],
        problem.z.shape[0], problem.target_count, problem.alpha.shape[0],
        problem.bias.size, problem.gamma.size, len(problem.mechanisms), term_count,
    )
    pieces = [HEADER.pack(MAGIC_RESULT, *counts, TYPE_CODE[np.dtype(output_dtype)], 2,
                          2 if repeat_vjp is not None else 1)]
    pieces.append(np.ascontiguousarray(output, dtype=np.dtype(output_dtype).newbyteorder("<")).tobytes())
    for gradients in (vjp,) if repeat_vjp is None else (vjp, repeat_vjp):
        for name in ("dx", "dz", "dalpha", "dbias", "dgamma"):
            pieces.append(np.ascontiguousarray(getattr(gradients, name), dtype="<f4").tobytes())
    return b"".join(pieces)


class FactorizedInteractionWireTests(unittest.TestCase):
    def test_request_round_trip_preserves_axes_terms_and_role_order(self):
        problem, dy = make_workload("correctness_ordered_role_context_variants",
                                    np.float16, np.float32)
        with tempfile.TemporaryDirectory() as directory:
            case_path = Path(directory) / "case.bin"
            identity = write_case(case_path, problem, dy, np.float16)
            restored, restored_dy, output_dtype = read_case(case_path)
            self.assertEqual(identity["bytes"], case_path.stat().st_size)
            self.assertEqual(output_dtype, np.dtype(np.float16))
            self.assertEqual(restored.mechanisms, problem.mechanisms)
            self.assertEqual(restored.target_count, problem.target_count)
            for name in ("x", "z", "alpha", "bias", "gamma"):
                np.testing.assert_array_equal(getattr(restored, name), getattr(problem, name))
            np.testing.assert_array_equal(restored_dy, dy)
            self.assertEqual(restored.mechanisms[0].participants, (0, 0, 1, 2))
            self.assertEqual(restored.mechanisms[1].participants, (0, 1, 0, 2))
            self.assertEqual(restored.mechanisms[0].context_ids, (0, 1))
            self.assertEqual(restored.mechanisms[1].context_ids, (1, 0))
            self.assertEqual(restored.x.shape, (3, 1, 1))
            self.assertEqual(restored.z.shape, (2, 1))

    def test_request_round_trip_preserves_long_context_order(self):
        problem, dy = make_workload("correctness_context_cancellation_middle",
                                    np.float32, np.float32)
        with tempfile.TemporaryDirectory() as directory:
            case_path = Path(directory) / "case.bin"
            write_case(case_path, problem, dy, np.float32)
            restored, _, _ = read_case(case_path)
        terms = restored.mechanisms[0].context_ids
        self.assertEqual(len(terms), 8193)
        self.assertEqual(terms[4095:4098], (4095, 4096, 4097))

    def test_request_round_trip_preserves_product_residual_term_order(self):
        problem, dy = make_workload(
            "correctness_fma_product_residual_cancellation", np.float32, np.float32)
        with tempfile.TemporaryDirectory() as directory:
            case_path = Path(directory) / "case.bin"
            write_case(case_path, problem, dy, np.float32)
            restored, _, _ = read_case(case_path)
        self.assertEqual(restored.mechanisms, problem.mechanisms)
        terms = restored.mechanisms[0].context_ids
        self.assertEqual(len(terms), 8192)
        self.assertEqual(terms[:6], (0, 1, 0, 1, 0, 1))

    def test_result_round_trip_has_fp32_gradients_and_fp16_output(self):
        problem, dy = make_workload("correctness_ordered_tied_zero", np.float16, np.float32)
        expected = evaluate(problem, dy, output_dtype=np.float16)
        payload = encode_result(problem, expected.y, expected.vjp, np.float16)
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.bin"
            result_path.write_bytes(payload)
            actual = read_result(result_path, problem, np.float16)
        np.testing.assert_array_equal(actual["y"], expected.y)
        for name in ("dx", "dz", "dalpha", "dbias", "dgamma"):
            self.assertEqual(actual[name].dtype, np.dtype("<f4"))
            np.testing.assert_array_equal(actual[name],
                                          getattr(expected.vjp, name).astype(np.float32))

    def test_result_parser_rejects_trailing_bytes(self):
        problem, dy = make_workload("correctness_empty_cells")
        expected = evaluate(problem, dy, output_dtype=np.float32)
        payload = encode_result(problem, expected.y, expected.vjp, np.float32) + b"x"
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.bin"
            result_path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "trailing bytes"):
                read_result(result_path, problem, np.float32)

    def test_result_parser_preserves_second_vjp_set(self):
        problem, dy = make_workload("correctness_ordered_tied_zero")
        expected = evaluate(problem, dy, output_dtype=np.float16)
        payload = encode_result(problem, expected.y, expected.vjp, np.float16,
                                expected.vjp)
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.bin"
            result_path.write_bytes(payload)
            actual = read_result(result_path, problem, np.float16)
        for name in ("dx", "dz", "dalpha", "dbias", "dgamma"):
            np.testing.assert_array_equal(actual[f"{name}_repeat"], actual[name])


if __name__ == "__main__":
    unittest.main()
