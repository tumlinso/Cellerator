"""Checks the optional ggml boundary without making it a core dependency."""
import unittest

import numpy as np

from cellerator.adapters import ggml


class GgmlAdapterTests(unittest.TestCase):
    def test_dependency_is_optional_and_failure_is_actionable(self):
        if ggml.available():
            return
        with self.assertRaisesRegex(ImportError, "CELLERATOR_ENABLE_GGML=ON"):
            ggml.quantize(np.zeros((2, 64), dtype=np.float32), "q8_0")

    def test_block_quantization_round_trip_preserves_row_major_shape(self):
        if not ggml.available():
            self.skipTest("optional native ggml bridge is not built")
        values = np.linspace(-1.0, 1.0, 2 * 3 * 64, dtype=np.float32).reshape(2, 3, 64)
        packed = ggml.quantize(values, "q8_0")
        restored = ggml.dequantize(packed)
        self.assertEqual(packed.format, "q8_0")
        self.assertEqual(packed.shape, values.shape)
        self.assertEqual(packed.payload.__class__, bytes)
        self.assertEqual(restored.shape, values.shape)
        self.assertLess(float(np.max(np.abs(restored - values))), 0.01)

    def test_block_size_is_checked(self):
        if not ggml.available():
            self.skipTest("optional native ggml bridge is not built")
        with self.assertRaises(ValueError):
            ggml.quantize(np.zeros((2, 31), dtype=np.float32), "q8_0")


if __name__ == "__main__":
    unittest.main()
