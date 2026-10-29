"""Host-only validation of the optional binding; never initialize CUDA.

Run with the built extension directory appended to cellerator.__path__, or
against an installed package. These checks intentionally do not test kernels.
"""
import unittest
import numpy as np

try:
    from cellerator import _modeling_cuda as native
except ImportError:
    native = None


@unittest.skipIf(native is None, "optional CUDA binding is not built")
class HostContractTests(unittest.TestCase):
    def setUp(self):
        self.stream = native.Stream.borrow(0, 0)

    def test_zero_allocation_and_borrowed_stream(self):
        self.assertFalse(self.stream.owns_handle)
        buffer = native.DeviceBuffer(0, 0)
        self.assertEqual(buffer.view().pointer, 0)
        self.assertEqual(buffer.view().nbytes, 0)

    def test_view_bounds(self):
        view = native.DeviceView(4096, 64, 0)
        self.assertEqual(view.subview(12, 8).pointer, 4108)
        with self.assertRaises(IndexError):
            view.subview(63, 2)
        with self.assertRaises(ValueError):
            native.DeviceView(0, 4, 0)

    def test_plan_rejects_inconsistent_reverse_map(self):
        with self.assertRaises(ValueError):
            native.DispatchPlan(1, [0], [1.0], [0, 1], [1], [0, 1], self.stream)
        with self.assertRaises(ValueError):
            native.DispatchPlan(2, [1], [1.0], [0, 1, 1], [0], [0, 1], self.stream)
        with self.assertRaises(ValueError):
            native.DispatchPlan(1, [0, 0], [1.0, 1.0], [0, 2], [0, 0], [0, 2], self.stream)

    def test_profile_rejects_invalid_host_metadata(self):
        for offsets, levels, mapping, bits in [
            ([0, 2], [1.0, 1.0], [0], 1),
            ([0, 2], [1.0, float("nan")], [0], 1),
            ([0, 2], [0.0, 1.0], [1], 1),
            ([0, 2], [0.0, 1.0], [0], 3),
        ]:
            with self.assertRaises(ValueError):
                native.QuantizationProfile(offsets, levels, mapping, bits, self.stream)

    def test_grouped_linear_rejects_invalid_metadata_before_cuda(self):
        with self.assertRaises(ValueError):
            native.GroupedLinear(0, 2, [0], [], native.DType.float32, 0)
        with self.assertRaises(ValueError):
            native.GroupedLinear(2, 2, [0, 1], [native.DeviceView(4097, 16, 0)],
                                 native.DType.float32, 0)

    def test_python_borrowed_shape_and_alignment(self):
        from cellerator.adapters.cuda import DeviceArray
        array = DeviceArray.borrow(4096, (2, 3), dtype=np.float16,
                                   device=0, stream=self.stream, owner=object())
        self.assertEqual(array.nbytes, 12)
        with self.assertRaises(ValueError):
            DeviceArray.borrow(4097, (1,), dtype=np.float32,
                               device=0, stream=self.stream, owner=object())
        with self.assertRaises(ValueError):
            DeviceArray.borrow(4096, (1,), dtype=np.float32,
                               device=0, stream=self.stream, owner=None)


if __name__ == "__main__":
    unittest.main()
