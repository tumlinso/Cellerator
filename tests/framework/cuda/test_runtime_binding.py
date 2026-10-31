"""Host-only validation of the optional binding; never initialize CUDA.

Run with the built extension directory appended to cellerator.__path__, or
against an installed package. These checks intentionally do not test kernels.
"""
import os
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


@unittest.skipUnless(native is not None and os.environ.get("CELLERATOR_TEST_CUDA") == "1",
                     "requires an admitted CUDA run with CELLERATOR_TEST_CUDA=1")
class DeviceIntegrationTests(unittest.TestCase):
    """Real wrapper calls checked against small, independent NumPy oracles."""

    def setUp(self):
        from cellerator.adapters.cuda import Stream
        self.device = int(os.environ.get("CELLERATOR_TEST_CUDA_DEVICE", "0"))
        self.stream = Stream(self.device)

    def test_dispatch_cross_stream_gather_combine_and_backward(self):
        from types import SimpleNamespace
        from cellerator.adapters.cuda import CudaDispatchPlan, DeviceArray

        host_plan = SimpleNamespace(
            origin_count=2,
            dispatch_origin_slots=(1, 0, 1),
            dispatch_weights=(0.5, -1.0, 0.25),
            origin_offsets=(0, 1, 3),
            origin_dispatch_indices=(1, 0, 2),
            expert_offsets=(0, 1, 3),
        )
        plan = CudaDispatchPlan(host_plan, stream=self.stream)
        producer = native.Stream(self.device)
        origins = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        x = DeviceArray.from_numpy(origins, stream=producer)
        gathered = plan.gather(x)
        expected_gathered = origins[[1, 0, 1]]
        np.testing.assert_allclose(gathered.to_numpy(), expected_gathered, rtol=0, atol=0)

        expert_output = np.asarray([[2, 1, 0], [3, 2, 1], [6, 5, 4]], dtype=np.float32)
        expert = DeviceArray.from_numpy(expert_output, stream=producer)
        combined = plan.combine(expert)
        expected_combined = np.zeros((2, 3), dtype=np.float32)
        for i, origin in enumerate(host_plan.dispatch_origin_slots):
            expected_combined[origin] += host_plan.dispatch_weights[i] * expert_output[i]
        np.testing.assert_allclose(combined.to_numpy(), expected_combined, rtol=1e-6, atol=1e-6)

        grad_origin = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        grad = DeviceArray.from_numpy(grad_origin, stream=producer)
        grad_expert, grad_weight = plan.combine_backward(grad, expert)
        expected_grad_expert = np.stack([
            grad_origin[origin] * weight
            for origin, weight in zip(host_plan.dispatch_origin_slots, host_plan.dispatch_weights)
        ])
        expected_grad_weight = np.asarray([
            np.dot(grad_origin[origin], expert_output[i])
            for i, origin in enumerate(host_plan.dispatch_origin_slots)
        ], dtype=np.float32)
        np.testing.assert_allclose(grad_expert.to_numpy(), expected_grad_expert, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(grad_weight.to_numpy(), expected_grad_weight, rtol=1e-6, atol=1e-6)

        gathered_grad = DeviceArray.from_numpy(expected_gathered, stream=producer)
        input_grad = plan.gather_backward(gathered_grad)
        expected_input_grad = np.zeros_like(origins)
        np.add.at(expected_input_grad, np.asarray(host_plan.dispatch_origin_slots), expected_gathered)
        np.testing.assert_allclose(input_grad.to_numpy(), expected_input_grad, rtol=1e-6, atol=1e-6)

    def test_grouped_linear_forward_backward_with_empty_expert(self):
        from cellerator.adapters.cuda import CudaGroupedLinear, DeviceArray

        for dtype, atol in ((np.float32, 3e-5), (np.float16, 2e-2)):
            with self.subTest(dtype=np.dtype(dtype).name):
                offsets = [0, 2, 2, 5]
                x_host = np.asarray([
                    [0.25, -0.5, 0.75], [1.0, 0.5, -0.25],
                    [-1.0, 0.25, 0.5], [0.5, 1.25, -0.75], [0.75, -0.25, 1.0],
                ], dtype=dtype)
                weights_host = [
                    np.asarray([[1, 0, -1], [0.5, 0.25, 0.75]], dtype=dtype),
                    np.asarray([[3, 2, 1], [1, 2, 3]], dtype=dtype),
                    np.asarray([[-0.5, 1, 0.25], [0.75, -1, 0.5]], dtype=dtype),
                ]
                x = DeviceArray.from_numpy(x_host, stream=native.Stream(self.device))
                weights = [DeviceArray.from_numpy(w, stream=self.stream) for w in weights_host]
                linear = CudaGroupedLinear(weights, offsets, stream=self.stream)
                actual = linear(x).to_numpy()

                x32 = x_host.astype(np.float32)
                weights32 = [w.astype(np.float32) for w in weights_host]
                expected = np.empty((len(x_host), 2), dtype=np.float32)
                for expert, (begin, end) in enumerate(zip(offsets, offsets[1:])):
                    expected[begin:end] = x32[begin:end] @ weights32[expert].T
                np.testing.assert_allclose(actual.astype(np.float32), expected,
                                           rtol=atol, atol=atol)

                grad_host = np.asarray([
                    [1, -0.5], [0.25, 0.75], [-1, 0.5], [0.5, 1.25], [0.75, -0.25],
                ], dtype=dtype)
                grad = DeviceArray.from_numpy(grad_host, stream=self.stream)
                grad_x, grad_weights = linear.backward(x, grad)
                grad32 = grad_host.astype(np.float32)
                expected_dx = np.empty_like(x32)
                expected_dw = []
                for expert, (begin, end) in enumerate(zip(offsets, offsets[1:])):
                    expected_dx[begin:end] = grad32[begin:end] @ weights32[expert]
                    expected_dw.append(grad32[begin:end].T @ x32[begin:end])
                np.testing.assert_allclose(grad_x.to_numpy().astype(np.float32), expected_dx,
                                           rtol=atol, atol=atol)
                for actual_dw, expected_expert_dw in zip(grad_weights, expected_dw):
                    np.testing.assert_allclose(actual_dw.to_numpy().astype(np.float32),
                                               expected_expert_dw, rtol=atol, atol=atol)

    def test_pinned_transfer_ticket_roundtrip(self):
        source = native.PinnedBuffer(16)
        destination = native.PinnedBuffer(16)
        source_values = np.frombuffer(source, dtype=np.float32)
        expected = np.asarray([1.25, -2.5, 3.75, 4.0], dtype=np.float32)
        source_values[:] = expected
        device = native.DeviceBuffer(16, self.device)
        upload = device.upload_pinned_async(source, self.stream)
        upload.wait()
        download = device.download_pinned_async(destination, self.stream)
        download.wait()
        np.testing.assert_array_equal(np.frombuffer(destination, dtype=np.float32), expected)


if __name__ == "__main__":
    unittest.main()
