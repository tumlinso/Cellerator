"""Focused CUDA checks for the optional resident Torch adapter."""
from __future__ import annotations

import gc
import numpy as np
import pytest
import weakref

torch = pytest.importorskip("torch")
from cellerator import cuda as ce_cuda  # noqa: E402
from cellerator.torch import cuda as torch_cuda  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not ce_cuda.resident_cuda_available,
    reason="requires Torch CUDA and the Cellerator resident CUDA capability",
)


def _tensor(values):
    return torch.tensor(values, dtype=torch.float32, device="cuda").contiguous()


def test_native_buffer_dlpack_is_zero_copy_and_keeps_native_owner():
    stream = torch_cuda.current_stream()
    owner = ce_cuda.Buffer((4,), stream)
    owner.upload(np.asarray([1, 2, 3, 4], dtype=np.float32))
    alias = torch_cuda.as_tensor(owner)
    assert alias.data_ptr() == owner.data_ptr()
    assert alias.tolist() == [1.0, 2.0, 3.0, 4.0]
    del owner
    assert alias.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_native_dlpack_hands_off_producer_work_to_another_torch_stream():
    producer = ce_cuda.Stream(0)
    left = ce_cuda.Buffer((4,), producer)
    right = ce_cuda.Buffer((4,), producer)
    output = ce_cuda.Buffer((4,), producer)
    left.upload(np.asarray([1, 2, 3, 4], dtype=np.float32))
    right.upload(np.asarray([2, 3, 4, 5], dtype=np.float32))
    ce_cuda.multiply_into(left, right, output, producer)

    consumer = torch.cuda.Stream(device=0)
    with torch.cuda.stream(consumer):
        alias = torch_cuda.as_tensor(output)
        output_ptr = output.data_ptr()
        assert alias.data_ptr() == output_ptr
        del output
        actual = alias.to("cpu")
    assert actual.tolist() == [2.0, 6.0, 12.0, 20.0]


def test_borrowed_arithmetic_matches_torch_and_rejects_output_alias():
    a = _tensor([[1, 2], [3, 4]])
    b = _tensor([[5, 6], [7, 8]])
    out = torch.empty_like(a)
    stream = torch_cuda.current_stream(a.device)
    torch_cuda.multiply_into(a, b, out, stream)
    assert torch.equal(out, a * b)
    torch_cuda.axpby_into(2.0, a, -0.5, b, out, stream)
    assert torch.equal(out, 2.0 * a - 0.5 * b)
    with pytest.raises(ValueError, match="overlap"):
        torch_cuda.multiply_into(a, b, a, stream)


def test_record_stream_keeps_temporary_inputs_live_across_allocator_reuse():
    producer = torch.cuda.Stream(device=0)
    allocator_pressure = torch.cuda.Stream(device=0)
    size = 1 << 18
    with torch.cuda.stream(producer):
        a = torch.arange(size, dtype=torch.float32, device="cuda")
        b = torch.full_like(a, 3.0)
        out = torch.empty_like(a)
        torch_cuda.multiply_into(a, b, out, torch_cuda.current_stream(0))
        del a, b

    with torch.cuda.stream(allocator_pressure):
        scratch = [torch.empty((size,), dtype=torch.float32, device="cuda") for _ in range(4)]
        for index, tensor in enumerate(scratch):
            tensor.fill_(float(index))
        allocator_pressure.wait_stream(producer)
        actual = out.to("cpu")

    expected = torch.arange(size, dtype=torch.float32) * 3.0
    assert torch.equal(actual, expected)


def test_borrowed_tensor_contract_rejects_invalid_layout_dtype_and_grad():
    stream = torch_cuda.current_stream()
    with pytest.raises(ValueError, match="float32"):
        torch_cuda.borrow(torch.ones(4, dtype=torch.float64, device="cuda"), stream)
    with pytest.raises(ValueError, match="contiguous"):
        torch_cuda.borrow(torch.ones((4, 4), device="cuda")[:, ::2], stream)
    with pytest.raises(ValueError, match="requires_grad"):
        torch_cuda.borrow(torch.ones(4, device="cuda", requires_grad=True), stream)
    with pytest.raises(ValueError, match="CUDA"):
        torch_cuda.borrow(torch.ones(4), stream)
    with pytest.raises(ValueError, match="current PyTorch"):
        other = torch.cuda.Stream(device="cuda")
        wrong = ce_cuda.Stream.borrow(0, int(other.cuda_stream), other)
        torch_cuda.borrow(torch.ones(4, device="cuda"), wrong)


def test_native_capability_guard_precedes_torch_device_queries(monkeypatch):
    monkeypatch.setattr(ce_cuda, "resident_cuda_available", False)

    def unexpected_device_query(*args, **kwargs):
        raise AssertionError("Torch CUDA device query ran before the capability guard")

    monkeypatch.setattr(torch.cuda, "current_device", unexpected_device_query)
    monkeypatch.setattr(torch.cuda, "current_stream", unexpected_device_query)
    with pytest.raises(RuntimeError, match="resident CUDA capability is unavailable"):
        torch_cuda.current_stream()


def test_prepared_csr_reuses_native_owner_and_publishes_new_generation():
    indptr = np.asarray([0, 2, 3], dtype=np.uint64)
    indices = np.asarray([0, 1, 1], dtype=np.uint64)
    weights = _tensor([0.25, 0.75, 1.0])
    adapter = torch_cuda.PreparedCsr(indptr, indices, weights, 2, 2)
    x = _tensor([[2, 4], [6, 8]])
    out = torch.empty_like(x)
    adapter.apply_into(x, out)
    assert torch.equal(out, torch.tensor([[5, 7], [6, 8]], device="cuda"))

    generation = adapter.generation
    replacement = _tensor([1.0, 0.0, 0.5])
    adapter.publish_values(replacement)
    assert adapter.generation == generation + 1
    adapter.apply_into(x, out)
    assert torch.equal(out, torch.tensor([[2, 4], [3, 4]], device="cuda"))
    assert adapter.native.generation == adapter.generation
    adapter.close()


def test_prepared_csr_close_releases_weight_owner_and_guards_use_after_close():
    indptr = np.asarray([0, 1, 2], dtype=np.uint64)
    indices = np.asarray([0, 1], dtype=np.uint64)
    weights = _tensor([1.0, 1.0])
    weights_ref = weakref.ref(weights)
    adapter = torch_cuda.PreparedCsr(indptr, indices, weights, 2, 1)
    del weights
    torch.cuda.synchronize()

    adapter.close()
    adapter.close()  # idempotent
    torch.cuda.synchronize()
    gc.collect()
    assert weights_ref() is None
    assert adapter._weights_owner is None
    assert adapter._weights_buffer is None
    with pytest.raises(RuntimeError, match="closed"):
        adapter.apply_into(None, None)
    with pytest.raises(RuntimeError, match="closed"):
        adapter.publish_values(None)
