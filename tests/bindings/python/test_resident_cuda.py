import numpy as np
import pytest

from cellerator import cuda


def test_host_only_import_reports_resident_capability_false():
    if cuda.resident_cuda_available:
        pytest.skip("resident CUDA bindings are present")
    with pytest.raises(RuntimeError, match="resident CUDA bindings are unavailable"):
        cuda.Stream(0)


requires_resident_cuda = pytest.mark.skipif(
    not cuda.resident_cuda_available,
    reason="requires the resident Cellerator CUDA binding",
)


@requires_resident_cuda
def test_resident_numeric_buffers_and_events():
    stream = cuda.Stream(0)
    left = cuda.Buffer((2, 3), stream)
    right = cuda.Buffer((2, 3), stream)
    output = cuda.Buffer((2, 3), stream)
    left_values = np.arange(6, dtype=np.float32).reshape(2, 3)
    right_values = np.full((2, 3), 2, dtype=np.float32)
    left.upload(left_values)
    right.upload(right_values)

    start = cuda.Event.record(stream)
    cuda.multiply_into(left, right, output, stream)
    stop = cuda.Event.record(stream)
    assert start.elapsed_ms(stop) >= 0.0
    np.testing.assert_array_equal(output.download(), left_values * right_values)
    assert output.shape == (2, 3)
    assert output.nbytes == 6 * np.dtype(np.float32).itemsize

    affine = cuda.Buffer((2, 3), stream)
    cuda.axpby_into(0.5, left, -1.0, right, affine, stream)
    np.testing.assert_array_equal(affine.download(), 0.5 * left_values - right_values)


@requires_resident_cuda
def test_prepared_csr_generation_and_nonblocking_value_leases():
    stream = cuda.Stream(0)
    indptr = np.array([0, 2, 3], dtype=np.uint64)
    indices = np.array([0, 1, 1], dtype=np.uint64)
    weights = cuda.Buffer((3,), stream)
    weights.upload(np.array([0.25, 0.75, 1.0], dtype=np.float32))
    prepared = cuda.PreparedCsr(indptr, indices, weights, 2, 2, stream)
    assert prepared.generation == 1
    assert prepared.prepared_bytes > 0

    source = cuda.Buffer((2, 2), stream)
    result = cuda.Buffer((2, 2), stream)
    source_values = np.array([[2, 4], [6, 8]], dtype=np.float32)
    source.upload(source_values)
    prepared.apply_into(source, result)
    expected = np.array([[5, 7], [6, 8]], dtype=np.float32)
    np.testing.assert_allclose(result.download(), expected, rtol=1e-6, atol=1e-6)

    replacement = cuda.Buffer((3,), stream)
    replacement.upload(np.array([0.5, 0.5, 1.0], dtype=np.float32))
    prepared.publish_values(replacement)
    for i in range(20):
        current = cuda.Buffer((3,), stream)
        current.upload(np.array([1.0 - i / 100, i / 100, 0.5], dtype=np.float32))
        prepared.publish_values(current)
    assert prepared.generation == 22
    prepared.apply_into(source, result)
    np.testing.assert_allclose(
        result.download(),
        np.array([[2 * 0.81 + 6 * 0.19, 4 * 0.81 + 8 * 0.19], [3, 4]], dtype=np.float32),
        rtol=1e-6, atol=1e-6,
    )
    stream.synchronize()
    assert prepared.pending_value_leases == 0
    prepared.close()
    with pytest.raises(ValueError, match="closed"):
        prepared.apply_into(source, result)


@requires_resident_cuda
def test_prepared_csr_rejects_output_overlap_and_wrong_context():
    stream = cuda.Stream(0)
    weights = cuda.Buffer((2,), stream)
    weights.upload(np.ones(2, dtype=np.float32))
    prepared = cuda.PreparedCsr(
        np.array([0, 1, 2], dtype=np.uint64),
        np.array([0, 1], dtype=np.uint64), weights, 2, 2, stream,
    )
    state = cuda.Buffer((2, 2), stream)
    with pytest.raises(ValueError, match="disjoint"):
        prepared.apply_into(state, state)
    prepared.close()


@requires_resident_cuda
def test_borrowed_buffer_rejects_invalid_capacity_alignment_and_extent():
    stream = cuda.Stream(0)
    owner = cuda.Buffer((4,), stream)
    pointer = owner.data_ptr()
    with pytest.raises(ValueError, match="capacity"):
        cuda.Buffer.borrow(pointer, (4,), 4, stream, owner)
    with pytest.raises(ValueError, match="aligned"):
        cuda.Buffer.borrow(pointer + 1, (1,), 4, stream, owner)
    with pytest.raises(OverflowError, match="extent overflow"):
        cuda.Buffer.borrow(pointer, ((1 << 63) - 1, 4), 16, stream, owner)


@requires_resident_cuda
def test_stream_and_prepared_relation_reject_invalid_device_and_wrong_stream():
    with pytest.raises(ValueError, match="device ordinal is out of range"):
        cuda.Stream.borrow((1 << 31) - 1, 0, object())
    with pytest.raises(ValueError, match="not valid on the declared device"):
        cuda.Stream.borrow(0, 1, object())

    stream = cuda.Stream(0)
    other_stream = cuda.Stream(0)
    weights = cuda.Buffer((2,), stream)
    weights.upload(np.ones(2, dtype=np.float32))
    prepared = cuda.PreparedCsr(
        np.array([0, 1, 2], dtype=np.uint64),
        np.array([0, 1], dtype=np.uint64), weights, 2, 1, stream,
    )
    values = cuda.Buffer((2, 1), other_stream)
    output = cuda.Buffer((2, 1), stream)
    with pytest.raises(ValueError, match="operation device and stream"):
        prepared.apply_into(values, output)
    prepared.close()
