"""CPU FP32 Product2 binding tests against independent formulas."""
from __future__ import annotations

import numpy as np
import pytest

from cellerator import (
    Axis, Identity, prepare_product2, product2_forward, product2_jvp, product2_vjp,
)


def _case():
    x = np.asarray([0.0, 0.4, -0.2], dtype=np.float32)
    k = np.asarray([0.7, -0.3], dtype=np.float32)
    a = np.asarray([1, 1], dtype=np.int64)
    b = np.asarray([2, 1], dtype=np.int64)
    return x, k, a, b


def _reference(x, k, a, b):
    return (k * x[a]) * x[b]


def test_native_prepared_product_forward_vjp_and_jvp():
    x, k, a, b = _case()
    prepared = prepare_product2(a, b, x.size, structure_generation=19)
    assert prepared.input_count == x.size
    assert prepared.packet_count == k.size
    assert prepared.structure_generation == 19
    expected = _reference(x, k, a, b)
    np.testing.assert_array_equal(prepared.forward(x, k), expected)
    cotangent = np.asarray([0.3, -0.7], dtype=np.float32)
    dx, dk = prepared.vjp(x, k, cotangent)
    expected_dx = np.zeros_like(x)
    expected_dk = cotangent * x[a] * x[b]
    for i in range(k.size):
        expected_dx[a[i]] += (cotangent[i] * k[i]) * x[b[i]]
        expected_dx[b[i]] += (cotangent[i] * k[i]) * x[a[i]]
    np.testing.assert_array_equal(dx, expected_dx)
    np.testing.assert_array_equal(dk, expected_dk)
    tangent_x = np.asarray([0.2, -0.5, 0.8], dtype=np.float32)
    tangent_k = np.asarray([0.1, 0.3], dtype=np.float32)
    tangent = prepared.jvp(x, k, tangent_x, tangent_k)
    expected_tangent = k * (tangent_x[a] * x[b] + x[a] * tangent_x[b]) + tangent_k * x[a] * x[b]
    np.testing.assert_array_equal(tangent, expected_tangent)
    np.testing.assert_array_equal(product2_forward(x, k, a, b), expected)
    np.testing.assert_array_equal(product2_vjp(x, k, a, b, cotangent)[0], expected_dx)
    np.testing.assert_array_equal(product2_jvp(x, k, a, b, tangent_x, tangent_k), expected_tangent)


@pytest.mark.parametrize("bad", [
    np.asarray([1, 2], dtype=np.int32),
    np.asarray([[1, 2]], dtype=np.int64),
    np.asarray([1, 1, 2, 2], dtype=np.int64)[::2],
])
def test_product2_rejects_malformed_index_buffers(bad):
    x, k, _, b = _case()
    with pytest.raises((TypeError, ValueError)):
        prepare_product2(bad, b[:len(bad)] if bad.ndim == 1 else b, x.size)


def test_product2_rejects_mismatched_directions_and_bad_values():
    x, k, a, b = _case()
    prepared = prepare_product2(a, b, x.size)
    with pytest.raises(ValueError):
        prepared.jvp(x, k, np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="packet index"):
        prepare_product2(np.asarray([x.size], dtype=np.int64),
                         np.asarray([0], dtype=np.int64), x.size)


def test_product2_rejects_unaligned_buffers_and_malformed_cotangents():
    x, k, a, b = _case()
    unaligned_a = np.ndarray((2,), dtype=np.int64, buffer=bytearray(17), offset=1)
    with pytest.raises(ValueError, match="aligned int64"):
        prepare_product2(unaligned_a, b, x.size)
    unaligned_x = np.ndarray((3,), dtype=np.float32, buffer=bytearray(13), offset=1)
    with pytest.raises(ValueError, match="aligned float32"):
        product2_forward(unaligned_x, k, a, b)
    unaligned_k = np.ndarray((2,), dtype=np.float32, buffer=bytearray(9), offset=1)
    with pytest.raises(ValueError, match="aligned float32"):
        product2_forward(x, unaligned_k, a, b)
    prepared = prepare_product2(a, b, x.size)
    with pytest.raises(ValueError, match="extent differs"):
        prepared.forward(x[:-1], k)
    with pytest.raises(ValueError, match="extent differs"):
        prepared.forward(x, k[:-1])
    with pytest.raises(ValueError, match="cotangent extent"):
        prepared.vjp(x, k, np.ones(1, dtype=np.float32))
    with pytest.raises(ValueError, match="contiguous rank-1"):
        prepared.vjp(x, k, np.ones((1, 2), dtype=np.float32))


def test_identity_words_require_exact_integers():
    with pytest.raises(TypeError, match="identity words must be integers"):
        Identity(1.9, 2)
    with pytest.raises(TypeError, match="identity words must be integers"):
        Identity((1.9), 2)
    with pytest.raises(TypeError, match="identity words must be integers"):
        Axis((1.9, 2), Identity(3, 4), Identity(5, 6), Identity(7, 8), 1)


@pytest.mark.parametrize("value", [1.5, True])
def test_product2_integer_arguments_require_exact_indices(value):
    _, _, a, b = _case()
    with pytest.raises(TypeError, match="input_count must be an integer"):
        prepare_product2(a, b, value)
    with pytest.raises(TypeError, match="structure_generation must be an integer"):
        prepare_product2(a, b, 3, structure_generation=value)


def test_product2_integer_arguments_check_unsigned_range():
    _, _, a, b = _case()
    with pytest.raises(ValueError, match="input_count must fit"):
        prepare_product2(a, b, 1 << 64)
    with pytest.raises(ValueError, match="structure_generation must fit"):
        prepare_product2(a, b, 3, structure_generation=-1)
