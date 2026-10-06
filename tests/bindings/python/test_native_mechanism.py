"""Torch-free Python tests for native indexed-mechanism lifecycle surfaces."""
from __future__ import annotations

import numpy as np
import pytest
import gc
import os

from cellerator import (
    Axis, Identity, Incidence, Mechanism, MechanismSpec, OutputContribution,
    mechanisms_available, prepare_mechanism,
)


requires_mechanisms = pytest.mark.skipif(not mechanisms_available(),
                                         reason="requires CUDA indexed mechanisms")
if os.environ.get("CELLERATOR_REQUIRE_NATIVE") == "1" and not mechanisms_available():
    raise RuntimeError("CELLERATOR_REQUIRE_NATIVE=1 requires CUDA indexed mechanisms")


def _identity(value: int) -> Identity:
    return Identity(value, value + 1000)


def _axis(value: int, extent: int) -> Axis:
    return Axis(_identity(value), _identity(value + 1), _identity(value + 2),
                _identity(value + 3), extent)


def _spec() -> MechanismSpec:
    x, y, k = _axis(10, 2), _axis(20, 1), _axis(30, 1)
    mechanism = Mechanism(
        _identity(40), _identity(50),
        (Incidence(0, _identity(60), 0), Incidence(1, _identity(61), 1)),
        (OutputContribution(0, _identity(70), 0, _identity(80)),),
    )
    return MechanismSpec(x, y, k, (_identity(50),), (mechanism,))


@requires_mechanisms
def test_native_numpy_forward_vjp_tape_and_owner_lifecycle():
    spec = _spec()
    handle = prepare_mechanism(spec, np.asarray([0.5], dtype=np.float32),
                              device=0, max_batch=2)
    assert handle.spec == spec
    assert handle.snapshot().tolist() == [0.5]
    input_values = np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    output, tape = handle.forward(input_values)
    np.testing.assert_allclose(output[:, 0], [3.0, 10.0], rtol=0, atol=1e-6)
    input_values.fill(-99.0)  # backward must use the forward tape's saved primal.
    with pytest.raises(RuntimeError):
        handle.preflight_write()
    input_gradient, coefficient_gradient = tape.backward(np.ones((2, 1), dtype=np.float32))
    np.testing.assert_allclose(input_gradient, [[1.5, 1.0], [2.5, 2.0]], rtol=0, atol=1e-6)
    np.testing.assert_allclose(coefficient_gradient, [26.0], rtol=0, atol=1e-6)
    assert tape.consumed
    with pytest.raises(RuntimeError, match="consumed"):
        tape.backward(np.ones((2, 1), dtype=np.float32))

    generation = handle.generation
    handle.begin_write()
    handle.publish_write()
    assert handle.generation == generation + 1
    handle.restore(np.asarray([0.25], dtype=np.float32))
    assert handle.generation == generation + 2
    np.testing.assert_allclose(handle.snapshot(), [0.25], rtol=0, atol=0)
    handle.restore(np.asarray([0.5], dtype=np.float32))
    assert handle.generation == generation + 3


@requires_mechanisms
def test_native_handle_rejects_invalid_numpy_shapes_and_precision():
    spec = _spec()
    handle = prepare_mechanism(spec, np.asarray([0.5], dtype=np.float32),
                              device=0, max_batch=2)
    with pytest.raises(ValueError, match="contiguous rank-2"):
        handle.forward(np.asarray([[1.0, 2.0]], dtype=np.float32)[:, ::-1])
    with pytest.raises(ValueError, match="float32"):
        handle.forward(np.asarray([[1.0, 2.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="float32 or float16"):
        handle.forward(np.asarray([[1.0, 2.0]], dtype=">f2"))


@requires_mechanisms
def test_native_tape_retains_program_after_handle_lifetime():
    spec = _spec()
    handle = prepare_mechanism(spec, np.asarray([0.5], dtype=np.float32),
                               device=0, max_batch=1)
    _, tape = handle.forward(np.asarray([[2.0, 3.0]], dtype=np.float32))
    del handle
    gc.collect()
    dx, dk = tape.backward(np.ones((1, 1), dtype=np.float32))
    np.testing.assert_allclose(dx, [[1.5, 1.0]], rtol=0, atol=1e-6)
    np.testing.assert_allclose(dk, [6.0], rtol=0, atol=1e-6)


def test_mechanism_capacity_inputs_are_exact_and_width_checked_before_capability():
    spec = _spec()
    values = np.asarray([0.5], dtype=np.float32)
    with pytest.raises(TypeError, match="max_batch must be an integer"):
        prepare_mechanism(spec, values, device=0, max_batch=1.5)
    with pytest.raises(TypeError, match="max_live_forwards must be an integer"):
        prepare_mechanism(spec, values, device=0, max_batch=1, max_live_forwards=True)
    with pytest.raises(ValueError, match="max_live_forwards must fit"):
        prepare_mechanism(spec, values, device=0, max_batch=1,
                          max_live_forwards=1 << 32)
    with pytest.raises(ValueError, match="max_batch must fit"):
        prepare_mechanism(spec, values, device=0, max_batch=1 << 63)
    with pytest.raises(ValueError, match="select a CUDA device"):
        prepare_mechanism(spec, values, device="cudabad", max_batch=1)
    if not mechanisms_available():
        with pytest.raises(RuntimeError, match="unavailable"):
            prepare_mechanism(spec, values, device=0, max_batch=1)
