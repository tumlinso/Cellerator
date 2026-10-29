"""Explicit optional CUDA storage, dispatch and state-profile quantization.

Importing this module needs NumPy but neither Torch nor a CUDA device. Creating
streams or allocations requires the optional ``cellerator._modeling_cuda``
extension and an admitted CUDA execution context. All arrays are row-major;
external CUDA pointers are borrowed with explicit owners and stream ordering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Sequence
import numpy as np


def _native():
    try:
        return import_module("cellerator._modeling_cuda")
    except ModuleNotFoundError as exc:
        if exc.name != "cellerator._modeling_cuda":
            raise
        raise ImportError(
            "The optional CUDA framework extension is unavailable. Build with "
            "CELLERATOR_ENABLE_FRAMEWORK_CUDA=ON and "
            "CELLERATOR_ENABLE_FRAMEWORK_CUDA_PYTHON=ON."
        ) from exc


def available() -> bool:
    """Whether the extension imports; does not probe devices or execute CUDA."""
    try:
        _native()
        return True
    except ImportError:
        return False


def Stream(device: int = 0, *, nonblocking: bool = True):
    return _native().Stream(device, nonblocking)


def borrow_stream(pointer: int, *, device: int):
    """Borrow an external stream. Its owner must outlive every submitted use."""
    return _native().Stream.borrow(pointer, device)


def Event(device: int = 0, *, timing: bool = False):
    return _native().Event(device, timing)


def PinnedBuffer(nbytes: int):
    """Pinned byte storage, exposing Python's buffer protocol."""
    return _native().PinnedBuffer(nbytes)


def _dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.dtype("float16"):
        return _native().DType.float16
    if dtype == np.dtype("float32"):
        return _native().DType.float32
    raise TypeError("CUDA numerical operations require float16 or float32 storage")


def _shape(shape: Sequence[int]) -> tuple[int, ...]:
    import operator
    result = tuple(operator.index(n) for n in shape)
    if any(n < 0 for n in result):
        raise ValueError("array dimensions must be nonnegative")
    return result


def _nbytes(shape, dtype) -> int:
    import math
    return math.prod(shape) * np.dtype(dtype).itemsize


@dataclass(frozen=True)
class DeviceArray:
    """Contiguous owned or borrowed CUDA array with retained dependencies.

    ``stream`` orders this array's most recent operation. Use ``on_stream`` to
    transfer dependency ordering explicitly before using another stream.
    """
    shape: tuple[int, ...]
    dtype: np.dtype
    stream: Any = field(repr=False)
    _view: Any = field(repr=False)
    _owner: Any = field(repr=False)
    _dependencies: tuple[Any, ...] = field(default=(), repr=False)

    def __post_init__(self):
        object.__setattr__(self, "shape", _shape(self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        if self.dtype.hasobject or self.dtype.fields or self.dtype.itemsize == 0:
            raise TypeError("device arrays require a plain fixed-size dtype")
        if self._view.nbytes != _nbytes(self.shape, self.dtype):
            raise ValueError("view extent does not match shape and dtype")
        if self.stream.device != self._view.device:
            raise ValueError("array and stream devices differ")
        if self._view.pointer % max(1, self.dtype.alignment):
            raise ValueError("borrowed device pointer is not dtype-aligned")

    @classmethod
    def empty(cls, shape, *, stream, dtype=np.float16):
        shape, dtype = _shape(shape), np.dtype(dtype)
        if dtype.hasobject or dtype.fields or dtype.itemsize == 0:
            raise TypeError("device arrays require a plain fixed-size dtype")
        owner = _native().DeviceBuffer(_nbytes(shape, dtype), stream.device)
        return cls(shape, dtype, stream, owner.view(), owner)

    @classmethod
    def from_numpy(cls, array, *, stream, dtype=None):
        """Copy host data explicitly; optional dtype conversion is caller chosen.

        Transfer synchronizes the supplied stream before returning, so temporary
        NumPy inputs can safely be released. No implicit float64 downcast occurs.
        """
        array = np.ascontiguousarray(array, dtype=dtype)
        result = cls.empty(array.shape, stream=stream, dtype=array.dtype)
        result._owner.upload(array, stream)
        return result

    @classmethod
    def borrow(cls, pointer: int, shape, *, dtype, device: int, stream, owner):
        """Borrow contiguous external storage and retain its supplied owner.

        The caller asserts the pointer's device, extent and readiness on stream.
        Read-only external arrays must not be used as writable outputs.
        """
        if owner is None:
            raise ValueError("borrowed storage requires an explicit retained owner")
        shape, dtype = _shape(shape), np.dtype(dtype)
        view = _native().DeviceView(pointer, _nbytes(shape, dtype), device)
        return cls(shape, dtype, stream, view, owner)

    @property
    def device(self):
        return self._view.device

    @property
    def pointer(self):
        return self._view.pointer

    @property
    def nbytes(self):
        return self._view.nbytes

    def on_stream(self, stream):
        if stream.device != self.device:
            raise ValueError("cross-device stream transfer requires an explicit copy")
        done = Event(self.device)
        done.record(self.stream)
        stream.wait(done)
        return DeviceArray(self.shape, self.dtype, stream, self._view,
                           self._owner, (self, done))

    def to_numpy(self):
        """Synchronize and explicitly copy the entire array to host memory."""
        # Borrowed owners need not implement the native buffer API.
        copy = _native().DeviceBuffer(self.nbytes, self.device)
        copy.copy_from_async(self._view, self.stream)
        return copy.download(self.dtype, self.shape, self.stream)

    def _retain(self, *dependencies):
        return DeviceArray(self.shape, self.dtype, self.stream, self._view,
                           self._owner, tuple(dependencies))


class CudaDispatchPlan:
    """Reusable device projection of a host conditional DispatchPlan.

    Upload preserves expert revisions, origin metadata and logical reduction
    order in the retained host plan; it does not select or replace experts.
    Route weights convert to FP32, accepting finite rounding/underflow to zero
    and rejecting overflow or nonfinite conversion results.
    """
    def __init__(self, host_plan, *, stream):
        self.host_plan, self.stream = host_plan, stream
        self._plan = _native().DispatchPlan(
            host_plan.origin_count, host_plan.dispatch_origin_slots,
            host_plan.dispatch_weights, host_plan.origin_offsets,
            host_plan.origin_dispatch_indices, host_plan.expert_offsets, stream)

    @property
    def expert_offsets(self):
        return tuple(self._plan.expert_offsets)

    def _input(self, array, rows):
        if len(array.shape) != 2 or array.shape[0] != rows:
            raise ValueError(f"expected a two-dimensional array with {rows} rows")
        if array.device != self._plan.device:
            raise ValueError("array and dispatch plan devices differ")
        return array.shape[1], _dtype(array.dtype)

    def gather(self, array: DeviceArray) -> DeviceArray:
        features, dtype = self._input(array, self._plan.origin_count)
        output = DeviceArray.empty((self._plan.selection_count, features),
                                   dtype=array.dtype, stream=array.stream)
        self._plan.gather(array._view, output._view, features, dtype, array.stream)
        return output._retain(self, array)

    def combine(self, array: DeviceArray) -> DeviceArray:
        features, dtype = self._input(array, self._plan.selection_count)
        output = DeviceArray.empty((self._plan.origin_count, features),
                                   dtype=array.dtype, stream=array.stream)
        self._plan.combine(array._view, output._view, features, dtype, array.stream)
        return output._retain(self, array)

    def gather_backward(self, gradient: DeviceArray) -> DeviceArray:
        features, dtype = self._input(gradient, self._plan.selection_count)
        output = DeviceArray.empty((self._plan.origin_count, features),
                                   dtype=gradient.dtype, stream=gradient.stream)
        self._plan.gather_backward(gradient._view, output._view, features, dtype, gradient.stream)
        return output._retain(self, gradient)

    def combine_backward(self, gradient: DeviceArray, expert_output: DeviceArray):
        features, dtype = self._input(gradient, self._plan.origin_count)
        expert_features, _ = self._input(expert_output, self._plan.selection_count)
        if expert_features != features or expert_output.dtype != gradient.dtype:
            raise ValueError("gradient and expert output shapes/dtypes disagree")
        if expert_output.stream.pointer != gradient.stream.pointer:
            expert_output = expert_output.on_stream(gradient.stream)
        expert_gradient = DeviceArray.empty(expert_output.shape, dtype=gradient.dtype, stream=gradient.stream)
        weight_gradient = DeviceArray.empty((self._plan.selection_count,), dtype=np.float32, stream=gradient.stream)
        self._plan.combine_backward(gradient._view, expert_output._view,
                                    expert_gradient._view, weight_gradient._view,
                                    features, dtype, gradient.stream)
        dependencies = (self, gradient, expert_output)
        return expert_gradient._retain(*dependencies), weight_gradient._retain(*dependencies)


class CudaQuantizationProfile:
    """FP32 device codebooks with explicit feature identities and code width."""
    def __init__(self, profile, *, stream, feature_ids=None):
        self.profile, self.stream = profile, stream
        self.feature_ids = tuple(profile.feature_ids if feature_ids is None else feature_ids)
        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise ValueError("physical feature identities must be unique")
        lookup = {identity: i for i, identity in enumerate(profile.feature_ids)}
        try:
            feature_map = [lookup[identity] for identity in self.feature_ids]
        except KeyError as exc:
            raise ValueError("physical feature identity is absent from profile") from exc
        offsets, levels = [0], []
        for book in profile.codebooks:
            levels.extend(np.asarray(book, dtype=np.float32).tolist())
            offsets.append(len(levels))
        self._profile = _native().QuantizationProfile(offsets, levels, feature_map, profile.bits, stream)

    def encode(self, array: DeviceArray, *, exact_mask: DeviceArray | None = None,
               check=True):
        if len(array.shape) != 2 or array.shape[1] != len(self.feature_ids):
            raise ValueError("state must have shape (rows, physical profile features)")
        if array.device != self._profile.device:
            raise ValueError("state and profile devices differ")
        if exact_mask is not None:
            if exact_mask.shape != array.shape or exact_mask.dtype != np.dtype("uint8"):
                raise ValueError("exact_mask must be a uint8 array with the state shape")
            if exact_mask.stream.pointer != array.stream.pointer:
                exact_mask = exact_mask.on_stream(array.stream)
        capacity = array.shape[0] * array.shape[1] if exact_mask is not None else 0
        native = _native().QuantizedState(self._profile, array.shape[0], capacity)
        native.encode(array._view, _dtype(array.dtype),
                      None if exact_mask is None else exact_mask._view, array.stream)
        result = CudaQuantizedState(native, self, array.stream, (array, exact_mask))
        if check:
            result.check()
        return result


class CudaQuantizedState:
    """Packed device codes with sparse FP32 exact-value overrides.

    Exact means preserved relative to FP32 sidecar storage. FP16 decoding rounds
    overrides to FP16; choose FP32 decoding when that conversion is unwanted.
    """
    def __init__(self, native, profile, stream, dependencies):
        self._state, self.profile, self.stream = native, profile, stream
        self._dependencies = dependencies

    @property
    def shape(self):
        return self._state.rows, self._state.features

    @property
    def packed_bytes(self):
        return self._state.packed_bytes

    @property
    def override_count(self):
        """Number of exact sidecar entries; synchronizes the state's stream."""
        return self._state.override_count(self.stream)

    def check(self):
        """Synchronize and reject malformed profiles, overflow or nonfinite data."""
        status = self._state.status(self.stream)
        if status:
            raise ValueError(f"CUDA quantization reported validation bits {status}")
        return self

    def decode(self, *, dtype=np.float16, check=True):
        if check:
            self.check()
        result = DeviceArray.empty(self.shape, dtype=dtype, stream=self.stream)
        self._state.decode(result._view, _dtype(dtype), self.stream)
        if check:
            self.check()
        return result._retain(self)


class CudaGroupedLinear:
    """cuBLAS linear maps over expert-major ragged rows on one fixed stream.

    Weights have shape ``(output_features, input_features)`` and share dtype
    and device. Pass a dispatch plan's ``expert_offsets`` as row_offsets.
    FP16 and FP32 storage are supported, with FP32 internal accumulation;
    gradients use the same storage dtype as weights. There is no implicit
    autograd registration, parameter update, bias or expert selection.
    Callers must order later external weight mutations against this stream;
    borrowed storage does not track mutations performed by another provider.
    """
    def __init__(self, weights: Sequence[DeviceArray], row_offsets, *, stream):
        weights = tuple(weights)
        if not weights:
            raise ValueError("the Python grouped-linear wrapper requires at least one expert")
        first = weights[0]
        if len(first.shape) != 2:
            raise ValueError("weights must have shape (output_features, input_features)")
        self.dtype, self.stream = first.dtype, stream
        for weight in weights:
            if weight.shape != first.shape or weight.dtype != first.dtype or weight.device != stream.device:
                raise ValueError("all expert weights must have matching shape, dtype and device")
        self.weights = tuple(w if w.stream.pointer == stream.pointer else w.on_stream(stream) for w in weights)
        self._operator = _native().GroupedLinear(
            first.shape[1], first.shape[0], list(row_offsets),
            [w._view for w in self.weights], _dtype(first.dtype), stream.device)

    def _prepare(self, array, shape):
        if array.shape != shape or array.dtype != self.dtype or array.device != self.stream.device:
            raise ValueError(f"grouped-linear array must have shape {shape}, dtype {self.dtype}, and matching device")
        return array if array.stream.pointer == self.stream.pointer else array.on_stream(self.stream)

    def forward(self, inputs: DeviceArray) -> DeviceArray:
        inputs = self._prepare(inputs, (self._operator.row_count, self._operator.input_features))
        result = DeviceArray.empty((self._operator.row_count, self._operator.output_features),
                                   dtype=self.dtype, stream=self.stream)
        self._operator.forward(inputs._view, result._view, self.stream)
        return result._retain(self, inputs)

    __call__ = forward

    def backward(self, inputs: DeviceArray, output_gradient: DeviceArray, *,
                 input_gradient: DeviceArray | None = None,
                 weight_gradients: Sequence[DeviceArray] | None = None,
                 accumulate_input: bool = False, accumulate_weights: bool = False):
        """Return ``(input_gradient, tuple(weight_gradients))``.

        Accumulation requires caller-supplied initialized destination buffers.
        Overwrite mode zeros weight gradients for experts with no selected rows.
        """
        input_shape = (self._operator.row_count, self._operator.input_features)
        output_shape = (self._operator.row_count, self._operator.output_features)
        inputs = self._prepare(inputs, input_shape)
        output_gradient = self._prepare(output_gradient, output_shape)
        if input_gradient is None:
            if accumulate_input:
                raise ValueError("input accumulation requires an initialized input_gradient")
            input_gradient = DeviceArray.empty(input_shape, dtype=self.dtype, stream=self.stream)
        input_gradient = self._prepare(input_gradient, input_shape)
        if weight_gradients is None:
            if accumulate_weights:
                raise ValueError("weight accumulation requires initialized weight_gradients")
            weight_gradients = tuple(DeviceArray.empty(w.shape, dtype=self.dtype, stream=self.stream)
                                     for w in self.weights)
        else:
            weight_gradients = tuple(weight_gradients)
        if len(weight_gradients) != len(self.weights):
            raise ValueError("one weight gradient is required per expert")
        weight_gradients = tuple(self._prepare(g, w.shape) for g, w in zip(weight_gradients, self.weights))
        native = _native()
        self._operator.backward(
            inputs._view, output_gradient._view, input_gradient._view,
            [g._view for g in weight_gradients],
            native.GradientMode.accumulate if accumulate_input else native.GradientMode.overwrite,
            native.GradientMode.accumulate if accumulate_weights else native.GradientMode.overwrite,
            self.stream)
        dependencies = (self, inputs, output_gradient, input_gradient, *weight_gradients)
        return (input_gradient._retain(*dependencies),
                tuple(g._retain(*dependencies) for g in weight_gradients))


__all__ = ["available", "Stream", "borrow_stream", "Event", "PinnedBuffer",
           "DeviceArray", "CudaDispatchPlan", "CudaQuantizationProfile", "CudaQuantizedState", "CudaGroupedLinear"]
