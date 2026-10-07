"""Thin optional PyTorch views over Cellerator's native resident CUDA API.

This module owns no CUDA math or prepared sparse topology. It validates and
borrows Torch storage, records allocator use on the active Torch stream, and
dispatches directly to :mod:`cellerator.cuda`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from cellerator import cuda as _cuda


def _require_native_cuda() -> None:
    if not _cuda.resident_cuda_available:
        raise RuntimeError(
            "the Cellerator resident CUDA capability is unavailable in this build"
        )


def _device_index(device: Any = None) -> int:
    _require_native_cuda()
    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, bool):
        raise ValueError("device must select a CUDA device")
    if isinstance(device, int):
        if device < 0:
            raise ValueError("device index must be nonnegative")
        return device
    value = torch.device(device)
    if value.type != "cuda":
        raise ValueError("device must select a CUDA device")
    return int(torch.cuda.current_device() if value.index is None else value.index)


def _torch_stream(device: Any = None):
    index = _device_index(device)
    return index, torch.cuda.current_stream(index)


def _native_stream_matches(stream: Any, device: int, torch_stream: Any) -> None:
    if getattr(stream, "device", None) != device:
        raise ValueError("native stream device must match the tensor device")
    if getattr(stream, "handle", None) != int(torch_stream.cuda_stream):
        raise ValueError("native stream must be the current PyTorch CUDA stream")


def current_stream(device: Any = None):
    """Borrow PyTorch's current stream as a non-owning Cellerator stream."""
    _require_native_cuda()
    index, torch_stream = _torch_stream(device)
    return _cuda.Stream.borrow(index, int(torch_stream.cuda_stream), torch_stream)


def _validate_tensor(tensor: Tensor, name: str) -> int:
    _require_native_cuda()
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must have dtype torch.float32")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.ndim not in (1, 2):
        raise ValueError(f"{name} must have rank one or two")
    if tensor.requires_grad:
        raise ValueError(f"{name} requires_grad is unsupported by this forward-only adapter")
    return int(tensor.device.index)


def _resolve_stream(tensors: tuple[Tensor, ...], stream: Any = None):
    _require_native_cuda()
    if not tensors:
        raise ValueError("at least one tensor is required")
    devices = {_validate_tensor(t, f"tensor[{i}]") for i, t in enumerate(tensors)}
    if len(devices) != 1:
        raise ValueError("all tensors must be on the same CUDA device")
    device = devices.pop()
    torch_stream = torch.cuda.current_stream(device)
    native_stream = current_stream(device) if stream is None else stream
    _native_stream_matches(native_stream, device, torch_stream)
    # PyTorch's allocator must keep every borrowed storage alive until work
    # queued on this stream completes. This does not establish data readiness
    # or grant permission to mutate a concurrently read buffer.
    for tensor in tensors:
        tensor.record_stream(torch_stream)
    return native_stream


def borrow(tensor: Tensor, stream: Any = None):
    """Return a native non-owning FP32 view while retaining the Torch owner."""
    device = _validate_tensor(tensor, "tensor")
    native_stream = _resolve_stream((tensor,), stream)
    return _cuda.Buffer.borrow(
        int(tensor.data_ptr()), tuple(int(x) for x in tensor.shape),
        int(tensor.numel() * tensor.element_size()), native_stream,
        owner=tensor,
    )


def _same_shape(tensors: tuple[Tensor, ...]) -> None:
    for i, tensor in enumerate(tensors):
        _validate_tensor(tensor, f"tensor[{i}]")
    shape = tuple(tensors[0].shape)
    if any(tuple(tensor.shape) != shape for tensor in tensors[1:]):
        raise ValueError("all tensors must have the same shape")


def _reject_output_alias(inputs: tuple[Tensor, ...], out: Tensor) -> None:
    out_begin = int(out.data_ptr())
    out_end = out_begin + int(out.numel() * out.element_size())
    for tensor in inputs:
        begin = int(tensor.data_ptr())
        end = begin + int(tensor.numel() * tensor.element_size())
        if begin < out_end and out_begin < end:
            raise ValueError("output storage must not overlap an input tensor")


def multiply_into(a: Tensor, b: Tensor, out: Tensor, stream: Any = None) -> None:
    """Write elementwise ``a * b`` using Cellerator's native FP32 kernel."""
    _same_shape((a, b, out))
    _reject_output_alias((a, b), out)
    native_stream = _resolve_stream((a, b, out), stream)
    _cuda.multiply_into(borrow(a, native_stream), borrow(b, native_stream),
                        borrow(out, native_stream), native_stream)


def axpby_into(alpha: float, a: Tensor, beta: float, b: Tensor, out: Tensor,
               stream: Any = None) -> None:
    """Write ``alpha * a + beta * b`` with the native FP32 AXPBY kernel."""
    _same_shape((a, b, out))
    _reject_output_alias((a, b), out)
    native_stream = _resolve_stream((a, b, out), stream)
    _cuda.axpby_into(
        float(alpha), borrow(a, native_stream), float(beta),
        borrow(b, native_stream), borrow(out, native_stream), native_stream,
    )


class PreparedCsr:
    """Torch-storage adapter around one native prepared CSR value owner."""

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, weights: Tensor,
                 source_count: int, feature_width: int, stream: Any = None):
        _require_native_cuda()
        self._validate_csr(indptr, indices, source_count, feature_width)
        _validate_tensor(weights, "weights")
        if weights.ndim != 1 or weights.numel() != indices.size:
            raise ValueError("weights must be a rank-one tensor matching the CSR index count")
        native_stream = _resolve_stream((weights,), stream)
        self.stream = native_stream
        self.device = int(weights.device.index)
        self._source_count = int(source_count)
        self._feature_width = int(feature_width)
        self._destination_count = int(indptr.size - 1)
        self._weights_owner = weights
        self._weights_buffer = borrow(weights, native_stream)
        self._native = _cuda.PreparedCsr(
            indptr, indices, self._weights_buffer, int(source_count),
            int(feature_width), native_stream,
        )
        self._closed = False

    @staticmethod
    def _validate_csr(indptr: np.ndarray, indices: np.ndarray, source_count: int,
                      feature_width: int) -> None:
        if not isinstance(indptr, np.ndarray) or indptr.dtype != np.uint64 \
                or indptr.ndim != 1 or not indptr.flags.c_contiguous:
            raise ValueError("indptr must be a contiguous rank-one NumPy uint64 array")
        if not isinstance(indices, np.ndarray) or indices.dtype != np.uint64 \
                or indices.ndim != 1 or not indices.flags.c_contiguous:
            raise ValueError("indices must be a contiguous rank-one NumPy uint64 array")
        try:
            if isinstance(source_count, bool) or int(source_count) != source_count:
                raise ValueError
            source_count = int(source_count)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("source_count must be a positive integer") from exc
        try:
            if isinstance(feature_width, bool) or int(feature_width) != feature_width:
                raise ValueError
            feature_width = int(feature_width)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("feature_width must be a positive integer") from exc
        if source_count <= 0:
            raise ValueError("source_count must be a positive integer")
        if feature_width <= 0:
            raise ValueError("feature_width must be a positive integer")
        if indptr.size < 1:
            raise ValueError("indptr must contain at least the initial zero")
        if int(indptr[0]) != 0 or int(indptr[-1]) != indices.size \
                or np.any(indptr[1:] < indptr[:-1]):
            raise ValueError("CSR indptr must be monotone and span all indices")
        if indices.size and int(indices.max()) >= int(source_count):
            raise ValueError("CSR column index exceeds source_count")

    @property
    def native(self):
        """The sole native prepared owner; exposed for direct binding use."""
        self._ensure_open()
        return self._native

    @property
    def generation(self) -> int:
        self._ensure_open()
        return int(self._native.generation)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("prepared CSR adapter is closed")
        try:
            # The native owner is intentionally exposed for direct use, so it
            # may have been closed independently of this Python wrapper.
            self._native.generation
        except ValueError as exc:
            if str(exc) == "prepared CSR is closed":
                raise RuntimeError("prepared CSR adapter is closed") from exc
            raise

    def apply_into(self, values: Tensor, out: Tensor) -> None:
        self._ensure_open()
        _validate_tensor(values, "values")
        _validate_tensor(out, "out")
        if values.ndim != 2 or out.ndim != 2:
            raise ValueError("prepared CSR apply requires rank-two values and output")
        if int(values.shape[0]) != self._source_count \
                or int(values.shape[1]) != self._feature_width:
            raise ValueError("values shape must match the prepared CSR source and feature extents")
        if tuple(out.shape) != (self._destination_count, self._feature_width):
            raise ValueError("output shape must match the prepared CSR destination and feature extents")
        _reject_output_alias((values, self._weights_owner), out)
        native_stream = _resolve_stream((values, out, self._weights_owner), self.stream)
        self._native.apply_into(
            borrow(values, native_stream), borrow(out, native_stream),
        )

    def publish_values(self, weights: Tensor) -> None:
        self._ensure_open()
        _validate_tensor(weights, "weights")
        if weights.device.index != self.device:
            raise ValueError("weights device must match the prepared CSR device")
        if weights.ndim != 1 or weights.numel() != self._weights_owner.numel():
            raise ValueError("weights shape must match the prepared CSR value extent")
        native_stream = _resolve_stream((weights,), self.stream)
        replacement = borrow(weights, native_stream)
        self._native.publish_values(replacement)
        self._weights_owner = weights
        self._weights_buffer = replacement

    def close(self) -> None:
        if self._closed:
            return
        self._native.close()
        self._weights_owner = None
        self._weights_buffer = None
        self._closed = True


def as_tensor(buffer: Any) -> Tensor:
    """Create a zero-copy Torch alias of a native-owned Cellerator buffer.

    DLPack carries the native buffer's owner lease into Torch and performs the
    stream handoff required by the DLPack protocol. The result aliases the same
    allocation; callers must keep normal read/write ordering and mutation
    ownership rules for that shared storage.
    """
    _require_native_cuda()
    return torch.utils.dlpack.from_dlpack(buffer)


__all__ = [
    "PreparedCsr", "as_tensor", "axpby_into", "borrow", "current_stream",
    "multiply_into",
]
