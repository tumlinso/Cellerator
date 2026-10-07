"""Resident CUDA access to Cellerator's native FP32 operations.

Buffers and prepared relations stay on one CUDA device and stream. Explicit
upload/download calls block on that stream; native operations only enqueue.
For borrowed buffers, ``capacity_bytes`` is a caller assertion, and the buffer,
its owner and its stream must remain alive until queued uses complete.
"""
from __future__ import annotations

from . import _native

resident_cuda_available = bool(_native.resident_cuda_available())

if resident_cuda_available:
    Stream = _native.Stream
    Buffer = _native.Buffer
    Event = _native.Event
    PreparedCsr = _native.PreparedCsr
    multiply_into = _native.multiply_into
    axpby_into = _native.axpby_into
else:
    def _unavailable(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "resident CUDA bindings are unavailable in this Cellerator build; "
            "rebuild with the prepared-relation and native-numeric CUDA targets"
        )

    class Stream:
        def __init__(self, *args, **kwargs):
            _unavailable(*args, **kwargs)

        @staticmethod
        def borrow(*args, **kwargs):
            _unavailable(*args, **kwargs)

    class Buffer:
        def __init__(self, *args, **kwargs):
            _unavailable(*args, **kwargs)

        @staticmethod
        def borrow(*args, **kwargs):
            _unavailable(*args, **kwargs)

    class Event:
        def __init__(self, *args, **kwargs):
            _unavailable(*args, **kwargs)

        @staticmethod
        def record(*args, **kwargs):
            _unavailable(*args, **kwargs)

    class PreparedCsr:
        def __init__(self, *args, **kwargs):
            _unavailable(*args, **kwargs)

    multiply_into = _unavailable
    axpby_into = _unavailable


__all__ = [
    "resident_cuda_available", "Stream", "Buffer", "Event", "PreparedCsr",
    "multiply_into", "axpby_into",
]
