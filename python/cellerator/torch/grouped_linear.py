"""Trainable expert-major linear maps with optional native CUDA execution.

Torch owns parameters, activations and gradients. The native operation borrows
their storage on the current Torch stream; it never creates another parameter
owner. ``implementation="reference"`` selects ordinary Torch operations for
comparison or for devices/dtypes outside the native CUDA contract.
"""
from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
import math
import operator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["grouped_linear", "GroupedLinear"]


def _integer(value, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _offsets(values: Sequence[int] | Tensor, experts: int, rows: int) -> tuple[int, ...]:
    if isinstance(values, Tensor):
        if values.device.type != "cpu" or values.ndim != 1:
            raise ValueError("row_offsets must be a host rank-1 integer sequence")
        if values.dtype not in (torch.int32, torch.int64):
            raise TypeError("row_offsets tensor must use int32 or int64")
        values = values.tolist()
    try:
        result = tuple(_integer(value, "row offset") for value in values)
    except TypeError as exc:
        raise TypeError("row_offsets must be a host integer sequence") from exc
    if len(result) != experts + 1:
        raise ValueError("row_offsets must contain one boundary per expert plus the final boundary")
    if result[0] != 0 or result[-1] != rows:
        raise ValueError("row_offsets must start at zero and end at the input row count")
    if any(right < left for left, right in zip(result, result[1:])):
        raise ValueError("row_offsets must be monotone")
    return result


def _native_module():
    try:
        return import_module("cellerator._modeling_cuda")
    except ModuleNotFoundError as exc:
        if exc.name != "cellerator._modeling_cuda":
            raise
        raise ImportError(
            "Native grouped_linear requires the optional _modeling_cuda extension. "
            "Build with CELLERATOR_ENABLE_FRAMEWORK_CUDA=ON and "
            "CELLERATOR_ENABLE_FRAMEWORK_CUDA_PYTHON=ON, or explicitly select "
            "implementation='reference'."
        ) from exc


def _borrow(native, tensor: Tensor, stream):
    # record_stream protects allocator reuse if the Python/autograd owner is
    # released before asynchronous native work finishes. Saved tensors retain
    # the forward operands and their ordinary mutation/version checks.
    tensor.record_stream(stream)
    return native.DeviceView(
        tensor.data_ptr(), tensor.numel() * tensor.element_size(), tensor.device.index,
    )


class _NativeGroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: Tensor, weight: Tensor, offsets: tuple[int, ...]):
        native = _native_module()
        stream = torch.cuda.current_stream(inputs.device)
        borrowed_stream = native.Stream.borrow(int(stream.cuda_stream), inputs.device.index)
        weight_views = [_borrow(native, expert, stream) for expert in weight.unbind(0)]
        dtype = native.DType.float16 if inputs.dtype == torch.float16 else native.DType.float32
        executor = native.GroupedLinear(
            inputs.shape[1], weight.shape[1], list(offsets), weight_views,
            dtype, inputs.device.index,
        )
        output = inputs.new_empty((inputs.shape[0], weight.shape[1]))
        executor.forward(
            _borrow(native, inputs, stream), _borrow(native, output, stream), borrowed_stream,
        )
        ctx.save_for_backward(inputs, weight)
        ctx.executor = executor
        ctx.native = native
        # The executor keeps borrowed pointers. Keep the views and stream
        # wrapper along with the operands until this invocation's tape retires.
        ctx.forward_dependencies = (weight_views, borrowed_stream, stream)
        return output

    @staticmethod
    def backward(ctx, output_gradient: Tensor):
        if torch.is_grad_enabled():
            raise RuntimeError("native grouped_linear supports first-order gradients only")
        inputs, weight = ctx.saved_tensors  # validates saved Tensor versions
        native = ctx.native
        gradient = output_gradient.contiguous()
        stream = torch.cuda.current_stream(inputs.device)
        borrowed_stream = native.Stream.borrow(int(stream.cuda_stream), inputs.device.index)
        input_gradient = torch.empty_like(inputs)
        weight_gradient = torch.empty_like(weight)
        input_view = _borrow(native, inputs, stream)
        gradient_view = _borrow(native, gradient, stream)
        input_gradient_view = _borrow(native, input_gradient, stream)
        weight_gradient_views = [
            _borrow(native, expert, stream) for expert in weight_gradient.unbind(0)
        ]
        # The native dX computation reads the saved parameter matrix as well.
        weight.record_stream(stream)
        ctx.executor.backward(
            input_view, gradient_view, input_gradient_view, weight_gradient_views,
            native.GradientMode.overwrite, native.GradientMode.overwrite, borrowed_stream,
        )
        return input_gradient, weight_gradient, None


def grouped_linear(
    inputs: Tensor,
    weight: Tensor,
    row_offsets: Sequence[int] | Tensor,
    *,
    implementation: str = "auto",
) -> Tensor:
    """Apply shared expert weights to an expert-major ragged row partition.

    ``inputs`` has shape ``[N, I]``, ``weight`` has shape ``[E, O, I]``, and
    host ``row_offsets`` has ``E + 1`` monotone boundaries from zero to ``N``.
    Segment ``e`` produces ``inputs[start:end] @ weight[e].T``. Empty segments
    and zero rows are valid. There is no routing, combination, bias, parameter
    mutation, implicit dtype cast or device transfer in this operation.

    ``auto`` uses the Torch reference on CPU and native execution on CUDA;
    CUDA availability failures do not silently switch implementations.
    ``reference`` explicitly selects Torch operations on the operand device.
    Native execution accepts matching CUDA FP16 or FP32 operands, accumulates
    internally in FP32, and returns values/first-order gradients in that operand
    dtype. Ordinary differentiable casts may connect FP32 master parameters to
    FP16 compute. Noncontiguous operands are explicitly made contiguous before
    entering the native autograd operation.
    """
    if implementation not in {"auto", "reference", "native"}:
        raise ValueError("implementation must be 'auto', 'reference', or 'native'")
    if not isinstance(inputs, Tensor) or not isinstance(weight, Tensor):
        raise TypeError("inputs and weight must be torch.Tensor objects")
    if inputs.ndim != 2 or weight.ndim != 3:
        raise ValueError("inputs must have shape [N, I] and weight [E, O, I]")
    if not inputs.is_floating_point() or not weight.is_floating_point():
        raise TypeError("inputs and weight must use floating point dtypes")
    if inputs.layout != torch.strided or weight.layout != torch.strided:
        raise ValueError("grouped_linear operands must use strided layouts")
    if inputs.device != weight.device or inputs.dtype != weight.dtype:
        raise ValueError("inputs and weight must have matching devices and dtypes")
    if min(weight.shape) <= 0 or inputs.shape[1] != weight.shape[2]:
        raise ValueError("expert and feature extents must be positive and input widths must match")
    offsets = _offsets(row_offsets, weight.shape[0], inputs.shape[0])
    selected = ("native" if inputs.is_cuda else "reference") if implementation == "auto" else implementation
    if selected == "reference":
        return torch.cat([
            F.linear(inputs[start:end], expert)
            for expert, start, end in zip(weight.unbind(0), offsets, offsets[1:])
        ], dim=0)
    if not inputs.is_cuda:
        raise ValueError("native grouped_linear requires CUDA tensors")
    if inputs.dtype not in (torch.float16, torch.float32):
        raise TypeError("native grouped_linear supports CUDA float16 and float32 only")
    return _NativeGroupedLinear.apply(inputs.contiguous(), weight.contiguous(), offsets)


class GroupedLinear(nn.Module):
    """Ordinary Torch-owned weights for :func:`grouped_linear`.

    The sole parameter has shape ``[experts, out_features, in_features]``.
    Standard optimizers, ``to()``, sharing and state dictionaries retain their
    ordinary Torch semantics. Construct parameters in the desired storage dtype;
    the functional operation does not implicitly recast them.
    """
    def __init__(self, experts: int, in_features: int, out_features: int,
                 *, device=None, dtype=None):
        super().__init__()
        self.experts = _integer(experts, "experts")
        self.in_features = _integer(in_features, "in_features")
        self.out_features = _integer(out_features, "out_features")
        if min(self.experts, self.in_features, self.out_features) <= 0:
            raise ValueError("expert and feature extents must be positive")
        self.weight = nn.Parameter(torch.empty(
            self.experts, self.out_features, self.in_features, device=device, dtype=dtype,
        ))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, inputs: Tensor, row_offsets: Sequence[int] | Tensor,
                *, implementation: str = "auto") -> Tensor:
        return grouped_linear(inputs, self.weight, row_offsets, implementation=implementation)

    def extra_repr(self) -> str:
        return (f"experts={self.experts}, in_features={self.in_features}, "
                f"out_features={self.out_features}")
