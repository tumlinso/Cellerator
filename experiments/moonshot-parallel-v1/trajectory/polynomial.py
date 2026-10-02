"""M02 CPU Torch reference for F(X) = L X + X R + X M X.

The supplied matrices define a restricted quadratic family. These functions
return fresh tensors, preserve Torch autograd graphs, and never update inputs.
"""

import torch


def _validate(**operands):
    reference = operands["X"]
    for name, value in operands.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a Torch tensor")
    if reference.ndim != 2 or reference.shape[0] != reference.shape[1] or not reference.numel():
        raise ValueError("X must be a nonempty square matrix")
    if reference.device.type != "cpu" or reference.dtype not in (torch.float32, torch.float64):
        raise ValueError("X requires CPU float32 or float64")
    for name, value in operands.items():
        if value.shape != reference.shape:
            raise ValueError(f"{name} must have the same square shape as X")
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f"{name} must have the same dtype and device as X")
        if value.layout != torch.strided:
            raise ValueError(f"{name} requires strided storage")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain finite values")


def polynomial(X, L, R, M):
    """Evaluate the square law using float32 or float64 CPU matrices."""
    _validate(X=X, L=L, R=R, M=M)
    return L @ X + X @ R + X @ M @ X


def polynomial_jvp(X, L, R, M, dX, dL, dR, dM):
    """Push forward simultaneous state and parameter tangents."""
    _validate(X=X, L=L, R=R, M=M, dX=dX, dL=dL, dR=dR, dM=dM)
    return (dL @ X + L @ dX + dX @ R + X @ dR
            + dX @ M @ X + X @ dM @ X + X @ M @ dX)


def polynomial_vjp(X, L, R, M, G):
    """Pull back G under the Frobenius inner product; order is X, L, R, M."""
    _validate(X=X, L=L, R=R, M=M, G=G)
    grad_X = L.T @ G + G @ R.T + G @ (M @ X).T + (X @ M).T @ G
    return grad_X, G @ X.T, X.T @ G, X.T @ G @ X.T
