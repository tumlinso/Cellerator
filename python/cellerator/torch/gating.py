"""Trainable gate values and explicit admission/estimator policies."""
from __future__ import annotations

import torch
from torch import Tensor


def learned_gates(scores: Tensor, *, temperature: float, threshold: float,
                  prune: bool, straight_through: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    """Return p, admission, and trainable q with the declared explicit STE."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    p = torch.sigmoid(scores / temperature)
    admitted = p >= threshold
    if not prune:
        return p, torch.ones_like(admitted), p
    q = p * admitted.to(p.dtype)
    if straight_through:
        q = p + (q - p).detach()
    return p, admitted, q
