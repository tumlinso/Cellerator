"""Explicit adapters for native CUDA buffers and ATen tokenization."""
from __future__ import annotations


def __getattr__(name: str):
    if name == "cuda":
        from . import cuda
        return cuda
    if name == "tokenization":
        try:
            from cellerator import tokenization
        except ImportError as exc:
            raise ImportError(
                "ATen tokenization is unavailable; build with CELLERATOR_ENABLE_TOKENIZATION"
            ) from exc
        return tokenization
    raise AttributeError(name)


__all__ = ["cuda", "tokenization"]
