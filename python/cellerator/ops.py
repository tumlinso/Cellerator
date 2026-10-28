"""Core native-backed operations available without importing PyTorch."""
from . import (
    prepare_product2, product2_forward, product2_jvp, product2_vjp,
)

__all__ = ["prepare_product2", "product2_forward", "product2_jvp", "product2_vjp"]
