"""Native-backed operations and their optional Torch autograd adapters."""
from .product2 import product2, product2_jvp, prepare_product2

__all__ = ["product2", "product2_jvp", "prepare_product2"]
