"""Native-backed operations and their optional Torch autograd adapters."""
from .product2 import product2, product2_jvp, prepare_product2


def sparse_latent_transition(module, state, inputs, affected_offsets,
                             affected_mechanisms):
    """Run the module's selected native transition mode through its flat owner.

    The native v2 operation owns all arithmetic and autograd. This adapter
    supplies mode-specific views into the existing canonical parameter leaf.
    """
    from .execution import SparseSupportModule

    if not isinstance(module, SparseSupportModule):
        raise TypeError("module must be a Cellerator SparseSupportModule")
    matrix_a, matrix_b, bias = module.transition_parameters()
    return module._torch_native.sparse_latent_transition_v2(
        module.transition_mode, state, inputs, affected_offsets,
        affected_mechanisms, matrix_a, matrix_b, bias, module._native)


__all__ = ["product2", "product2_jvp", "prepare_product2", "sparse_latent_transition"]
