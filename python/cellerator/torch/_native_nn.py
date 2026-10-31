"""Lazy access to shared native functions, without another parameter owner."""

from importlib import import_module


def validate_implementation(value: str) -> str:
    if value not in {"auto", "reference", "native"}:
        raise ValueError("implementation must be 'auto', 'reference', or 'native'")
    return value


def native_nn(implementation: str):
    validate_implementation(implementation)
    if implementation == "reference":
        return None
    try:
        return import_module("cellerator._nn_native")
    except ModuleNotFoundError as exc:
        if exc.name != "cellerator._nn_native":
            raise
        if implementation == "native":
            raise ImportError(
                "Native NN execution requires the optional _nn_native extension; "
                "build with CELLERATOR_ENABLE_NN=ON and CELLERATOR_ENABLE_PYTHON=ON, "
                "or select implementation='reference'."
            ) from exc
        return None


def functional_children_supported(*children) -> bool:
    """Functional borrowing must not bypass customized child computations."""
    from torch import nn
    from torch.nn.modules import module as module_hooks

    if any(getattr(module_hooks, name, {}) for name in (
        "_global_forward_pre_hooks", "_global_forward_hooks",
        "_global_backward_hooks", "_global_backward_pre_hooks",
    )):
        return False
    for child in children:
        if child is None:
            continue
        if type(child) not in (nn.Linear, nn.Embedding) or "forward" in child.__dict__:
            return False
        if any(getattr(child, name, {}) for name in (
            "_forward_pre_hooks", "_forward_hooks", "_backward_hooks", "_backward_pre_hooks",
        )):
            return False
    return True
