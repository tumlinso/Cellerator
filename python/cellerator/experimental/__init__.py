"""Experimental compositions that are not yet part of the stable API."""
from importlib import import_module

__all__ = ["accessibility", "fields", "inheritance", "response_precision"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
