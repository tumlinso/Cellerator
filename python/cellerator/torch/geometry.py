"""Explicit CE-GEO projection preparation from logical support identities."""
from __future__ import annotations

from .structure import SupportSpec


def compile_sparse_support_ce_geo(support: SupportSpec, *, source_count: int,
                                  destination_count: int,
                                  projection_id: int = 1) -> dict:
    """Compile a logical pair support with Cellerator's native CE-GEO builder.

    The returned projection image and group maps can be passed to
    :func:`cellerator.torch.routing.prepare_sparse_slice` as explicit
    preparation inputs. This API does not perform automatic runtime discovery.
    """
    import importlib

    native = importlib.import_module("cellerator.torch._torch")
    return native.compile_sparse_support_ce_geo(
        list(support.edge_ids), list(support.sources), list(support.destinations),
        source_count, destination_count, support.structure_epoch, projection_id)


__all__ = ["compile_sparse_support_ce_geo"]
