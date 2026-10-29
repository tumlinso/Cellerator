"""Explicit boundaries to upstream data and learning libraries."""

# Provider modules import their dependency on first use; importing the adapter
# namespace does not initialize AnnData or PyTorch.

__all__: list[str] = []
