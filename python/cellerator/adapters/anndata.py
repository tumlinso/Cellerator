"""Explicit AnnData boundary for model state and observation exchange.

This module selects data; it does not preprocess measurements or guess their
biological meaning. Sparse matrices remain sparse, and all row/column movement
is performed by caller-visible identities rather than assumed axis order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Literal, Mapping, Sequence


def _unique(values: Iterable[Hashable], name: str) -> tuple[Hashable, ...]:
    result = tuple(values)
    try:
        if len(set(result)) != len(result):
            raise ValueError(f"{name} must be unique")
    except TypeError as exc:
        raise TypeError(f"{name} must contain hashable identities") from exc
    return result


def _select_positions(axis_ids: tuple[Hashable, ...], selected: Sequence[Hashable] | None,
                      name: str) -> tuple[int, ...]:
    if selected is None:
        return tuple(range(len(axis_ids)))
    requested = _unique(selected, name)
    positions = {identity: index for index, identity in enumerate(axis_ids)}
    try:
        return tuple(positions[identity] for identity in requested)
    except KeyError as exc:
        raise KeyError(f"unknown {name[:-1]} identity: {exc.args[0]!r}") from exc


def _slice_matrix(matrix: Any, rows: tuple[int, ...], cols: tuple[int, ...]) -> Any:
    """Slice without densifying; convert only to indexed SciPy sparse form."""
    try:
        from scipy import sparse
    except ImportError:
        sparse = None
    if sparse is not None and sparse.issparse(matrix):
        # SciPy sparse matrices do not support paired advanced indexing. The
        # two-stage slice preserves CSR/CSC family and never materializes dense.
        return matrix[list(rows), :][:, list(cols)]
    return matrix[list(rows), :][:, list(cols)]


@dataclass(frozen=True)
class AnnDataView:
    """A selected matrix with the exact identities and source that define it."""

    values: Any
    cell_ids: tuple[Hashable, ...]
    feature_ids: tuple[Hashable, ...]
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_ids", _unique(self.cell_ids, "cell identities"))
        object.__setattr__(self, "feature_ids", _unique(self.feature_ids, "feature identities"))
        if getattr(self.values, "shape", None) != (len(self.cell_ids), len(self.feature_ids)):
            raise ValueError("matrix shape must match cell and feature identity extents")

    @classmethod
    def from_adata(
        cls,
        adata: Any,
        *,
        layer: str | None = None,
        use_raw: bool = False,
        cell_ids: Sequence[Hashable] | None = None,
        feature_ids: Sequence[Hashable] | None = None,
        copy: bool = False,
    ) -> "AnnDataView":
        """Select `X`, one named layer, or `.raw.X` explicitly.

        Selection by identity retains the requested identity order. No
        normalization, transformation, imputation, or layer inference occurs.
        Sparse slices remain sparse. `copy=False` means the adapter avoids an
        additional defensive copy; sparse/dense indexing may itself allocate.
        """
        if layer is not None and use_raw:
            raise ValueError("choose either layer or use_raw, not both")
        if use_raw:
            raw = getattr(adata, "raw", None)
            if raw is None:
                raise ValueError("AnnData has no .raw matrix")
            matrix, all_features, source = raw.X, tuple(raw.var_names), "raw.X"
        elif layer is not None:
            if layer not in adata.layers:
                raise KeyError(f"AnnData layer {layer!r} does not exist")
            matrix, all_features, source = adata.layers[layer], tuple(adata.var_names), f"layers[{layer!r}]"
        else:
            matrix, all_features, source = adata.X, tuple(adata.var_names), "X"
        all_cells = _unique(adata.obs_names, "AnnData obs_names")
        all_features = _unique(all_features, "selected AnnData var_names")
        rows = _select_positions(all_cells, cell_ids, "cell_ids")
        cols = _select_positions(all_features, feature_ids, "feature_ids")
        values = _slice_matrix(matrix, rows, cols)
        if copy:
            values = values.copy()
        selected_cells = tuple(all_cells[i] for i in rows)
        selected_features = tuple(all_features[i] for i in cols)
        return cls(values, selected_cells, selected_features, source)

    def row(self, cell_id: Hashable, *, copy: bool = False) -> Any:
        """Return one cell's measurements with its complete feature axis intact."""
        try:
            i = self.cell_ids.index(cell_id)
        except ValueError as exc:
            raise KeyError(f"unknown cell identity: {cell_id!r}") from exc
        # A two-dimensional single-row result preserves sparse matrix shape and
        # carries no implied dense-cell ontology.
        value = self.values[i:i + 1, :]
        return value.copy() if copy else value

    def to_cell_states(self, *, field: str = "measurement", copy: bool = False) -> dict[Hashable, dict[str, Any]]:
        """Create model-neutral state records while retaining the feature axis."""
        return {
            identity: {
                field: self.row(identity, copy=copy),
                "feature_ids": self.feature_ids,
                "observation_source": self.source,
            }
            for identity in self.cell_ids
        }

    def to_cells(self, model: Any, *, state_factory: Any = None, copy: bool = False,
                 start_time: float = 0.0) -> tuple[Any, ...]:
        """Bind selected measurement states to `cellerator.modeling.Cell` objects.

        A caller may supply ``state_factory(cell_id, row, feature_ids, source)``
        to construct domain-specific, latent, or externally owned state.
        """
        from cellerator.modeling import Cell

        if state_factory is None:
            states = self.to_cell_states(copy=copy)
        else:
            states = {identity: state_factory(identity, self.row(identity, copy=copy),
                                               self.feature_ids, self.source)
                      for identity in self.cell_ids}
        return tuple(Cell(model, states[identity], id=identity, time=start_time)
                     for identity in self.cell_ids)


def from_adata(adata: Any, **kwargs: Any) -> AnnDataView:
    """Convenience spelling for :meth:`AnnDataView.from_adata`."""
    return AnnDataView.from_adata(adata, **kwargs)


def write_observation(
    adata: Any,
    values: Any,
    *,
    cell_ids: Sequence[Hashable],
    target: Literal["obs", "obsm", "layer"],
    key: str,
    feature_ids: Sequence[Hashable] | None = None,
    fill_value: Any = 0,
) -> Any:
    """Write model output to an explicit AnnData target with identity alignment.

    `obs` stores one scalar per supplied cell. `obsm` stores arbitrary-width
    rows. `layer` additionally requires and aligns feature identities against
    `adata.var_names`. Sparse matrix inputs stay sparse when reindexed.
    Missing target cells/features receive `fill_value`; this defaults to zero
    and is explicit in the function signature. The assigned output is returned.
    """
    ids = _unique(cell_ids, "cell_ids")
    all_ids = _unique(adata.obs_names, "AnnData obs_names")
    if target not in {"obs", "obsm", "layer"}:
        raise ValueError("target must be 'obs', 'obsm', or 'layer'")
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")
    if target == "obs":
        if feature_ids is not None:
            raise ValueError("feature_ids apply only to matrix outputs")
        if getattr(values, "ndim", 1) != 1 or len(values) != len(ids):
            raise ValueError("obs output must be one scalar value per cell_id")
        import numpy as np

        try:
            dtype = values.dtype
        except AttributeError:
            dtype = np.asarray(values).dtype
        out = np.full(len(all_ids), fill_value, dtype=dtype)
        positions = {identity: i for i, identity in enumerate(all_ids)}
        for source_i, identity in enumerate(ids):
            if identity not in positions:
                raise KeyError(f"unknown cell identity: {identity!r}")
            out[positions[identity]] = values[source_i]
        adata.obs[key] = out
        return out

    if not hasattr(values, "shape"):
        import numpy as np

        values = np.asarray(values)
    if getattr(values, "ndim", 2) != 2 or values.shape[0] != len(ids):
        raise ValueError("matrix output rows must match cell_ids")
    all_positions = {identity: i for i, identity in enumerate(all_ids)}
    if any(identity not in all_positions for identity in ids):
        missing = next(identity for identity in ids if identity not in all_positions)
        raise KeyError(f"unknown cell identity: {missing!r}")
    row_map = {i: all_positions[identity] for i, identity in enumerate(ids)}

    if target == "layer":
        if feature_ids is None:
            raise ValueError("layer output requires explicit feature_ids")
        features = _unique(feature_ids, "feature_ids")
        if values.shape[1] != len(features):
            raise ValueError("matrix columns must match feature_ids")
        all_features = tuple(adata.var_names)
        feature_positions = {identity: i for i, identity in enumerate(all_features)}
        if any(identity not in feature_positions for identity in features):
            missing = next(identity for identity in features if identity not in feature_positions)
            raise KeyError(f"unknown feature identity: {missing!r}")
        col_map = {i: feature_positions[identity] for i, identity in enumerate(features)}
        shape = (len(all_ids), len(all_features))
    else:
        col_map = {i: i for i in range(values.shape[1])}
        shape = (len(all_ids), values.shape[1])

    try:
        from scipy import sparse
    except ImportError:
        sparse = None
    if sparse is not None and sparse.issparse(values):
        if fill_value != 0:
            raise ValueError("sparse output supports zero fill_value only")
        coo = values.tocoo(copy=False)
        rr = [row_map[int(i)] for i in coo.row]
        cc = [col_map[int(j)] for j in coo.col]
        out = sparse.coo_matrix((coo.data.copy(), (rr, cc)), shape=shape)
        if sparse.isspmatrix_csr(values):
            out = out.tocsr()
        elif sparse.isspmatrix_csc(values):
            out = out.tocsc()
        else:
            out = out.asformat(values.format)
    else:
        import numpy as np

        source = np.asarray(values)
        out = np.full(shape, fill_value, dtype=source.dtype)
        for source_i, target_i in row_map.items():
            for source_j, target_j in col_map.items():
                out[target_i, target_j] = source[source_i, source_j]
    if target == "layer":
        adata.layers[key] = out
    else:
        adata.obsm[key] = out
    return out


__all__ = ["AnnDataView", "from_adata", "write_observation"]
