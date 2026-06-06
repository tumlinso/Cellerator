from __future__ import annotations

from os import PathLike
from typing import Any

from ._cellerator import (
    PreprocessOptions,
    compile_default_qc_feature_group_masks,
    compile_qc_feature_group_masks,
    plan_cellshard_adapter_stage,
    preprocess_cellshard,
    validate_raw_count_state,
)


def _dataset_path(data: Any, cellshard_path: str | None = None) -> str:
    if cellshard_path is not None:
        return str(cellshard_path)
    if isinstance(data, (str, PathLike)):
        return str(data)
    path = getattr(data, "path", None)
    if path is not None and str(path).endswith(".csh5"):
        return str(path)
    uns = getattr(data, "uns", None)
    if isinstance(uns, dict):
        for key in ("cellshard_path", "cellerator_cellshard_path"):
            if key in uns:
                return str(uns[key])
    raise TypeError(
        "GPU preprocessing requires a CellShard .csh5 path, a cellshard.Dataset, "
        "or an AnnData object with uns['cellshard_path']; AnnData matrices are "
        "not converted through SciPy in this hot path"
    )


def _make_options(options: PreprocessOptions | None, kwargs: dict[str, Any]) -> PreprocessOptions:
    if options is not None and kwargs:
        raise ValueError("pass either options=... or keyword option overrides, not both")
    if options is not None:
        return options
    out = PreprocessOptions()
    for key, value in kwargs.items():
        if not hasattr(out, key):
            raise TypeError(f"unknown PreprocessOptions field: {key}")
        setattr(out, key, value)
    return out


def plan(data: Any = None, *, path: str | None = None, format: str = "h5ad",
         matrix_source: str = "counts", allow_processed: bool = False):
    source_path = path if path is not None else getattr(data, "filename", None)
    if source_path is None and isinstance(data, (str, PathLike)):
        source_path = str(data)
    return plan_cellshard_adapter_stage(
        str(source_path or ""),
        format=format,
        matrix_source=matrix_source,
        allow_processed=allow_processed,
    )


def preprocess(data: Any, *, cellshard_path: str | None = None,
               options: PreprocessOptions | None = None, copy: bool = True,
               inplace: bool = False, autotune: bool | None = None, **kwargs):
    if inplace:
        raise ValueError("inplace=True is metadata-only future work; GPU preprocessing returns a session")
    if not copy:
        raise ValueError("copy=False would imply hidden mutation; use the returned PreprocessSession")
    active_options = _make_options(options, kwargs)
    if autotune is not None:
        active_options.autotune = bool(autotune)
    return preprocess_cellshard(_dataset_path(data, cellshard_path), active_options)


def qc(data: Any, *, cellshard_path: str | None = None,
       options: PreprocessOptions | None = None, autotune: bool | None = None, **kwargs):
    session = preprocess(data, cellshard_path=cellshard_path, options=options, autotune=autotune, **kwargs)
    return session.metrics()


def normalize_log1p(data: Any, *, cellshard_path: str | None = None,
                    options: PreprocessOptions | None = None, autotune: bool | None = None, **kwargs):
    return preprocess(data, cellshard_path=cellshard_path, options=options, autotune=autotune, **kwargs)


__all__ = [
    "PreprocessOptions",
    "compile_default_qc_feature_group_masks",
    "compile_qc_feature_group_masks",
    "normalize_log1p",
    "plan",
    "plan_cellshard_adapter_stage",
    "preprocess",
    "qc",
    "validate_raw_count_state",
]
