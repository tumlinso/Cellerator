"""Small PyTorch composition for named views of one shared model result.

This adapter is intentionally just normal ``nn.Module`` composition: every
prediction calls the base model, selects a declared source, and applies the
requested readout while preserving the ordinary autograd graph. It owns no
cell state, cache, or training policy.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from torch import nn

from cellerator.biology import FeatureBinding, Prediction, PredictionQuery


def _declared_value(record: Any, name: str) -> Any:
    if isinstance(record, Prediction):
        return getattr(record, name)
    if isinstance(record, Mapping):
        return record.get(name)
    return None


def _ids_tuple(values: Any, name: str) -> tuple[Any, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or hasattr(values, "device"):
        raise TypeError(f"{name} must be a host sequence of identities")
    try:
        return tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a host sequence of identities") from exc


def _offset_tuple(offsets: Any, row_count: int | None = None) -> tuple[int, ...] | None:
    if offsets is None:
        return None
    if isinstance(offsets, (str, bytes)) or hasattr(offsets, "device"):
        raise TypeError("row_offsets must be a host sequence of integers")
    try:
        result = tuple(offsets)
    except TypeError as exc:
        raise TypeError("row_offsets must be a host sequence of integers") from exc
    if not result or result[0] != 0:
        raise ValueError("row_offsets must begin at zero")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in result):
        raise TypeError("row_offsets entries must be integers")
    if any(left > right for left, right in zip(result, result[1:])):
        raise ValueError("row_offsets must be nondecreasing")
    if row_count is not None and result[-1] != row_count:
        raise ValueError("row_offsets terminal offset must match the prediction row count")
    return result


def _validate_prediction_coordinates(
    values: Any,
    sample_ids: tuple[Any, ...] | None,
    row_offsets: tuple[int, ...] | None,
) -> None:
    shape = getattr(values, "shape", None)
    row_count = None if shape is None or len(shape) == 0 else int(shape[0])
    offsets = _offset_tuple(row_offsets, row_count)
    if sample_ids is None:
        return
    expected = len(offsets) - 1 if offsets is not None else row_count
    if expected is None:
        expected = 1 if len(sample_ids) == 1 else None
    if expected != len(sample_ids):
        raise ValueError("sample_ids must label prediction rows or row-offset groups in output order")


class _View(nn.Module):
    def __init__(self, module: nn.Module,
                 source: str | Callable[[Any, Any], Any] | None,
                 output_binding: FeatureBinding | None,
                 queryable: bool,
                 source_id: str | None) -> None:
        super().__init__()
        self.module = module
        self.output_binding = output_binding
        self.queryable = bool(queryable)
        self.source_id = source_id
        if isinstance(source, nn.Module):
            # A trainable source transform is itself part of the shared module
            # tree and must remain visible to optimizers and checkpoints.
            self.source_module = source
            self._source = None
        else:
            self.source_module = None
            # Plain callables are selectors, not parameter owners. Parameterized
            # selectors should be supplied as nn.Modules.
            object.__setattr__(self, "_source", source)

    def select_source(self, result: Any, inputs: Any) -> Any:
        source = self.source_module if self.source_module is not None else self._source
        if source is None:
            value = result.values if isinstance(result, Prediction) else result
        elif isinstance(source, str):
            if not isinstance(result, Mapping):
                raise TypeError("a named view source requires a mapping base result")
            try:
                value = result[source]
            except KeyError as exc:
                raise KeyError(f"base result has no view source {source!r}") from exc
        else:
            value = source(result, inputs)
        return value

    def forward(self, result: Any, inputs: Any) -> Any:
        return self.module(self.select_source(result, inputs))


class ModelWithViews(nn.Module):
    """Attach shared readout modules to selected results of one base model.

    ``forward`` delegates directly to the base model. ``predict`` evaluates the
    base once per call and runs only the requested views. A string request
    returns that view's value; a sequence returns a name-to-value dictionary in
    request order. Repeated calls recompute the base and retain no hidden cache.

    A view source may be the complete base result (``None``), a key from a
    mapping result, or a callable ``source(result, inputs)``. If the selector
    itself has trainable parameters, supply it as an ``nn.Module`` so PyTorch
    can discover its parameters. ``query`` adds an optional feature-identity
    boundary without changing the tensor-level ``forward`` and ``predict`` APIs.
    """

    def __init__(self, base: nn.Module, *,
                 conditioner: Callable[[Any, Any], Any] | nn.Module | None = None) -> None:
        super().__init__()
        if not isinstance(base, nn.Module):
            raise TypeError("base must be a torch.nn.Module")
        self.base = base
        if conditioner is not None and not callable(conditioner):
            raise TypeError("conditioner must be callable or None")
        if isinstance(conditioner, nn.Module):
            self.conditioner_module = conditioner
            object.__setattr__(self, "_conditioner_fn", None)
        else:
            self.conditioner_module = None
            object.__setattr__(self, "_conditioner_fn", conditioner)
        self._views = nn.ModuleList()
        self._view_indices: dict[str, int] = {}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Return the base model result unchanged."""
        return self.base(*args, **kwargs)

    def add_view(self, name: str, module: nn.Module,
                 source: str | Callable[[Any, Any], Any] | None = None, *,
                 output_binding: FeatureBinding | None = None,
                 queryable: bool | None = None,
                 source_id: str | None = None) -> None:
        """Register one named readout and the source it consumes."""
        if not isinstance(name, str) or not name:
            raise ValueError("view name must be a nonempty string")
        if name in self._view_indices:
            raise ValueError(f"view {name!r} is already registered")
        if not isinstance(module, nn.Module):
            raise TypeError("view module must be a torch.nn.Module")
        if source is not None and not isinstance(source, str) and not callable(source):
            raise TypeError("source must be None, a mapping key, or callable")
        if output_binding is not None and not isinstance(output_binding, FeatureBinding):
            raise TypeError("output_binding must be a FeatureBinding or None")
        module_binding = getattr(module, "binding", None)
        if isinstance(module_binding, FeatureBinding) and output_binding is not None:
            try:
                module_binding.assert_compatible(output_binding, require_order=True)
            except ValueError as exc:
                raise ValueError("output_binding must match the readout module's FeatureBinding") from exc
        if source_id is not None and (not isinstance(source_id, str) or not source_id):
            raise ValueError("source_id must be a non-empty string or None")
        supports_query = callable(getattr(module, "query", None))
        if queryable is None:
            queryable = supports_query
        elif not isinstance(queryable, bool):
            raise TypeError("queryable must be a Boolean or None")
        if queryable and not supports_query:
            raise TypeError("queryable views must implement query(source, target_indices=..., query=...)")
        if queryable and output_binding is None:
            raise ValueError("queryable views require an explicit output_binding")
        self._view_indices[name] = len(self._views)
        self._views.append(_View(module, source, output_binding, queryable, source_id))

    def biological_view_descriptors(self, *, identity_codec: Any = None) -> tuple[dict[str, Any], ...]:
        """Describe registered readouts without serializing executable code.

        Callable selectors require an explicit stable ``source_id`` before a
        manifest descriptor can be produced. Output bindings contribute their
        JSON-safe biological axis descriptor, including identity order and
        model indices.
        """
        descriptors = []
        for name, view in zip(self._view_indices, self._views):
            source = view.source_module if view.source_module is not None else view._source
            if source is None:
                source_kind, source_key = "base_result", None
            elif isinstance(source, str):
                source_kind, source_key = "mapping_key", source
            else:
                source_kind, source_key = "callable", None
                if not view.source_id:
                    raise ValueError(
                        f"view {name!r} uses a callable selector; register source_id before manifest capture"
                    )
            binding = view.output_binding
            binding_descriptor = (
                None if binding is None else binding.to_manifest(codec=identity_codec)
            )
            descriptors.append({
                "name": name,
                "output_binding": binding_descriptor,
                "queryable": view.queryable,
                "source_kind": source_kind,
                "source_key": source_key,
                "source_id": view.source_id,
            })
        return tuple(descriptors)

    def _alignment(self, view: _View, query: PredictionQuery):
        binding = view.output_binding
        if binding is None:
            if query.feature_ids is not None:
                raise ValueError(
                    f"view {query.view!r} has no output_binding for feature-specific prediction"
                )
            if view.queryable:
                raise ValueError(f"queryable view {query.view!r} has no output_binding")
            return None
        requested = binding.identities if query.feature_ids is None else query.feature_ids
        return binding.prepare(binding.identities, requested)

    def _validate_query_batch_ids(self, batch: Any, query: PredictionQuery) -> None:
        input_ids = _ids_tuple(_declared_value(batch, "sample_ids"), "batch sample_ids")
        if query.sample_ids is not None and input_ids is not None:
            if tuple(query.sample_ids) != input_ids:
                raise ValueError("query sample_ids must match the declared batch order; select the batch first")
        declared_prediction_offsets = _offset_tuple(
            _declared_value(batch, "prediction_row_offsets")
        )
        if (query.row_offsets is not None and declared_prediction_offsets is not None
                and query.row_offsets != declared_prediction_offsets):
            raise ValueError("query row_offsets do not match declared input prediction_row_offsets")

    def _evaluate_query(self, view: _View, result: Any, batch: Any,
                        query: PredictionQuery, alignment: Any) -> Prediction:
        source = view.select_source(result, batch)
        output_prediction = None
        if view.queryable:
            query_method = getattr(view.module, "query")
            indices = alignment.model_indices
            output = query_method(source, target_indices=indices, query=query)
        else:
            output = view.module(source)
        if isinstance(output, Prediction):
            output_prediction = output
            values = output.values
        else:
            values = output

        feature_ids = None
        if alignment is not None:
            feature_ids = alignment.requested_ids
            if not view.queryable:
                # Ordinary heads compute their full output. The declared
                # output axis is last; queryable modules receive model indices
                # and avoid this full-output gather.
                values = alignment.gather(values, axis=-1)
        elif output_prediction is not None:
            feature_ids = output_prediction.feature_ids

        result_samples = _ids_tuple(_declared_value(result, "sample_ids"), "result sample_ids")
        input_samples = _ids_tuple(_declared_value(batch, "sample_ids"), "batch sample_ids")
        if result_samples is not None and input_samples is not None and result_samples != input_samples:
            raise ValueError("base result sample_ids do not match the declared input order")
        declared_samples = result_samples if result_samples is not None else input_samples
        if query.sample_ids is not None and declared_samples is not None:
            if tuple(query.sample_ids) != declared_samples:
                raise ValueError("query sample_ids must match declared output order; implicit reordering is not supported")
        module_samples = None if output_prediction is None else output_prediction.sample_ids
        if module_samples is not None and declared_samples is not None and tuple(module_samples) != declared_samples:
            raise ValueError("readout sample_ids do not match base output order")
        sample_ids = (
            query.sample_ids if query.sample_ids is not None else
            module_samples if module_samples is not None else declared_samples
        )

        result_offsets = _offset_tuple(_declared_value(result, "row_offsets"))
        module_offsets = (
            None if output_prediction is None
            else _offset_tuple(output_prediction.row_offsets)
        )
        declared_offsets = tuple(
            offsets for offsets in (
                query.row_offsets, module_offsets, result_offsets,
                _offset_tuple(_declared_value(batch, "prediction_row_offsets")),
            ) if offsets is not None
        )
        if declared_offsets and any(offsets != declared_offsets[0] for offsets in declared_offsets[1:]):
            raise ValueError("query, readout, base, and declared prediction row_offsets must agree")
        row_offsets = (
            query.row_offsets if query.row_offsets is not None else
            module_offsets if module_offsets is not None else result_offsets
        )
        _validate_prediction_coordinates(values, sample_ids, row_offsets)

        result_context = _declared_value(result, "context")
        module_context = None if output_prediction is None else output_prediction.context
        context = (
            query.context_metadata if query.context_metadata else
            module_context if module_context is not None else result_context
        )
        return Prediction(
            values=values,
            feature_ids=feature_ids,
            sample_ids=sample_ids,
            row_offsets=row_offsets,
            context=context,
        )

    def query(
        self,
        batch: Any,
        query: PredictionQuery | Sequence[PredictionQuery],
    ) -> Prediction | tuple[Prediction, ...]:
        """Run identity-labelled readouts, reusing base results by condition identity.

        Query sample IDs label rows in the caller-provided batch order; this
        method does not implicitly reorder or select input samples. Use a
        registered ``conditioner(batch, conditioning)`` for actual model input
        conditioning. Context metadata alone is descriptive and is never fed to
        the base model as conditioning.
        """
        single = isinstance(query, PredictionQuery)
        queries = (query,) if single else tuple(query)
        if not all(isinstance(item, PredictionQuery) for item in queries):
            raise TypeError("query must be a PredictionQuery or a sequence of PredictionQuery values")
        if not queries:
            return ()

        records: list[tuple[_View, Any]] = []
        for item in queries:
            if item.view not in self._view_indices:
                raise KeyError(f"unknown view {item.view!r}")
            view = self._views[self._view_indices[item.view]]
            if item.conditioning is not None and self.conditioner_module is None and self._conditioner_fn is None:
                raise ValueError("non-None query conditioning requires a registered conditioner")
            self._validate_query_batch_ids(batch, item)
            alignment = self._alignment(view, item)
            records.append((view, alignment))

        # Conditioning is compared by object identity. Equal-valued but distinct
        # contexts are separate model inputs unless the caller shares one object.
        groups: list[tuple[Any, list[int]]] = []
        for index, item in enumerate(queries):
            for condition, members in groups:
                if item.conditioning is condition:
                    members.append(index)
                    break
            else:
                groups.append((item.conditioning, [index]))

        predictions: list[Prediction | None] = [None] * len(queries)
        for conditioning, members in groups:
            conditioned_batch = batch
            if conditioning is not None:
                conditioner = (
                    self.conditioner_module if self.conditioner_module is not None
                    else self._conditioner_fn
                )
                conditioned_batch = conditioner(batch, conditioning)
            result = self(conditioned_batch)
            for index in members:
                view, alignment = records[index]
                predictions[index] = self._evaluate_query(
                    view, result, batch, queries[index], alignment
                )
        values = tuple(prediction for prediction in predictions if prediction is not None)
        return values[0] if single else values

    def predict(self, batch: Any, view: str | Sequence[str] | None = None) -> Any:
        """Run the base batch, optionally applying explicitly requested views.

        With ``view=None``, return the base result unchanged. Otherwise one
        base evaluation is shared by the requested readout modules.
        """
        if view is None:
            return self(batch)
        single = isinstance(view, str)
        names = (view,) if single else tuple(view)
        if not all(isinstance(name, str) for name in names):
            raise TypeError("view must be a name or a sequence of names")
        if len(set(names)) != len(names):
            raise ValueError("view names in one prediction request must be unique")
        missing = [name for name in names if name not in self._view_indices]
        if missing:
            raise KeyError(f"unknown view {missing[0]!r}")
        if not names:
            return {}

        # Each requested head shares this invocation's result and graph. Nothing
        # is retained after return, so parameter changes cannot stale a cache.
        result = self(batch)
        values = {
            name: self._views[self._view_indices[name]](result, batch)
            for name in names
        }
        return values[names[0]] if single else values


__all__ = ["ModelWithViews"]
