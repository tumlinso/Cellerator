"""Torch-free biological identities, alignment records, and prediction queries.

These records keep biological identity separate from a model's integer
coordinates.  They describe meaning at the data boundary; numerical values
remain owned by the caller and may be dense, sparse, differentiable, or
externally managed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Callable, Literal

from ._declarations import Axis, Identity


def _hashable_unique(values: Sequence[Any], name: str) -> tuple[Any, ...]:
    result = tuple(values)
    try:
        if len(set(result)) != len(result):
            raise ValueError(f"{name} must be unique")
    except TypeError as exc:
        raise TypeError(f"{name} must contain hashable values") from exc
    return result


def _nonnegative_index(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a nonnegative integer")
    try:
        result = int(value.__index__())
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a nonnegative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _freeze_json(value: Any, path: str = "metadata") -> Any:
    """Copy JSON-shaped metadata into recursively immutable containers."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} cannot contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            frozen[key] = _freeze_json(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item, f"{path}[]") for item in value)
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _json_safe(value: Any, path: str = "value") -> Any:
    """Validate a custom codec payload without coercing opaque objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} cannot contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} mapping keys must be strings")
        return {key: _json_safe(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item, f"{path}[]") for item in value]
    raise TypeError(f"{path} is not JSON serializable")


@dataclass(frozen=True)
class IdentityCodec:
    """Explicit, caller-defined codec for custom feature identity types.

    ``encode`` must return JSON-compatible data and ``decode`` must reconstruct
    the identity.  The codec name and version are declarations; users remain
    responsible for supplying equivalent code when a manifest is restored.
    """

    name: str
    encode: Callable[[Any], Any] = field(compare=False, repr=False)
    decode: Callable[[Any], Any] = field(compare=False, repr=False)
    version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("codec name must be a non-empty string")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version <= 0:
            raise ValueError("codec version must be a positive integer")
        if not callable(self.encode) or not callable(self.decode):
            raise TypeError("codec encode and decode must be callable")


def _encode_builtin_identity(value: Any) -> dict[str, Any]:
    if isinstance(value, Identity):
        return {"type": "identity128", "low": value.low, "high": value.high}
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, float) and math.isfinite(value):
        return {"type": "float", "value": value}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode_builtin_identity(item) for item in value]}
    raise TypeError(
        f"identity type {type(value).__name__} needs an explicit IdentityCodec"
    )


def _decode_builtin_identity(value: Mapping[str, Any]) -> Any:
    if not isinstance(value, Mapping):
        raise TypeError("encoded identity must be a mapping")
    kind = value.get("type")
    if kind == "identity128":
        return Identity(value["low"], value["high"])
    if kind == "none":
        return None
    if kind == "bool":
        return bool(value["value"])
    if kind == "int":
        return int(value["value"])
    if kind == "str":
        return str(value["value"])
    if kind == "float":
        result = float(value["value"])
        if not math.isfinite(result):
            raise ValueError("encoded identity float must be finite")
        return result
    if kind == "tuple":
        return tuple(_decode_builtin_identity(item) for item in value["items"])
    raise ValueError(f"unsupported encoded identity type: {kind!r}")


def _encode_identity(value: Any, codec: Any = None) -> dict[str, Any]:
    if codec is None:
        return _encode_builtin_identity(value)
    if isinstance(codec, str):
        if codec != "json":
            raise ValueError("the only named built-in identity codec is 'json'")
        return _encode_builtin_identity(value)
    if not all(hasattr(codec, name) for name in ("name", "version", "encode", "decode")):
        raise TypeError("identity_codec must be IdentityCodec or a compatible codec object")
    encoded = _json_safe(codec.encode(value), "encoded identity")
    return {"type": "custom", "codec": codec.name,
            "version": int(codec.version), "value": encoded}


def _decode_identity(value: Mapping[str, Any], codecs: Mapping[str, Any] | None = None) -> Any:
    if value.get("type") != "custom":
        return _decode_builtin_identity(value)
    name = value.get("codec")
    if codecs is None or name not in codecs:
        raise ValueError(f"custom identity codec {name!r} was not supplied")
    codec = codecs[name]
    if int(value.get("version", -1)) != int(codec.version):
        raise ValueError(f"identity codec version mismatch for {name!r}")
    return codec.decode(value.get("value"))


@dataclass(frozen=True)
class PreparedAlignment:
    """A reusable mapping from a source occurrence order to model coordinates.

    ``source_positions`` and ``model_indices`` are parallel and preserve the
    requested order.  ``source_order`` and ``requested_ids`` contain canonical
    identities when resolution succeeded; repeated values are retained for
    event occurrences.  Numerical arrays are not stored or copied.
    """

    source_order: tuple[Any, ...]
    requested_ids: tuple[Any, ...]
    source_positions: tuple[int, ...]
    model_indices: tuple[int, ...]
    canonical_indices: tuple[int | None, ...] = ()
    dropped_ids: tuple[Any, ...] = ()
    binding_signature: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_order", tuple(self.source_order))
        object.__setattr__(self, "requested_ids", tuple(self.requested_ids))
        object.__setattr__(self, "source_positions", tuple(
            _nonnegative_index(value, "source position") for value in self.source_positions
        ))
        object.__setattr__(self, "model_indices", tuple(
            _nonnegative_index(value, "model index") for value in self.model_indices
        ))
        canonical = tuple(self.canonical_indices)
        if not canonical and self.model_indices:
            canonical = (None,) * len(self.model_indices)
        if len(canonical) != len(self.model_indices):
            raise ValueError("canonical_indices must align with model_indices")
        if len(self.requested_ids) != len(self.model_indices) or len(self.source_positions) != len(self.model_indices):
            raise ValueError("requested IDs, source positions, and model indices must align")
        if any(value is not None and _nonnegative_index(value, "canonical index") < 0
               for value in canonical):
            raise ValueError("canonical indices must be nonnegative or None")
        object.__setattr__(self, "canonical_indices", tuple(
            None if value is None else _nonnegative_index(value, "canonical index")
            for value in canonical
        ))
        object.__setattr__(self, "dropped_ids", tuple(self.dropped_ids))
        object.__setattr__(self, "binding_signature", tuple(self.binding_signature))
        if any(position >= len(self.source_order) for position in self.source_positions):
            raise ValueError("source position is outside source_order")

    def __len__(self) -> int:
        return len(self.model_indices)

    def index_tensor(
        self,
        *,
        kind: Literal["source", "model"] = "source",
        device: Any = None,
    ) -> Any:
        """Explicitly materialize indices as a Torch tensor on ``device``.

        Torch is imported only by this method.  No input tensor is inspected,
        copied back to the host, or moved between devices implicitly.
        """
        import torch

        if kind == "source":
            values = self.source_positions
        elif kind == "model":
            values = self.model_indices
        else:
            raise ValueError("kind must be 'source' or 'model'")
        return torch.tensor(values, dtype=torch.long, device=device)

    def gather(self, values: Any, *, axis: int = -1) -> Any:
        """Select aligned entries while preserving the source storage family."""
        shape = getattr(values, "shape", None)
        if shape is None:
            raise TypeError("values must expose a shape")
        ndim = len(shape)
        normalized_axis = axis + ndim if axis < 0 else axis
        if normalized_axis < 0 or normalized_axis >= ndim:
            raise ValueError("axis is outside values dimensions")
        if int(shape[normalized_axis]) != len(self.source_order):
            raise ValueError("values axis extent does not match the prepared source order")

        # Keep SciPy matrices sparse; importing SciPy is optional and localized.
        try:
            from scipy import sparse
        except ImportError:
            sparse = None
        if sparse is not None and sparse.issparse(values):
            if ndim != 2:
                raise ValueError("SciPy sparse values must be two-dimensional")
            indices = list(self.source_positions)
            return values[:, indices] if normalized_axis == 1 else values[indices, :]

        # Torch indexing remains differentiable and uses the tensor's device.
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(values, torch.Tensor):
            indices = self.index_tensor(kind="source", device=values.device)
            return values.index_select(normalized_axis, indices)

        if hasattr(values, "take"):
            return values.take(self.source_positions, axis=normalized_axis)
        if normalized_axis == 0:
            return values[list(self.source_positions)]
        raise TypeError("values must support axis-aware take/indexing")


class FeatureBinding:
    """Immutable model-local mapping between biological IDs and model indices.

    Canonical biological identities, optional source-vocabulary canonical
    indices, and model/embedding indices are distinct coordinate systems.
    Aliases resolve to canonical identities using canonical-ID precedence;
    aliases resolving to multiple identities fail as ambiguous.
    """

    __slots__ = (
        "identities", "model_indices", "canonical_indices", "domain",
        "metadata", "aliases", "unknown_policy", "unknown_model_index",
        "model_capacity", "identity_codec", "axis", "_positions", "_alias_targets",
        "_model_targets", "_signature", "_frozen",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("use FeatureBinding.from_ids() or from_gene_vocabulary()")

    @classmethod
    def from_ids(
        cls,
        identities: Sequence[Any],
        model_indices: Sequence[int] | None = None,
        *,
        domain: str = "feature",
        metadata: Mapping[str, Any] | None = None,
        aliases: Mapping[Any, Sequence[Any]] | None = None,
        canonical_indices: Sequence[int] | None = None,
        axis: Axis | None = None,
        model_capacity: int | None = None,
        unknown_policy: Literal["error", "drop", "map_to_token"] = "error",
        unknown_model_index: int | None = None,
        identity_codec: Any = None,
    ) -> "FeatureBinding":
        ids = _hashable_unique(identities, "feature identities")
        if not ids:
            raise ValueError("at least one feature identity is required")
        model = tuple(range(len(ids))) if model_indices is None else tuple(
            _nonnegative_index(value, "model index") for value in model_indices
        )
        if len(model) != len(ids):
            raise ValueError("model_indices must match the identity count")
        canonical = None if canonical_indices is None else tuple(
            _nonnegative_index(value, "canonical index") for value in canonical_indices
        )
        if canonical is not None:
            if len(canonical) != len(ids):
                raise ValueError("canonical_indices must match the identity count")
            if len(set(canonical)) != len(canonical):
                raise ValueError("canonical_indices must be unique")
        if not isinstance(domain, str) or not domain:
            raise ValueError("domain must be a non-empty string")
        if axis is not None:
            if not isinstance(axis, Axis):
                raise TypeError("axis must use the existing Cellerator Axis declaration")
            if axis.extent != len(ids):
                raise ValueError("axis extent must match the ordered feature identities")
        if unknown_policy not in {"error", "drop", "map_to_token"}:
            raise ValueError("unknown_policy must be error, drop, or map_to_token")
        fallback = None if unknown_model_index is None else _nonnegative_index(
            unknown_model_index, "unknown_model_index"
        )
        if (unknown_policy == "map_to_token") != (fallback is not None):
            raise ValueError("map_to_token requires unknown_model_index, used only for that policy")
        capacity = None if model_capacity is None else _nonnegative_index(
            model_capacity, "model_capacity"
        )
        if capacity == 0:
            raise ValueError("model_capacity must be positive")
        if not isinstance(metadata, Mapping) and metadata is not None:
            raise TypeError("metadata must be a mapping")
        frozen_metadata = _freeze_json(metadata or {})

        positions = {identity: position for position, identity in enumerate(ids)}
        alias_targets: dict[Any, list[Any]] = {}
        aliases_by_identity: dict[Any, tuple[Any, ...]] = {}
        for identity, names in (aliases or {}).items():
            if identity not in positions:
                raise KeyError(f"aliases refer to unknown canonical identity {identity!r}")
            try:
                alias_names = tuple(names)
            except TypeError as exc:
                raise TypeError("each alias collection must be iterable") from exc
            try:
                if len(set(alias_names)) != len(alias_names):
                    raise ValueError(f"aliases for {identity!r} must be unique")
            except TypeError as exc:
                raise TypeError("aliases must be hashable identities") from exc
            aliases_by_identity[identity] = alias_names
            for alias in alias_names:
                try:
                    alias_targets.setdefault(alias, []).append(identity)
                except TypeError as exc:
                    raise TypeError("aliases must be hashable identities") from exc

        model_targets: dict[int, list[int]] = {}
        for position, index in enumerate(model):
            model_targets.setdefault(index, []).append(position)
        if capacity is not None and max((*model, *(x for x in (fallback,) if x is not None))) >= capacity:
            raise ValueError("model index exceeds model_capacity")

        frozen = object.__new__(cls)
        object.__setattr__(frozen, "identities", ids)
        object.__setattr__(frozen, "model_indices", model)
        object.__setattr__(frozen, "canonical_indices", canonical)
        object.__setattr__(frozen, "domain", domain)
        object.__setattr__(frozen, "metadata", frozen_metadata)
        object.__setattr__(frozen, "aliases", MappingProxyType(aliases_by_identity))
        object.__setattr__(frozen, "unknown_policy", unknown_policy)
        object.__setattr__(frozen, "unknown_model_index", fallback)
        object.__setattr__(frozen, "model_capacity", capacity)
        object.__setattr__(frozen, "identity_codec", identity_codec)
        object.__setattr__(frozen, "axis", axis)
        object.__setattr__(frozen, "_positions", MappingProxyType(positions))
        object.__setattr__(frozen, "_alias_targets", MappingProxyType({
            alias: tuple(targets) for alias, targets in alias_targets.items()
        }))
        object.__setattr__(frozen, "_model_targets", MappingProxyType({
            token: tuple(targets) for token, targets in model_targets.items()
        }))
        object.__setattr__(frozen, "_signature", (
            domain, ids, model, canonical, axis, tuple(aliases_by_identity.items()),
            _metadata_signature(frozen_metadata), unknown_policy, fallback,
        ))
        object.__setattr__(frozen, "_frozen", True)
        return frozen

    @classmethod
    def from_gene_vocabulary(
        cls,
        vocabulary: Any,
        *,
        axis: Axis | None = None,
        model_capacity: int | None = None,
        identity_codec: Any = None,
    ) -> "FeatureBinding":
        """Snapshot a native ``GeneVocabulary`` without retaining its owner.

        Canonical index is the sorted vocabulary position; model index is the
        actual token ID stored on each record. Metadata, aliases, and unknown
        token policy are copied once, so later mutation/destruction of the
        vocabulary wrapper cannot change this binding.
        """
        if not all(hasattr(vocabulary, attr) for attr in ("genes", "metadata", "integer_vocabulary")):
            raise TypeError("vocabulary must expose GeneVocabulary records and metadata")
        records = tuple(vocabulary.genes)
        ids = tuple(record.gene_id for record in records)
        if any(record.token_id is None for record in records):
            raise ValueError("GeneVocabulary records must have resolved token IDs")
        model_indices = tuple(_nonnegative_index(record.token_id, "gene token ID")
                              for record in records)
        canonical_indices = tuple(range(len(records)))
        aliases = {
            record.gene_id: tuple(record.aliases) + ((record.symbol,) if record.symbol else ())
            for record in records
        }
        metadata_record = vocabulary.metadata
        metadata = {
            "source": str(metadata_record.source),
            "organism": str(metadata_record.organism),
            "assembly": str(metadata_record.assembly),
            "release": str(metadata_record.release),
        }
        native = vocabulary.integer_vocabulary("cpu")
        policy = getattr(native, "unknown_policy", "error")
        policy_name = getattr(policy, "name", None)
        if not policy_name:
            policy_name = str(policy).rsplit(".", 1)[-1]
        policy_name = str(policy_name).lower()
        if policy_name not in {"error", "drop", "map_to_token"}:
            raise ValueError(f"unsupported native unknown identity policy {policy_name!r}")
        unknown_token = getattr(native, "unknown_token_id", None)
        return cls.from_ids(
            ids,
            model_indices,
            domain="gene",
            metadata=metadata,
            aliases=aliases,
            canonical_indices=canonical_indices,
            axis=axis,
            model_capacity=model_capacity,
            unknown_policy=policy_name,
            unknown_model_index=unknown_token,
            identity_codec=identity_codec,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("FeatureBinding is immutable")
        object.__setattr__(self, name, value)

    def validate_model_capacity(self, capacity: int) -> None:
        """Check model rows once when the binding is attached to an encoder."""
        size = _nonnegative_index(capacity, "model capacity")
        if size <= 0:
            raise ValueError("model capacity must be positive")
        indices = self.model_indices + (() if self.unknown_model_index is None
                                        else (self.unknown_model_index,))
        if any(index >= size for index in indices):
            raise ValueError("FeatureBinding model index exceeds encoder vocabulary size")

    def assert_compatible(self, other: "FeatureBinding", *, require_order: bool = True) -> None:
        """Raise when two equal-shaped bindings declare incompatible meaning."""
        if not isinstance(other, FeatureBinding):
            raise TypeError("other must be a FeatureBinding")
        if self.domain != other.domain:
            raise ValueError("feature binding domains differ")
        if require_order:
            compatible = self.identities == other.identities
        else:
            compatible = set(self.identities) == set(other.identities)
        if not compatible:
            raise ValueError("feature binding identities or ordering differ")
        if self.model_indices != other.model_indices or self.canonical_indices != other.canonical_indices:
            raise ValueError("feature binding model/canonical indices differ")
        if self.axis != other.axis:
            raise ValueError("feature binding typed axes differ")
        if self._signature[5:] != other._signature[5:]:
            raise ValueError("feature binding aliases, metadata, or unknown policy differ")

    def validate_alignment(self, alignment: PreparedAlignment) -> None:
        if not isinstance(alignment, PreparedAlignment):
            raise TypeError("alignment must be a PreparedAlignment")
        if alignment.binding_signature != self._signature:
            raise ValueError("prepared alignment belongs to an incompatible feature binding")

    def _resolve(self, identity: Any, *, id_space: str = "identity") -> tuple[Any, int | None, int]:
        if id_space == "model":
            try:
                model_id = _nonnegative_index(identity, "model identity")
            except (TypeError, ValueError) as exc:
                raise TypeError("model-space identities must be nonnegative integer IDs") from exc
            positions = self._model_targets.get(model_id, ())
            if len(positions) > 1:
                raise ValueError(f"ambiguous model index {model_id}")
            if positions:
                position = positions[0]
                canonical = (None if self.canonical_indices is None
                             else self.canonical_indices[position])
                return self.identities[position], canonical, model_id
            if self.unknown_policy == "drop":
                return identity, None, -1
            if self.unknown_policy == "map_to_token":
                return identity, None, int(self.unknown_model_index)
            raise KeyError(f"unknown model index: {model_id}")

        try:
            position = self._positions.get(identity)
        except TypeError as exc:
            raise TypeError("feature identities must be hashable") from exc
        if position is not None:
            canonical = None if self.canonical_indices is None else self.canonical_indices[position]
            return self.identities[position], canonical, self.model_indices[position]
        targets = self._alias_targets.get(identity, ())
        if len(targets) > 1:
            raise ValueError(f"ambiguous feature alias: {identity!r}")
        if targets:
            canonical_id = targets[0]
            position = self._positions[canonical_id]
            canonical = None if self.canonical_indices is None else self.canonical_indices[position]
            return canonical_id, canonical, self.model_indices[position]
        if self.unknown_policy == "drop":
            return identity, None, -1
        if self.unknown_policy == "map_to_token":
            return identity, None, int(self.unknown_model_index)
        raise KeyError(f"unknown feature identity: {identity!r}")

    def model_index(self, identity: Any, *, id_space: str = "identity") -> int:
        """Resolve one canonical ID/alias (or explicit model-space ID)."""
        _canonical, _canonical_index, model_index = self._resolve(identity, id_space=id_space)
        if model_index < 0:
            raise KeyError(f"feature identity was dropped by unknown_policy: {identity!r}")
        return model_index

    def prepare(
        self,
        source_ids: Sequence[Any],
        requested_ids: Sequence[Any] | None = None,
    ) -> PreparedAlignment:
        """Prepare unique feature-axis alignment in requested biological order."""
        raw_source = tuple(source_ids)
        canonical_source: list[Any] = []
        source_positions: dict[Any, int] = {}
        dropped: list[Any] = []
        for position, identity in enumerate(raw_source):
            canonical, _canonical_index, _model_index = self._resolve(identity)
            if self.unknown_policy == "drop" and _model_index < 0:
                dropped.append(identity)
                canonical_source.append(identity)
                continue
            if canonical in source_positions:
                raise ValueError("source feature identities must be unique after alias resolution")
            source_positions[canonical] = position
            canonical_source.append(canonical)

        raw_requested = raw_source if requested_ids is None else tuple(requested_ids)
        requested: list[Any] = []
        positions: list[int] = []
        model_indices: list[int] = []
        canonical_indices: list[int | None] = []
        for identity in raw_requested:
            canonical, canonical_index, model_index = self._resolve(identity)
            if self.unknown_policy == "drop" and model_index < 0:
                dropped.append(identity)
                continue
            try:
                source_position = source_positions[canonical]
            except KeyError as exc:
                raise KeyError(f"requested feature is absent from source axis: {identity!r}") from exc
            requested.append(canonical)
            positions.append(source_position)
            model_indices.append(model_index)
            canonical_indices.append(canonical_index)

        if len(set(requested)) != len(requested):
            raise ValueError("requested feature identities must be unique")
        return PreparedAlignment(
            tuple(canonical_source), tuple(requested), tuple(positions),
            tuple(model_indices), tuple(canonical_indices), tuple(dropped), self._signature,
        )

    def map_occurrences(
        self,
        source_ids: Sequence[Any],
        *,
        id_space: Literal["identity", "model"] = "identity",
    ) -> PreparedAlignment:
        """Map ordered occurrences without deduplicating repeated identities."""
        if id_space not in {"identity", "model"}:
            raise ValueError("id_space must be 'identity' or 'model'")
        raw_source = tuple(source_ids)
        source_order: list[Any] = []
        requested: list[Any] = []
        source_positions: list[int] = []
        model_indices: list[int] = []
        canonical_indices: list[int | None] = []
        dropped: list[Any] = []
        for position, identity in enumerate(raw_source):
            canonical, canonical_index, model_index = self._resolve(identity, id_space=id_space)
            if model_index < 0:
                dropped.append(identity)
                source_order.append(identity)
                continue
            source_order.append(canonical)
            requested.append(canonical)
            source_positions.append(position)
            model_indices.append(model_index)
            canonical_indices.append(canonical_index)
        return PreparedAlignment(
            tuple(source_order), tuple(requested), tuple(source_positions),
            tuple(model_indices), tuple(canonical_indices), tuple(dropped), self._signature,
        )

    def to_manifest(self, *, codec: Any = None) -> dict[str, Any]:
        """Return a strict JSON-safe semantic descriptor for checkpoint pairing."""
        selected_codec = self.identity_codec if codec is None else codec
        encode = lambda value: _encode_identity(value, selected_codec)
        alias_pairs = []
        for identity in self.identities:
            names = self.aliases.get(identity, ())
            if names:
                alias_pairs.append({"identity": encode(identity),
                                    "aliases": [encode(alias) for alias in names]})
        return {
            "schema_version": 1,
            "domain": self.domain,
            "identities": [encode(identity) for identity in self.identities],
            "canonical_indices": (None if self.canonical_indices is None
                                  else list(self.canonical_indices)),
            "axis": (None if self.axis is None else {
                "domain": _encode_identity(self.axis.domain),
                "order": _encode_identity(self.axis.order),
                "geometry": _encode_identity(self.axis.geometry),
                "partition": _encode_identity(self.axis.partition),
                "extent": self.axis.extent,
            }),
            "model_indices": list(self.model_indices),
            "aliases": alias_pairs,
            "metadata": _thaw_json(self.metadata),
            "unknown_policy": self.unknown_policy,
            "unknown_model_index": self.unknown_model_index,
            "model_capacity": self.model_capacity,
        }

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
        *,
        codecs: Mapping[str, Any] | None = None,
    ) -> "FeatureBinding":
        """Restore binding meaning using built-in or explicitly supplied codecs."""
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported FeatureBinding manifest version")
        decode = lambda value: _decode_identity(value, codecs)
        identities = tuple(decode(item) for item in manifest["identities"])
        aliases = {
            decode(item["identity"]): tuple(decode(alias) for alias in item["aliases"])
            for item in manifest.get("aliases", ())
        }
        codec = None
        for encoded in manifest["identities"]:
            if encoded.get("type") == "custom":
                codec_name = encoded["codec"]
                codec = None if codecs is None else codecs.get(codec_name)
                break
        axis_data = manifest.get("axis")
        axis = None if axis_data is None else Axis(
            _decode_builtin_identity(axis_data["domain"]),
            _decode_builtin_identity(axis_data["order"]),
            _decode_builtin_identity(axis_data["geometry"]),
            _decode_builtin_identity(axis_data["partition"]),
            axis_data["extent"],
        )
        return cls.from_ids(
            identities,
            manifest["model_indices"],
            domain=manifest["domain"],
            metadata=manifest.get("metadata", {}),
            aliases=aliases,
            canonical_indices=manifest.get("canonical_indices"),
            axis=axis,
            model_capacity=manifest.get("model_capacity"),
            unknown_policy=manifest.get("unknown_policy", "error"),
            unknown_model_index=manifest.get("unknown_model_index"),
            identity_codec=codec,
        )


def _metadata_signature(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((key, _metadata_signature(item)) for key, item in value.items()))
    if isinstance(value, tuple):
        return tuple(_metadata_signature(item) for item in value)
    return value


def _offsets(value: Sequence[int] | None, sample_count: int | None, name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    offsets = tuple(_nonnegative_index(item, f"{name} entry") for item in value)
    if not offsets or offsets[0] != 0:
        raise ValueError(f"{name} must begin at zero")
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError(f"{name} must be nondecreasing")
    if sample_count is not None and len(offsets) != sample_count + 1:
        raise ValueError(f"{name} needs one boundary per sample plus a final boundary")
    return offsets


@dataclass(frozen=True)
class PredictionQuery:
    """One requested model view and its explicit feature/sample coordinates."""

    view: str
    feature_ids: tuple[Any, ...] | Sequence[Any] | None = None
    sample_ids: tuple[Any, ...] | Sequence[Any] | None = None
    row_offsets: tuple[int, ...] | Sequence[int] | None = None
    conditioning: Any = None
    context_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.view, str) or not self.view:
            raise ValueError("view must be a non-empty string")
        features = None if self.feature_ids is None else _hashable_unique(
            self.feature_ids, "query feature IDs"
        )
        samples = None if self.sample_ids is None else tuple(self.sample_ids)
        metadata = _freeze_json(self.context_metadata or {}, "context_metadata")
        offsets = _offsets(self.row_offsets, None if samples is None else len(samples), "row_offsets")
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "row_offsets", offsets)
        object.__setattr__(self, "context_metadata", metadata)


@dataclass(frozen=True)
class Prediction:
    """Borrowed differentiable values with explicit biological coordinates.

    ``values`` and ``context`` are retained by reference: this record never
    copies, moves, or detaches them. Repeated sample IDs are valid for distinct
    windows/observations; use row offsets or caller-declared keys to distinguish
    them when needed.
    """

    values: Any
    feature_ids: tuple[Any, ...] | Sequence[Any] | None = None
    sample_ids: tuple[Any, ...] | Sequence[Any] | None = None
    row_offsets: tuple[int, ...] | Sequence[int] | None = None
    context: Any = None

    def __post_init__(self) -> None:
        features = None if self.feature_ids is None else _hashable_unique(
            self.feature_ids, "prediction feature IDs"
        )
        samples = None if self.sample_ids is None else tuple(self.sample_ids)
        offsets = _offsets(self.row_offsets, None if samples is None else len(samples), "row_offsets")
        object.__setattr__(self, "feature_ids", features)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "row_offsets", offsets)


__all__ = [
    "FeatureBinding", "IdentityCodec", "PreparedAlignment", "Prediction",
    "PredictionQuery",
]
