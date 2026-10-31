"""Safe, versioned sidecars for the biological meaning declared by a model.

The manifest records declarations exposed by public module hooks. It does not
serialize modules, code, parameter values, training examples, cell histories,
contexts, or latent activations. Ordinary Torch and native-owner checkpoint
formats remain responsible for weights.

Modules may expose cellerator_biological_descriptors(*, identity_codec=None)
returning a mapping from declaration kind to a JSON-compatible descriptor.
View-composing modules may expose biological_view_descriptors with the same
codec keyword. These hooks are explicit declarations; undeclared modules remain
valid and simply add no biological declaration to the sidecar.

View selector IDs are caller-supplied labels for reconstruction and
compatibility checks. Matching labels do not establish that two selector
implementations are behaviorally equivalent.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from torch import nn

MANIFEST_FORMAT = "cellerator-biological-model-manifest"
MANIFEST_VERSION = 1


class BiologicalManifestError(ValueError):
    """A semantic manifest is malformed or does not match its model."""


_PAYLOAD_KEYS = frozenset({
    "history", "histories", "context", "contexts", "conditioning",
    "latent", "latents", "activation", "activations", "cell_state",
    "cell_states", "event_values", "observations", "measurement_values",
    "values", "sample_values", "example", "examples", "batch",
    "batch_values", "latent_activations", "history_values",
})


def _codec_fields(codec: Any) -> tuple[str, Any, Any]:
    if codec is None:
        raise TypeError("custom biological identities require an explicit identity codec")
    name = getattr(codec, "name", None)
    version = getattr(codec, "version", None)
    encode = getattr(codec, "encode", None)
    if not isinstance(name, str) or not name or not callable(encode):
        raise TypeError("identity_codec must expose a nonempty name and callable encode")
    if isinstance(version, bool) or not isinstance(version, (str, int)) or not version:
        raise TypeError("identity_codec.version must be a nonempty string or integer")
    return name, version, encode


def _json_value(value: Any, codec: Any = None) -> Any:
    """Copy a declaration into safe JSON values, with tagged identity forms."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise TypeError("biological manifest values must be finite JSON numbers")
        return numeric
    if isinstance(value, tuple):
        return {"$cellerator_type": "tuple",
                "items": [_json_value(part, codec) for part in value]}

    from cellerator._declarations import Axis, Identity
    if isinstance(value, Identity):
        return {"$cellerator_type": "identity128",
                "words": [str(value.low), str(value.high)]}
    if isinstance(value, Axis):
        return {
            "$cellerator_type": "axis",
            "domain": _json_value(value.domain, codec),
            "order": _json_value(value.order, codec),
            "geometry": _json_value(value.geometry, codec),
            "partition": _json_value(value.partition, codec),
            "extent": int(value.extent),
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("biological manifest object keys must be strings")
            if key == "$cellerator_type":
                raise TypeError("'$cellerator_type' is reserved in biological declarations")
            if key.casefold() in _PAYLOAD_KEYS:
                raise TypeError(f"biological manifest must not serialize payload field {key!r}")
            result[key] = _json_value(item, codec)
        return result
    if isinstance(value, list):
        return [_json_value(item, codec) for item in value]

    if codec is not None:
        name, version, encode = _codec_fields(codec)
        encoded = encode(value)
        if encoded is value:
            raise TypeError("identity codec must return a JSON-safe encoding, not the original object")
        return {
            "$cellerator_type": "custom_identity",
            "codec": {"name": name, "version": version},
            "value": _json_value(encoded),
        }
    raise TypeError(
        f"unsupported biological manifest value {type(value).__name__}; "
        "use a JSON-safe identity or supply identity_codec"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _call_descriptor_hook(hook: Any, identity_codec: Any, *, name: str) -> Any:
    if not callable(hook):
        raise BiologicalManifestError(f"{name} descriptor hook must be callable")
    return hook(identity_codec=identity_codec)


def _module_declarations(module: nn.Module, identity_codec: Any) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    try:
        modules = module.named_modules(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with older Torch
        modules = module.named_modules()
    for path, submodule in modules:
        hook = getattr(submodule, "cellerator_biological_descriptors", None)
        if hook is None:
            continue
        raw = _call_descriptor_hook(
            hook, identity_codec, name="cellerator_biological_descriptors"
        )
        if not isinstance(raw, Mapping):
            raise BiologicalManifestError(
                "cellerator_biological_descriptors must return a mapping"
            )
        for kind, descriptor in raw.items():
            if not isinstance(kind, str) or not kind:
                raise BiologicalManifestError("biological declaration kinds must be nonempty strings")
            key = (path, kind)
            if key in seen:
                raise BiologicalManifestError(
                    f"duplicate biological declaration {kind!r} at module path {path!r}"
                )
            seen.add(key)
            declarations.append({
                "module_path": path,
                "kind": kind,
                "descriptor": _json_value(descriptor, identity_codec),
            })
    return declarations


def _view_declarations(module: nn.Module, identity_codec: Any) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    try:
        modules = module.named_modules(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with older Torch
        modules = module.named_modules()
    for path, submodule in modules:
        hook = getattr(submodule, "biological_view_descriptors", None)
        if hook is None:
            continue
        raw = _call_descriptor_hook(hook, identity_codec, name="biological_view_descriptors")
        if not isinstance(raw, (list, tuple)):
            raise BiologicalManifestError(
                "biological_view_descriptors must return an ordered sequence"
            )
        for descriptor in raw:
            if not isinstance(descriptor, Mapping):
                raise BiologicalManifestError("each biological view descriptor must be a mapping")
            item = _json_value(descriptor, identity_codec)
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise BiologicalManifestError("each biological view needs a nonempty name")
            source_kind = item.get("source_kind")
            if source_kind in ("callable", "module"):
                source_id = item.get("source_id")
                if not isinstance(source_id, str) or not source_id:
                    raise BiologicalManifestError(
                        f"view {name!r} uses a {source_kind} selector and requires an explicit source_id"
                    )
            views.append({"module_path": path, **item})
    return views


def _explicit_relations(relations: Mapping[str, Any] | None,
                        identity_codec: Any) -> list[dict[str, Any]]:
    if relations is None:
        return []
    if not isinstance(relations, Mapping):
        raise TypeError("relations must be a mapping of stable names to descriptors")
    result: list[dict[str, Any]] = []
    for name, relation in relations.items():
        if not isinstance(name, str) or not name:
            raise TypeError("explicit relation names must be nonempty strings")
        to_manifest = getattr(relation, "to_manifest", None)
        if callable(to_manifest):
            descriptor = to_manifest(codec=identity_codec)
        else:
            descriptor = relation
        result.append({"name": name, "descriptor": _json_value(descriptor, identity_codec)})
    return result


def _weight_record(weights_path: str | os.PathLike[str] | None) -> dict[str, str] | None:
    if weights_path is None:
        return None
    path = Path(weights_path)
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise FileNotFoundError(f"cannot pair manifest with weights at {path}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError("biological manifest pairing currently supports regular weight files")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"filename": path.name, "sha256": digest.hexdigest()}


def biological_manifest(
    module: nn.Module,
    *,
    identity_codec: Any = None,
    relations: Mapping[str, Any] | None = None,
    weights_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Snapshot explicit biological declarations as a versioned JSON object.

    Raw nn.Module compositions need no biological annotations and produce empty
    declaration arrays. Runtime values, including contexts and cell histories,
    are rejected rather than captured by default.
    """
    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    manifest = {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "declarations": _module_declarations(module, identity_codec),
        "relations": _explicit_relations(relations, identity_codec),
        "views": _view_declarations(module, identity_codec),
        "weights": _weight_record(weights_path),
    }
    return json.loads(_canonical_json(manifest))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BiologicalManifestError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise BiologicalManifestError(f"non-finite JSON constant {value!r} is unsupported")


def _validated_shape(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise BiologicalManifestError("biological manifest must be a JSON object")
    try:
        copied = json.loads(_canonical_json(manifest))
    except (TypeError, ValueError) as exc:
        raise BiologicalManifestError(f"manifest is not safe JSON: {exc}") from exc
    if copied.get("format") != MANIFEST_FORMAT:
        raise BiologicalManifestError("unsupported biological manifest format")
    version = copied.get("version")
    if isinstance(version, bool) or version != MANIFEST_VERSION:
        raise BiologicalManifestError(f"unsupported biological manifest version {version!r}")
    for key in ("declarations", "relations", "views"):
        if not isinstance(copied.get(key), list):
            raise BiologicalManifestError(f"manifest field {key!r} must be an array")
    weights = copied.get("weights")
    if weights is not None:
        if (not isinstance(weights, dict) or not isinstance(weights.get("filename"), str)
                or not weights["filename"] or Path(weights["filename"]).name != weights["filename"]
                or not isinstance(weights.get("sha256"), str)
                or len(weights["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in weights["sha256"])):
            raise BiologicalManifestError("manifest weights must name a file and lowercase SHA-256 digest")
    elif "weights" not in copied:
        raise BiologicalManifestError("manifest must declare weights as a file record or null")
    return copied


def _verify_weights(manifest: Mapping[str, Any], weights_path: str | os.PathLike[str] | None) -> None:
    expected = manifest.get("weights")
    if expected is None:
        if weights_path is not None:
            raise BiologicalManifestError("manifest does not declare a paired weight file")
        return
    if weights_path is None:
        raise BiologicalManifestError(
            "weights_path is required to verify the manifest's paired weight file"
        )
    actual = _weight_record(weights_path)
    if actual != expected:
        raise BiologicalManifestError("weight file name or SHA-256 does not match biological manifest")


def validate_biological_manifest(
    module: nn.Module,
    manifest: Mapping[str, Any],
    *,
    identity_codec: Any = None,
    relations: Mapping[str, Any] | None = None,
    weights_path: str | os.PathLike[str] | None = None,
) -> None:
    """Validate declared meaning and optional weight pairing before loading.

    This function never calls load_state_dict. Call it before delegating to
    ordinary Torch or native-owner checkpoint loading.
    """
    expected = _validated_shape(manifest)
    actual = biological_manifest(
        module, identity_codec=identity_codec, relations=relations,
    )
    for key in ("declarations", "relations", "views"):
        if actual[key] != expected[key]:
            raise BiologicalManifestError(f"model biological {key} do not match manifest")
    _verify_weights(expected, weights_path)


def save_biological_manifest(
    module: nn.Module,
    path: str | os.PathLike[str],
    *,
    identity_codec: Any = None,
    relations: Mapping[str, Any] | None = None,
    weights_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Atomically write a UTF-8 JSON sidecar and return the saved declaration."""
    manifest = biological_manifest(
        module, identity_codec=identity_codec, relations=relations,
        weights_path=weights_path,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(manifest) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        raise
    return manifest


def load_biological_manifest(
    path: str | os.PathLike[str],
    *,
    weights_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Read and validate a JSON sidecar without importing or executing code."""
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = json.load(
                stream, object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
    except BiologicalManifestError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BiologicalManifestError(f"cannot read biological manifest: {exc}") from exc
    manifest = _validated_shape(raw)
    if weights_path is not None:
        _verify_weights(manifest, weights_path)
    return manifest


__all__ = [
    "MANIFEST_FORMAT", "MANIFEST_VERSION", "BiologicalManifestError",
    "biological_manifest", "validate_biological_manifest",
    "save_biological_manifest", "load_biological_manifest",
]
