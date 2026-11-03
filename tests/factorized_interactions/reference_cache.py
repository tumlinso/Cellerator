"""Packet-local cache for the independent factorized-interaction oracle.

Cache entries are bound to exact CFINT001 bytes, the current oracle and
workload-generator source, and the workload manifest. The archive contains
only numeric NumPy arrays and JSON metadata; loading never enables pickle.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from case_io import CODE_TYPE, HEADER, MAGIC_INPUT
from reference import Evaluation, Problem, VJP, VJPScale, evaluate


SCHEMA = "cellerator.factorized_interactions.reference_cache.v1"
_ARRAY_NAMES = (
    "y_float64", "y", "sum_abs_y_contributions",
    "vjp_dx", "vjp_dz", "vjp_dalpha", "vjp_dbias", "vjp_dgamma",
    "scale_dx", "scale_dz", "scale_dalpha", "scale_dbias", "scale_dgamma",
)
_VJP_NAMES = ("dx", "dz", "dalpha", "dbias", "dgamma")
_INPUT_NAMES = ("x", "z", "alpha", "bias", "gamma", "dy")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {
        "reference_sha256": _sha256_file(here / "reference.py"),
        "workloads_sha256": _sha256_file(here / "workloads.py"),
        "workload_manifest_sha256": _sha256_file(
            here.parents[1] / "bench" / "factorized_interactions" / "workloads.json"),
    }


def _input_identity(case_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Hash the exact ordered input stream and check its supplied identity."""
    if not isinstance(case_identity, Mapping):
        raise TypeError("case_identity must be the identity returned by case_io")
    parts = case_identity.get("parts")
    if parts is None:
        path = case_identity.get("path")
        if path is None:
            raise ValueError("case identity must include path or parts")
        parts = (path,)
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)) or not parts:
        raise ValueError("case identity parts must be a nonempty sequence")

    digest = hashlib.sha256()
    byte_count = 0
    header = bytearray()
    for raw_path in parts:
        path = Path(raw_path)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                byte_count += len(block)
                if len(header) < HEADER.size:
                    header.extend(block[:HEADER.size - len(header)])
    actual_sha256 = digest.hexdigest()
    actual = {"sha256": actual_sha256, "bytes": byte_count}
    if case_identity.get("sha256") != actual_sha256 or \
            case_identity.get("bytes") != byte_count:
        raise ValueError("CFINT001 input bytes differ from their supplied identity")
    if len(header) < HEADER.size:
        raise ValueError("CFINT001 input lacks a complete header")
    fields = HEADER.unpack(header)
    if fields[0] != MAGIC_INPUT:
        raise ValueError("case identity does not contain CFINT001 bytes")
    output_code = int(fields[-1])
    if output_code not in CODE_TYPE:
        raise ValueError("CFINT001 input declares an unsupported output dtype")
    actual["output_dtype"] = CODE_TYPE[output_code].str
    return actual


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _problem_identity(problem: Problem, dy: np.ndarray | None) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {
        name: np.asarray(getattr(problem, name))
        for name in ("x", "z", "alpha", "bias", "gamma")
    }
    if dy is not None:
        arrays["dy"] = np.asarray(dy)
    identities = {
        name: {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _array_digest(value),
        }
        for name, value in arrays.items()
    }
    structure = [
        [int(mechanism.target),
         [int(value) for value in mechanism.participants],
         [int(value) for value in mechanism.context_ids],
         [int(value) for value in mechanism.gamma_ids],
         int(mechanism.bias_id), int(mechanism.alpha_id)]
        for mechanism in problem.mechanisms
    ]
    structure_bytes = json.dumps(
        [int(problem.target_count), structure], separators=(",", ":")).encode("ascii")
    return {
        "arrays": identities,
        "structure_sha256": hashlib.sha256(structure_bytes).hexdigest(),
        "target_count": int(problem.target_count),
        "mechanism_count": len(problem.mechanisms),
    }


def _flatten(evaluation: Evaluation) -> dict[str, np.ndarray]:
    arrays = {
        "y_float64": np.asarray(evaluation.y_float64),
        "y": np.asarray(evaluation.y),
        "sum_abs_y_contributions": np.asarray(evaluation.sum_abs_y_contributions),
    }
    if evaluation.vjp is not None:
        if evaluation.vjp_scale is None:
            raise ValueError("reference evaluation has VJPs without contribution scales")
        arrays.update({f"vjp_{name}": np.asarray(getattr(evaluation.vjp, name))
                       for name in _VJP_NAMES})
        arrays.update({f"scale_{name}": np.asarray(getattr(evaluation.vjp_scale, name))
                       for name in _VJP_NAMES})
    return arrays


def _expected_names(with_gradients: bool) -> tuple[str, ...]:
    return _ARRAY_NAMES if with_gradients else _ARRAY_NAMES[:3]


def _cache_root(packet_output_dir: Path) -> Path:
    requested = Path(packet_output_dir).expanduser().absolute()
    if any(component.is_symlink() for component in (requested, *requested.parents)):
        raise ValueError("packet output directory path cannot contain symbolic links")
    requested.mkdir(parents=True, exist_ok=True)
    if requested.is_symlink():
        raise ValueError("packet output directory cannot be a symbolic link")
    root = requested.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"packet output directory is not a directory: {root}")
    return root


def cache_path(input_sha256: str, output_dtype: np.dtype | type,
               with_gradients: bool, packet_output_dir: Path) -> Path:
    """Return the deterministic cache path inside the supplied packet output dir."""
    if len(input_sha256) != 64 or any(c not in "0123456789abcdef" for c in input_sha256):
        raise ValueError("input_sha256 must be a lowercase SHA256 digest")
    dtype = np.dtype(output_dtype)
    dtype_tag = hashlib.sha256(dtype.str.encode("ascii")).hexdigest()[:12]
    gradient_tag = "grad" if with_gradients else "forward"
    root = _cache_root(Path(packet_output_dir))
    return root / f".factorized-reference-{input_sha256}-{dtype_tag}-{gradient_tag}.npz"


def _request_manifest(input_identity: dict[str, Any],
                      problem_identity: dict[str, Any],
                      output_dtype: np.dtype,
                      with_gradients: bool) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "input": input_identity,
        **_source_hashes(),
        "output_dtype": output_dtype.str,
        "gradients": with_gradients,
        "problem_arrays": problem_identity,
    }


def _manifest_for(input_identity: dict[str, Any], problem_identity: dict[str, Any],
                  output_dtype: np.dtype, with_gradients: bool,
                  arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    entries = {}
    for name, value in arrays.items():
        entries[name] = {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256": _array_digest(value),
        }
    return {**_request_manifest(input_identity, problem_identity, output_dtype,
                                with_gradients),
            "arrays": entries}


def _write_cache(path: Path, manifest: dict[str, Any],
                 arrays: dict[str, np.ndarray]) -> None:
    metadata = np.frombuffer(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".factorized-reference-tmp-",
                                         suffix=".npz", dir=path.parent,
                                         delete=False) as stream:
            temporary_name = stream.name
            np.savez_compressed(stream, __manifest__=metadata, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _load_cache(path: Path, expected_identity: dict[str, Any],
                expected_shapes: dict[str, tuple[int, ...]],
                expected_output_dtype: np.dtype,
                with_gradients: bool) -> Evaluation:
    if path.is_symlink():
        raise ValueError("reference cache path must not be a symbolic link")
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            if "__manifest__" not in keys:
                raise ValueError("reference cache lacks its manifest")
            raw_manifest = np.asarray(archive["__manifest__"])
            if raw_manifest.dtype != np.dtype(np.uint8) or raw_manifest.ndim != 1:
                raise ValueError("reference cache manifest has an invalid representation")
            manifest = json.loads(raw_manifest.tobytes().decode("utf-8"))
            expected_names = _expected_names(with_gradients)
            if keys != set(expected_names) | {"__manifest__"}:
                raise ValueError("reference cache array set differs from the request")
            if not isinstance(manifest, dict) or \
                    set(manifest) != set(expected_identity) | {"arrays"} or \
                    any(manifest.get(name) != value
                        for name, value in expected_identity.items()):
                raise ValueError("reference cache identity differs from the current request")
            if set(manifest.get("arrays", {})) != set(expected_names):
                raise ValueError("reference cache manifest array set is invalid")

            arrays: dict[str, np.ndarray] = {}
            for name in expected_names:
                value = np.asarray(archive[name]).copy()
                record = manifest["arrays"].get(name)
                if not isinstance(record, dict) or \
                        set(record) != {"shape", "dtype", "sha256"}:
                    raise ValueError(f"reference cache metadata for {name} is invalid")
                if value.shape != expected_shapes[name] or \
                        record.get("shape") != list(expected_shapes[name]):
                    raise ValueError(f"reference cache shape mismatch for {name}")
                if record.get("dtype") != value.dtype.str:
                    raise ValueError(f"reference cache dtype metadata mismatch for {name}")
                expected_dtype = expected_output_dtype if name == "y" else np.dtype(np.float64)
                if value.dtype != expected_dtype:
                    raise ValueError(f"reference cache dtype mismatch for {name}")
                if record.get("sha256") != _array_digest(value):
                    raise ValueError(f"reference cache content hash mismatch for {name}")
                arrays[name] = value
    except ValueError:
        raise
    except Exception as error:
        # Normalize damaged ZIP/NPY payloads into a clear cache-validation error.
        raise ValueError("reference cache archive is damaged or unreadable") from error

    if not with_gradients:
        return Evaluation(arrays["y_float64"], arrays["y"],
                          arrays["sum_abs_y_contributions"])
    vjp = VJP(*(arrays[f"vjp_{name}"] for name in _VJP_NAMES))
    scale = VJPScale(*(arrays[f"scale_{name}"] for name in _VJP_NAMES))
    return Evaluation(arrays["y_float64"], arrays["y"],
                      arrays["sum_abs_y_contributions"], vjp, scale)


def evaluate_cached(problem: Problem, dy: np.ndarray | None,
                     output_dtype: np.dtype | type,
                     case_identity: Mapping[str, Any],
                     packet_output_dir: Path) -> Evaluation:
    """Return the scalar oracle result, using a verified packet-local cache.

    ``case_identity`` is returned by ``case_io.write_case`` or
    ``case_io.write_sharded_case``.  Its exact ordered bytes are re-hashed on
    every call, while cache files are stored in ``packet_output_dir`` so
    candidate-specific case directories can share a suite-level cache.
    Passing ``dy=None`` requests a forward-only entry; otherwise all five VJPs
    and contribution-scale arrays are retained.
    """
    dtype = np.dtype(output_dtype)
    if dtype.kind != "f":
        raise TypeError("output_dtype must be floating")
    input_identity = _input_identity(case_identity)
    if input_identity["output_dtype"] != dtype.str:
        raise ValueError("requested output dtype differs from the CFINT001 header")
    with_gradients = dy is not None
    problem_identity = _problem_identity(problem, dy)
    expected_identity = _request_manifest(input_identity, problem_identity, dtype,
                                          with_gradients)
    path = cache_path(input_identity["sha256"], dtype, with_gradients,
                      packet_output_dir)
    expected_shapes = {
        "y_float64": (problem.target_count, problem.x.shape[1], problem.x.shape[2]),
        "y": (problem.target_count, problem.x.shape[1], problem.x.shape[2]),
        "sum_abs_y_contributions": (
            problem.target_count, problem.x.shape[1], problem.x.shape[2]),
    }
    if with_gradients:
        expected_shapes.update({
            f"vjp_{name}": tuple(np.asarray(getattr(problem, input_name)).shape)
            for name, input_name in zip(_VJP_NAMES, _INPUT_NAMES[:5])
        })
        expected_shapes.update({
            f"scale_{name}": tuple(np.asarray(getattr(problem, input_name)).shape)
            for name, input_name in zip(_VJP_NAMES, _INPUT_NAMES[:5])
        })

    if path.exists() or path.is_symlink():
        return _load_cache(path, expected_identity, expected_shapes, dtype,
                           with_gradients)

    evaluation = evaluate(problem, dy=dy, output_dtype=dtype)
    arrays = _flatten(evaluation)
    if tuple(arrays) != _expected_names(with_gradients):
        raise ValueError("scalar reference returned an unexpected array set")
    manifest = _manifest_for(input_identity, problem_identity, dtype,
                             with_gradients, arrays)
    _write_cache(path, manifest, arrays)
    # Re-read the completed archive through the same validation path used for
    # future cache hits so writes cannot silently establish malformed entries.
    return _load_cache(path, _request_manifest(input_identity, problem_identity,
                                               dtype, with_gradients),
                       expected_shapes, dtype, with_gradients)

