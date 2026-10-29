"""Reusable immutable per-feature calibration and contextual precision profiles."""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence
from typing import Any
import hashlib
import json
import numbers

import numpy as np

from .quantization import QuantizationDiagnostics, QuantizedState, _unique_ids, quantize_state


@dataclass(frozen=True)
class QuantizationProfile:
    """Reusable immutable per-feature codebooks shared across state payloads.

    `namespace` and optional `species` delimit the biological meaning of the
    feature identities. `measurement_id` and `transform_id` identify how
    calibration values were obtained and transformed; compatibility is checked
    explicitly before encoding. Profile arrays are shared by reference from
    each encoded QuantizedState, so they are stored once across many cells.
    """

    namespace: str
    feature_ids: tuple[Any, ...]
    codebooks: tuple[np.ndarray, ...]
    bits: int = 4
    species: str | None = None
    measurement_id: str = "unspecified"
    transform_id: str = "identity"
    version: str = "1"
    codebook_dtype: Any = np.float32

    def __post_init__(self) -> None:
        if (not isinstance(self.namespace, str) or not isinstance(self.measurement_id, str)
                or not isinstance(self.transform_id, str) or not isinstance(self.version, str)
                or not self.namespace or not self.measurement_id or not self.transform_id or not self.version):
            raise ValueError("profile namespace, measurement, transform, and version must be non-empty")
        if self.species is not None and (not isinstance(self.species, str) or not self.species):
            raise ValueError("species must be None or a non-empty identifier")
        if isinstance(self.bits, bool) or self.bits not in (1, 2, 4, 8):
            raise ValueError("profile bits must be one of 1, 2, 4, or 8")
        try:
            codebook_dtype = np.dtype(self.codebook_dtype)
        except TypeError as exc:
            raise TypeError("profile codebook_dtype must be a NumPy floating dtype") from exc
        if codebook_dtype.kind != "f" or codebook_dtype.itemsize < 4:
            raise TypeError("profile codebook_dtype must be float32 or float64")
        ids = _unique_ids(self.feature_ids, len(self.feature_ids))
        if len(self.codebooks) != len(ids):
            raise ValueError("profile needs one codebook per feature identity")
        books = []
        for levels in self.codebooks:
            book = np.asarray(levels, dtype=codebook_dtype)
            if (book.ndim != 1 or not 1 <= len(book) <= (1 << self.bits)
                    or not np.isfinite(book).all() or np.any(np.diff(book) <= 0)):
                raise ValueError("profile codebooks must be finite, sorted, unique, and fit the bit width")
            book = book.copy()
            book.flags.writeable = False
            books.append(book)
        object.__setattr__(self, "feature_ids", ids)
        object.__setattr__(self, "codebooks", tuple(books))
        object.__setattr__(self, "codebook_dtype", codebook_dtype)

    @classmethod
    def from_codebooks(cls, feature_ids: Sequence[Any],
                       codebooks: Mapping[Any, Sequence[float]] | Sequence[Sequence[float]], *,
                       namespace: str = "generic", species: str | None = None,
                       measurement_id: str = "unspecified", transform_id: str = "identity",
                       bits: int = 4, version: str = "1",
                       codebook_dtype: Any = np.float32) -> "QuantizationProfile":
        ids = _unique_ids(feature_ids, len(feature_ids))
        if isinstance(codebooks, Mapping):
            if set(codebooks) != set(ids):
                raise ValueError("codebooks must cover exactly the profile feature identities")
            books = tuple(codebooks[identity] for identity in ids)
        else:
            books = tuple(codebooks)
        return cls(namespace, ids, books, bits, species, measurement_id, transform_id, version,
                   codebook_dtype)

    @classmethod
    def calibrate(cls, values: Any, feature_ids: Sequence[Any], *,
                  method: str = "quantile", levels: int = 16, bits: int = 4,
                  iterations: int = 32, namespace: str = "generic", species: str | None = None,
                  measurement_id: str = "unspecified", transform_id: str = "identity",
                  version: str = "1", codebook_dtype: Any = np.float32) -> "QuantizationProfile":
        """Derive per-feature codebooks with deterministic quantiles or 1-D Lloyd steps."""
        data = np.asarray(values, dtype=np.float64)
        ids = _unique_ids(feature_ids, data.shape[-1] if data.ndim else -1)
        if data.ndim < 2 or data.shape[-1] != len(ids) or data.size == 0 or not np.isfinite(data).all():
            raise ValueError("calibration values must be finite samples with final axis matching feature IDs")
        if bits not in (1, 2, 4, 8) or not 1 <= levels <= (1 << bits):
            raise ValueError("levels must be positive and fit the requested profile bit width")
        if method not in ("quantile", "lloyd"):
            raise ValueError("calibration method must be 'quantile' or 'lloyd'")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        samples = data.reshape(-1, len(ids))
        books = []
        for column in samples.T:
            unique = np.unique(column)
            count = min(levels, unique.size)
            if method == "quantile":
                book = np.unique(np.quantile(column, np.linspace(0.0, 1.0, count)))
            elif unique.size <= count:
                book = unique
            else:
                centers = np.quantile(column, np.linspace(0.0, 1.0, count))
                for _ in range(iterations):
                    assignment = np.abs(column[:, None] - centers[None, :]).argmin(axis=1)
                    updated = np.asarray([column[assignment == i].mean() if np.any(assignment == i)
                                          else centers[i] for i in range(count)])
                    updated.sort()
                    if np.array_equal(updated, centers):
                        break
                    centers = updated
                book = np.unique(centers)
            books.append(book)
        return cls("" if namespace is None else namespace, ids, tuple(books), bits, species,
                   measurement_id, transform_id, version, codebook_dtype)

    @property
    def codebook_bytes(self) -> int:
        return sum(book.nbytes for book in self.codebooks)

    @property
    def profile_id(self) -> str:
        digest = hashlib.sha256()
        for value in (self.namespace, self.species or "", self.measurement_id,
                      self.transform_id, str(self.bits), self.version, self.codebook_dtype.str,
                      repr(self.feature_ids)):
            digest.update(value.encode("utf-8")); digest.update(b"\0")
        for book in self.codebooks:
            digest.update(str(book.shape).encode("ascii")); digest.update(b"\0")
            digest.update(book.tobytes())
        return digest.hexdigest()

    def compatible_with(self, other: "QuantizationProfile") -> bool:
        """Whether the scientific domains and all per-identity codebooks agree."""
        if not isinstance(other, QuantizationProfile):
            return False
        if ((self.namespace, self.species, self.measurement_id, self.transform_id, self.bits,
             self.codebook_dtype)
                != (other.namespace, other.species, other.measurement_id,
                    other.transform_id, other.bits, other.codebook_dtype)
                or self.version != other.version or set(self.feature_ids) != set(other.feature_ids)):
            return False
        left = dict(zip(self.feature_ids, self.codebooks))
        right = dict(zip(other.feature_ids, other.codebooks))
        return all(np.array_equal(left[key], right[key]) for key in left)

    def select(self, feature_ids: Sequence[Any]) -> "QuantizationProfile":
        """Return an explicitly ordered view with shared codebook arrays."""
        requested = tuple(feature_ids)
        lookup = dict(zip(self.feature_ids, self.codebooks))
        if len(set(requested)) != len(requested) or not set(requested).issubset(lookup):
            raise ValueError("selected feature identities must be unique members of this profile")
        result = replace(self, feature_ids=requested,
                         codebooks=tuple(lookup[identity] for identity in requested))
        # Restore the original immutable arrays after dataclass validation copies them.
        object.__setattr__(result, "codebooks", tuple(lookup[identity] for identity in requested))
        return result

    def storage_bytes_for(self, state_shape: Sequence[int], *, state_count: int,
                          exact_override_bytes_per_state: int = 0) -> int:
        """Estimate numeric array bytes for N equally shaped payloads plus one profile.

        This includes packed state codes, exact numeric overrides specified by
        the caller, and shared per-feature codebook arrays. It excludes Python
        object headers and the encoded storage for identity strings/objects.
        """
        shape = tuple(int(dimension) for dimension in state_shape)
        if not shape or shape[-1] != len(self.feature_ids) or any(d < 0 for d in shape):
            raise ValueError("state_shape must end with the profile feature count")
        if state_count < 0 or exact_override_bytes_per_state < 0:
            raise ValueError("state counts and override bytes must be nonnegative")
        elements = int(np.prod(shape, dtype=np.int64))
        packed = (elements * self.bits + 7) // 8
        return self.codebook_bytes + state_count * (packed + exact_override_bytes_per_state)

    def save(self, path: str, *, identity_encoder: Any | None = None) -> None:
        """Persist profile metadata and codebooks without pickling Python objects.

        Built-in stable identity forms (strings, integers, None, and nested
        tuples of those forms) round-trip directly. Supply an encoder for a
        project's typed identity objects; pair it with ``load``'s decoder.
        """
        def encode(value: Any) -> Any:
            if identity_encoder is not None:
                return {"custom": identity_encoder(value)}
            if value is None or isinstance(value, str):
                return {"kind": "scalar", "value": value}
            if isinstance(value, bool):
                raise TypeError("boolean feature identities are unsupported")
            if isinstance(value, numbers.Integral):
                return {"kind": "int", "value": str(int(value))}
            if isinstance(value, tuple):
                return {"kind": "tuple", "items": [encode(item) for item in value]}
            raise TypeError("supply identity_encoder for this feature identity type")

        metadata = {
            "format": "cellerator-state-quantization-profile-v1",
            "namespace": self.namespace, "species": self.species,
            "measurement_id": self.measurement_id, "transform_id": self.transform_id,
            "version": self.version, "bits": self.bits,
            "codebook_dtype": self.codebook_dtype.str,
            "feature_ids": [encode(identity) for identity in self.feature_ids],
            "profile_id": self.profile_id,
        }
        arrays = {f"codebook_{i}": book for i, book in enumerate(self.codebooks)}
        with open(path, "wb") as stream:
            np.savez_compressed(stream, metadata=np.asarray(json.dumps(metadata)), **arrays)

    @classmethod
    def load(cls, path: str, *, identity_decoder: Any | None = None) -> "QuantizationProfile":
        """Load a saved profile without enabling NumPy object deserialization."""
        def decode(item: Any) -> Any:
            if "custom" in item:
                if identity_decoder is None:
                    raise ValueError("custom saved feature IDs require identity_decoder")
                return identity_decoder(item["custom"])
            kind = item.get("kind")
            if kind == "scalar":
                return item["value"]
            if kind == "int":
                return int(item["value"])
            if kind == "tuple":
                return tuple(decode(part) for part in item["items"])
            raise ValueError("saved profile contains an unsupported feature identity encoding")

        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("format") != "cellerator-state-quantization-profile-v1":
                raise ValueError("unsupported quantization profile file format")
            ids = tuple(decode(item) for item in metadata["feature_ids"])
            books = tuple(archive[f"codebook_{i}"].copy() for i in range(len(ids)))
        profile = cls(metadata["namespace"], ids, books, metadata["bits"],
                      metadata["species"], metadata["measurement_id"],
                      metadata["transform_id"], metadata["version"],
                      np.dtype(metadata["codebook_dtype"]))
        if profile.profile_id != metadata.get("profile_id"):
            raise ValueError("saved profile ID does not match its metadata/codebooks")
        return profile

    def encode(self, values: Any, *, feature_ids: Sequence[Any] | None = None,
               exact_mask: Any | None = None, override_dtype: Any | None = None) -> QuantizedState:
        """Encode a cell payload against shared codebooks, preserving chosen values."""
        ids = self.feature_ids if feature_ids is None else _unique_ids(feature_ids, len(feature_ids))
        if not set(ids).issubset(set(self.feature_ids)):
            raise ValueError("encoded feature identities are outside this quantization profile")
        data = np.asarray(values)
        if data.ndim < 1 or data.shape[-1] != len(ids):
            raise ValueError("values final axis must match the supplied feature order")
        lookup = dict(zip(self.feature_ids, self.codebooks))
        books = {identity: lookup[identity] for identity in ids}
        quantized = quantize_state(data, bits=self.bits, feature_ids=ids,
                                   codebooks=books, exact_mask=exact_mask,
                                   override_dtype=override_dtype)
        shared_books = tuple(lookup[identity] for identity in ids)
        state = replace(quantized, codebooks=shared_books, profile_id=self.profile_id)
        decoded = state.dequantize().astype(np.float64)
        actual_error = np.abs(decoded - data.astype(np.float64))
        if state.exact_overrides:
            for index, value in state.exact_overrides.items():
                actual_error.reshape(-1)[index] = abs(float(value) - data.astype(np.float64).reshape(-1)[index])
        diagnostics = replace(
            quantized.diagnostics, metadata_array_bytes=0,
            shared_profile_bytes=self.codebook_bytes,
            mean_absolute_error=float(actual_error.mean()) if actual_error.size else 0.0,
            root_mean_square_error=float(np.sqrt(np.mean(actual_error ** 2))) if actual_error.size else 0.0,
            maximum_absolute_error=float(actual_error.max(initial=0.0)))
        return replace(state, diagnostics=diagnostics)


__all__ = ["QuantizationProfile"]
