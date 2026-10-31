"""Feature-identity-aware encoders for biological model inputs.

The encoder deliberately does not own a gene-name registry or infer a feature
ordering. Callers map their declared identities to integer IDs and pass the
aligned values and attributes they want the model to use.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from cellerator.biology import FeatureBinding, PreparedAlignment


def _flatten_feature_ids(values):
    """Return a rectangular Python identity tree's shape and leaf order."""
    if hasattr(values, "tolist") and not isinstance(values, (tuple, list, str, bytes)):
        values = values.tolist()
    if isinstance(values, (tuple, list)):
        if not values:
            return (0,), []
        parts = [_flatten_feature_ids(value) for value in values]
        child_shape = parts[0][0]
        if any(shape != child_shape for shape, _ in parts[1:]):
            raise ValueError("named feature IDs must form a rectangular sequence")
        return (len(values),) + child_shape, [item for _, items in parts for item in items]
    return (), [values]


class FeatureEncoder(nn.Module):
    """Encode registered feature IDs together with optional aligned data.

    ``feature_ids`` may have any leading shape (commonly ``[batch, features]``
    or a packed ``[occurrences]`` vector). ``values`` aligns with those axes;
    for a scalar measurement per feature it may either match ``feature_ids`` or
    append a singleton value dimension. Multi-channel measurements use a final
    dimension of ``value_size``. Attributes, when configured, use the same
    leading axes and a final dimension of ``attribute_size``.

    Missingness is explicit. ``observed`` is a Boolean mask over feature axes,
    and hidden values are replaced with zero before any learned projection.
    A learned status embedding distinguishes an observed numeric zero from a
    missing measurement. No biological meaning is inferred from IDs, values,
    or attributes.
    """

    def __init__(
        self,
        vocabulary_size: int,
        width: int,
        *,
        value_size: int | None = 1,
        attribute_size: int | None = None,
        binding: FeatureBinding | None = None,
    ) -> None:
        super().__init__()
        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        if value_size is not None and value_size <= 0:
            raise ValueError("value_size must be positive or None")
        if attribute_size is not None and attribute_size <= 0:
            raise ValueError("attribute_size must be positive or None")
        if binding is not None:
            if not isinstance(binding, FeatureBinding):
                raise TypeError("binding must be a FeatureBinding")
            binding.validate_model_capacity(vocabulary_size)
        self.vocabulary_size = vocabulary_size
        self.width = width
        self.value_size = value_size
        self.attribute_size = attribute_size
        self.binding = binding
        self.feature_embedding = nn.Embedding(vocabulary_size, width)
        self.status_embedding = nn.Embedding(2, width)
        self.value_projection = (
            nn.Linear(value_size, width, bias=False)
            if value_size is not None else None
        )
        self.attribute_projection = (
            nn.Linear(attribute_size, width, bias=False)
            if attribute_size is not None else None
        )

    def _features(self, feature_ids: Tensor) -> None:
        if not isinstance(feature_ids, Tensor):
            raise TypeError("feature_ids must be a Torch Tensor")
        if feature_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("feature_ids must have dtype int32 or int64")
        if feature_ids.ndim < 1:
            raise ValueError("feature_ids must have at least one dimension")
        if feature_ids.device != self.feature_embedding.weight.device:
            raise ValueError("feature_ids and module parameters must share a device")

    def forward(
        self,
        feature_ids: Tensor,
        values: Tensor | None = None,
        *,
        observed: Tensor | None = None,
        attributes: Tensor | None = None,
    ) -> Tensor:
        """Return an ordinary Torch tensor with shape ``[*feature_axes, width]``."""
        self._features(feature_ids)
        encoded = self.feature_embedding(feature_ids.to(dtype=torch.long))

        if observed is None:
            observed = (
                torch.ones_like(feature_ids, dtype=torch.bool)
                if values is not None else
                torch.zeros_like(feature_ids, dtype=torch.bool)
            )
        elif not isinstance(observed, Tensor) or observed.dtype != torch.bool:
            raise TypeError("observed must be a Boolean Torch Tensor")
        elif observed.shape != feature_ids.shape:
            raise ValueError("observed must have the same shape as feature_ids")
        elif observed.device != feature_ids.device:
            raise ValueError("observed and feature_ids must share a device")

        status_ids = observed.to(dtype=torch.long)
        encoded = encoded + self.status_embedding(status_ids)

        if values is not None:
            if self.value_projection is None:
                raise ValueError("values were supplied but value_size=None")
            if not isinstance(values, Tensor):
                raise TypeError("values must be a Torch Tensor")
            if values.device != feature_ids.device:
                raise ValueError("values and feature_ids must share a device")
            if not values.is_floating_point():
                raise TypeError("values must have a floating-point dtype")
            if self.value_projection.weight.dtype != values.dtype:
                raise TypeError("values and module parameters must share a dtype")

            value_shape = tuple(feature_ids.shape)
            if self.value_size == 1 and tuple(values.shape) == value_shape:
                values = values.unsqueeze(-1)
            expected_shape = value_shape + (self.value_size,)
            if tuple(values.shape) != expected_shape:
                raise ValueError(
                    f"values must have shape {value_shape} or {expected_shape}"
                )
            # `where` blocks NaN/Inf values at masked positions before the
            # projection, so hidden targets cannot leak through arithmetic.
            visible_values = torch.where(
                observed.unsqueeze(-1), values, torch.zeros_like(values)
            )
            encoded = encoded + self.value_projection(visible_values)
        if attributes is not None:
            if self.attribute_projection is None:
                raise ValueError("attributes were supplied but attribute_size=None")
            if not isinstance(attributes, Tensor):
                raise TypeError("attributes must be a Torch Tensor")
            if tuple(attributes.shape) != tuple(feature_ids.shape) + (self.attribute_size,):
                raise ValueError(
                    "attributes must align with feature_ids and end in attribute_size"
                )
            if attributes.device != feature_ids.device:
                raise ValueError("attributes and feature_ids must share a device")
            if attributes.dtype != self.attribute_projection.weight.dtype:
                raise TypeError("attributes and module parameters must share a dtype")
            encoded = encoded + self.attribute_projection(attributes)
        return encoded

    def encode_bound(
        self,
        feature_ids,
        values: Tensor | None = None,
        *,
        observed: Tensor | None = None,
        attributes: Tensor | None = None,
    ) -> Tensor:
        """Encode named biological identities through the registered binding.

        Raw integer-ID calls continue to use :meth:`forward`. This method maps
        canonical IDs or declared aliases to model/embedding rows at the data
        boundary. Passing a ``PreparedAlignment`` reuses an existing mapping;
        sequence inputs may contain repeated occurrences and retain their
        order. Unknown identities selected by a ``drop`` policy are rejected
        here because silently shortening a dense measurement axis would detach
        it from its values; callers can explicitly gather with the alignment.
        """
        if self.binding is None:
            raise RuntimeError("encode_bound requires FeatureEncoder(binding=...)")
        if isinstance(feature_ids, PreparedAlignment):
            alignment = feature_ids
            self.binding.validate_alignment(alignment)
            shape = (len(alignment.model_indices),)
            indices = alignment.model_indices
        else:
            if isinstance(feature_ids, Tensor):
                raise TypeError(
                    "encode_bound accepts host biological IDs; use forward for integer model-ID tensors"
                )
            shape, leaves = _flatten_feature_ids(feature_ids)
            if not shape:
                raise ValueError("named feature IDs must have at least one feature axis")
            alignment = self.binding.map_occurrences(leaves)
            if alignment.dropped_ids:
                raise ValueError(
                    "unknown features were dropped; gather aligned values explicitly before encoding"
                )
            indices = alignment.model_indices
        if len(indices) != _product(shape):
            raise ValueError("prepared model indices do not match the named feature shape")
        feature_tensor = torch.tensor(
            indices, dtype=torch.long, device=self.feature_embedding.weight.device
        ).reshape(shape)
        return self.forward(feature_tensor, values, observed=observed, attributes=attributes)

    def cellerator_biological_descriptors(self, *, identity_codec=None):
        """Expose an explicit manifest descriptor when this encoder is bound."""
        if self.binding is None:
            return {}
        return {"feature_binding": self.binding.to_manifest(codec=identity_codec)}


def _product(shape):
    result = 1
    for extent in shape:
        result *= extent
    return result


__all__ = ["FeatureEncoder"]
