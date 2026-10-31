"""Optional Torch model modules with stock ``torch.nn`` ownership semantics."""
from __future__ import annotations

from importlib import import_module

_FOUNDATION = {
    "FeatureEncoder": ".torch.features",
    "RelationAttention": ".torch.relation_attention",
    "PreparedRelation": ".torch.relations",
    "InteractionPrograms": ".torch.programs",
    "FeatureReadout": ".torch.readouts",
    "ModelWithViews": ".torch.views",
    "biological_manifest": ".torch.manifest",
    "validate_biological_manifest": ".torch.manifest",
    "save_biological_manifest": ".torch.manifest",
    "load_biological_manifest": ".torch.manifest",
    "align_prediction_targets": ".torch.supervision",
    "categorical_target_columns": ".torch.supervision",
    "reduce_prediction_loss": ".torch.supervision",
    "grouped_linear": ".torch.grouped_linear",
    "GroupedLinear": ".torch.grouped_linear",
}


def __getattr__(name: str):
    if name in _FOUNDATION:
        return getattr(import_module(_FOUNDATION[name], "cellerator"), name)
    if name == "Product2Module":
        return getattr(import_module(".torch.nn", "cellerator"), name)
    if name in {"MechanismModule", "SharedSupportRelation", "BiologicalTensor",
                "RankedGeneSlice", "KaryoplastEventSlice"}:
        return getattr(import_module(".torch.nn", "cellerator"), name)
    raise AttributeError(name)


__all__ = [
    *_FOUNDATION,
    "Product2Module", "MechanismModule", "SharedSupportRelation",
    "BiologicalTensor", "RankedGeneSlice", "KaryoplastEventSlice",
]
