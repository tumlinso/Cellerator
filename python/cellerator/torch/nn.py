"""Cellerator model modules built on Torch's Module and optimizer contracts."""
from importlib import import_module

_EXPORTS = {
    "Product2Module": ".product2",
    "BiologicalTensor": ".mechanism",
    "MechanismModule": ".mechanism",
    "SharedSupportRelation": ".biology",
    "RankedGeneSlice": ".routing",
    "KaryoplastEventSlice": ".routing",
    "FeatureEncoder": ".features",
    "RelationAttention": ".relation_attention",
    "PreparedRelation": ".relations",
    "InteractionPrograms": ".programs",
    "FeatureReadout": ".readouts",
    "ModelWithViews": ".views",
    "biological_manifest": ".manifest",
    "validate_biological_manifest": ".manifest",
    "save_biological_manifest": ".manifest",
    "load_biological_manifest": ".manifest",
    "align_prediction_targets": ".supervision",
    "categorical_target_columns": ".supervision",
    "reduce_prediction_loss": ".supervision",
    "grouped_linear": ".grouped_linear",
    "GroupedLinear": ".grouped_linear",
}


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name], __package__), name)
    raise AttributeError(name)


__all__ = list(_EXPORTS)
