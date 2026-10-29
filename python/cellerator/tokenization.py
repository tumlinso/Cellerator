"""ATen backed tokenization operations and sequence representations.

This submodule is built only when Cellerator's optional tokenization component
is enabled. Its tensors are ordinary PyTorch tensors and remain owned by ATen.
"""

# Register PyTorch's Python Tensor caster before importing the native module.
import torch as _torch

from ._tokenization import (
    AppendResult,
    FeatureStatistics,
    EventStreamGeneTokenizer,
    GeneAnnotationFormat,
    GeneAnnotationOptions,
    GeneRecord,
    GeneTokenizer,
    GeneVocabulary,
    GeneVocabularyMetadata,
    PaddedBatch,
    RankedFeatureOptions,
    RankedGeneTokenizer,
    RankedSelection,
    SequenceBuffer,
    TokenBatch,
    UnknownIdentityPolicy,
    Vocabulary,
    add_boundaries,
    event_stream,
    event_stream_into,
    geneformer_style,
    make_vocabulary,
    map_identities,
    nonzero_feature_median,
    normalize_features,
    pack_padded,
    pad_sequences,
    ranked_feature,
    ranked_feature_into,
    select_ranked_features,
)

__all__ = [
    "AppendResult",
    "FeatureStatistics",
    "EventStreamGeneTokenizer",
    "GeneAnnotationFormat",
    "GeneAnnotationOptions",
    "GeneRecord",
    "GeneTokenizer",
    "GeneVocabulary",
    "GeneVocabularyMetadata",
    "PaddedBatch",
    "RankedFeatureOptions",
    "RankedGeneTokenizer",
    "RankedSelection",
    "SequenceBuffer",
    "TokenBatch",
    "UnknownIdentityPolicy",
    "Vocabulary",
    "add_boundaries",
    "event_stream",
    "event_stream_into",
    "geneformer_style",
    "make_vocabulary",
    "map_identities",
    "nonzero_feature_median",
    "normalize_features",
    "pack_padded",
    "pad_sequences",
    "ranked_feature",
    "ranked_feature_into",
    "select_ranked_features",
]
