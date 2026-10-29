"""CPU-side intake and exact sparse retrieval for supplied tokenized data.

This package assigns no biological meaning to IDs, ranks, events, predicates,
or targets. It validates and indexes caller-provided records, then writes a
hash-checked raw bundle for existing sparse-support consumers.
"""

from .inputs import (
    CapacityExceeded,
    CapacityLimits,
    EvaluationBundle,
    prepare_events,
    prepare_ranked,
    load_raw_bundle,
    write_raw_bundle,
)
from .catalog import mechanism_catalog

__all__ = [
    "CapacityExceeded", "CapacityLimits", "EvaluationBundle",
    "prepare_events", "prepare_ranked", "load_raw_bundle",
    "write_raw_bundle", "mechanism_catalog",
]
