"""Experiment with alternative daughter-state rules without fixing biology."""
import numpy as np

from cellerator.experimental.inheritance import (
    BinomialPartition,
    DivisionContext,
    FieldwiseInheritance,
    FractionPartition,
    ResetField,
    divide_cell,
)
from cellerator.modeling import Cell, Model


def identity(state, context):
    return state


# These names and rules are a model's choices; Cellerator treats them as keys.
parent = Cell(
    Model(identity, parameters={"shared": "model-owned"}),
    {
        "particle_count": 19,
        "continuous_pool": 8.0,
        "accessible": True,
        "regulatory_memory": ["input-A", "activation-B"],
    },
    id="parent",
)

inheritance = FieldwiseInheritance({
    "particle_count": BinomialPartition(fraction=0.35),
    "continuous_pool": FractionPartition(fraction=0.7),
    "accessible": ResetField(lambda old, division, daughter: daughter == 0),
    # Unspecified regulatory_memory is copied into independently owned histories.
})

left, right = divide_cell(
    parent,
    inheritance,
    daughter_ids=("daughter-A", "daughter-B"),
    rng=np.random.default_rng(41),
)

assert left.model is right.model is parent.model
assert left.state["particle_count"] + right.state["particle_count"] == 19
assert left.state["continuous_pool"] + right.state["continuous_pool"] == 8.0
assert left.state["regulatory_memory"] is not right.state["regulatory_memory"]
print(left.id, left.state)
print(right.id, right.state)
