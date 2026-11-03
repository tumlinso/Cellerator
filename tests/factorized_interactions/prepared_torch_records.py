"""Pure host conversion from oracle records to the prepared Torch API."""

from __future__ import annotations


def prepared_factorized_records(problem) -> list[dict[str, object]]:
    """Build adapter records with stable, nonzero mechanism identities.

    Participant, target, and parameter-bank indices remain zero-based logical
    coordinates.  Only the stable mechanism identity is shifted because the
    native preparation contract reserves zero as invalid.
    """
    return [
        {
            "logical_id": ordinal + 1,
            "inputs": list(mechanism.participants),
            "target": mechanism.target,
            "alpha": mechanism.alpha_id,
            "bias": mechanism.bias_id,
            "terms": list(zip(mechanism.context_ids, mechanism.gamma_ids)),
        }
        for ordinal, mechanism in enumerate(problem.mechanisms)
    ]
