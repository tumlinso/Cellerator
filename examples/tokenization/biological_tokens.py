"""Synthetic encoding examples; inputs already have caller-defined meaning."""
from __future__ import annotations

import torch
from cellerator import tokenization as tk


def main() -> None:
    # The caller associates these stable identities with genes or peaks.
    feature_ids = torch.tensor([100, 200, 300], dtype=torch.int64)
    vocabulary = tk.make_vocabulary(feature_ids, torch.tensor([10, 20, 30]))

    counts = torch.tensor([[20., 100., 0.], [0., 4., 8.]], dtype=torch.float64)
    medians = torch.tensor([2., 200., 4.], dtype=torch.float64)
    ranked = tk.geneformer_style(
        counts, counts.sum(1), medians, feature_ids, vocabulary, max_per_row=2
    )
    print("ranked feature tokens:", ranked.token_ids.tolist(), ranked.offsets.tolist())

    # These occurrences have already been inferred/defined by the model owner.
    events = tk.event_stream(
        torch.tensor([100, 200, 100, 300]), torch.tensor([0, 3, 4]), vocabulary,
        attributes={"time": torch.tensor([0.1, 0.2, 0.8, 0.3])},
    )
    history = tk.SequenceBuffer.create(
        [8, 4], attribute_examples={"time": torch.empty(0)}
    )
    history.append_checked(torch.tensor([0, 1]), events)

    # Another caller-supplied occurrence enters the ordinary encoding path.
    injected = tk.event_stream(
        torch.tensor([200]), torch.tensor([0, 1]), vocabulary,
        attributes={"time": torch.tensor([1.0])},
    )
    history.append_checked(torch.tensor([0]), injected)
    packed = history.compact()
    print("event history:", packed.token_ids.tolist(), packed.offsets.tolist())


if __name__ == "__main__":
    main()
