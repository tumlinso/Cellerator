# CE-SIG1-N01: Add native latent-state operations and uncompiled application slices

Implement indexed latent gather, the shared-parameter transition `z' = tanh(Az + Bu + b)`, and its VJP under the existing operation layer. Build the Karyoplast-style positive-event slice with identity-preserving occurrences, reverse mechanism postings, optional timestamps and caller-defined candidate windows. Build the ranked-gene slice over existing ranked-token batches, preserving gene identity/rank without temporal semantics. Provide the two synthetic trainable fixtures and validator-only hidden support.

Acceptance: native Torch-free forward/reference behavior; optional Torch binding exposes the same canonical parameter owners; both slices train uncompiled, including overlapping latent mechanisms, repeated events, multi-participant conditions, rank-aware and rank-independent structure. No fixed chromatin state is assumed.
