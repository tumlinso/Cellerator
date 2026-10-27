# CE-SIG1-P01: Integrate packed forward and backward execution

Extend existing CE-GEO relation cover, value maps, and width-64 physical consumer as the first backend. Bind direct logical-to-physical maps during preparation; support per-cell gates and batched forward, input transpose gradients, logical-order relation-value gradients, gate/scorer gradients including derivative-only candidates, and recurrent latent backward through saved events. Retain sparse SIMT and account separately for the existing MMA residual region and adaptive support delta.

Acceptance: compiled output and declared surrogate gradients match an equivalent Torch reference for identical candidate sets and pruning decisions. Preparation, execution streams, temporary storage, and complete costs are explicit. Irregular activity retains an executable SIMT route.
