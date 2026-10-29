# CE-SIG1-A01: Add continual support adaptation and logical checkpointing

Maintain a compiled stable base and separate sparse support delta. Freeze support, activity decisions and parameter generations for each forward/backward window; tapes retain those revisions. Publish revisions only after dependent tapes retire. Route new/revoked support through the delta, enforce disjoint logical ownership, and merge stable support into a replacement projection at publication boundaries. Invalidate caches affected by parameter updates. Checkpoint parameters, supported Adam state, latent/event/window state, support identities, conditions, exploration RNG, and compilation metadata; restore logical state before rebinding/rebuilding physical projections.

Acceptance: live-tape edits are rejected, optimizer parameter identity is retained, support promotion/revocation/repacking preserve unchanged logical outputs and declared gradients, and checkpoint continuation matches an uninterrupted reference.
