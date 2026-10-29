# Scaled shared-condition probe

This isolated experiment uses the frozen scaled-fixture v2 small and event
bundles plus the v3 ranked bundle. The v3 ranked inputs separate the 512-rank
presence view from the generic observed-predicate view. The probe uses rank
presence as the local gate state, but deliberately treats it as a set: rank
order is outside this condition-DAG experiment. The v2 event and small cases
use their generic observed-predicate state. Input manifests and selected-array
hashes are retained in `evidence/input/`.

The probe compares the public native full/dirty condition-bitmap APIs with
binding search and a prepared direct edge-to-root map, plus a compiled
binary-root cache with predicate-to-root and root-to-edge postings. The
candidate compiles the two all-of predicate participants from the fixture's
condition arrays; it does not use gene or module labels in its kernel logic.

Workloads are the small control `(E=96, B=8)`, large ranked `(E=262,144,
B=32)`, and large event `(E=1,280,000, B=8)` fixtures. Each carries exactly
two predicate participants per edge. For each shape, the harness checks
initial full bitmap construction, local-gene updates, and shared-latent toggles
against an independent CPU oracle. The event and ranked fixtures provide
distinct local-change and cohort-synchronized latent-change fanout patterns.

The candidate's preparation includes root-pair compilation, predicate-to-root
postings, root-to-edge postings, device copies, and root-cache allocation.
Steady full timing includes root-cache evaluation and bitmap materialization.
Steady dirty timing includes applying the state changes and updating affected
roots and edge bits. The native dirty timing includes its GPU worklist build
and the public dirty updater. Preparation, worklist construction, bitmap
output, complete path, peak working storage, and break-even reuse are reported
separately. All paths use identical resident states and updates.

Inputs and source identities, independent-oracle results, exact scheduler
receipts, CUDA-event samples, host-wall samples, and sanitizer output belong in
`evidence/`. Build in a probe-owned directory; this is a design probe only and
does not modify production source or a shared build target.
