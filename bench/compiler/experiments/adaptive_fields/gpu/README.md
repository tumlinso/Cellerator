# Adaptive fields GPU probes

This directory contains isolated GPU harnesses for E5, E6's stream/lifetime
portion, and E7. These probes consume the existing width-64 native sparse
support owner and CE-GEO/SIMT implementations. They do not change production
APIs, kernels, defaults, or application algorithms.

Current GPU decisions, exact evidence links, and incomplete E7 coverage are
recorded in [REPORT.md](REPORT.md). The follow-up horizon and adaptive-trace
commands below describe unexecuted source only; they are not GPU results.

## Current execution boundary

All files here are preflight-only until the root task owner submits an
exclusive, UUID-bound Device Scheduler job and grants this branch its execution
window. Do not run `torch.cuda`, `nvidia-smi`, or any GPU binary from an
interactive shell. CPU-only manifest and syntax checks are safe.

The current frozen environment discovered for this branch is:

| Item | Value |
|---|---|
| Python | `/home/tumlinson/Software/venvs/cellerator-ml2-py313-cu126/bin/python` |
| Frozen package | `/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v14-optional-policies/package` |
| Identity manifest | `/home/tumlinson/Cellerator/build-sparse-support/evidence/torch-v14-optional-policies/identity.json` |
| Scaled fixture root | `/tmp/cellerator_scaled_sparse_fixtures_887437fc` |
| CUDA toolkit | `/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9` |
| Intended device | `GPU-d745da9c-7649-8334-41f1-483afa6f3206` (parent-owned admission) |

The v13 owner-validation snapshot is stale for the currently required
`relation_input` and `workset_union` policies and must not be used. The v14
optional-policies snapshot's `sparse_support.py` hash matches the current
checkout and its extension hash matches v13. A runner must still revalidate
the source/package/fixture hashes before computation and record the
Device Scheduler job and attempt IDs. Scheduler admission, quiescence, KDA
records, receipts, and cleanup are owned by the root task.

## E5: field-visible sparse training

The direct baseline is the existing `training_step` path in
`bench/sparse_support/packed_training_benchmark.py`: candidate scoring, STE
gates, shared edge values, native relation, loss, native VJP, then guarded Adam
through the canonical `parameters_flat` owner. The experiment-local field plan
may expose this existing connected sequence as named stages, but must not
claim that stage labels make it compiler-lowered. Until the CPU semantic
experiment establishes a shared operation contract, this branch records the
field plan as an explicit adapter over the same native calls.

`e5_stage_adapter.py` is the experiment-local event composition runner. It
spells out provider → latent transition → context → gate scoring → workset
selection → edge values → relation as dependency-checked operations and
compares that path with the current `KaryoplastEventSlice.forward`. It retains
the exact primitive inputs/outputs, worksets, gates, owner gradients, and
fixture tensors in checksummed NPZ files. This is not yet a Torch-independent
oracle; the separate verifier consumes those raw records.

For small deterministic fixtures, validate three seeds. Large event and
ranked cases use their frozen fixture manifests. Exercise current-event and
eligible-history input policies as separate declared inputs, exact primal and
derivative worksets, dormant sampling, repeated gate indices, full-support and
indexed-union value paths, one gate recovery, one structural publication
between pinned windows, and sparse checkpoint restore. Compare direct and
field-plan forward values, surrogate probability VJPs, recurrent VJPs, owner
updates, generation counters, and checkpoint state. The hard threshold's
surrogate gradient is checked against explicit Torch composition, not finite
differences.

No result is scientific validation: these are synthetic fixture execution
fidelity checks.

### E5 execution status and transition boundary

The admitted smoke at `evidence/e5_smoke.json` executed one small current-event
case. The direct module and the experiment-local field-stage adapter agreed on
the captured outputs, loss, gradients, and owner update. This demonstrates
path parity only; both paths call the same native transition and relation
primitives.

Independent CPU replay of the retained raw arrays found that the transition
parameter contract is inconsistent across the native and Python surfaces.
`include/Cellerator/compute/operation/sparse_support/training.hh` documents
shared transition A/B/bias matrices. The native CUDA/reference implementations
index those matrices as shared values, while
`python/cellerator/torch/sparse_support.py::sparse_parameter_layout` stores
per-mechanism blocks shaped `[mechanisms,width,width]` and `[mechanisms,width]`
and the binding accepts those rank-three/rank-two tensors. The raw fixture
includes affected mechanisms beyond mechanism zero, so it distinguishes the
two readings. The verifier's independent replay using the documented shared
block-0 matrices matched final output (max abs `3.05e-7`), loss (`2.38e-7`),
and owner gradient (`3.13e-8`); replay using each mechanism's corresponding
matrix did not. This establishes an ABI/storage mismatch, not a native kernel
math defect.

Per the experiment boundary, this branch does not choose or repair the model
contract. Recurrent E5 expansion, including ranked/history/value-policy
matrices, structural publication continuation, and checkpoint restoration,
is stopped pending resolution by the production owners. The captured smoke and
raw counterexample are retained. The independent gate-estimator checks remain
useful, but they do not resolve the transition parameter contract.

## E7: bounded execution portfolio

Each of the small, large-event, and large-ranked fixtures is frozen before the
GPU run. The six candidates are the two indexer families (scan and ordered
postings) crossed with SIMT/full-support values, SIMT/indexed-union values, or
CE-GEO forward with indexed-union values and the existing SIMT backward.
CE-GEO/full-support values are outside this bounded portfolio. Ordered posting
preparation has an experiment-local lazy entry point, so a fixed scan candidate
does not pay for an unused posting projection; adaptive policies may include
that preparation when considering a later switch.

`e7_relation_runner.py --validate-only --case <fixture>` validates all six
candidates, exact identities, and a forced ordered-posting fallback without
timing. `--benchmark --case <fixture>` collects paired complete-step samples
for all six candidates against scan/SIMT/full, including the per-step C ABI
query, native synchronization and D2H, host materialization, and host-to-device
workset construction. The reusable support, query, output, optional posting,
geometry, and module preparation costs are reported separately. KDA can invoke
one collector round per request and retain its 21 paired samples for each of
the six internal candidates. `--include-horizons` runs actual 1, 16, and 256
step reuse sequences; these are separate from repeated KDA timing requests.
Each timed pair restores both native owners and Adam states from captured
`NativeOwnerProtocol` snapshots before measuring either implementation. The
full checkpoint is retained once for resume and horizon evidence, not reloaded
for each sample. Snapshot restoration is reported separately as harness
overhead. `--validate-reset-protocol --case small` compares this fixed-structure
reset against a full checkpoint restore, checks exact support-manager state,
owner values, Adam state, generations, and a subsequent native forward/VJP/
update, then confirms that a changed structure epoch rejects the stale
snapshot. The fast path is experiment-local and restores through the canonical
owner plus optimizer state APIs; it does not write Parameter storage directly
or rebind the support projection. Candidate cold cost includes shared
indexer preparation plus candidate-only postings, CE-GEO geometry, and module
preparation; baseline cold cost includes the same shared preparation plus its
SIMT module. Peak device memory is reported for the joint comparison process,
not as a per-candidate value.

The E7 relation-only runner does not call the recurrent latent transition.
This isolates indexer/value/backend economics while the Python/native
transition parameter layout mismatch blocks E5 recurrent interpretation.

Unsupported indexer/backend combinations remain explicit in the report. Each
candidate uses the same restored initial owner, fixture, and masks. The
confirmation schedule is five warmups and 21 paired samples in each of two
exploration and two confirmation rounds. Each candidate/fixture therefore has
84 paired observations; setup and complete costs are reported separately.
Cold setup and reuse horizons 1, 16, and 256 are measured. The density trace is
low → high → low with one explicit support mutation, and a cost-table policy
may re-evaluate every 16 steps only when estimated remaining savings exceed
preparation and switching cost.

NVCC and NVRTC compare the same small generated device region and bindings;
the already-precompiled path is the baseline. This branch adds no alternate
kernel algorithm or direct-PTX campaign.

## E6 GPU lifetime slice

The GPU slice uses two streams and one background compiler worker with at most
one replacement pending. Deterministic event/barrier checkpoints cover old work
in flight, forward-tape ownership through backward, compile failure/cancel,
duplicate requests, stale topology, and value-only changes that remain
compatible. It records artifact/module/buffer references, completion events,
publication and retirement order, and fallback selection. Memory, race, and
synchronization sanitizer records are required before a pass decision. If the
existing executable API cannot support this without production redesign, report
the precise missing capability and mark the GPU slice unresolved.

## Run order and acceptance

The root coordinates one serial scheduler window at a time. CPU-only launch
and identity checks precede each admitted run. Raw logs, receipts, generated
artifacts, per-run manifests, and reports are retained here. The GPU reports
must distinguish existing behavior, experiment-local adapters, demonstrated
behavior, negative findings, and unanswered questions. No thresholds become
production defaults.
