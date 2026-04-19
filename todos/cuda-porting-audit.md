---
slug: "cuda-porting-audit"
status: "done"
execution: "closed"
owner: "codex"
created_at: "2026-04-16T09:10:38Z"
last_heartbeat_at: "2026-04-16T09:13:21Z"
last_reviewed_at: "2026-04-16T09:13:21Z"
stale_after_days: 14
objective: "Audit CPU-centric Cellerator subsystems for CUDA port candidates and prioritize a Volta-native migration order"
---

# Current Objective

## Summary
Audit CPU-centric Cellerator subsystems for CUDA port candidates and prioritize a Volta-native migration order

## Quick Start
- Why this stream exists: _Summarize the domain boundary and why it was split out._
- In scope: _List the work this stream owns._
- Out of scope / dependencies: _List handoffs, upstream dependencies, or adjacent streams._
- Required skills: _List the exact repo-local skills to read before starting._
- Required references: _List the exact repo-local references to read before starting._
- Use the native system route and Volta router: this audit assumes the 4x Tesla V100 16 GB host and should classify each subsystem as keep-on-CPU control plane, move-to-CUDA hot path, or already-correct mixed boundary.

## Planning Notes
- Inventory current CUDA-backed surfaces first, then identify CPU orchestration that still sits in hot paths, and only then rank candidate ports by expected payoff and decomposition cost.

## Assumptions
- This task is an audit/report, not an implementation slice; done means a repo-grounded port map with priorities, not code changes.
- Blocked-ELL is the native sparse execution layout, CSR/compressed is fallback or interop, and any new CUDA recommendation should respect that direction.

## Suggested Skills
- `cuda` - Use the native V100 system route and Volta CPU-porting doctrine for the audit recommendations.
- `todo-orchestrator` - Keep this repo-wide audit resumable in the existing ledger structure.

## Useful Reference Files
- `optimization.md` - Current repo-wide bottleneck map and subsystem-level CUDA posture.
- `custom_torch_ops.md` - Existing model-op and sparse-runtime CUDA boundaries that should be extended rather than bypassed.
- `todos/cellshard-blocked-ell-ingest-runtime.md` - Active workstream already targeting the CPU-only ingest blocked builder.

## Plan
- Inspect repo ledgers, optimization notes, and the active CUDA workstreams so the audit matches current migration priorities.
- Map each top-level subsystem under src/ to current execution posture: CPU-only, mixed, or CUDA-backed.
- Pull line-level evidence from hot-path files where CPU control flow or sparse-layout churn still blocks GPU residency.
- Produce a Volta-oriented porting map that distinguishes control-plane code that should stay on CPU from compute that should move to CUDA or a custom Torch/CUDA op.

## Tasks
_None recorded yet._

## Blockers
_None recorded yet._

## Progress Notes
- Created the audit workstream and anchored it to the existing native-Volta migration doctrine before starting the repo scan.
- Completed the repo-wide CUDA porting audit. Highest-value ports are the dataset ingest blocked builder and rebucketing path, the Torch model sparse-projection/loss boundaries in developmental_time and dense_reduce, and the forward-neighbor query orchestration path; text/HDF5 parsing and the explicit torch export bridge should remain CPU control-plane surfaces.

## Next Actions
- Finish the static audit of ingest, preprocess, models, neighbors, trajectory, workbench, and graph surfaces, then write the prioritized CUDA port map.
- No further action in this audit stream unless the user wants one of the recommended port slices turned into an implementation workstream.

## Done Criteria
- The audit clearly labels each major subsystem as keep on CPU, partially port, or fully port, with the recommended CUDA boundary and expected limiter called out.
- A Volta-oriented repo map exists that separates keep-on-CPU control plane code from the CPU hot paths that should move into CUDA/autograd/custom-op surfaces.
