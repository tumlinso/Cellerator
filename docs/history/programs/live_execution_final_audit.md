# CE-LIVE final audit

CE-LIVE closes with one planner-backed quantitative biological execution path,
one bounded native training path, persistence/replay and concurrency evidence,
one measured Tensor Core decision, and a thin CelleraTorch adapter. This is a
bounded activation result, not a claim that every historical Cellerator surface
has migrated or every biological workload is supported.

## Accepted state

| Area | Final result | Evidence boundary |
|---|---|---|
| Biological relation | Forward is feature/gene source to row/cell/module destination; transpose/backward retains logical edge identity through explicit maps | Shape alone never establishes compatibility. |
| Quantitative forward | PBMC3K computational fixture passed widths 1, 16, 17, 31, 32, 48, 64; reuse 1, 8, 1024; two generations; independent CSR referee | The fixture supports no biological interpretation. The built-in catalog currently exposes one legal FMP1 schedule per tested request. |
| Conventional fallback | Session-backed cuSPARSE CSR preparation/execution exists with explicit complete costs and no per-run descriptor/device setup | The quantitative built-in SpMM catalog does not yet expose cuSPARSE as a second legal candidate for every request. |
| Planning | Existing planner consumes real candidates, projections, complete phase costs, reuse horizons, analytical/measured evidence, and exposes its decision | CE-LIVE does not add a general DAG planner. Zero regret in the forward sweep is over the actual legal set. |
| Training | Native five-stage forward/epilogue/explicit-transpose/update/readiness program passed parity and multi-generation tests | N=16 only; not a general-width training engine or Torch-owned optimizer. |
| Persistence/replay | Pointer-free CPE2 reloads through existing opaque CellShard CPEXEC01 transport, activates non-owning typed views, and reproduces CUDA output | CPEXEC02 remains independent CellShard work; CellShard does not interpret projections or select kernels. |
| Runtime | Same-stream and cross-stream readiness, two-stream reuse, pointer relocation, fixed-transition CUDA Graph capture, stale identity/generation rejection, and sanitizer coverage passed | No hidden hot-path allocation, device selection, descriptor creation, structural hashing, or device-wide synchronization. |
| Tensor Core | `v100-wmma-dense-fragment-f16-f32` received a measured `evaluated_not_promoted` result | PBMC3K produced zero tiles at the frozen 50% density threshold; the candidate remains absent from the built-in catalog. |
| CelleraTorch views | Native dense operands and parameters are exposed as lifetime-bound, zero-copy Torch aliases with explicit metadata and failure cases | Torch does not become canonical storage or parameter owner. The copied CPU CSR exporter remains compatibility/debug only. |
| CelleraTorch forward | Torch tensors bind to the prepared native executable program on the current Torch CUDA stream | No second planner/runtime, hidden conversion, canonicalization, private stream, or device synchronization. |
| CelleraTorch autograd | Supported N=16 autograd calls native training, returns the dense-input gradient, preserves native updates and readiness | Cellerator owns relation values and bias; applying a Torch optimizer to the same aliases would be a duplicate update and is unsupported. |
| CelleraTorch quantitative | 126 paired native/adapter PBMC3K forward comparisons passed; adapter GPU/enqueue deltas were indistinguishable from timing noise; zero-copy view metadata cost about 3.47 us median | PBMC3K sweep is forward-only; autograd parity uses the supported native N=16 envelope. |

## Evidence index

- Wave C aggregate evidence: `bench/ce_live/evidence/ce_live_evidence_v1.json`
  and `docs/CE_LIVE_EVIDENCE.md`.
- Quantitative forward records:
  `bench/ce_live/forward/pbmc3k_forward_v1.jsonl`.
- Native training evidence:
  `bench/ce_live/training/native_training_v1_evidence.json`.
- Replay evidence: `docs/CE_LIVE_REPLAY.md`.
- Concurrency evidence:
  `bench/ce_live/concurrency/acceptance_evidence.json`.
- Tensor Core decision:
  `bench/ce_live/tensor_core/campaign/v100_decision_v1.json`; controller
  `39cd21c2-ef11-4b1e-a23d-fe49bc838794`; memcheck
  `2803c563-faeb-4358-a471-70c69118f4d0`.
- CelleraTorch quantitative result:
  `bench/ce_live/celleratorch/quantitative_results_v1.json`; clean-source
  controller `b7cf7354-ddce-4afb-89ec-0f912a8724a1`; exact CUDA 12.9 memcheck
  completed with `ERROR SUMMARY: 0 errors` under the benchmark mutex.

## Explicit unsupported frontier

- Training widths other than N=16 and broader optimizer/model envelopes.
- Broader built-in candidate plurality for the quantitative SpMM requests,
  including catalog exposure of the retained conventional path where legal.
- Automatic promotion of the experimental V100 Tensor Core candidate.
- General DAG, distributed, or multi-node execution.
- CellShard CPEXEC02 source/residency integration.
- Broader Baseplane sequence-engine work beyond the frozen Cellerator seam.
- Biological conclusions from the PBMC3K computational fixture.

These are future tasks, not hidden CE-LIVE completion claims. Cellerator remains
the owner of biological identity, structures, projections, planning, prepared
programs, runtime resources, parameters, value generations, and reusable math.
CelleraTorch remains an adapter.

With all CE-LIVE leaf and rendezvous tasks terminal, the frozen
`cellerator-cellera-torch-entry-v1` interface intact, the Tensor Core decision
terminal, and the ledger reconciled, CE-LIVE-45 may publish `CE_LIVE_COMPLETE`.
