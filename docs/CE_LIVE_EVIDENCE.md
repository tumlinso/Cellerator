# CE-LIVE end-to-end evidence v1

CE-LIVE-36 is the Wave C evidence rendezvous. It combines the checksum-pinned
quantitative forward slice, bounded native training slice, CPE2 opaque replay,
and adversarial stream/readiness acceptance into one reproducible foreground
campaign. The machine-readable fan-in is
`bench/ce_live/evidence/ce_live_evidence_v1.json`.

## Evidence boundary

The PBMC3K subset is computational evidence only: 512 cell/row destinations,
32,738 feature/gene sources, and 433,808 logical edges. Forward orientation is
feature/gene source to cell/row destination. Transpose/backward uses the same
logical edge identity through CTP1's explicit map.

The forward operation covers `N = 1, 16, 17, 31, 32, 48, 64`, reuse horizons
`1`, `8`, and `1024`, and two mutable value generations over one structure.
Every output matches the independent CSR-coordinate referee. The retained
records report 0% planner regret, but the result is qualified: the built-in
executable catalog currently exposes exactly one legal FMP1 schedule for each
request. There is no legal conventional SpMM catalog entry to compare in this
program, so this is complete regret over the legal set—not evidence that FMP1
beats cuSPARSE universally.

Forward totals are Level-2 operation evidence. FMP1 construction, static value
packing, and kernels were measured; host/backend preparation and transfer
terms are explicit analytical inputs to the complete-cost planner. Input and
output canonicalization, communication, and transient workspace are zero by
the tested contract. The FMP1 projection occupies 3,073,340 bytes and each
f16 value generation occupies 867,616 bytes. Dense input/output traffic scales
explicitly with `N` in the machine-readable artifact.

The bounded N=16 training pipeline is Level-3 evidence. Its five native stages
cover forward, epilogue, explicit transpose/backward, sparse update, and bias
update before readiness publication. The retained V100 comparison measured
19.584 us native versus 28.352 us for the persistent conventional CSR path
(1.45x), with equal preparation exclusion and independent correctness. This is
only a small-module N=16 result.

CPE2 replay remains pointer-free and uses the unchanged opaque CPEXEC01
transport. Runtime acceptance covers same-stream chaining, cross-stream event
ordering, one fixed-generation CUDA Graph transition, pointer relocation,
stale identity rejection, and absence of hidden allocation, device selection,
or device synchronization. The replay and concurrency device paths passed
Compute Sanitizer in their owning tasks.

## Reproduction

Run the authoritative one-V100 foreground campaign with:

```bash
python /home/tumlinson/.agents/skills/cuda/scripts/cuda_controller.py run \
  --spec bench/ce_live/evidence/cuda_controller.json --json
```

The controller builds without a GPU lease, then holds the benchmark mutex,
requires three idle samples, runs one correctness pass, and performs five
foreground repeats. It records exact source, toolchain, device, command, and
contamination identity. The accepted CE-LIVE-36 evidence ID is added here only
after that committed campaign succeeds.

Checkpoint: `CELLERATOR_LIVE_EVIDENCE_V1`.
