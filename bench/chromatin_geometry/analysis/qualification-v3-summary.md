# V3 bounded milestone evidence

The v3 paired measurement is **unqualified**. Job `45bab455-239f-4e32-8ebd-fd33ae03e979` completed the raw matrices but its sampled telemetry recorded a foreign GPU compute process. The worker failed the guard. No latency, staging, update-strategy, or performance conclusion is drawn from this packet.

The [qualification export](qualification-v3.json) retains the 48 base-matrix and 96 update-ablation records, their paths and hashes, the worker outcome, and telemetry evidence. Their recorded per-arm completion does not make them eligible samples. The telemetry is sampled and cannot establish host-wide exclusivity or exclude activity between observations. The earlier v2 qualification remains separate.

Four representative complete-chain sanitizer modes independently passed on the **exact v3 archive and binary**, with verified Scheduler cleanup. The [sanitizer export](sanitizer-coverage-v3.json) records memcheck, racecheck, initcheck and synccheck receipts; all six supplied/STE/teacher and product/linear cases used the cached two-tile backend, FP16 synthetic input, tile 64 and one stage. This is representative software coverage, not a performance result or full-matrix qualification.

- Source archive SHA-256: `f131948000849270e22c392459264fa599566f4a1d51492a09e72d9e1d8f72c7`.
- Binary SHA-256: `b0292b309e12751655b8a476c4d34e23d1c877549e175d33c5dcdca12aa25ddf`.
- Frozen contract SHA-256: `4033eea4a8168f5e898212873252c8782943d12f5325e5a31ca45487f9751f06`.
- Triage job: `384b43b3-3229-47cb-b91b-7aeb3db25df4`. Sanitizer jobs: memcheck `720ad1a6-616b-4fa7-b3d3-fca24a91aaad`, racecheck `622a7cb7-e319-42f3-86e5-6e5f1d757059`, initcheck `5031bf45-2e2d-4ff0-9cba-5ad26775301c`, synccheck `5b6d63ea-f325-4f3f-9625-b3f0bcf334ec`.

The ledger also records two cancelled unreserved auto-submissions with verified cleanup, preserved under `runs/failures/`: `f16ecce8-8c95-42d4-8a6e-6c56bdf84b9c` was charged 22.4117 seconds; `1dd519df-ba2b-4883-945d-9208dbea3492` was charged zero and has no Scheduler attempt window. Their ledger rows lack receipt paths, so the export explicitly retains a separate global serialization reconstruction gap. The failure receipts were retained; no attempt window is inferred for the pre-attempt cancellation.

The remaining workload matrix and profiling coverage are unfinished. No further measurement submission is included in this milestone.
