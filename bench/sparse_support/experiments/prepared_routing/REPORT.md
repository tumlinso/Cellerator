# Prepared entity-to-mechanism routing qualification

## Decision

Keep `affected_mechanism_workset(..., strategy="prepared")` opt-in; the
existing Python strategy remains the default. The prepared posting path has a
narrow measured long-reuse win for ranked `[32,8]` routing, an inconclusive
result for event `[32,12]`, and loses to the dense Torch baseline at the larger
synthetic shapes. This evidence does not support default promotion or a full
model-speed claim.

## Source and run identity

- Installed Torch extension: SHA-256
  `8db8763e1b440b3efba569de82b616ff3fc09d28001d269b7bbee3b25b5df8be`.
- Route module source: `execution.py` SHA-256
  `98b995c18b5382c48968d91a813c7ec3ec193cf9ce580574c95f8a33cf87d107`;
  prepared adapter SHA-256
  `3079ffc0330a0d46da130af323ccc9de64727013da93186e6094133fe92392cd`.
- Prepared native host/CUDA sources: `prepared_routing.cc`
  `dcc5c96cb6135c92bc5b86bd4ab8bb62af8fbe284dbb99491d143d43dd3ee7d8`;
  `prepared_routing.cu`
  `1c9d2de66e380e174cde159cecb86dbc827d19864295795a6cd0fd6807b01cb9`;
  binding `prepared_routing.cc`
  `9562ca394ac9bb4feb27ca3fa4a2fbcf9d8ca1def67f0190f7d705399fa1cbb4`.
  Existing reverse-posting CUDA source SHA-256:
  `d034ac6815e070b356d08b3f8852cf03d7ab39a91d008ab12008bbf30c92f99c`.
- Benchmark source SHA-256:
  `b91b252f46a2ab6c87cb28625a54c0e3de979a3f4004aae7443f26cc5613177a`.
- Raw result: [paired-route-paths.json](evidence/paired-route-paths.json),
  SHA-256
  `4bca4e2b11794a1f52926a5002acf16e35704960b0a6e344f5e9871e9e6e1783`.
- Timing job `09d3e070-c6ca-4db6-8114-3b51354c14f9` succeeded with verified
  cleanup. It used a V100-SXM2-16GB, Torch 2.7.0+cu126, 5 query warmups,
  21 paired repeats, seed 14491, 4096 entities, 16 mechanisms, and route degree
  at most 2. The benchmark first checked results against the independent
  Python oracle.

The timing is owner-cold, not process-cold: PyTorch, the extension, CUDA
context, route rows, and correctness checks were already initialized. Each
candidate sample includes fresh-owner construction and its first complete
query, followed by `H-1` paired warm-query measurements. Thus prepared cost
includes index construction/upload and the first query's workspace reservation;
dense cost includes route-matrix construction/upload and its first query. The
incumbent cost is `H` calls through the existing Python route path. Repeat IDs,
input hashes, measurement order, and per-horizon paired lifecycle costs are in
the raw JSON. The horizon totals are estimates formed from measured cold-first
and warm samples; they are not sequential 64-step training runs.

## Lifecycle results

Horizon 64 median total milliseconds; the final column is the paired prepared
versus Python-route improvement with its bootstrap 95% interval.

| Input shape | Python route | Prepared postings | Dense Torch | Prepared vs Python |
|---|---:|---:|---:|---:|
| ranked `[32,8]` | 14.28 | 13.14 | 24.03 | +9.21% [7.79, 10.36] |
| event `[32,12]` | 16.28 | 16.72 | 23.98 | -0.87% [-2.47, 2.10] |
| stress `[256,64]` | 215.98 | 56.59 | 24.47 | +73.63% [73.41, 73.89] |
| stress `[1024,128]` | 1488.95 | 212.80 | 25.47 | +85.70% [85.59, 85.85] |

At horizon 1, prepared cold-first medians are 2.56 ms for ranked `[32,8]` and
2.60 ms for event `[32,12]`, versus Python route medians of 0.22 ms and
0.25 ms. At the stress shapes, the stronger dense alternative also has much
lower warm query medians: 0.350 ms versus 0.846 ms for `[256,64]`, and
0.364 ms versus 3.289 ms for `[1024,128]`. It remains faster in lifecycle cost
at horizon 64, despite its cold matrix construction.

## Qualification and limits

The final frozen binary passed the 16-test suite and exact prepared CUDA
correctness check in job `85ba8634-7761-4eb5-a376-52137d79d040`. Separate
Compute-Sanitizer jobs passed:

- memcheck: `9ae53356-9256-40ff-9bb8-0dc795f3e950`;
- racecheck: `ca486ff7-6b02-4008-96a3-15f62b7830ee`;
- synccheck: `ed28c4ea-641c-4b17-9c25-752b107c840d`.

An earlier synccheck submission, `95004b71-a8ad-494f-9bcf-9b0dbc8ede43`, was
blocked because the execution precondition could not verify an active resource
lease; it had zero attempts and did not execute. This was a scheduler
precondition failure, not an engine or sanitizer failure; the later retry
passed. An earlier test snapshot also had an assertion comparing two
references to the same mutable cache; the test was corrected before the final
16/16 pass.

The prepared query performs one explicit synchronization per nonempty query to
read the exact output count before allocating its CSR result. Native shape and
capacity checks fail closed at the supported CUDA grid and tensor-index limits.
The timing starts from the already available route table; score refresh is
shared setup and was excluded. Inputs cover the documented valid ordinals and
negative padding. The dense baseline does not implement the prepared path's
explicit invalid-ID prevalidation and matching exception contract, so the
comparison establishes valid-input route-union cost, not complete API
substitutability. Synthetic route shapes are plumbing evidence, not biological
or end-to-end training performance.
