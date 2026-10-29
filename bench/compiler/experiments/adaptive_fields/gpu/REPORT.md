# GPU experiment decisions

This report records GPU-side evidence for the bounded adaptive-fields probes.
Every performance result is synthetic. None is biological validation. The
production API, defaults, and numerical kernels were not changed.

## Decisions

| Probe | Decision | Evidence boundary |
|---|---|---|
| E5 field-visible sparse training | **Unresolved** beyond the smoke | One admitted current-event/all-values smoke showed path parity, but both paths call the same native primitives. Recurrent training expansion stopped at a Python/native transition-layout contract mismatch. |
| E6 executable replacement on GPU | **Unresolved** | Sanitizers reported no errors in the exercised path, but `cuModuleLoad` waited for the callback watchdog to release queued work in both sanitized and normal runs. Replacement while old work remains in flight was not demonstrated. |
| E7 numerical/indexer correctness | **Supported** for the six bounded candidates | All six candidates passed exact workset/mask checks and numerical checks across three deterministic small seeds; the fixed-structure reset protocol and the same-source sanitizer suite passed. |
| E7 fixed-structure value/optimizer reset | **Supported** for the small fixture | The existing `NativeOwnerProtocol` snapshot/restore plus Adam state restore matched full checkpoint restoration and native forward/VJP/update. A stale epoch was rejected. |
| E7 performance portfolio | **Unresolved** | One 21-pair exploration round completed for small and large-events. The second exploration and both confirmation rounds did not run; large-ranked was interrupted before a complete candidate aggregate. |
| E7 reuse horizons and GPU adaptation trace | **Unresolved; not run** | No GPU horizon 1/16/256 sequence, measured low/high cost calibration, or 48-step support-mutation trace was executed. The two-indexer trace source is not evidence of a six-candidate policy. |
| E4 NVCC/NVRTC device-region comparison | **Supported** as a bounded compile/load comparison | NVRTC reduced cold compile time for one generated region, with no measured steady-state execution advantage. |

## E5: field-visible sparse training

The admitted small current-event smoke is in [e5_smoke.json](evidence/e5_smoke.json),
with retained arrays in [raw/e5_seed20261027_current_event_all.npz](evidence/raw/e5_seed20261027_current_event_all.npz).
It demonstrates direct-versus-adapter parity for that trace only. The independent
replay and source analysis are recorded in [e5_contract_boundary.json](evidence/e5_contract_boundary.json)
and the [estimator report](../verification/estimator/README.md).

The native header and CUDA transition implementation use shared A/B/bias
blocks, while the Python binding accepts and allocates per-mechanism blocks.
Independent Torch replay of the retained smoke using the documented shared
block matched final output within `3.05e-7`, loss within `2.39e-7`, and owner
gradient within `3.13e-8` maximum absolute error. The intended parameter
contract is not resolved, so recurrent fixture expansion, history/value-policy
matrices, and structural-publication continuation are not claimed.

## E6: executable lifetime on GPU

The v3 lifetime path completed the available memory, race, and synchronization
sanitizers with zero reported errors, but did not establish overlap. The normal
run reproduced the sanitizer observation: module load remained blocked until
the callback watchdog released queued work at about ten seconds. The test's
unconditional replacement-success field was corrected in
[RESULT_CORRECTION.md](runtime/lifetime/RESULT_CORRECTION.md); retain the frozen
v3 source and scheduler evidence in the [normal receipt](../evidence/scheduler/runtime_lifetime_normal_terminal.json),
the [sanitizer receipt](../evidence/scheduler/runtime_lifetime_v3_terminal.json),
and the [independent source review](../verification/lifetime/E6_GPU_SOURCE_REVIEW.md).
The GPU slice therefore does not demonstrate safe
replacement with old work pending. The separate CPU executable-lifetime
prototype passed its deterministic ownership/replacement cases, but does not
prove CUDA stream or tape behavior.

## E7: correctness, reset, and incomplete economics

The final qualification used the E7 driver source SHA-256
`30ca61ad230cb21fa2470fb9b77a5828150a27e966d9ef50ac4b3c60593c6efe`.
All six scan/ordered-postings × SIMT-full/SIMT-union/CE-GEO-union candidates
passed numeric validation for seeds 20261027, 20261028, and 20261029. The
small fixture reset test also passed: full checkpoint restore took `3.923 ms`
and fixed-structure numerical/Adam restore `0.469 ms` in this single run. The
restored native generation matched (`2 → 3`), output and update max errors were
zero, gradient max error was `1.78e-15`, optimizer state matched, support and
epoch matched, and epoch 1 → 2 invalidated the saved snapshot. These are
single-fixture protocol measurements, not a general speedup claim. The reset
record is [reset_protocol.json](../evidence/e7_final_qualification_v2/normal/reset_protocol.json);
the three sanitizer summaries are collected in
[qualification summary](../evidence/e7_final_qualification_v2/summary.json).

The KDA v2 run was stopped at the user-directed time boundary and its terminal
receipt verifies cleanup. It retained one complete 21-pair round for small and
large-events, with five warmups per candidate. The measured paired median
candidate-minus-scan/SIMT/full deltas (milliseconds) were:

| Candidate | Small | Large events |
|---|---:|---:|
| scan / SIMT / full | -0.046 | -0.875 |
| scan / SIMT / indexed union | +0.344 | -8.357 |
| scan / CE-GEO / indexed union | +0.393 | -9.238 |
| ordered postings / SIMT / full | +0.022 | -3.545 |
| ordered postings / SIMT / indexed union | +0.337 | -14.159 |
| ordered postings / CE-GEO / indexed union | +0.411 | -11.966 |

These are exploratory medians from one round, without the required second
exploration and two confirmation rounds. Do not select a universal execution
policy from them. Large-ranked retained partial raw/checkpoint artifacts only;
it has no complete timing aggregate. The two completed portfolio records are
[small](../kda/workspace/runs/kda-experiments/attempts/6c3b810c-cd47-4ad5-bf1f-efb27cde71e2/stages/portfolio_collector/exploration/round-000/81db8ebbbbc69c6c6ad4/requests/df69d4ed-46cd-4ccf-82f8-c416e80edaf3-small-portfolio.json)
and [large-events](../kda/workspace/runs/kda-experiments/attempts/6c3b810c-cd47-4ad5-bf1f-efb27cde71e2/stages/portfolio_collector/exploration/round-000/622b9b114f2bc8b337cb/requests/930bb64f-7af0-45e1-9ad1-b0cfb4c1e9cd-large_events-portfolio.json).
The cancelled run's [terminal receipt](../evidence/scheduler/e7_kda_v2_terminal.json)
records `cleanup_verified: true`.

Reuse horizons 1/16/256 and the low → high → low GPU trace were not run. The
experiment-local trace/calibration scripts remain unqualified source. Root
review also found that the trace compares one fixed baseline rather than both
fixed candidates, and charges preprepared posting cost again in its switch
hurdle. No trace result or adaptation-policy conclusion is claimed.

## E4: NVCC and NVRTC comparison

For one small generated region on `compute_70`, both routes and value rebinding
matched the FP64 reference with zero maximum absolute error. Cold NVCC compile
took `449.839 ms`; cold NVRTC compile took `18.998 ms`. Module load took
`0.059 ms` for the precompiled NVCC fatbinary and `0.126 ms` for NVRTC. Across
84 paired execution samples the median NVCC/NVRTC wall ratio was `0.9917`, so
the faster compilation did not produce a measured execution advantage. See
[comparison.json](../evidence/runtime_clean/comparison.json). This supports
reusing the current code-generation route for a bounded JIT probe, not a claim
that runtime compilation improves steady-state execution generally.

## GPU validation performed

- Final E7 normal validation: three deterministic small-fixture seeds, all six
  candidates, plus the fixed-structure reset check and a one-pair protocol
  smoke.
- Final E7 `memcheck`, `racecheck`, and `synccheck`: all completed with zero
  reported errors on that source snapshot. The scheduler run's terminal
  evidence is under `../evidence/scheduler/e7_final_qualification_v2_terminal.json`.
- E7 KDA: one completed 21-pair exploration round for small and large-events;
  large-ranked was partial. The run was cancelled at the user's time boundary
  and cleanup was verified.
- E4 NVCC/NVRTC: 84 paired executions after numerical and rebind checks.
- E6 lifetime: normal and sanitizer attempts retained; sanitizer zero-error
  does not change the unresolved overlap result (see the receipts and source
  review above).

No GPU horizon diagnostic or GPU trace was launched. The synthetic portfolio
evidence is incomplete and does not establish biological accuracy or a
production specialization threshold.
