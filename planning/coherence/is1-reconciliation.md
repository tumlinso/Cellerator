# CE-IS1 reconciliation proposal

This proposal maps the five still-blocked IS1 obligations to the narrower
CE-COHERE plan. It does not accept, complete, supersede, release, or edit native
Project Control state. The root must apply any changes through a fresh supported
plan preview/apply. The current workflow read showed all five old tasks blocked
at semantic revision 19.

## Task mapping

| Old task | Evidence already represented | Proposed destination | Still open |
|---|---|---|---|
| `CE-IS1-TORCH` | Real CPU FP32 Product2 module with first-order gradients; implemented sparse/mechanism adapters and bounded integration checks. See `docs/current_implementation.qmd`, `docs/development/python-bindings.md`, `tests/prepared_routing/`, and `tests/sparse_support/`. | Carry the coherent package slice into `CE-COHERE-04`; parameter sharing and reusable sparse/execution work into `CE-COHERE-05/06`. Supersede the blanket “native mathematics through Torch” obligation. | Not every native family has a Torch wrapper. General Torch JVP coverage and a uniform installed operation/device/precision matrix are not established. Treat future adapters as feature requests with real consumers. |
| `CE-IS1-QUALIFY` | Bounded sparse performance reports and source-bound correctness/sanitizer records; `planning/coherence/qualification/native/receipt.json` documents CPU/static builds and installed SDK consumers, with no device execution in those commands. Prepared routing separately has a 16-test pass, exact CUDA correctness and all three sanitizer passes, plus an owner-cold lifecycle comparison across four synthetic shapes/horizons (`bench/sparse_support/experiments/prepared_routing/REPORT.md`). The results do not support a universal strategy. | Carry strategy-specific complete-cost evidence to `CE-COHERE-06`; final independent qualification to `CE-COHERE-09`. Retain the custom-packer/new-operator exercise only when a concrete extension is selected. | No unified cold/warm/training/changing-structure campaign is established. Prepared routing's benchmark excludes shared score refresh and is not full training or biological validation. `docs/current_implementation.qmd` still says its performance qualification is pending and should be corrected against the later report. Existing measurements remain workload-specific. |
| `CE-IS1-DOCS` | User/developer orientation, current implementation boundaries, routing map, and focused guides exist in `README.md`, `scope.md`, `docs/current_implementation.qmd`, `docs/getting_started.qmd`, and `docs/development/`. | Carry documentation acceptance to `CE-COHERE-08`; final consistency check to `CE-COHERE-09`. | Render/link validation and the GitHub description update were not verified in this audit. Do not assume every stale status claim was removed. |
| `CE-IS1-INTEGRATE` | Native matrix/process extraction and actual installed SDK consumer checks are recorded in `planning/coherence/extraction-map.json` and `planning/coherence/qualification/native/`. | Carry native capability into `CE-COHERE-03`; actual Torch slices into `CE-COHERE-04/05/06`; final capability handoff into `CE-COHERE-09`. | No integrated-substrate `provider.json` was identified. Installed SDK checks do not establish a published stable ABI or comprehensive Torch matrix. Preserve this as a gap unless CE-COHERE explicitly produces an equivalent source/build-bound receipt. |
| `CE-IS1-CLOSE` | Preservation verification, branch disposition, source map, and native checks are recorded in `planning/coherence/preservation-validation.json`, `planning/coherence/branches.md`, `planning/coherence/archive-disposition.md`, and `planning/coherence/qualification/native/receipt.json`. | Supersede broad closure with `CE-COHERE-09`, while preserving already represented slices in `CE-COHERE-00` through `08`. | Final package/source-map/docs qualification remains open. Keep the managed IS1 worktree and records because current compact workflow has no supported cleanup route. Unknown legacy effects remain unverified. |

## Acceptance boundaries

- Native host math and installed SDK checks passed for their recorded source and
  consumers. They do not establish Torch wrappers for all native families.
- The integrated native receipt is CPU/static for its listed commands. GPU
  evidence in older source-bound records does not extend to new binaries or
  workloads. Prepared routing has bounded GPU correctness, sanitizer, and
  synthetic route-lifecycle performance evidence; it is not full training or
  biological performance evidence.
- Historical benchmark comparisons support regime-specific choices and retained
  fallbacks, not a universal packing or sparse strategy.
- Preservation and ancestry prove retained artifacts or source history, not
  completion of old native tasks and not the truth of unknown legacy effects.
- `CE-COHERE-08` still needs render/link and external package-identity checks;
  `CE-COHERE-09` must reconcile evidence from the final integrated source.

## Root plan-change constraints

1. Use the host's fresh preview and apply only the returned proposal reference.
2. Preserve all evidence links and state that these are supersession/carry-forward
   proposals, not retrospective acceptance.
3. Keep the missing provider receipt and Torch coverage explicit if the new
   outcomes do not consume them.
4. Keep the protected worktree until a supported managed cleanup operation is
   available; do not replay unknown legacy actions.
