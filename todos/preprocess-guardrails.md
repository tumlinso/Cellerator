# Current Objective

## Summary
Protect preprocessing and sparse logical indexing while Blocked-ELL becomes the native sparse type and compressed survives only as a fallback.

## Planning Notes
- The Blocked-ELL-first rewrite cannot split logical cells and genes or silently degrade preprocessing-oriented sparse workflows that still depend on compressed fallback.

## Assumptions
_None recorded yet._

## Suggested Skills
- `cuda-v100` - Sparse phase boundary and layout advice for omics workflows on V100.

## Useful Reference Files
- `src/apps/series_workbench_cuda.cu` - Preprocess and browse-cache surfaces that still assume compressed shards.
- `optimization.md` - Current guidance that compressed remains the general sparse path unless repeated SpMM justifies promotion.

## Plan
- Identify preprocessing and indexing paths that must keep compressed support.
- Add validation and tests so execution-format promotion does not break those paths.

## Tasks
- [ ] Audit preprocessing and browse-cache code paths for compressed-specific assumptions.
- [ ] Define regression checks for logical indexing and preprocessing behavior under promoted execution layouts.

## Blockers
_None recorded yet._

## Progress Notes
- The workbench browse-cache path no longer depends on compressed parts; it now reloads native Blocked-ELL headers and uses Blocked-ELL device views for feature sums and sample extraction.
- The main remaining compressed-only guardrail is `run_preprocess_pass()` and the lower-level preprocess operators under `src/compute/preprocess/`.

## Next Actions
- Port or explicitly gate the compressed-only preprocess path so native Blocked-ELL series files do not silently depend on CSR-only internals.
- Add a focused runtime check once the preprocess decision is made.

## Done Criteria
- Compressed-path preprocessing and logical indexing remain correct and explicitly guarded after planner changes.
