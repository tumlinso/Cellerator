# Current Objective

## Summary
Build a biology-first public embryo dataset shortlist, record additive manuscript append material, and define the first benchmark package without touching the main manuscript source.

## Planning Notes
- First execution pass should stop at discovery, ranking, and append-ready specs rather than downloads or manuscript rewrites.

## Assumptions
- A dataset root has not been provided yet, so manifests and fetch actions are intentionally deferred.
- Biology-first coverage is more important than processed-file convenience for the shortlist stage.
- Primate secondary-modality coverage will be macaque-heavy in the first pass because direct human embryo multiome data are sparse.

## Suggested Skills
- `public-omics-intake` - Drive official GEO and SRA metadata discovery and ranking.
- `quarto-manuscript` - Keep manuscript additions source-first and additive relative to docs/main.qmd.
- `todo-orchestrator` - Keep the ledger current while this multi-step dataset and manuscript track evolves.

## Useful Reference Files
- `docs/main.qmd` - Current manuscript scaffold with explicit dataset and figure placeholders.
- `docs/public_omics/ranked/all_candidates.tsv` - Combined shortlist output for all current public candidate pools.
- `docs/public_omics/benchmark_spec.md` - First benchmark contract for the selected datasets.

## Plan
- Generate reproducible shortlist artifacts from accession-curated GEO and SRA candidate pools.
- Record additive manuscript section text and figure specs without editing docs/main.qmd.
- Map selected datasets to existing benchmark binaries and the next intake step.

## Tasks
- [x] Generate shortlist script and ranked outputs.
- [x] Write additive manuscript append material and figure specs.
- [x] Record benchmark mapping for selected datasets.

## Blockers
_None recorded yet._

## Progress Notes
- Added scripts/build_public_omics_shortlists.py to resolve curated GEO and SRA accessions, apply embryo-specific metadata overrides, and emit reproducible shortlist artifacts under docs/public_omics.
- Generated ranked candidate sets for mouse scRNA, mouse secondary modalities, primate scRNA, and primate secondary modalities, plus a combined all_candidates.tsv summary.
- Added additive manuscript append text, benchmark spec text, and two figure spec files without modifying docs/main.qmd.

## Next Actions
- Get a dataset root from the user, then use the canonical anchor set to build layout plans, manifests, and the first processed-file fetch scope.
- After intake starts, turn the figure specs into rendered dataset-landscape and benchmark-readiness figures.

## Done Criteria
- Repo contains a reproducible first-pass shortlist with machine-readable query specs and ranked outputs.
- Repo contains additive manuscript and figure planning artifacts that do not modify docs/main.qmd.
- The next intake action is explicit and blocked only on dataset-root selection.
