# Filling results without rewriting the presentation

The prepared records in `inputs/results/` are the data source for tables, captions and figures. Edit these JSON files, not the exported chart. The record is intentionally small; it is not a replacement for the original benchmark's raw format.

`evidence_state` is one of `recorded_summary`, `verified_historical`, `fresh_measurement`, or `pending`. The first is the preparation-time state of retrieved summaries, not a new run. `pending` must have an empty `rows` array. Do not fill absent measurements with zero.

Keep the original source path/revision and raw evidence identity distinct from the revision at which someone read or verified it. A historical result remains historical after its JSON is copied. Use portable public evidence paths, not links into `.todo-orchestrator/runtime` or a user's home directory. Preserve the original archived receipt and add a path translation note when files move; never silently rewrite a signed/hashed old measurement to fit a new layout.

Each row has a `case`, a `values` map for every declared series, and optional per-series `bounds`. Bounds must retain their meaning (the GlassHelix record uses observed minima/maxima of five samples, not confidence intervals). All measurements in a chart share the displayed units and stated timing scope. Do not pool unlike workloads into a performance ranking.

Before selecting a record, set its `review` with the actual reviewer, verified source HEAD, source and method checks, and verified `public_evidence_paths`. Provide a runnable reproduction description using the existing scheduler. Set `selection.selected` only after comparing the candidates using `shared/RESULTS_POLICY.md`; at most two studies per repository. Update README result links/callouts to match the selection.

Fresh timing is limited to the selected existing fixture shapes, with small repetitions and adverse regimes where relevant. It is not permission to tune until a positive result appears. A useful negative result, a preserved capability without speedup, or a documented lack of publishable evidence can all close the corresponding evaluation.

Run the final renderer to build Markdown, JSON, PNG and SVG from those records. All stages require a readable unit, comparator, workload, source, cost boundary and limitation. Inspect the figures at README width and verify the numbers against the retained raw rows. There is no benchmark auto-execution in this package.
