# Cellerator: prepared migration decisions

This is a source-informed shortlist, not a complete file inventory. `tools/inventory.py` produces the full tracked/untracked listing at execution time. Entries from an old map remain **existence/consumer checks** until inspected locally. No source move is scheduled automatically.

| Observed material | Intended destination/status | Action and guard |
|---|---|---|
| README.md | Human entry | Replace with prepared question-first text; separate current host SDK from GPU runtime. |
| AGENTS.md (404+ lines) | Agent entry + durable design links | Preserve managed Project Control block; move rationale/status out of the operational contract. |
| scope.md; docs/architecture.qmd; docs/biological_execution_model.qmd | Developer design | Read full QMD locally (observer denied this file type). Consolidate overlapping introductions; preserve technical value/frontmatter. |
| docs/current_implementation.qmd; docs/migration_roadmap.qmd | Dated status / historical migration | Reconcile against final source; one human snapshot, not several contradictory “current” pages. |
| docs/CE_LIVE_FINAL_AUDIT.md; docs/CE_GEO_PROGRAM.md | Campaign evidence/planning | Preserve source-specific claims, link from result pages and archive index. Do not turn one campaign into universal architecture. |
| scripts/ce_geo/validate_documentation.py | Living documentation validator | It currently requires exact old README/QMD headings and `22/22`. Retarget checks to new evidence/status pages without weakening actual numerical/source evidence validation. |
| src/compiler; include/Cellerator/compiler; cmake/compiler | Current compiler | Keep ownership. Optional small API/driver filename cleanup only; avoid renaming hundreds of task-derived files in this pass. |
| src/geometry; src/execution; src/compute | Current numerical substrate | Keep conceptually useful separation. No wholesale tree redesign. |
| compat; components/CelleraTorch | Compatibility / adapter | Keep distinct and indexed; no second runtime. No writes to these optional/subordinate trees are included in the default cleanup. Resolve any symlinks and the actual owning scope before a separately qualified change. |
| root ce-*-plan.json; planning; todos* | Planning/generated | Classify root plan clutter. Move only non-live plans with references repaired; generated state is never manually moved. |

## Default source decision

Documentation reorganization first. Keep the current component structure unless the one bounded cluster described above materially improves navigation. Record final paths in `inputs/moves.json`; update bindings after moves. Preserve immutable evidence paths or provide an explicit old→new resolution map. Never change code semantics merely to make the tree look tidy.
