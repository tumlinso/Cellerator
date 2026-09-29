# Cellerator: prepared migration decisions

Outcome 10 disposition, checked 2026-09-29 at source HEAD `e4135674446ab9e32717f8b5ade04c1c3648326a`. The generated `inputs/inventory.json` lists 6,021 tracked and one untracked path; it is a path inventory, not proof every file was semantically reviewed. No source or documentation move was made in this outcome. The focused decisions below record what remains and what later outcomes own.

| Observed material | Intended destination/status | Action and guard |
|---|---|---|
| README.md | Human entry | Applied prepared question-first text; host SDK/CUDA boundary is explicit, current capabilities link to dated status. |
| AGENTS.md | Agent entry + durable design links | Replaced with concise contract and links; managed Project Control block preserved byte-for-byte. |
| scope.md; docs/architecture.qmd; docs/biological_execution_model.qmd | Scope contract and detailed design | Keep scope.md as intent/ownership; preserve technical Quarto chapters; concise design overview routes to them. |
| docs/current_implementation.qmd; docs/migration_roadmap.qmd | Dated history / target migration | Label current_implementation.qmd as 2026-09-06 evidence; new status/current.md is the 2026-09-29 concise snapshot. Roadmap remains target context. |
| docs/CE_LIVE_FINAL_AUDIT.md; docs/CE_GEO_PROGRAM.md | Campaign evidence/planning | Preserve source-specific claims, link from result pages and archive index. Do not turn one campaign into universal architecture. |
| scripts/ce_geo/validate_documentation.py | Living campaign validator | Retargeted to the new entry/status pages and retained detailed campaign documents; exact forward, fusion, preprint, sanitizer, and acceptance evidence assertions remain unchanged. |
| src/compiler; include/Cellerator/compiler; cmake/compiler | Current compiler | Keep ownership. Optional small API/driver filename cleanup only; avoid renaming hundreds of task-derived files in this pass. |
| src/geometry; src/execution; src/compute | Current numerical substrate | Keep conceptually useful separation. No wholesale tree redesign. |
| compat; components/CelleraTorch | Compatibility / adapter | Keep distinct and indexed; no second runtime. No writes to these optional/subordinate trees are included in the default cleanup. Resolve any symlinks and the actual owning scope before a separately qualified change. |
| root ce-*-plan.json; planning; todos* | Planning/generated | Keep with Todo/package owners; do not move or hand-edit generated authority. Further inventory/classification is not a source-move prerequisite. |

## Outcome 20 source decision

No source move was useful. Retain the current compiler/API/CMake relationship (`src/compiler`, `include/Cellerator/compiler`, `cmake/compiler`) and the separate geometry/execution areas (`src/geometry`, `src/execution`). These groupings already aid development; renaming would add consumer churn without improving navigation. CE-ML2-TRAIN remains in progress and owns `CMakeLists.txt`, `src/execution`, tests, and CelleraTorch interfaces, so leaving paths stable keeps its work continuable. `inputs/moves.json` records the decision and binds the inspected source, build, example, adapter, compatibility, and public-document paths. No numerical code, test semantics, or evidence paths changed.
