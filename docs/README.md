# Cellerator Documentation

This directory contains the authoritative architecture, implementation-status, performance, and migration documentation for Cellerator.

## Authority

The active documentation spine is:

1. `../AGENTS.md`
2. `../scope.md`
3. `architecture.qmd`
4. `current_implementation.qmd`
5. `biological_execution_model.qmd`
6. `cellpack_cp_bp.qmd`
7. `core_execution_cp_math.qmd`
8. `baseplane_integration.qmd`
9. `storage_distribution_and_interop.qmd`
10. `performance_validation.qmd`
11. `migration_roadmap.qmd`
12. `developer_reference.qmd`

`index.qmd` is the reader entry point. `_quarto.yml` defines the rendered book.

## Current state versus target state

Documentation must distinguish:

- **implemented now**;
- **transitional implementation**;
- **target architecture**;
- **historical evidence**.

Do not describe a planned ABI as implemented. Do not treat a current implementation detail as a permanent architectural rule.

`current_implementation.qmd` is the explicit implementation snapshot. Update it when a migration phase materially changes the repository.

## Historical material

Previous architecture reports, root strategy documents, closed TODO ledgers, and one-off performance investigations should be preserved under `docs/history/` when they contain useful evidence.

Every historical file must begin with a banner:

```text
Historical document. It describes an earlier repository state and is not
authoritative for current architecture or implementation decisions.
```

Historical material must not appear in the main Quarto chapter list.

## Local operational material

`docs/pipeline/` may remain for local operations. Its documentation lives in
`operations/local_pipeline_ops.qmd`, outside the main architecture book.

These scripts and notes are not product or architecture dependencies. They must not be imported by CMake targets or cited as core architecture.

## Authoring rules

- Prefer one authoritative explanation over several overlapping summaries.
- Link to the central document rather than copying its architecture into a local README.
- Record current source paths only where they help readers navigate the implementation.
- Keep persistent identifiers, ownership, order, and lifetime explicit.
- State which claims are measured and which are proposed.
- Include performance mechanisms, not only design motives.
- Avoid declaring a universal format or backend unless the planner and benchmark evidence establish that scope.
- Update docs in the same change that changes an ABI, ownership boundary, or persistent image.
