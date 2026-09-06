# Semantic Spine v1: planning, manual bootstrap and small accelerator demo

**Delivered 6 September 2026. Planning artifacts only. Nothing has been applied or implemented.**

This package defines **29 atomic tasks plus one epic record, 30 Todo records total**, in **seven first-class lanes**. After a short common-contract phase, native execution, compiler lowering, algebra repair and independent validation proceed concurrently. The narrow witness is forward and transpose application of one biological relation on an actual **sm70 GPU**, with reusable topology and changing values.

## Important readiness distinction

The package has an internally checked native **schema-3** plan. **Live Cellerator Todo authority was unavailable during construction. Native validator acceptance, live ID collision checks, and the additive application diff therefore are not certified.** The supplied wrapper refuses application until the installed native validator and coherent workflow authority are available. Do not downgrade to schema 2 to bypass this: schema 2 loses first-class run/lane semantics.

The example is a **post-epic acceptance program**. Its normal build intentionally requires the proposed Cellerator API that these tasks implement. No fake backend or silent CPU fallback is provided. A separate reference-only mode has been executed locally to check fixture mathematics; that is not a Cellerator/GPU test.

## Copy into the repository

The archive contains two repository-relative directories:

```text
planning/semantic-spine-v1/
examples/semantic_spine_v1/
```

The main program belongs at **`examples/semantic_spine_v1/regulatory_reuse.cc`**, beside its `fixture.hh`, `CMakeLists.txt` and `README.md`. An `examples/` hierarchy already exists; there is no need for a new top-level `demo/` directory. Extraction does not edit the root build. I04 adds the minimal opt-in wiring later.

Preserve and review the pre-existing `library/cellerator.cell` and `library/cellerator/cellerator.ceh` edits. Commit the reviewed package, demo and preserved edits before a live preview. This does not implement the epic. The `.ceh` conversion is an implementation task, not permission for package extraction to delete source.

## Read in this order

1. `01_SCOPE_AND_DECISIONS.md`: the scope, choices and stop line.
2. `02_SEMANTIC_AND_NATIVE_CONTRACT.md`: concrete mathematical and provisional native contracts.
3. `03_PARALLEL_LANES_AND_INTEGRATION.md`: ownership, first-class agents and actual dependency graph.
4. `04_VALIDATION_AND_DEMO.md`: meaningful tests and accelerator acceptance.
5. `05_MANUAL_BOOTSTRAP.md`: safe installation, preview, separate application and stop.
6. `06_DEFERRED_WORK.md` and `07_SOURCE_LEDGER.md`: what is not done and what supports the design.

`proposed-todos/` contains a substantive task sheet for every record. `handoff/` contains per-lane playbooks and an execution prompt, which is **not** an instruction to execute now. `contracts/` contains declaration-only sketches to make the expected endpoint concrete without prematurely installing a new public API. The unmodified comprehensive review is retained in `basis/architecture-review.html`.

## Machine files

**The only native plan file to apply is:**

```text
planning/semantic-spine-v1/machine/semantic-spine-v1.todo-plan.json
```

`machine/proposed_todos.json` is a richer **non-authoritative pre-ledger**, not native Todo input. The CSVs and other JSON catalogs are consistency-checked views. `external_dependency_receipts.csv` deliberately has only a header: this epic does not mutate or depend on another project's authority.

Do not use `project-control plan compile` on this package. The inspected generic pre-ledger compiler emits schema 2. This package's `scripts/compile_plan.py` preserves native schema-3 runs and lanes.

## First command after placing the files

From the repository root:

```bash
python3 planning/semantic-spine-v1/scripts/todo_bootstrap.py validate --source-root .
```

This checks package integrity and placement, not live Todo readiness. Follow `05_MANUAL_BOOTSTRAP.md` for the separate native preview and explicit application. **Application does not authorize activation, implementation worktrees, agent dispatch or source changes.**

## Completion will mean

C++ and the bounded parser/Sema/IR route construct one authoritative relation meaning; actual forward and transpose GPU implementations consume it; two generations reuse prepared structure; false primitive/composition equivalences and the contraction split law are corrected; the small demo works; and full `.cell` execution is still clearly marked as future work. It will not mean full compiler, installed SDK, general algebra or broad performance completion.
