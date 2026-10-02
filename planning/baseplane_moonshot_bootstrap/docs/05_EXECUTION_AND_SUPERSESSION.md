# Execution and old-program supersession

## What this package does and does not do

`machine/baseplane.todo-plan.json` is a schema-3 successor plan with 16 records: one aggregate epic, one foundation outcome, 12 family outcomes, one composition outcome and one integration/handoff outcome. It supplies one conservative serial lane. Family scopes are disjoint so the root can authorize useful bounded delegation without creating another planning authority.

The package is **not applied**. The exposed Project Control connector is read-only. Its empty diagnostic preview returned `bounded_read_failed`; a subsequent two-record schema-3 preview timed out. Neither result validates this full plan, and no exact handoff authorization was returned. The local package validator checks structure and consistency, not the installed Todo runtime's complete semantics.

The native plan was constructed from the inspected current schema-3 example at `project-control:planning/pce2/machine/project-control.todo-plan.json`. The commands below are from inspected `planning/pce2/BOOTSTRAP.md` and `docs/MAINTENANCE.md` at Project Control commit `366e85e88d1585e511c12bb856f7fd64e77ff3d5`. The Project Control working tree was dirty; these are observed instructions, not a claim that an installed runtime cannot have changed. Use the existing verified launcher, not a newly improvised interpreter.

## One successor, not an appended competing roadmap

The live source run is `compat-v2`, with root `BITOP-00`. The proposed successor is `BP-MOON-RUN-1`, root `BP-MOON-000`. The complete 33-record old-to-new map is mandatory input to adoption. Preserve all done/superseded records and historical lab/docs outcomes. The qualified production obligations are retained as Q01–Q05, explicitly outside the workshop's scheduled execution.

The cutover also needs the reviewed policy amendments. In particular, the old isolated experimental exception must include `experiments/moonshot`, and evidence of a speedup must no longer be a prerequisite for implementing an authorized idea. Importing new task rows alone does not apply those policy changes. Do not let a stale invariant silently forbid the campaign, and do not silently ignore a still-authoritative invariant.

## Adoption route for the implementing root

1. **Inspect current identity once.** Confirm the target repository, actual Todo authority, source run, claims, dirty work, pending artifacts and current instructions. Compare with `evidence/observed-state.json`; it is a baseline, not a reusable signed precondition. Material drift requires reconciliation, not blind replay.

2. **Stage source documents and review the intended changes.** Put this package under `planning/bp-moonshot-20261002` through the normal owned workspace process. It is source material, not a database. Read `machine/old-to-new.json`, `machine/policy-amendments.json` and the actual target-specific plan diff. Preserve all existing code/results. The first implementation outcome installs `seed/` into the isolated `experiments/moonshot/` tree.

3. **Validate and stage the successor through the canonical front door.** The source plan is a complete candidate payload intended for native import, not a hand-built SQLite patch:

   ```sh
   python3 planning/bp-moonshot-20261002/tools/check_package.py
   project-control plan validate --project baseplane --file planning/bp-moonshot-20261002/machine/baseplane.todo-plan.json
   ```

   Inspect the actual validation/diff response. Resolve exact schema, scope, dependency or interface errors; do not replace an unreadable authority with another database. Apply only after target validation and review:

   ```sh
   project-control plan apply --project baseplane --file planning/bp-moonshot-20261002/machine/baseplane.todo-plan.json
   ```

   Do not claim or dispatch either new or old implementation while the cutover remains incomplete. The new run must exist for the documented signed supersession route. Apply the reviewed policy changes and old-record lifecycle dispositions using supported current authority operations, never generated-view edits. If these require an additional typed plan/maintenance action, inspect its current schema rather than guessing a payload from this policy-intent document.

4. **Use the supported audited replacement operation.** Prepare a fresh assignment after the reviewed state is staged:

   ```sh
   project-control admin prepare-supersession \
     --repo "$BASEPLANE_REPO" \
     --intent planning/bp-moonshot-20261002/machine/supersession-intent.json \
     --recipient "$OPERATOR_PRINCIPAL"
   ```

   `$BASEPLANE_REPO` and `$OPERATOR_PRINCIPAL` refer to the actual configured repository and trusted operator, not a guessed path or model name. Follow the returned fixed operator launch and opaque-grant `maintain_execution` call. The host derives complete unfinished source membership and current identities. Do not assemble row digests, broaden grants, refresh signed requests automatically, or clean retained work. A newly discovered retained dirty workspace needs an explicit handoff/adoption decision; the provided intent deliberately does not invent one.

5. **Verify the receipt and postconditions, then claim the exact successor.** Confirm only the successor can drive the replaced scope; completed old outcomes remain intact; deferred qualifications are not silently running; the stale `CE-BITOP-44 -> CE-BITOP-40` binding is not imported as a new prerequisite; policy amendments are actually in effect. If the replacement retires execution membership but leaves historical task rows blocked, reconcile their disposition through the supported lifecycle path using the map. Do not mark old executable/adapter checkpoints reached merely because new seeds exist. Use the exact `next_task` recommendation in the successful maintenance receipt when present.

This is a staged reviewed adoption, not a claim that plan import and run replacement form one cross-repository transaction. The Baseplane run replacement has no cross-repository transaction. A separate reviewed Cellerator companion plan adds the numerical research tasks under its own authority; it does not replace ML2 or any unrelated Cellerator run. If the canonical route is unavailable, preserve the staged package and report the specific blocker rather than inventing another runtime or editing state directly.

## Default execution order

The native lane runs foundation, then family outcomes, then compositions and handoff. That is a convenient safe starting queue, not a scientific ordering of merit. The root may move a ready family forward, create a bounded local delegate, or begin a composition early through supported workflow changes.

Each family has four designed cards, but a card is not a separate native TODO. Implementation can reveal a better decomposition. Preserve the distinct questions: combining shared implementation is fine; collapsing several different ideas into one generic wrapper and declaring them all done is not.

Keep shared include/build seams owned by foundation/integration. Family agents write inside their family directory and return proposed shared changes rather than concurrently editing the root CMake file or public Baseplane headers. Public ABI changes remain separately reviewed against the actual owner. Baseplane tasks do not claim Cellerator paths; companion CE-MOON tasks do own experimental numerical/learned mechanisms. Neither plan rewrites unrelated general planning, GlassHelix theory or CellShard persistence.

## Effort allocation

No numerical token allowance was specified by the user, so this package does not invent one. The user's direction is to spend available effort on many interesting implementations with little qualification. Keep a visible aggregate usage receipt when the runtime provides one, and report gaps rather than implying exact metering.

A useful qualitative allocation is: most effort on mechanism implementation, a meaningful second share on cross-family compositions, and only the small remainder on common scaffolding and essential probes. If tests, orchestration or a generic framework are consuming the campaign, simplify the workflow—not the truth of the results.

## Completion means experimental substance, not production proof

Each artifact records authored/compiled/run/compared separately. A kernel can be authored and not compiled because the toolchain is absent. A mechanism can be compiled but not timed. A synthetic fit can be demonstrated without any biological generalization claim. All are honest experimental states.

A family outcome closes only with tangible mechanisms or concrete individual dispositions. A rejected idea needs a reason tied to its semantics, implementation or scope, not a vague claim that it is speculative. The aggregate handoff reports how many catalogue items actually received code, which combinations ran, and what remains. It does not announce “48 implemented” just because 48 cards exist.

## Paired numerical research adoption

The user explicitly clarified that experimental learned mechanisms and ML-like mathematics are also in Cellerator's scope. Adopt `machine/cellerator.todo-plan.json` independently in Cellerator, following `docs/07_CELLERATOR_RESEARCH_SCOPE.md`. It has seven local records and never imports the old Baseplane CE-prefixed placeholders as Cellerator authority. This supplement is part of the research campaign, not a requirement to wait until a Baseplane prototype is mature enough to migrate. The Baseplane replacement still covers exactly the old 33-record programme.
