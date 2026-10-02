# Integrated substrate bootstrap readiness

Observed 2026-10-02, 15:02–15:06 UTC. Verdict: extracted and host-checked; **not ready for native adoption or implementation**.

## Extraction and evidence

The sealed 104-file package is present at `planning/integrated-substrate-v1` in Cellerator, Baseplane and GlassHelix. Each copy contains all shared documents, scripts and three native plans; adopt only the plan belonging to each repository. All 103 manifest-listed hashes verified at each destination. No plan import, task dispatch, commit or authority mutation occurred.

Official staging preview passed in Baseplane and refused tracked dirty state in Cellerator and GlassHelix. The user's explicit extraction request was fulfilled using exclusive create-only copies. Existing tracked file hashes were unchanged during each copy. Concurrent work continues. The sealed package was not modified. See [staging receipt](staging-receipt.json).

Fresh package/DAG validation, 52 Python tests, strict C++17 compilation/conformance and host demonstration passed. See [validation and durable logs](validation.json). No native repository, GPU, framework or scientific qualification was performed. The archive's older static report counts 92 manifest files; fresh verification covers the final 103 manifest entries.

## Intent

Integrate existing numerical and dynamical work into Cellerator's normal preparation/execution/derivative paths. Preserve semantic axes, parameter ownership, tapes, structure epochs and value generations. Packing remains pluggable rather than imposing Cellpack on every operator. Baseplane retains exact sequence semantics and adds coherent representations and queries; GlassHelix supplies scientific experiment specification, actual data, inference and replay. Independent CE and BP cores plus an optional bridge avoid a package dependency cycle. Preserve both moonshot portfolios, current CellTag ML2 work, evidence and unfinished research.

## Live predecessor barrier

| Project | Required root | Observed state |
|---|---|---|
| Cellerator | CE-ML2-0000 | planned, unfinished |
| Cellerator | CE-MOON-000 | done; run completed |
| Cellerator | CE-MOON-0000 | planned, unfinished |
| Baseplane | BP-MOON-000 | done; run completed |
| GlassHelix | GH-ML2-0000 | planned, unfinished |
| GlassHelix | GH-MOON-0000 | planned, unfinished |

Cellerator revision 7630, HEAD `009a5579c1facec8c3bce3c36f6764c9f49ab97c`: CE-ML2-BIO has active ownership of CMakeLists.txt, directly overlapping CE-IS1-ADOPT. CE-MOON-NATIVE-READY remains blocked on CE-MOON-OLD-OWNER-DONE; native readiness, integration, closure, rewrite and trajectory work remains.

GlassHelix revision 334, HEAD `19707ee3e0741c64bbb82621a1fea4fe9554a5fa`: GH-ML2-DATA has active ownership of docs/learning. GH-MOON-RECEIPT has failed CE capability and old-owner gates; receipt, consumer and closure work remains. GH progressed during this inspection, so these observations require refresh at adoption.

Baseplane revision 376, HEAD `e7e225dcbb6e5c7ca61b6a3b77d6ea9eb1026749`: BP-MOON-RUN-1 completed; no active dispatch observed. The global barrier still prevents independent IS1 activation here.

## Other active or scheduled work

- CE compat-v2, POST-REMAP and BIOPREP extraction runs retain active labels with observed closed lanes/completed queues. PTR has a ready coordinator and closed implementation lanes. These labels alone do not justify restarting work.
- CE-AMP-RUN-V1 retains queued work across seven ready lanes. Preserve its permission gate; IS1 does not authorize Ampere.
- CE-GEO and CE-JBC had sampled closed/completed lanes, but complete residual reconciliation remains necessary. GEO overview still showed 108 done, 8 planned, 3 blocked.
- BP-CUDA-LAB-RUN-1 retains an active label without observed queued lane work; its separate closure run completed. Preserve deferred BitOp intent and historical dispositions.
- Preserve frozen CE sequence-predicate ABI, BP sequence-event/predicate/ownership interfaces and GH NF1 interfaces. Resolve actual changes through owners.
- Existing accepted/pending artifacts and worktrees need a complete adoption review. Bounded rich reads were truncated; this report does not certify all historical patches or every scheduled obligation.

Re-fetch through Project Control `project_overview`, `coordination_view(detail="expanded", max_items=100)`, and `inspect(kind="task"|"run", target=<IDs above>)` for each lowercase project ID. This is a bounded observation digest, not an activation approval.

## Native plan compatibility blocker

All three original plans fail the installed `todo_orchestrator.plan.validate_plan`: ADOPT's predecessor dependencies are not task IDs inside the submitted plan. Live database existence does not satisfy this validator. CE references CE-ML2-0000, CE-MOON-000 and CE-MOON-0000; BP references BP-MOON-000; GH references GH-ML2-0000 and GH-MOON-0000.

Implementation reference: `/home/tumlinson/.agents/skills/todo-orchestrator/todo_orchestrator/plan.py`, validate_plan at line 81 and dependency checks at lines 189–202. An in-memory diagnostic removing these dependencies passed structural validation for CE20/BP14/GH14 tasks, isolating this defect; those altered plans were neither saved nor adopted. Dropping the dependencies is not an accepted repair.

Project Control native preview also returned `internal_error` / `bounded_read_failed` for all three plans at 15:04:41–43 UTC; this separate preview failure must be resolved before adoption. No successful native diff review exists yet.

A candidate repair is to compose a proposal from a fresh supported native export plus the successor, retaining exact predecessor records and dependencies. Review the full additive diff and current preconditions through supported plan administration. Do not fabricate predecessor stubs: upsert can overwrite their fields. This repair has not been qualified; keep sealed originals unchanged and prepare derived inputs separately after predecessor completion.

## Parallel implementation recommendation

Use at most eight live agents including root, and one clean GPU timing job unless resource authority explicitly permits more. Root keeps cross-project contracts, integration and final acceptance. Configured Codex workers receive bounded ownership and accepted base commits; no local worker implementation. A parallel head is optional only when sustained coordination savings justify its seat.

1. Complete the global barrier and three small ADOPT foundations; integrate accepted contract/base commits.
2. Schedule 17 independent domains in batches: CE state, packing, operators, effects, execution, build; BP sequence, hierarchy, index, reuse, learning, build; GH science, data, models, analysis, build. Initially favor four CE workers and three BP/GH workers under root, adapting to readiness and conflicts.
3. Integrate each repository's MERGE-A. Shared build files, registries and common headers have one integration owner. Leaf workers propose shared changes rather than concurrently editing them.
4. Run CE strategies, mathematics, differentiation, adaptation and lowering in parallel from the accepted MERGE-A commit. Then MERGE-B, TORCH and INTEGRATE publish the installed CE capability receipt.
5. BP BRIDGE and GH BRIDGE join that real CE provider, then BP COMPOSE and GH PIPELINE proceed independently with qualification/docs. Native and framework claims require actual tested call paths.
6. BP CLOSE publishes its final capability; GH JOINT consumes BP and CE receipts before GH closure.

A completed task in an isolated worktree is insufficient: merge its actual patch before creating dependent worktrees. Serialize integration into shared source lineages, not all independent implementation. Failed domains block only their dependent region. Preserve the 92-entry inventory and 33 historical BitOp dispositions, expanding them from completed predecessor source and newly scheduled work.

## Remaining activation steps

1. Let existing ML2 and moonshot work finish; do not supersede, resume or repair it as part of this preflight.
2. Refresh all live authorities, claims, source heads, worktrees, pending patches, interfaces and remaining obligations. Back up current plans/generated state and untracked/ignored evidence before substantial adoption or implementation.
3. Resolve native proposal composition and preview failures. Require reviewed additive diffs for all three authorities without changing completed history or deferred work.
4. Prepare ignored explicit local configuration and a fresh source-bound activation review from actual evidence. Verify the installed SemanticReader interface and executable gate paths. No approved review was manufactured during this preflight.
5. After appropriate planning commits, refresh exact-source evidence. Independently import the three accepted plans through supported operations, retain separate runs/authorities, and check global activation before dispatch.
6. Bind real acceptance commands and provider receipts to the lifecycle as implementation becomes available. Host seed checks do not satisfy native, SM70, framework or scientific acceptance.
