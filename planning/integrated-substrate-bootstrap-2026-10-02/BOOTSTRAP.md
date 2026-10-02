# IS1 bootstrap handoff — 2 October 2026

All three repaired native plans were validated, independently reviewed and imported through `project-control plan apply`. Sealed inputs in `../integrated-substrate-v1` remain unchanged. Derived inputs add exact predecessor root definitions so native dependency validation succeeds without removing the predecessor barrier.

| Authority | Native revision | New tasks including closure epic | Run | Lanes |
|---|---:|---:|---|---:|
| cellerator | 7678 | 20 | CE-IS1-RUN-V1 | 16 |
| baseplane | 377 | 14 | BP-IS1-RUN-V1 | 10 |
| glasshelix | 405 | 14 | GH-IS1-RUN-V1 | 9 |

Post-import preservation verification passed for 1,475 existing tasks, 121 interfaces and 27 runs. Existing definitions, completion results, evidence, frozen interfaces and pending work were retained. Only the six replayed predecessor roots received expected version/revision/time updates, with closure gate revision changes where applicable. Each repository contains `preservation-verification.json` and ignored exact before/after native exports.

The private preservation backup is `/home/tumlinson/is1-bootstrap-backups/20261002T163131Z`; its manifest, verification and README describe copied evidence and excluded rebuildable files. Native Todo exports and recovery snapshots were preserved without direct database edits.

## Existing work and authority boundaries

All six mandatory predecessor epics are done. The reviewed 27 original runs are classified in `reviewed-run-inventory.json`. CE Ampere remains permission-locked; GEO aggregate closure is preserved for separate reconciliation. GH-PREPRINT and GH-SCIENCE-FOUNDATIONS remain independently available and carry their original ownership/authorization. Bootstrap does not publish or edit a preprint.

Eight historical pending commits are ancestors of the accepted source heads and their owner tasks are done. Preserve their patch/workspace/integration metadata; do not apply these commits again. Historical NF1 queued integration entries and frozen contracts remain owned by their original workflows. Implementation starts from the accepted current lineage and reconciles any actual interface change through its owner.

## Activation and dispatch

No implementation task was dispatched by bootstrap. Each run starts with its local `*-IS1-ADOPT` task; ADOPT is substantive integration work, not completed by importing this plan.

`prepare_activation.py` checks the reviewed original inventory against full native read-only authority, requires all six predecessors done and no active owners, verifies pending commit ancestry, rejects unexpected runs or unresolved integration records, and checks explicit clean source heads. Its approval flag is for a controller that has reviewed the actual evidence. The local configuration and source-bound review live in ignored files; they are not a portable or permanent approval. The freshness limit is 600 seconds.

For each repository, run the existing package activation gate before explicit dispatch:

```sh
python3 -B planning/integrated-substrate-v1/tools/gates.py activation --config planning/integrated-substrate-v1/local-config.json
```

If evidence has expired or source/authority changed, review the changes and refresh the activation capture first. Root uses `next_task` with explicit repo_root, run_id and task_id; never rely on whichever historical run an unqualified call selects. Check workflow workspace readiness as part of that dispatch; bootstrap has not launched numerical execution or produced implementation acceptance.

## Parallel execution

Keep eight live agents total including root. After three bounded ADOPT foundations and accepted source commits, schedule 17 independent CE/BP/GH domains in batches. Favor CE's critical path with roughly four CE and three BP/GH worker seats initially. Shared build files, registries and headers have one integrator. Every isolated leaf hands its actual patch to MERGE-A or MERGE-B; dependent worktrees start from the accepted integrated commit.

CE's second wave covers strategies, math, differentiation, adaptation and lowering, then framework and installed integration. BP/GH provider joins wait for the tested CE receipt. GH JOINT also waits for BP's final capability. Serialize clean GPU timing, preserve losing alternatives and keep native/framework/science claims distinct from package reference tests.

No repository numerical source changed during bootstrap. Existing package checks passed 52 Python tests, strict C++17 conformance and static validation during preflight. No GPU benchmark or scientific qualification was run here.
