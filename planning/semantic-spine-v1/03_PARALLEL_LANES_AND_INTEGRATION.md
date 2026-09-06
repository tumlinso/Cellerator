# 3. First-class parallel lanes and integration

## Topology

One native schema-3 run, `CE-SS1-RUN-V1`, contains the following lanes. The parent lane is `CE-SS1-L-COORD`. All work queues are serial within a lane; task-level `parallel_safe` allows different lanes to advance together.

| Lane | Role | Tasks | Owned responsibility |
|---|---|---|---|
| COORD | coordinator | root epic | Dispatch/observe/coordinate; no source implementation |
| CORE | implementer | C01–C04 | Shared mathematical descriptor, validation and provisional signatures |
| NATIVE | implementer | N01–N06 | Prepared forward/transpose pair, value refresh and existing GPU adapters |
| FRONTEND | implementer | F01–F04 | Existing parser/Sema/IR relation bridge and source-specific tests |
| ALGEBRA | specialist | A01–A04 | Contraction split law and truthful composition/effect classification |
| VERIFY | validator | V01–V04 | Independent oracle, negative controls and integrated test probes |
| INTEGRATE | integrator | I01–I07 | Baseline, shared build wiring, .ceh conversion, joins and final hardware acceptance |

The graph is:

```text
I01 -> C01 -> C02 -> C03 -> C04 -> I02 (integrated foundation)
                                         |
             +---------------------------+-------------------------+
             |                 |                 |                 |
          N01..N06          F01..F04          A01..A04          V01..V04
             |                 |                 |                 |
             +-----------------+-----------------+-----------------+
                                         |
                                I04 -> I05 -> I06 -> I07

I02 -> I03 (.ceh conversion, independent of the four lanes) -> I04
```

The root epic has no outgoing prerequisite edge to children. It closes only after I07. Neither NATIVE nor FRONTEND waits for VERIFY's entire lane, and VERIFY does not wait for their completed implementation to write independent probes. Linking/running those integrated probes occurs at I04/I05. This is deliberate useful parallelism, not labels attached to a serial task graph.

## JBC mechanics retained, campaign scale not retained

The JBC planning map specifies serial task queues inside first-class lanes, a few genuine interface/checkpoint barriers, independent provider work and explicit integration [S01]. Use those mechanics here. The much larger CCP plan supplies a native schema-3 shape and useful catalogs, but neither its phase hierarchy nor its heavy serial dependencies are copied.

Only one internal interface is published: `CE-SS1-I-RELATION`, owned by I02 and hashed from the two real shared headers. The publication freezes a working interface for consumers of this epic. It is not public API stabilization. Consumers must invalidate assumptions and reconcile if that interface is revised.

## First-class Codex agents are not local-worker children

Applying a native plan creates records, not running subagents. An execution-authorized coordinator must use the installed first-class agent/run/lane facilities to start or bind one Codex agent per ready lane. Each agent uses the canonical `next_task`, `inspect_task`, `coordinate_task` and `finish_task` workflow for its actual assigned lane.

In the inspected Project Control protocol, `delegate_task` starts a bounded subordinate local worker; it does not create an independent first-class lane [S15]. It must not be used as a substitute. This package supplies lane identities, scopes, queues and playbooks, but does not invent an unverified command-line `spawn` interface or dispatch anything now. Resolve the exact installed first-class agent binding surface at execution authorization and record the agent-to-run/lane mapping.

## Workspace decision

The native plan uses **contract_split** for the five implementation/verification lanes, **exclusive** for the integrator, and **read_shared** for the coordinator. This is an accepted native-v3 pattern with explicit per-task file ownership [S02]. It avoids making this small epic depend on an unverified new automatic merge scheduler.

The modes express authority and isolation intent; they do not create Git worktrees. The execution owner must prepare/register workspaces using the installed canonical mechanism. In isolated worktree deployments, all post-foundation consumers start from the integrated I02 revision. Do not initialize those worktrees at the pre-foundation revision and then let them silently compile stale headers. The owner should record base commit and interface hash in each lane's context.

Every task's exclusive paths are in the native plan and detailed task sheet. Separate lanes do not edit the shared semantic headers after I02. The native lane owns its implementation, the frontend lane owns relation IR lowering, the algebra lane owns operation classification and contraction semantics, and verification owns its tests. The integrator alone edits root/central CMake and the example integration points. I03's `library/` and `stdlib/` scope is limited by its objective to `.ceh` content/reference disposition, not arbitrary library rewriting.

## Handoff and joins

Core supplies a committed source artifact to I02. I02 verifies and integrates it before releasing downstream agents. Later each lane supplies its concrete revision/patch, tests and source hashes to I04. The coordinator can collect and inspect pending artifacts while a join task is waiting; it must not wait for integration-dependent closure before even examining an artifact.

The native plan does not use `isolated_merge` with an unverified circular `integration_task_id` requirement. If the local execution configuration imposes an integration prerequisite on producer completion, use its canonical artifact-ready/queue-drain protocol before waiting for terminal task states. Do not mark tasks done artificially or edit the database to break a cycle. A needed change of workspace topology is a reviewed plan revision, not a silent executor reinterpretation.

Final integrated tests run after the join. Leaf authors supply real local evidence, while pending integrated tests are explicitly pending. A shared-build conflict or new common contract change returns to its owner; an integrator should not quietly make divergent semantic choices to make a merge compile.

## Hardware and resource coordination

This plan deliberately does not redefine the project's global GPU inventory from stale source. Existing GPU availability and leases must be resolved through the canonical live resource service at execution time. NATIVE validation, baseline GPU tests and final evidence require a real device lease. The named `ce-ss1-accelerator-evidence` lock serializes this epic's final evidence but does not itself exclude unrelated processes or lease a GPU.

For the supplied gate runner, the execution owner sets `CELLERATOR_SS1_GPU_LEASE_RECEIPT` to the actual local lease receipt and configures the leased device mapping, for example through the existing launcher’s `CUDA_VISIBLE_DEVICES`. The file path is provenance, not a claim that an environment variable enforces a lease. A missing/unverifiable resource must block the hardware gate. CPU syntax/test preparation continues where independent.

## Pause, commit and stop behavior

After each completed leaf, commit the scoped changes and evidence and push through the project's configured remote workflow when available. Do not claim a push succeeded if it did not. Preserve unrelated branches/worktrees. An execution instruction to pause takes precedence over “continue until done”: stop new claims/launches, preserve state, and report outstanding work. Finish this epic, publish the candid completion receipt, and stop before any next-epic work.
