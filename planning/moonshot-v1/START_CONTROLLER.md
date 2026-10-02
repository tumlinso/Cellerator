# Controller handoff: build the single-cell moonshot

You are the implementation owner for a paired Cellerator/GlassHelix experimental program. The user wants unusual, elegant numerical mechanisms, not another documentation-only planning cycle. Start from the supplied code and make useful prototypes callable. Do not rerun the entire research survey.

## Preserve the actual objective

Read REFERENCE_INTENT.md, then DESIGN.md. The examples are not fixed ontology or ABI requirements. Be willing to replace a struct, tile shape or factorization with a better implementation. Tensor Cores are a first-class design resource; use the rest of Volta wherever it helps. Do not fall back to the claim that one cell is necessarily an unacceleratable vector. Distinguish reshaping data from choosing or exploiting a matrix-native operator.

Both a plain learned predictor and an externally refactorable hidden-state model are required consumers. Local hidden bases remain private; extraction acts on supplied computational structure rather than globally canonical columns.

## First bounded action

Refresh each registered repository HEAD, dirty work, AGENTS/instructions, relevant current tasks/claims and source interfaces. The observation snapshot is a precondition hint, not current authority. CE and GH are separate authorities. Read the active BIO/DESIGN/LEARN scopes before taking integration writes.

Run package integrity, preview installation, then use the existing registered Project Control launcher. The following plan commands were observed in the live repository instructions; resolve the actual installed executable and environment rather than substituting an arbitrary toolchain:

```sh
project-control plan validate --project cellerator --file <CE_ROOT>/planning/moonshot-v1/machine/cellerator.todo-plan.json
project-control plan validate --project glasshelix --file <GH_ROOT>/planning/moonshot-v1/machine/glasshelix.todo-plan.json
```

CE preview failed during package preparation with `bounded_read_failed`; obtain a successful CE-target preview before importing. Do not work around a failed authority read by editing Todo storage.

After reviewing fresh additive diffs and reconciling applicable owners, use the same front door:

```sh
project-control plan apply --project cellerator --file <CE_ROOT>/planning/moonshot-v1/machine/cellerator.todo-plan.json
project-control plan apply --project glasshelix --file <GH_ROOT>/planning/moonshot-v1/machine/glasshelix.todo-plan.json
```

These are future controller actions, not actions performed by this archive. Do not replay these files after tasks have acquired new runtime bindings. Amend through the supported lifecycle.

## Work style

Use the coarse tasks and briefs. One active writable lane per repository is the default. Configured Codex subagents can handle a kernel, model witness or isolated review when useful; the owner retains task acceptance. Do not use the read-only local investigator as an implementation agent. Do not spend the sprint creating an army of overlapping TODOs.

Stage the matching payload into a new claimed experimental directory using `bootstrap/stage_source.py`. The code is deliberately simple and self-contained for transfer. Adapt it to existing CE interfaces before public exposure; it is not a mandate to maintain duplicate abstractions.

## First useful implementation

Make M01 matrix patches and M04 quad micro-MMA callable on V100, with M06 as a contrasting ordinary-ALU/JVP primitive. The raw-MMA layout is uncompiled: compile and run its included tiny mapping smoke before trusting it. Use the existing CUDA12 toolchain, not CUDA13.

Then implement M05's private-hidden-state message interface and the chosen first-order derivative routes through current parameter/tape ownership. GH can prototype ordinary Torch models concurrently while waiting for the CE capability receipt; it must not advertise native execution before it uses it.

Next demonstrate M08 using a supplied exact map and live residual regrowth. Do not turn this into an unsolved automatic scientific-discovery prerequisite. Extend to nonlinear approximate refactors only with an explicit approximation label.

M03 exact local flows and M10 delta ledgers follow naturally. Choose at least one additional unusual card to make concrete; favor a different computational mechanism, not a superficial variant of the same matmul.

## Minimal checks, not a qualification campaign

Use a small same-policy numerical witness, one gradient/direction check where implemented, invalid-shape/alias/generation admission, and a runnable consumer. Preserve existing tests affected by integration. A sanitizer run is warranted for a suspicious memory/concurrency issue, not as an excuse for endless validation before the first API works.

Do not require a real biological dataset, a speedup, broad autotuning or publication-quality plots for prototype completion. Do not mark an uncompiled kernel validated. Performance tables may say `not_measured`. An engineering hypothesis may remain anonymous and scientifically unqualified.

If a branch is unproductive, keep its reference and an explicit experimental selector or record a bounded rejection reason. Do not collapse all fourteen ideas into a new universal framework. Build a few strong, distinct mechanisms.

## Publish

CE publishes a capability receipt with exact source, callable path, precision, supported shape and derivatives. GH consumes it and publishes its model/evidence role. Reconcile old ML2 tasks only where replacement evidence satisfies their actual scope; leave data/science qualification separate.

Finish with working code, concise usage examples and a clear unsupported list. The point is not to prove everything. It is to leave the user with interesting, mathematically coherent systems they can actually work with.
