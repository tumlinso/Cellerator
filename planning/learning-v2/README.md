# GlassHelix ML2 — biology-purpose learning bootstrap

**Six GlassHelix outcomes; two Cellerator outcomes; separate schema-v3 Todo authorities.**

This replaces the earlier *proposed* L1 architecture. It does not mutate, retire or reapply any imported L1/NF1A work. GlassHelix is a scientific model/inference library using **PyTorch + CelleraTorch**. Cellerator remains the native biology-aware programming, compiler and execution system. The first CellTag client remains independent of the core libraries.

The package contains a worked biological-learning construction, rather than an instruction to invent an unspecified ML kernel. It also addresses an actual integration gap: the inspected CelleraTorch adapter performs a combined native backward/update and does not expose native-parameter gradients to an ordinary Torch optimizer. See `SOURCE_REVIEW.md` and `TORCH_BRIDGE.md`.

## Adopt

Copy this `learning-v2` directory to both `<GlassHelix>/planning/` and `<Cellerator>/planning/`. The same documents support both authorities; each imports **only its own** plan.

From either copy, check archive integrity first:

```sh
python3 planning/learning-v2/scripts/check_package.py
```

Then use the existing verified Project Control launcher, validate the actual targets and inspect the fresh additive diffs:

```sh
project-control plan validate --project glasshelix --file <GlassHelix>/planning/learning-v2/machine/glasshelix.todo-plan.json
project-control plan validate --project cellerator --file <Cellerator>/planning/learning-v2/machine/cellerator.todo-plan.json
```

After review, apply through the same canonical front door:

```sh
project-control plan apply --project glasshelix --file <GlassHelix>/planning/learning-v2/machine/glasshelix.todo-plan.json
project-control plan apply --project cellerator --file <Cellerator>/planning/learning-v2/machine/cellerator.todo-plan.json
```

Replace angle-bracket paths. These are instructions, not actions taken by this archive. Give the main Codex agent `START_CONTROLLER.md`; focus runs are `GH-ML2-RUN-V1` and `CE-ML2-RUN-V1`. Native implementation is optional where reuse suffices; **the CT/CE integration outcome is not an optional afterthought**.

## Outcomes

| GH | Result |
|---|---|
| DESIGN | One evidence-qualified biological learning formulation and executable reference |
| DATA | Independent client, source joins and actual evidence coverage |
| REPAIR | Two retained native inference correctness fixes |
| LEARN | Scientific fitting/conditioning/results over Torch/CT, connected to NF1A |
| PILOT | One bounded experimental arm and same-model computation comparison |
| ACCEPT | Installed integration and honest scientific/performance handoff |

| CE | Result |
|---|---|
| TRAIN | Composable native-parameter gradients, safe optimizer publication and actual Torch access |
| BIO | One structured-learning construction realized or reused and assessed, not a kernel quota |

Each authority also has a closure-only epic. One writable implementer lane per authority is the cheap default. Bounded children and justified plan amendments remain available; no mandatory army of agents or inspect/build/test/retry microtasks.

## Read in layers

Start with `START_CONTROLLER.md` and `DESIGN.md`. The difficult reasoning is in `BIOLOGICAL_ML.md` and `TORCH_BRIDGE.md`; use those at the corresponding tasks. `EXPERIMENT.md` governs the research client. `ACCEPTANCE.md`, `CROSS_PROJECT.md`, `SOURCE_REVIEW.md` and `SOURCES.md` provide exact limits and re-fetch locations. The reference witness is an algebra test, not a native implementation.

`validation/STATUS.md` records the actual checks performed on this delivery. A package check is not a product, GPU or biological test. Revalidate live previews before applying; after adoption, use supported amendments instead of reapplying the original files and overwriting newly bound gates.

## Preparation-time validation

GH passed a live target-specific native preview. The CE payload passed the shared canonical schema validator, but the actual CE preview returned `bounded_read_failed`. A successful CE-target validation is required before its apply; never apply it to GH. See `validation/STATUS.md`.
