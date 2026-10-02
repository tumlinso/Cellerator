# Single-cell moonshot: Cellerator × GlassHelix

**Research and implementation bootstrap · 2 October 2026 · experimental, not performance-qualified**

The target is the useful mathematics of **one evolving cell**: ordinary learned inference, hidden-state evolution, forward/backward, selected responses, and increasingly explicit dynamical structure. Tensor Cores are a design input, not a batching afterthought. The entire Volta machine is available to the design. Biological interpretations remain the model owner's responsibility.

The strongest proposal is to compile a cell into **small compute regions with explicit interfaces**. Different regions can use matrix-algebra models, local hidden-state messages, process packets, or an exact sparse fallback. Do not force all models into one representation.

## Start here

- [DESIGN.md](DESIGN.md): the synthesis and recommended implementation order.
- [MECHANISMS.md](MECHANISMS.md): fourteen concrete mechanism cards, including algebra, layout, derivatives, and limits.
- [START_CONTROLLER.md](START_CONTROLLER.md): the implementing agent's handoff.
- [INTEGRATION.md](INTEGRATION.md): actual library seams and payload-to-source map.
- [RECONCILIATION.md](RECONCILIATION.md): outstanding TODOs, preservation, integration and conditional supersession.
- [research/PROJECT_REVIEW.md](research/PROJECT_REVIEW.md): fresh source/instruction review.
- [research/SOURCES.md](research/SOURCES.md): primary research and hardware sources.
- [VALIDATION.md](VALIDATION.md): exactly what was and was not executed.

`machine/` contains **separate schema-v3 plans** for the two existing Todo authorities, plus an observation snapshot and reconciliation request. `payload/` contains actual CPU references, Torch prototypes, C++ representation seeds, three CUDA source prototypes, and lightweight checks. The CUDA sources are **not compiled or run in this delivery**.

## The first three bets

1. **Matrix patches and matrix flows:** arrange existing scalar coordinates into patches and use a structured operator such as `L X R`, `L X + X R`, or `X M X`. A genuine matrix computation within one cell, with no invented batch. Choosing this family changes the model unless its factorization was already supplied.
2. **Four independent micro-MMAs per Volta warp:** use the documented `m8n8k4` layout to group four small contractions, with distinct parameters, from the same cell.
3. **Local hidden coordinates, explicit message interfaces:** each actor keeps its own basis. Small encoders/decoders expose a computational interface for exchange, without canonicalizing latent biology.

Then add the learning loop: externally approved extraction, atomic state/parameter remapping, and immediate residual-capacity reuse. Keep delta execution, implicit feedback islands, texture-lookup rates and precision expansion as buildable additional experiments, not requirements that delay the first useful model.

## Installation is inert by default

```sh
python3 bootstrap/check_package.py
python3 bootstrap/install.py --cellerator /actual/Cellerator --glasshelix /actual/GlassHelix
```

The second command previews only. Add `--apply` to copy this package into **new** `planning/moonshot-v1` directories. It does not import Todo plans, edit library source, create agents, reset queues, or replace existing files. A nonmatching source revision requires an explicit `--accept-source-drift` after review. Read the controller before using the repository's existing, verified plan front door.

No live repository or persistent Todo authority was modified to produce this delivery.
