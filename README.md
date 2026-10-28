# Cellerator

**What if the organization of a biological system could determine the program that computes it?**

Cellerator is intended to become a **broad, high-performance computational framework for biological single-cell modeling**, providing reusable mathematical and execution primitives to represent, analyze and simulate biological systems.

The name combines “cell” and “accelerator.” There is also a “cell-torch” joke hiding in it: `cell-(a)-(ra)-tor(ch)`.

AnnData is the standard Python data interface. In the intended workflow, users call RAPIDS-singlecell for upstream analysis, then bring their AnnData objects to Cellerator for modeling and simulation. RAPIDS-singlecell is an upstream workflow tool, not a Cellerator dependency. Direct AnnData preparation remains to be implemented; see the [Python binding guide](docs/development/python-bindings.md).

A cell's measurements may arrive as a large matrix, but that matrix is not the whole computational story. The same relations can recur across cells and time points; topology may remain useful while expression, activity and learned weights change. Cellerator explores whether those regularities can be discovered once and compiled into work that an accelerator performs well.

One architectural direction is a **compiler and execution substrate for biologically structured computation**. It explores how useful structure can become execution order, physical layouts and generated or assembled kernels, alongside conventional dense and sparse computation and specialized biological kernels. The central question is how discovering and organizing structure can let us avoid unnecessary computation. Learned packing and gated tile execution are exploratory directions within the broader framework.

## The design in one pass

```text
system structure and workload
    → discover reusable organization
    → compile geometry and executable alternatives
    → bind current values and run
```

Structure and value have different lifetimes. A new cell state need not force the system to rediscover a reusable relation. The longer-term design also allows state-dependent gates to activate or reorganize prepared computation. Those gates are discovered before execution; their outcomes depend on runtime state. The full adaptive system is an intent, not a blanket claim about every current path.

Biology and hardware are considered together: repeated supports, order and modular structure matter when they can become cheaper memory access, reusable work, or a better kernel arrangement. A vendor library remains the right answer when it wins the complete comparison.

## What exists now

The repository now contains a host-first compiler/SDK layer as well as the CUDA numerical execution substrate, typed biological identities, geometry preparation, and bounded source/native relation-update paths. The Python binding architecture is one `cellerator` package, with optional Torch integration under `cellerator.torch`. Historical backends remain for evidence and compatibility. These components do not prove that the whole target language, general learning path or adaptive execution model is finished.

## Evidence, not just a design

The dated [implementation record](docs/current_implementation.qmd) describes
the historical N=64 exact-cover study, its source and limits. It is not a
universal layout or biological-performance claim.

## Where to start

Start with the [architecture](docs/architecture.qmd), then use the
[developer reference](docs/developer_reference.qmd) for build and source entry
points. The [current implementation record](docs/current_implementation.qmd)
separates dated implementation evidence from longer-term design intent. Python
and optional Torch setup is documented in the [binding guide](docs/development/python-bindings.md).

[CelleraTorch](CelleraTorch/README.md) is a separate experimental workspace
inside this repository for exploring a smaller biological numerical runtime.
Its [design notes](CelleraTorch/docs/design.md) describe PyTorch-owned
orchestration, scheduling, autograd and graph execution around Torch-independent
native kernels and structure-aware representations. Only documentation has
been added; implementation is on hold. The folder shares this repository's Git
history and is outside its component, build and package registration.

Cellerator provides the numerical execution beneath [GlassHelix](https://github.com/tumlinso/GlassHelix)'s questions about dynamics and mechanism. [Baseplane](https://github.com/tumlinso/Baseplane) develops sequence-grounded representations and shares the numerical boundary rather than becoming a second general runtime. See [Baseplane integration](docs/baseplane_integration.qmd) and [storage and framework boundaries](docs/storage_distribution_and_interop.qmd).

This is active research software. Its ambition is broader than its measured execution envelopes.
