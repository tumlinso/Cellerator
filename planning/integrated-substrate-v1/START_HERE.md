# Integrated Substrate Bootstrap — Cellerator · Baseplane · GlassHelix

**IS1 · 2 October 2026 · successor package, not an applied plan**

Turn the finished predecessor work into coherent tools without making future invention harder. Cellerator is the reusable mathematical/dynamical library. GlassHelix is the scientific experiment using it. Baseplane is the sequence-grounded representation/query layer with its own exact core and shared Cellerator numerical mechanisms.

## What is here

Three native schema-3 plan files, **45 substantive implementation/integration outcomes plus three closure epics**, 35 lane definitions, explicit intermediate source handoffs and cross-authority provider gates. After the three foundations, 17 independent domains are available; actual concurrency follows a bounded agent/resource budget.

The design includes pluggable packing beyond Cellpack, integration of existing mathematical/state/response/runtime work, an acyclic installed-component graph, usable Baseplane sequence tools, and a coherent GH scientific experiment surface. Twelve precise new mathematical/computational cards accompany the earlier portfolios. A 92-entry seed ledger covers all 48 Baseplane cards, 13 single-cell mathematical cards, five qualification obligations and other predecessor/frontier responsibilities. A separate map retains all 33 historical BitOp disposition entries. Activation must expand the inventory for work completed after this package was written.

Executable seeds include independent Python packing strategies and user-supplied injection, zero-safe contribution gradients, effect composition, hybrid finite/continuous maps, quadratic response/delta rules, remembered-transmission behavior, multilevel application and Schur port solves. A C++17 cold-strategy contract and a small authored SM70 kernel illustrate integration seams. These are semantic scaffolds, not a replacement runtime or a claim of native repository integration.

## Reading path

Read [the design overview](DESIGN_OVERVIEW.md) for the architectural decisions, or [START_CONTROLLER.md](START_CONTROLLER.md) for the implementing root. Architecture and packing are in [the architecture](docs/02_ARCHITECTURE.md) and [packing design](docs/03_PACKING.md); concrete new math is in [the frontier cards](docs/05_MATH_FRONTIER.md). [Parallel execution](docs/07_PARALLEL_EXECUTION.md) explains the barrier, budgets, worktrees and provider receipts. [The task map](docs/08_TASK_MAP.md) links every outcome and the dependency graph. Bounded inspected source and technical references live in `evidence/`.

## Local checks

From the unpacked package directory:

```sh
python3 tools/validate_package.py --verify-manifest
PYTHONPATH=seed/python python3 -m unittest discover -s tests -v
PYTHONPATH=seed/python python3 seed/python/demo.py
c++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
    seed/cpp/packing_conformance.cc -o /tmp/is1_packing_conformance
/tmp/is1_packing_conformance
```

The mathematical tests need NumPy. The package checks and activation tools use the Python standard library apart from the explicitly installed Todo read-only provider. `evidence/validation.json` records the actual authoring checks, including what was **not** tested. No CUDA compiler was available in the authoring container, so the CUDA seed has not been compiled or executed there. Its optional compile-only command is:

```sh
python3 tools/compile_cuda.py --nvcc /explicit/cuda-12.x/bin/nvcc --out /explicit/output
```

This does not choose or launch a GPU. Native/GPU/framework/scientific acceptance is work for the implementing agents using the actual repositories and resource leases.

The authoring checks passed **52 Python tests**, the strict C++17 conformance build/run, a C++ undefined-behavior-sanitizer build/run, Python syntax checks and static package/DAG validation. The local semantic demonstration also ran. These results do not qualify native Todo plans, CUDA execution, framework integration or biological science; see [the exact validation record](evidence/validation.json).

## Installation and execution boundary

Do not copy planning commits into live provider repos while the current source-bound predecessor checks are running. Wait for the current programme barrier. Then preview create-only staging:

```sh
python3 tools/stage.py --repo /explicit/path/to/Cellerator --project cellerator
python3 tools/stage.py --repo /explicit/path/to/Baseplane --project baseplane
python3 tools/stage.py --repo /explicit/path/to/GlassHelix --project glasshelix
```

Add `--apply` only after reviewing the destinations. This creates a new `planning/integrated-substrate-v1` directory and refuses overwrite. It neither imports a plan nor starts work. The root then supplies current ignored configuration, checks live prerequisites, natively previews/validates the plans and uses installed Project Control to adopt them. See `START_CONTROLLER.md`.

Do not treat the JSON plans or source snapshots as current authority. Do not hand-edit Todo databases or generated views. Backward compatibility is not the objective; useful behavior, scientific/numerical meaning, real consumers, future extension and performance are.
