# Demonstration and claims

## GlassHelix example

The post-epic `mechanistic_toolbox.cc` builds four continuous state quantities, one joint three-input nonlinear mechanism and a supplied forcing. Two parameter instances share definition/support while retaining independent values. Their a+b combination creates an exact known indistinguishability for the selected outputs. Width 33 rejects a permanent N16 ontology.

The program executes RK4, observes one quantity, compares against an independent scalar oracle, evaluates parameter JVP/VJP and a local numerical null space, updates one instance without contaminating another, compares exact masked and compacted support realizations, and reweights two supplied hidden-state alternatives after an observation. Default execution requires CUDA; explicit `--cpu` is a separate real production backend case.

The Python example expresses the same system through the native binding. It is not an independent Python numerical implementation. API spelling is prospective. Implementers can revise it through the contract owners and update both examples together; all semantic assertions and qualification obligations remain.

## Cellerator separate consumer

`combinatorial_consumer.cc` links Cellerator alone. It evaluates the three-input expression for 33 instances with multiple gain bindings and verifies a nonzero parameter derivative at zero gain. One structural preparation and zero hot allocations are required. This is a reusable numerical boundary witness, not a bespoke GH-only kernel.

## What has been checked at delivery

The separate scalar reference executable is compiled and run locally. Both normal example translation units receive declaration-only syntax checks against guarded planning headers. Normal package configuration is expected to fail without the future real libraries. Local evidence records actual outcomes. No linked Cellerator/GH, CUDA, Python binding or sanitizer result is claimed during package construction.

## What completes this evidence

Future agents implement and link actual targets. CTest registers exact CPU and CUDA demo tests. Independent qualification checks no source-level include of `.cu`, no fallback, correct providers and value lifetimes. Gate receipts retain source, binary, tool, build and device identities. Demonstration mathematics is not new biological evidence.
