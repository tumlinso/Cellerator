# Experimental runtime direction

## Purpose

Explore an aggressive simplification of Cellerator into a fast, maintainable
biological numerical runtime. This folder records intent; it does not establish
an implementation, API, package, build component, or performance result.

The design should start from biological numerical workloads and the
infrastructure already available. Reuse existing execution, memory, and
framework facilities when they satisfy the need. Add or reshape Cellerator
mechanisms only where evidence shows that the existing facilities leave a
concrete gap. Avoid preserving complexity without a demonstrated user or
workload need.

## Responsibility boundary

PyTorch is the framework layer for orchestration, scheduling, automatic
differentiation, and graph execution. Cellerator remains responsible for
Torch-independent native kernels, arbitrary internal representations suited
to the operation, and optimizations that exploit known biological structure.
Native numerical code must not depend on Torch or assume that every internal
representation is a tensor.

The intended relationship is a thin framework-facing path into reusable native
operations. Where a native operation needs derivatives, its derivative
semantics should be made available to PyTorch's differentiation machinery,
with gradient ownership kept in the framework layer.
Which operations need native derivative rules, and at what granularity, remains
an experimental question.

This direction is separate from the current optional `python/cellerator/torch`
and `bindings/torch` adapters, and from the former `components/CelleraTorch`
component. Those existing and historical paths remain evidence about prior
interfaces and behavior, not an implementation mandate for this workspace.

## Contracts to resolve through evidence

Any future framework-to-native boundary will need clear ownership and lifetime
rules for data and prepared structures, plus explicit stream and device
coordination. Structure identity and mutable numerical values should remain
distinct: structural epochs describe reusable layout or topology, while value
generations describe changing parameters or data bound to that structure.
Changing values should not silently imply that the structure changed.

The right boundary may differ by operation. Keep granularity open between
individual kernels and larger biological operations; decide from measured full
costs, including preparation, conversion, allocation, memory traffic, and
launch or scheduling overhead. Preserve useful native consumers that do not
use Torch.

## Evidence expected before conclusions

Compare candidate paths against an independent numerical reference and the
existing native or upstream behavior for representative biological cases.
Check repeated value updates over reused structures, derivative behavior,
ownership and stream correctness, and actual framework graph execution where
relevant. Measure complete workload cost on the assigned resources before
making speed claims. Record unsupported regimes and limits alongside results.

No implementation, benchmark, or biological validation is implied by this
design note.
