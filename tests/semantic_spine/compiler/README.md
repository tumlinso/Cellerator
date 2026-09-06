# Bounded source-origin conformance

Run from any directory:

```sh
python3 tests/semantic_spine/compiler/run_tests.py --receipt /tmp/compiler-validation.json
```

The harness compiles existing parsers, numerical Sema, the narrow binding
adapter and existing relation IR lowering with assertions enabled and CUDA
12.9 headers. It uses all available host processors and runs four executables.
It does not run GPU kernels. Integration must compare these source-origin
canonical descriptors with independent C++ descriptors and execute them
through the same native prepared relation pair.

`source_slice_test.cc` supplies a complete named metadata environment and
forward/transpose source examples. `diagnostic_test.cc` checks semantic loss
and source diagnostics. `lowering_test.cc` checks direct IR-to-C++ descriptor
parity; the retained existing relation IR test remains part of the harness.
A receipt records commands, source hashes, compiler identity and observed
results. Source provenance remains separate from canonical mathematical
identity; changing a launch's value generation does not require reparsing.

Only plain domain/axis/relation/state declarations and one assigned relation
application are supported by the adapter. Runtime bindings provide exact
identity, order, extent and numerical metadata. Undeclared or ambiguous names,
unsupported declarations, qualifiers, filters, expression chains and trailing
statements are rejected. Full `.cell` execution and installed SDK completion
remain deferred. See the authoritative [architecture](../../../docs/architecture.qmd)
and [Semantic Spine contract](../../../planning/semantic-spine-v1/02_SEMANTIC_AND_NATIVE_CONTRACT.md).

F05 additionally checks that canonical arithmetic and retained v2 operation/algebra
transport agree, including distinct relation/input/output precision and nonfinite
handling. `lowered_relation_apply_v1::transport_status` reports whether the legacy
transport exists. A successful canonical lowering with restricted FMA or
reassociation permissions retains that mathematical descriptor and reports
`unsupported_arithmetic_policy`; its legacy operation and algebra carry invalid
schema/kind and no bindings. Consumers must use the canonical descriptor or test
transport availability. Canonical validity does not promise a native candidate.
