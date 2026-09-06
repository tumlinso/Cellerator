# Relation Update Spine compiler boundary

The embedded bridge parses a bounded, straight-line relation closure and lowers
it to the same value-owned calculus and generation effects available to native
C++. It does not compile a complete language translation unit. See the
[architecture](../architecture.qmd), [core execution](../core_execution_cp_math.qmd)
and [biological execution model](../biological_execution_model.qmd) for ownership
and mathematical authority.

## Supported source

The embedding host declares relation/input/cotangent/destination names, output
names, topology, numerical permissions, width and symbolic generations through
`relation_update_source_environment`. These are metadata, not prebuilt native
operations or provider choices. Default output and coefficient names are
`Y`, `dX`, `g`, `delta` and `alpha`; all can be renamed explicitly. No two
bindings may shadow the same symbol.

```cpp
Y = X -[R]-> Genes;
dX = ce::transpose(R, dY);
g = ce::contract_on(R, X, dY);
ce::gradient_step(R, g, alpha);
ce::publish_generation(R);
ce::observe_generation(R);
Y = X -[R]-> Genes;
```

Replace the gradient-step statement with `ce::apply_value_delta(R, delta);`
for caller-supplied delta addition. The complete closure still contains forward,
transpose and scalar edge VJP. Widths N1 and N16 and both explicit VJP operand
profiles are validated by the core calculus. Half-rounded VJP is an explicitly
quantized-operand computation, not the exact derivative of a rounding function.
No normalization or edge-channel expansion is implied.

The source is tokenized using the existing lossless raw tokenizer; forward
syntax uses the existing relation parser. Bounded call parsing checks complete
statements and declared symbol roles. Source byte ranges and stage identities
remain provenance. Unknown operations, undeclared outputs, flipped operands,
shadowed symbols, numeric mismatches, stale generation transitions and malformed
order fail with diagnostics. `alpha` names a launch-time scalar supplied by the
caller, not a parsed arbitrary expression.

## Lowering and execution

`lower_relation_update_source_slice_v1` returns canonical `semantic`, `effects`
and validated compiler `program` records. The realization overload
`lower_relation_update(source, out)` revalidates all three so modifying a
semantic field or effect cannot retain a stale successful-lowering flag. The
native overload takes canonical calculus and effects directly, without compiler
IR construction. Both produce the same `lowered_relation_update` binding recipe.

Update and publication remain separate mathematical effects. The current core
`enqueue_value_update` submits the mutation and then records real readiness, so
one recipe action implements the adjacent update/publication pair. There is no
fabricated publication kernel and no reuse of `publish_values` for an in-place
update: that existing entry point is for initial/replacement values.

The optional observation must immediately follow its publication, at most once
per publication in this recipe. It maps to `begin_value_read`; after consuming
the const physical view the embedding caller must call `end_value_read` on the
same consumer stream before another mutation. Repeated or displaced observation
is rejected by realization, not silently moved. A publication is an enqueued
readiness promise, not host evidence of completed device execution.

`enqueue_relation_action` is a stateless binding adapter to the existing core
`enqueue`, `enqueue_edge_gradient`, `enqueue_value_update` and `begin_value_read`
entry points. The caller executes the ordered recipe, checks each status and
returns its lease. It owns no numerical callback, scheduler, prepared-training
object, stream or values. Preparation uses the same `prepare_relation_pair` and
`prepare_relation_gradient` APIs as native requests. Device pointers, operand
versions, gradient stamps, scalar coefficients and streams remain launch facts;
physical order and provider legality remain core validation responsibilities.

## Explicit limits

The slice rejects full `.cell`/C++ translation units, imports, cross-TU work,
declarations/initializers, namespaces, macros, comments, general expressions,
support filters, implicit outputs, arbitrary calls, relation chains, implicit
canonicalization and accumulation syntax. The bounded source closure contains
one update/publication cycle; a subsequent forward demonstrates new-generation
reuse. General loops, optimizers, topology mutation and compiler-owned training
execution are absent. Full mutable CUDA Graph capture, multi-reader leases,
multiple devices and concurrent relation execution remain outside this bridge.

The general language compiler, imports and cross-TU/IR optimization are deferred
work. Their absence is not disguised by attaching a compiler-origin flag to a
native operation.

## Conformance and integration handoff

`source_bridge_test.cc` checks independent native equivalence for both update
forms and VJP policies, renamed symbols, copy/move safety, precise diagnostic
ranges and source rejection. `dual_origin_test.cc` checks fieldwise recipe
equivalence, generation dependencies, update/publication fusion, observation and
semantic/effect tampering. `compiler_conformance_test.cc` checks N1/N16, both
numeric profiles, nonconsecutive generations, illegal stage kinds/dependencies,
axis/edge-order mismatches and unsupported operations. The conformance binary
was run with host AddressSanitizer and UndefinedBehaviorSanitizer; this is not
Compute Sanitizer or real GPU acceptance.

F01-F04 host evidence establishes parsing, semantic lowering and recipe behavior.
Actual V100 provider attribution, numerical execution, readiness timing,
Compute Sanitizer, WMMA/hybrid execution and native/source device convergence
remain required in integrated I05/I08 acceptance. They cannot be inferred from
these host tests or from inline core-call names.

The old `compare_gradient_program_with_training_v2` and
`lower_gradient_publication_stage_kind_v1` mappings were removed. Their only
code callers found by the frontend reference audit are in
`tests/compiler/semantic_ir/implement_gradient_and_publication_operations_test.cc`,
whose explicit target is in that directory's `CMakeLists.txt`. Integration must
migrate that target to current conformance before the full build: the old test
asserts the removed training authority and publication alias. Useful closure,
identity, numerical and invalid-order checks are preserved and strengthened in
the new suite. No wider training implementation or legacy kernel was removed.
