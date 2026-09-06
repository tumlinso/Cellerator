# Semantic Spine v1 algebra conformance

The bounded algebra repair distinguishes scalar support dots from edge-channel
products and removes executable substitutes from source classification. The
architectural authority remains [the architecture spine](../architecture.qmd)
and the epic's [semantic contract](../../planning/semantic-spine-v1/02_SEMANTIC_AND_NATIVE_CONTRACT.md).

## Counterexamples and disposition

| Previous behavior | Counterexample | Accepted behavior |
| --- | --- | --- |
| An ambiguous support contraction accepted disjoint output concatenation. | At K=2, scalar partial dots sum to -9; at K=17 they sum to 2261. Concatenating partial dots changes the output shape and meaning. | Explicit scalar-dot result requires partial summation and partial-algebra flags. Explicit edge-channel result requires disjoint channel assembly and preserves K outputs. Unspecified result/assembly is rejected. |
| The host scalar contraction accepted a nonzero neutral value. | Adding an offset independently to each panel changes the result with the number of panels. | Scalar-dot neutral value must be zero; rejected calls leave output untouched. |
| Chain, moments, hierarchy, exchange, gradient and publication carried representative primitive opcodes plus a boolean. | Ignoring the boolean could turn exchange into sparse update or gradient into transpose. | A variant holds exactly primitive, composition or effect meaning. The optional primitive accessor returns no opcode for all five compositions and the publication effect. |
| Graph chain/incidence lowering fabricated normalization/contraction compositions; unknown kinds became an epilogue. | A general chain is not a normalization followed by apply; incidence pooling is not a contraction followed by a segment operation. | Lowering returns an optional composition: unsupported cases return no result. Source graph meaning stays available. Unknown graph kinds fail validation. |

The 14-family coverage test checks eight actual primitives, five explicit
compositions and one publication effect. Bundle, paired-moments and typed-exchange
graph kinds retain their exact composition classifications. Classification is not
proof that a corresponding accelerator lowering exists. Apply and transpose
primitive resolution remains usable.

## Validation

All six focused and affected existing tests passed with C++17 and assertions
enabled. [The machine-readable receipt](algebra_validation.json) records exact
commands, compiler, outputs and unchanged foundational header hashes. The support
test independently evaluates whole and panel-wise scalar contractions through the
existing IR interpreter, sums real partial values, computes channel panel products,
checks shape, rejects overlapping panels and checks the precise failure code for
scalar concatenation. Small integer inputs make these double results exact;
general floating-point reassociation remains a separate numerical-policy question.

No contraction GPU provider was integrated, deleted or modified. These are host
semantic tests, not accelerator or performance evidence. The native and final
integration lanes remain responsible for the epic's CUDA 12.9/sm70 acceptance.

## Migration and bounded remainder

The old support-embedding test now declares edge-channel shape and concatenation
explicitly. Resolution consumers use the optional primitive accessor and the
typed variant. Graph lowering callers must handle an empty optional. These are
intentional corrections to bad contracts, with no operation opcode renumbering
and no replacement operation-core schema. Project Control revision 6812 added
only the required matching declaration and two legacy tests to A03 ownership.

The frozen relation/prepared-pair interface is byte-for-byte unchanged from the
integrated foundation. Remaining contraction GPU portfolio integration, general
chain/hierarchy/gradient execution, exchange optimization, effect scheduling,
full source-language execution, planner expansion and SDK work are deferred.
Nothing in this lane activates those programs.
