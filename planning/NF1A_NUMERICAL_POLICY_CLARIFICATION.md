# NF1A numerical representation clarification

User instruction dated 2026-09-16. This clarification governs NF1A implementation and qualifies any inherited wording about FP32 authority or RNE half projections. Read it before the package's START_CONTROLLER handoff. It does not authorize implementation; bootstrap-only remains in effect until the user explicitly permits implementation.

NF1A establishes a working mixed-precision execution foundation while keeping biological meaning independent of the numeric encoding used for storage or execution. Numerical representation is an explicit Cellerator policy/capability, not an intrinsic property of biological state.

FP32 may remain the reliable baseline and promotion type. FP16 must be usable as a first-class storage/execution representation where appropriate, especially for normalized biological data. Accumulation and numerically sensitive operations may promote to FP32 as needed. Preserve valid existing numerical behavior without treating a current FP32 owner plus half projection as the permanent biological storage model.

Downstream APIs, provenance, state identity, parameter identity, prepared-program contracts and GlassHelix scientific objects must not assume biological values are permanently stored as IEEE floats. Keep scientific identity separate from encoding; record the applicable representation and conversion/promotion policy explicitly enough for capability checking and execution provenance. Scientific identity independence does not make execution/cache compatibility or numerical equivalence automatic across representations.

Keep the existing numerical/storage boundary open to additional Cellerator policies such as scaled integers, per-feature affine quantization, block-floating/shared-exponent data and delta encodings, without changing GlassHelix scientific semantics or requiring major Cellerator redesign. Use only the minimal policy/metadata distinctions needed by present NF1A capabilities; do not introduce speculative framework abstractions.

Do not add a quantization subsystem, block-floating implementation, per-gene scaling scheme, custom mantissa/exponent representation or similar work during NF1A unless strictly necessary for an existing NF1A requirement. This clarification adds no implementation workstream or successor task and does not expand the current epic.

The supplied planning/nf1-adaptive-v1 package and its checksums remain intact. This adjacent user clarification takes precedence where its policy differs from inherited implementation-specific wording.
