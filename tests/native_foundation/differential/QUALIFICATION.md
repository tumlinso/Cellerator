# Local arithmetic and derivative actions

D01's initial vocabulary is elementwise add, multiply and tanh, uniform host
FP32 or FP64, equal extents, overwrite destinations and no output/input alias.
There is no implicit broadcasting, reduction, quantization, masking, saturation,
CUDA execution or second derivative. Empty batches are accepted. Repeated input
arguments are legal; VJP returns separate contributions, which the caller must
explicitly add for a shared argument. Output contributions cannot alias.

The sole primal implementation is `native_numeric/local_arithmetic.cc`.
`differential/local_arithmetic.cc` adds mathematical JVP/VJP actions, calling
that primal owner for tanh values. The tanh slope uses `1 - y*y`; near saturation
this has ordinary cancellation/rounding limits and can become zero before the
exact real derivative does. No derivative through float storage rounding is
claimed. NaNs/infinities propagate through arithmetic; zero times infinity is
not silently erased. Host rounding must be nearest-even. Fast-math is unsupported.

`make_local_block` validates the existing NF1 contract and registers only
requested supported callbacks. The existing `bind_compiled_stage` and
`execute_prepared_program_v2` execute them. There is no graph evaluator, registry,
allocator or session here. Contracts and payloads are borrowed and immutable
through calls. Each local action validates shapes/overlap before writing its
outputs. Global program preflight, generation tracking and session lifetime
remain their existing owners; these stateless local functions do not certify
that a caller-supplied primal is current. Fusion/result reuse flags remain
conservative rather than claiming provenance validation.

`ce_nf1_d01` runs real compositions of primitive stages for
`tanh(x*p+c*c)` at width 33 in FP32 and FP64. Independently evaluated double
finite differences check state and parameter directions separately, and state,
parameter and repeated-argument adjoints. JVP/VJP duality is checked as a scalar
inner-product identity. Tolerances are `2e-6*(1+abs(expected))` for FP32 and
`1e-9*(1+abs(expected))` for FP64. Negative controls include shape, partial-range
alias, shared adjoint outputs, unsupported policy/opcode, and forward-only
capability rejection without changing the destination stage.

Standalone configuration source-links actual owners for the isolated lane.
M20 must add `native_numeric/local_arithmetic.cc` to the existing
`cellerator_native_numeric` target, then add the differential owner subdirectory.
The test fragment automatically links `Cellerator::local_differential` when
available; it does not define a competing primal target. Compile these owners
without fast-math and with contraction disabled for this qualified profile.

```sh
cmake -S tests/native_foundation/differential -B /tmp/ce-d01 -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/ce-d01 -j 4
ctest --test-dir /tmp/ce-d01 --output-on-failure --no-tests=error
```

Four jobs bound this small host build while other first-class lanes use the
shared host. No GPU or performance evidence is claimed.
