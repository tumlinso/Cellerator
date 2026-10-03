# Guarded SM70 scaled tanh

This bounded lowering seam implements `z = state * parameters; y = tanh(z)`.
It preserves ordered coordinates, live trainable parameters and the saved
intermediate `z`. The fused provider saves one launch and the read of `z`;
it still writes `z` for the response owner. It allocates no memory.

The preparation guard contains extent, support generation and arithmetic
policy. A changed extent or support generation chooses the two-kernel direct
route. A changed arithmetic policy is rejected and requires preparation. No
parameter value or parameter generation participates in specialization: every
launch reads current parameters. The caller retains ownership and generation
admission; this seam does not introduce a detached cache or replace the native
prepared program's lifetime checks.

Both CUDA routes use nearest-rounding float multiplication and CUDA libdevice
`tanhf`, without fast math or floating reassociation across the composition.
`tanhf` is a library approximation to real tanh, not a correctly rounded
transcendental guarantee. Direct and fused results must match each other;
the host tanh reference permits absolute error 2e-7 for this gate's inputs.
NaNs/infinities propagate, signed zero survives, and outputs may not overlap
inputs or each other. Repeated input roles are permitted. The provider exposes
forward only. Mathematical chain-rule JVP/VJP remains owned by the existing
local derivative implementation at saved primal values; it is not the
bit-level derivative of libdevice's approximation. No new derivative callbacks
or higher-order capability are advertised here.

`compose_support_row` computes exact OR-of-AND for a declared finite Boolean
relation, with an explicit validity result for out-of-universe inputs. It does
not infer response support from primal zeros. The SM70 instruction family filter
allows scalar float, popcount and FP16/FP32 MMA, and rejects binary MMA, TF32 MMA
and asynchronous copy from later architectures. Only the scalar composition is
implemented by this provider.

The task's other provider families remain explicit mapped alternatives:

| Family | Status here | Numerical and response policy |
| --- | --- | --- |
| M04 native matrix/process provider | Existing separate owner; no substitution made | Its declared axes/ordered roles and policy must be preserved |
| M12 texture atlas | Deferred alternative | Interpolation primal and sampled derivative require a separate declared convention |
| M13 hi/lo contractions | Deferred alternative | Four accumulated contractions do not claim exact FP32; derivative follows the declared approximation |
| BP Boolean numerical machine | Finite support-row composition only | Exact OR-of-AND support; no arbitrary semiring or inferred support equivalence |

Run `sh tests/substrate/lowering/build_gate.sh` from the workspace. It runs the
host guard test and compiles the GPU gate; it does not execute GPU code. Build
artifacts are in ignored `build-is1-lowering`. The compile log reports registers
and spills, and `scaled_tanh.sass` records emitted instructions. The current
SM70 compile uses 15 registers for fusion, 12 for each direct kernel, and no
spills. Emitted multiply is FMUL; libdevice tanh uses MUFU.EX2/MUFU.RCP and FFMA.

The root controller must run `build-is1-lowering/scaled_tanh_gate` through its
foreground CUDA lease. The gate checks 262147 elements, a partial final block,
nonfinite/signed-zero behavior, direct/fused equality, host reference agreement,
parameter mutation, shape/support fallback and alias/policy rejection. It emits
one bounded direct/fused CUDA-event comparison after warmup. That comparison
covers resident device execution with saved-intermediate writes; it excludes
allocation and transfer. It establishes no end-to-end or scientific claim.
