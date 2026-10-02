# Numerical effects research

Link `Cellerator::moonshot_effects` and include `<ce_moon/effects.hpp>`.
All provider inputs are numerical. Baseplane retains sequence interpretation.

E09 exercises the foundation monomial affine effect with sequential comparison
and an order-sensitive example. E10 adds `BlockAffine<Blocks,Width>` with an
explicit permutation of blocks, dense square matrices per block and offsets.
It preserves the fixed block partition. Tests compare three composed 2x2-block
operators with independently expanded dense sequential application.

E11 provides scalar quadratic `Jet` values at explicit expansion points.
Composition requires the second expansion point to equal the first value and
truncates higher powers. `query` reports a trust-neighborhood flag; it does not
certify response error. The composed-square probe reports error 0.0041 at 1.1.
An out-of-radius query is tagged for consumer refinement. No automatic
recomputation or scientific calibration is supplied.

E12 `Checkpoints<N>` retains sparse prefix/suffix monomial maps and local effects.
Queries reconstruct a boundary from a checkpoint with fewer than stride replay
steps. Right queries apply effects from the queried boundary toward the end;
they do not invert effects. Initial construction is linear and retained local
effects are real storage cost. Tests compare both directions at every boundary.

E08 support supplies `Weighted<N>`, `Algebra::{probability,max_plus,boolean}`,
`compose` and `prune_below`. Rows denote source states, columns destinations;
composition visits the left operator then the right. Probability uses sum/product
of nonnegative binary64 weights, without normalization; max-plus uses max/sum
and negative infinity for absent edges; Boolean entries are exactly zero/one.
Pruning reports removed entries, removed probability mass and an approximation
tag. Removed mass is an accounting quantity, not a downstream error bound.
Tests demonstrate that pruning changes a later answer from 0.42 to 0.06.

E13 support uses existing guarded lifting and explicitly demonstrates residual
omission and restoration. E15 `precision_interval` provides refinable binary
intervals for a retained scalar in `[0,1]`; this fixture does not compress source
storage. E14 learned segmentation stays with the learning provider; E24 sampled
matrix response stays with the tensor provider. E39 can consume the explicit
order-sensitive effect already exercised here. These are support interfaces,
not completion claims for their separate sequence experiments.

Host operators reject invalid domains and nonfinite arithmetic results.
Floating algebraic closure is approximate due to rounding. The optional CUDA
translation unit compiles the existing representative monomial/lifting kernels
for sm_70; the new block, jet, checkpoint and weighted operators are host paths.
No GPU runtime, speedup, gradients, trained effect parameters or biological
validation is claimed. Parameters are explicit and hand supplied in the probes.
