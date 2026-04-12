# `compute/autograd`

Pointer-first sparse building-block library for Volta `sm_70`.

This is not a generic autograd framework.

What lives here:

- copyable sparse forward and backward kernel patterns
- explicitly isolated hot paths under `kernels/`
- tiny device helpers under `primitives/`
- minimal non-hot runtime support for streams, scratch, cuSPARSE caches, and multi-GPU launch plumbing
- base single-GPU copies and distributed pair / 4-GPU reference copies

What does not live here:

- framework-style tape orchestration
- wrapper-heavy tensor APIs in hot code
- hidden layout repairs
- hidden host round-trips

Design rules:

- raw pointers first
- FP16 storage with FP32 accumulate is the primary path
- row-compressed CSR is the optimized default
- distributed traffic is pair-local first on `0 <-> 2` and `1 <-> 3`
- 4-GPU reduction is hierarchical: pair-local work first, then leader merge

How to use it:

- `base::` functions are single-GPU reference copies
- `dist::launch_*` functions launch one base copy per selected device slot
- `dist::reduce_sum_to_leader_f32()` performs the explicit leader merge step for feature-sharded forwards or row-sharded dense-gradient reductions

The intent is that later model-specific kernels copy or inline these patterns rather than call them blindly from a hot training loop.
