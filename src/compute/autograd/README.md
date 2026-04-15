# `compute/autograd`

Pointer-first sparse building-block library with a generic default path and a
native V100 fast path.

This is not a generic autograd framework.

What lives here:

- copyable sparse forward and backward kernel patterns
- explicitly isolated hot paths under `kernels/`
- tiny device helpers under `primitives/`
- minimal non-hot runtime support for streams, scratch, cuSPARSE caches, and multi-GPU launch plumbing
- base single-GPU copies and distributed pair / 4-GPU reference copies
- Blocked-ELL `SpMM` reference copies for Tensor Core-oriented sparse projection work on Volta

What does not live here:

- framework-style tape orchestration
- wrapper-heavy tensor APIs in hot code
- hidden layout repairs
- hidden host round-trips

Design rules:

- raw pointers first
- FP16 storage with FP32 accumulate is the primary path
- Blocked-ELL is the native sparse execution layout
- row-compressed CSR exists as a secondary compatibility and fallback path
- `generic` mode discovers visible slots and peer-link quality at runtime
- `native` and `native-extreme` may use the pair-local `0 <-> 2` and `1 <-> 3`
  hierarchy only when the discovered topology matches the native 4x V100 host
- 4-GPU reduction is still hierarchical when discovery finds strong pairs;
  otherwise the runtime falls back to a direct leader merge

How to use it:

- `base::` functions are single-GPU reference copies
- `base::blocked_ell_spmm_fwd_f16_f32_lib()` is the library-backed Volta sparse Tensor Core path
- `dist::launch_*` functions launch one base copy per selected device slot
- `dist::reduce_sum_to_leader_f32()` performs the explicit leader merge step for feature-sharded forwards or row-sharded dense-gradient reductions
- `default_mode_pair_slots()` and `default_mode_fleet_slots()` expose the mode-aware slot order explicitly instead of hiding native topology assumptions in the default path

The intent is that later model-specific kernels copy or inline these patterns rather than call them blindly from a hot training loop.
