# Native host numeric qualification

`ce_nf1_n01` links `Cellerator::native_numeric` and exercises the actual
`host_relation` API from `compute/operation/native_numeric/host_relation.hh`.
The semantic owner remains `relation_semantics.hh`; architecture and identity
rules remain in the repository's authoritative scope and architecture documents.

Preparation copies the logical source/destination endpoints and builds stable
per-destination edge indices. Duplicate edges remain separate contributions in
the declared logical edge order. Preparation is the only allocation phase.
Run is synchronous, host-only, allocation-free and caller-serialized, with
row-major independent dense channels. Value/input buffers are borrowed until
return; output is overwritten after complete metadata/input admission. Callers
may release preparation inputs and change launch values without rebuilding.
No new allocator, session, tensor identity, device owner or compiler adapter is
introduced. The compiled library is a real native CPU route, not a CUDA fallback
claim or a mathematical oracle used as a provider.

Both homogeneous FP32 and homogeneous FP64 are supported, including multiply,
accumulation and output. Fast-math, reassociation and fused contraction are not
used. The cancellation test distinguishes the arithmetic types: the ordered sum
`1e8 + 1 - 1e8` is zero for FP32 and one for FP64. Mixed/reduced storage,
transpose, accumulation/affine effects and alias permissions are explicitly
unsupported here; later N tasks own their implementations. Width must be positive
per shared semantics. Empty support overwrites destinations with zero, and empty
source/destination axes are legal only without edges.

Bindings must match exact source/destination domain, order, geometry and
partition; value structure, epoch and edge order must match preparation. Values
carry a nonzero generation identifying the present call. This synchronous API
has no result cache, monotonic generation publisher or asynchronous readiness
state; no global freshness assertion is inferred from a generation number.
Output overlap with either input or values is rejected. Metadata failures and
nonfinite-input rejection preserve all output bytes. Failed preparation preserves
the prior prepared instance. Moved-from instances reject execution safely.

The propagate nonfinite policy follows IEEE multiplication and ordered addition.
The reject policy preflights nonfinite bound inputs; finite-input arithmetic may
still overflow or produce a nonfinite result, and no error bound or result-finite
guarantee is claimed. Predicate exclusion is not implemented as multiplication
by zero here. Host spans must refer to live accessible host memory; this API
cannot validate OS memory lifetime or inspect arbitrary device pointers.

The independent FP64 oracle uses named logical edges and semantic endpoints.
It does not reuse prepared CSR offsets, edge maps, or production evaluation.
Tests use a different physical source/destination order, widths 1/3/15/16/17/33/65,
duplicate edges, an empty row, independent value rebinding and intentionally
permuted endpoint/overwrite-reduction controls. FP32 comparisons use 5e-6
absolute-plus-relative tolerance, FP64 uses 3e-13; shape, status, sentinel,
zero-output and cancellation checks are exact. No GPU/performance claim follows.

`prepare.py` provides source-linked standalone qualification while the root
build hook is integrated separately by B01. It requires a clean commit,
configures this directory with the native production fragment, checks the exact
CMake source path, rebuilds, verifies compile_commands entries all belong to the
actual worktree and include the native library, shared semantics and test source,
and records source/cache/bindings hashes externally. The normal integrated root
build registers this same numeric test fragment. Native run_gate.py then executes
the exact real CTest inventory with no skipped tests. The build helper is adapted
from CE T01's provenance helper; it shares no numerical algorithm or oracle.
