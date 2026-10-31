# Equal-row batched grouped-linear forward probe

## Objective and boundary

Prototype an internal forward-only path for `GroupedLinear` workloads where at
least eight experts each have the same positive row count. Keep the existing
per-expert cuBLAS path as the fallback for ragged partitions, empty experts,
fewer than eight eligible experts, and every backward call. Do not change the
public API, numerical policy, source manifest, or production files in this
probe. The caller owns all input, output, and weight allocations; computation
remains asynchronous on the supplied stream and uses the selected device.

The operation is row-major `Y[e] = X[e] W[e]^T`, mapped to column-major
`Y[e]^T = W[e] X[e]^T`. FP32 uses FP32 operands/accumulation. FP16 uses
represented half inputs and weights, FP32 accumulation, and half output. The
independent oracle decodes the represented operands to FP32, accumulates in
FP32, then converts to the output type. The prototype must not modify inputs,
weights, or output memory outside the corresponding forward result.

## Baseline and opportunity

The strongest current baseline is one `cublasGemmEx` call per nonempty expert.
The task owner reports these V100 measurements from the canonical
`cellerator-framework-v100` evidence: at 32 experts × 2 rows × 64 input/output,
FP32 current forward is 0.203 ms, batched GEMM with resident pointers is
0.0063 ms, and batched GEMM with input/output pointer refresh is 0.0177 ms. At
8 experts × 16 rows × 128 input/output, the corresponding times are 0.0505,
0.0064, and 0.0264 ms. Grouped cuBLAS was often slower than conventional
batched GEMM. These are evidence for a bounded equal-row probe, not a general
crossover claim.

The candidate keeps the expert weight-pointer array on device for the prepared
executor lifetime. A small kernel derives input and output pointer arrays from
the current base views and equal row stride on each call, then one
`cublasGemmBatchedEx` call executes the group on that same stream. This avoids
host-side pointer refresh while preserving caller-owned activations and
outputs. The pointer-fill launch remains part of per-call timing. Constructor
allocation, handle creation, and weight-pointer preparation are reported as
setup/cold cost. One executor is used for one concurrent submission stream, in
keeping with the existing executor contract.

## Risks and unknowns

- The pointer-fill kernel plus batched call may still lose to the per-expert
  route for small groups or larger GEMMs; the threshold is only a candidate
  boundary and requires admitted, matched V100 measurements.
- Pointer-array memory must remain valid and stream ordered until cuBLAS
  consumes it. Reusing one executor concurrently across streams is unsupported
  by the existing contract; callers use one executor per concurrent stream.
- The borrowed expert-weight allocations must outlive the prepared executor.
- cuBLAS pointer-array placement and math-mode requirements must compile and
  pass on the target CUDA/cuBLAS version; no runtime capability is inferred
  from compilation.
- The fast path recognizes only globally equal, positive row counts. It does
  not group subsets of a ragged partition and does not change backward.
- Root owns Device Scheduler admission, V100 correctness/sanitizer work, and
  timing. This probe performs compilation only and makes no runtime or speedup
  claim.

## Candidate directions, ranked

1. Conventional `cublasGemmBatchedEx` with persistent weight pointers and
   device-generated per-call input/output pointers. This directly tests the
   measured leading batched candidate while removing host refresh copies.
2. Host-built asynchronous pointer-array refresh, as the measured reference
   for the exact same operation. Keep this in the trusted comparator rather
   than duplicating it as a production alternative.
3. `cublasGemmGroupedBatchedEx` and a custom fused/Tensor Core implementation
   are excluded from this bounded probe: existing evidence weakens the grouped
   API lead, while a custom GEMM changes ownership and expands correctness
   risk before the library composition is exhausted.

## First implementation steps

1. Add an isolated candidate library and CUDA test executable under this
   directory only; leave production and shared benchmark files unchanged.
2. Implement the eligibility predicate and batched forward helper, including
   validation for device, view size, alignment, and disjoint input/output.
3. Add host tests for the eligibility/fallback boundary and GPU-capable test
   cases for exact FP32 and FP16 represented-input oracle comparison, repeated
   calls with changed input/output allocations, stream ordering, and untouched
   weight/input buffers.
4. Compile for `sm_70` with the repository CUDA toolchain. Do not launch this
   executable; the task owner supplies the admitted validation route.

## Validation and evaluation

From the repository root, configure and build only this isolated target:

```sh
cmake -S planning/v100-framework-2026-10-29/grouped_linear/batched_candidate \
  -B /tmp/cellerator-grouped-linear-batched-candidate-build \
  -DCMAKE_CUDA_COMPILER=/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9/bin/nvcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build /tmp/cellerator-grouped-linear-batched-candidate-build --parallel 1
```

The build proves only host/device compilation and linkability. Before promotion,
the owner must run the candidate and incumbent with the independent oracle on
V100 for both reported exact shapes, both FP32 and FP16, plus all fallback
cases, through the admitted Scheduler/Autobench route. Measure warm resident
calls with pointer fill included, setup/cold cost separately, and a
submission-and-synchronization metric. Include the existing per-expert path
and host-refreshed batched path in matched comparisons; require correctness
before timing and preserve raw timing samples and source identity.

## Decision criteria

Keep the prototype as an integration candidate only if the V100 oracle and
required sanitizers pass, supported cases improve the matched forward metric
after pointer-fill overhead, and unsupported cases take the unchanged fallback
without numerical or ownership changes. Reject it if pointer preparation
erases repeat-use savings, the fill kernel erases batched-launch savings, any
stream/lifetime condition cannot be preserved, or the smallest supported
workloads regress relative to per-expert GEMM. No promotion is authorized by
this isolated prototype.

## Source and workflow identities

- Cellerator checkout: `acf6d972e4eda36fdbb8dc670eb8707d6e48e28b`;
  production grouped-linear source blob:
  `24c21381e412cba2757707b687293e53bde1dad1`.
- Existing comparison harness:
  `benchmark.cu` SHA-256
  `94042980901f6265ceea4054c064c6270a6a61ec3cd39c5261d730cc5e313beb`;
  CMake SHA-256
  `cd4765f020d44940939773b558ce0fa7fc7c79de78cb55d8851f2bd3347a4d16`.
- Parent-provided V100 comparison evidence:
  `.autobench/evidence/cellerator-framework-v100/5/*/stdout.log`.
- KDA workflow checkout: `277c79b8cdd1d13c34a0dfd132b8bb504bcb5afc`.
- CUDA skill `SKILL.md` SHA-256
  `3992c5b0e85409cce8002fcfe2c6caed63df6451daad9bae28ccd64a381d717a`;
  Volta router SHA-256
  `54f91d1e42395eac485862a0970d74ea641642f63b96ffbdc2381ca52a0096b7`;
  host-execution guide SHA-256
  `311edd1ea9055c70914bddebe7b5c6fbe36789cfaafa4491c153b2e713fbc90d`.
