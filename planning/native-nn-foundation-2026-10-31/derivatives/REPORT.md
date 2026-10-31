# Native first-order derivative qualification

The native C++ adapters preserve ordinary Torch parameter identity/autograd and execute the existing numerical operations. CPU Product2 and admitted V100 correctness qualification passed. The final source passed all eleven normal behavioral groups. Separate memory qualification passed nine groups with zero memory errors and two explicit instrumentation-incompatible fixture skips. No performance or biological validity claim is made.

## Public boundary

`include/Cellerator/nn/ops.hh`, namespace `cellerator::nn::ops`:

- `product2_cpu(x[N], coefficient[M], lhs[M], rhs[M]) -> [M]`, CPU FP32 values and CPU int64 topology. Native gathered products and VJP; repeated indices accumulate.
- `grouped_linear_cuda(input[R,I], weights[E,O,I], host_row_offsets[E+1]) -> [R,O]`, matching FP16/FP32 storage and native FP32 GEMM accumulation. Empty experts/zero rows are valid.
- `csr_scaled_dot_product_attention_cuda(q[Q,H,D], k[N,H,D], v[N,H,V], offsets[Q+1], key_ids[edges], scale, dropout_p=0) -> [Q,H,V]`. CUDA int64 CSR is snapshotted and validated before native execution; raw admission may synchronize.
- `csr_scaled_dot_product_attention_cuda_prepared` has the same signature and requires previously validated immutable contiguous CSR. It checks tensor metadata only: no CSR content reads, host transfers, snapshots, or second preparation/cache object. The identity-bound `PreparedRelation` wrapper owns structural validation/version authority.

Attention uses only the existing staged kernel. Duplicates are separate softmax edges, empty rows produce zero, and FP32 native attention VJPs are cast to matching input storage dtype. Nonzero dropout and create_graph/higher-order differentiation fail explicitly. There are no numerical-kernel or legacy guarded-owner changes.

## Ownership and streams

All original tensors are saved, even when native execution needs contiguous/head-packed copies. Saved-variable version checks run before backward pointer use. Each grouped invocation retains its own executor; each attention invocation retains its own probabilities/workspace, with separate scratch per backward replay. No pointer cache or fused native parameter update is introduced.

The current guarded Torch stream is passed to native execution. Torch-owned allocations use allocator stream recording. Foreign from_blob/DLPack storage is retained through completion events, including inference and immediately discarded graphs; a private host cleanup worker releases owners outside CUDA callbacks. Grouped executor handles are also event-retained through queued forward/backward use. Event/enqueue/thread failure synchronizes the relevant stream before release. The FIFO cleanup worker can delay later releases behind an earlier long stream; it does not control placement or execute numerical work. Process teardown drains the queue, and qualification is bounded by the scheduler timeout.

Supported active Torch allocator backends are `native` and `cudaMallocAsync`. Arbitrary globally installed pluggable allocators fail clearly because their optional stream-recording implementation may be absent. Foreign tensor ownership is supported under the declared backends; qualification used the default native allocator, while cudaMallocAsync was not separately exercised. External custom-deleter tests used from_blob as the foreign-ownership representative; no separate DLPack round-trip test was run. CUDA graph capture, compiler integration, higher-order gradients, dropout, broad precision regimes, and performance are outside qualification.

## Validation and evidence

Focused qualification build: `scripts/CMakeLists.txt`, `/tmp/cellerator-native-derivatives`. The public NN CMake/export integration belongs to the build owner; this harness compiles the same adapter source against the unchanged native numerical owners.

- `evidence/cpu-qualified-final.log`: CPU ATen forward/VJP comparison, repeated/zero indices, noncontiguous input/topology versions, empty products, invalid input and first-order rejection: PASS.
- `evidence/cuda-status-r5.json`: admitted correctness job `2ddb686b-d493-4bb5-bcb8-9fe7504e17eb`, Tesla V100-SXM2-16GB, UUID `GPU-d745da9c-7649-8334-41f1-483afa6f3206`, compute 7.0: 11 groups passed, 0 failed, exit 0, cleanup verified.
- Independent oracles: ordinary ATen equations, FP32 calculations on the uploaded FP16/FP32 values, then explicit output/gradient storage conversion. Bounded fixtures use FP32 absolute/relative tolerance 2e-4; FP16 4e-3. Seed `0xCE11`.
- Cases include grouped forward/input/weight VJP, empty expert/zero rows, ordinary SGD, staged multihead attention q/k/v VJP with value widths 1/2/33, duplicates/empty support/zero queries, raw/prepared parity, live and retained graphs, original tensor/CSR mutation checks, explicit unsupported contracts, nondefault streams and discarded graphs, creation-stream/execution-stream allocator churn with an independent CPU oracle, and gated foreign-owner custom-deleter inference/graph disposal.
- `evidence/memcheck-status-r4.json`: memory job `bfbe6b36-52b7-4fba-b0db-961b9dbbc8eb`, same binary and V100 UUID under `/usr/local/cuda/compute-sanitizer/compute-sanitizer --tool memcheck --error-exitcode 86`: nine groups passed, zero failed, two explicit skips, ERROR SUMMARY 0 errors, exit 0, cleanup verified. The two host-gated asynchronous-submission fixtures remain required and passed in normal mode; sanitizer instrumentation blocks/intercepts launches and forces their gate deadlines, so `--memcheck` explicitly excludes those two fixtures. Nondefault-stream/drop-graph/churn coverage still runs under memcheck. This is a declared coverage limit, not an all-eleven memory claim.
- `evidence/identity-qualified-final.json`: exact source/binary SHA256 identities; `environment.json`, `linked-libraries.txt`, `cuda-elf-architectures.txt`, build logs, submission/status/log receipts, and `docs/plan.md` reconstruct the attempt. Installed Torch is 2.7.0+cu126, ABI 1. Only SM70 CUDA objects are present in the final executable.

## Preserved unsuccessful attempts

The CLI default principal rejected an initial submit before creation; the existing Cellerator client configuration resolved it without credential changes. Job `ae76e5ce-e64c-42ec-a020-a917422ef2ff` for UUID 211 was blocked before admission, then cancelled with cleanup verified. The first admitted job `3d308d91-1dd8-4a95-8be5-89b5aeeebd6a` failed because installed Torch CMake replaced requested architecture flags with distribution defaults omitting SM70, and one zero-row test fixture supplied unsupported requires_grad TensorOptions. That job cleaned up. The harness now sets `TORCH_CUDA_ARCH_LIST=7.0` before finding Torch and restores CMake architecture 70 afterward; ELF metadata confirms final SM70. The fixture now sets requires_grad on the resulting Tensor. All unsuccessful attempt receipts remain in evidence; no unmanaged GPU execution occurred.

Independent review also identified the `PreparedRelation` before-forward mutation boundary. Biology/root own that integration fix: preparation-time version capture and per-use metadata/version checks preserve immutable CSR without device content reads. The low-level prepared function remains an explicitly trusted interface.

The first memcheck wrapper attempt selected an absent CUDA13.1 sanitizer and failed before test launch; the installed CUDA12.9 sanitizer resolved the executable path. Two subsequent attempts with three-second and thirty-second host gate budgets each reported zero memory errors but failed the two gated asynchronous-submission checks. Their receipts and cleanup remain preserved. Root explicitly accepted the final nine-pass/two-skip memory boundary alongside eleven required normal behavioral passes. An additional source audit added original tensor stream records before preparation copies, covering exceptional preparation paths; final normal and memcheck jobs above were rebuilt from that source.
