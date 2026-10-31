# Native derivative adapters: initial qualification draft

Objective: ordinary Torch autograd and optimizers around existing native Product2 CPU, cuBLAS grouped linear, and staged CSR scaled dot product attention. No numerical kernel changes, fused native updates, performance claims, or attention dropout.

Baseline: Product2 prepared_owner exposes synchronous FP32 forward/VJP; grouped linear borrows weights and executes asynchronous forward/VJP with matching FP16/FP32 storage and FP32 GEMM accumulation; staged attention has 2D per-head forward and FP32 VJP workspace. Existing Python adapters and native smoke sources provide orientation; independent ATen compositions are the qualification oracle.

Contracts: retain original arguments for Torch saved-tensor version checks even when preparing contiguous copies. Each invocation owns its executor/workspace; use current guarded device/stream and allocator stream records for every borrowed allocation. Attention q/k/v are [Q,H,D], [N,H,D], [N,H,V]; CSR duplicates remain distinct edges and empty rows produce zero. Raw CSR is copied into private contiguous snapshots and validated before use. Native backward rejects create_graph/higher order explicitly.

Ranked direction: (1) custom torch::autograd::Function over established native forward/VJP, preserving original tensors and separate prepared storage; (2) ATen reference as independent oracle only. Existing native kernels are fixed, so candidate benchmarking and Tensor Core redesign are outside scope.

Risks: tensors released before asynchronous completion; mutable original tensors hidden by contiguous conversion; repeated backward sharing mutable workspace; non-default streams and early graph disposal; head packing and duplicate/empty CSR support; FP16 output gradient conversion. Use per-call workspace, no persistent pointer cache, and explicit allocator stream recording. Native executor handle is retained per graph.

Qualification cases: FP32 CPU forward and x/coefficient VJP, duplicates and noncontiguous original version checks. CUDA FP16/FP32 grouped forward and x/weight VJP with empty experts; attention forward/q/k/v VJP with duplicate edges, empty rows, multiple heads and value widths; ordinary optimizer training; two outstanding graphs; retain_graph repeat; explicit higher-order rejection; original mutation rejection; non-default-stream release/allocator churn. Compare FP32 within 2e-4, FP16 within 4e-3 absolute/relative for bounded fixtures. Run focused Compute Sanitizer memcheck under separate admission if available.

Execution plan: publish ops.hh; implement CPU and CUDA adapters; build via native NN CMake target; execute CPU test binary directly; execute CUDA test binary and memcheck only as finite authenticated Device Scheduler UUID-selected exclusive jobs (2 GiB admission estimate, <=300 seconds). Preserve full submission/status/log and cleanup receipts. Exact binary paths and commands will be captured in docs/plan.md once integrator target is available. No GPU execution is permitted by visibility alone.

Promotion: source compiles, independent numerical/VJP and lifecycle tests pass, admission and cleanup receipts retained. A blocked scheduler or missing tool is reported without substituting unmanaged GPU execution.

Consumed guidance: CUDA SKILL.md and Torch extensions/native system/host execution/custom operation registry references; KDA agent-entry/basic-flow/agent-flow. These are adapter qualifications, not algorithm or profiling campaigns.

## Concrete build and execution commands

- Configure: `cmake -S planning/native-nn-foundation-2026-10-31/derivatives/scripts -B /tmp/cellerator-native-derivatives -DCMAKE_PREFIX_PATH=/home/tumlinson/Software/miniconda3/lib/python3.13/site-packages/torch/share/cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=RelWithDebInfo`
- Build: `cmake --build /tmp/cellerator-native-derivatives --target native_nn_ops_cpu_test native_nn_ops_cuda_test -j2`
- CPU: `env LD_LIBRARY_PATH=/home/tumlinson/Software/miniconda3/lib/python3.13/site-packages/torch/lib:/usr/local/cuda/lib64 /tmp/cellerator-native-derivatives/native_nn_ops_cpu_test`
- GPU: `device-scheduler job submit --payload-file planning/native-nn-foundation-2026-10-31/derivatives/evidence/cuda-job.json`; job executes only `/tmp/cellerator-native-derivatives/native_nn_ops_cuda_test --device` with matching Torch library environment and explicit UUID visibility inside its admitted grant.
- Memory qualification: same executable under `/usr/local/cuda/compute-sanitizer/compute-sanitizer --tool memcheck --error-exitcode 86` in its own finite exclusive admission.

Trusted prepared CSR route (added at root's direction): `csr_scaled_dot_product_attention_cuda_prepared` takes the same arguments but requires already validated contiguous immutable CSR, checks only metadata, and performs no CSR copies or content reads. It preserves the existing PreparedRelation preparation system rather than inventing another cache. Both routes save original tensors for version/lifetime checks.

Guidance revisions: KDA `277c79b8cdd1d13c34a0dfd132b8bb504bcb5afc`; skills checkout `15c987d717fd205ef08a518647cd51d563c7d978`. Native target is SM70. No kernel redesign/timing/profiling is part of qualification.

## Review-driven lifetime repair

Independent review identified that allocator stream recording does not retain foreign from_blob/DLPack owners. Scope was extended with root approval to a private adapter completion queue: only external allocator/deleter tensors are retained through an event recorded at scope exit, including no-grad/early graph disposal and exceptions; a host cleanup thread waits for the event and releases owners outside CUDA callbacks. Queue teardown drains and joins. Event/enqueue/thread allocation failure uses a stream synchronization fallback before releasing owners. Ordinary allocator-owned tensors keep stream recording without worker submissions. Finite Device Scheduler timeout bounds the qualification process including teardown; this is not a public runtime service.

Qualification adds foreign-owner deleter ordering and cross-stream creation/execution/churn cases. A separate prepared relation finding (mutation before forward) belongs to biology/root; preparation-version checks repair that boundary without content reads or copying.

Allocator boundary: supported active Torch CUDA backends are `native` and `cudaMallocAsync`; arbitrary globally installed pluggable allocators are rejected because their optional stream-recording callback may be absent. Foreign from_blob/DLPack tensor storage is supported through explicit completion retention under these backends. Grouped cuBLAS executors are also event-retained through every queued forward/backward, including discarded no-grad contexts, to avoid early handle destruction. The private FIFO cleanup worker may delay later owner release behind an earlier long stream, but does not control execution or placement.

Scheduler uses the existing `/home/tumlinson/.config/device-scheduler/cellerator-client.toml` configuration. An initial wrong-principal CLI-default submit was rejected before creation. UUID 211 was unavailable to the submitted request, so that unadmitted job was cancelled with verified cleanup; final qualification uses explicit UUID d745 and captures final source/binary hashes in `identity-final.json`.

Final memory qualification uses the explicit test executable `--memcheck` mode: nine ordinary numerical/autograd/nondefault-stream cases run, and two host-gated async-submission fixtures are explicitly reported skipped because sanitizer instrumentation blocks/intercepts launches enough to force their gate deadlines. All eleven behavioral cases remain mandatory in normal `--device` mode. Both failed instrumentation-gate attempts reported zero memory errors and are preserved; longer timeout did not resolve the tool interaction. Root accepted this stated coverage boundary.
