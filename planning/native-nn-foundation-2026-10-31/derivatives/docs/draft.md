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
