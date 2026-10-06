# Native neighborhood computation probe

This example is a reference workload for reusable Cellerator operations. Its
input fields and outputs are generic; scVelo and CellRank own the scientific
interpretation, upstream comparisons, and data preparation. The probe does
not add a moments API to Cellerator.

The **ceNativeNeighborhoodMoments** executable links
**Cellerator::prepared_relation_cuda** and **Cellerator::native_numeric**. It
prepares one paired CSR topology, publishes one authoritative FP32 value
generation, then reuses that relation and its caller-owned buffers for every
warmup and measured application. Each composition performs five relation
applications, six elementwise multiplications, and five affine combinations.
For row-major fields L and R and relation W, its ten output fields are:

| Output | Expression |
| --- | --- |
| mean_left, mean_right | W L, W R |
| second_left, cross_second, second_right | W(L*L), W(L*R), W(R*R) |
| variance_left, variance_right, covariance | second_left - mean_left*mean_left, second_right - mean_right*mean_right, cross_second - mean_left*mean_right |
| affine_second_left, affine_cross_second | 2*second_left - mean_left, 2*cross_second - mean_right |

The final pair represents the upstream adjusted-second-moment convention;
there is no adjusted right-right moment in this probe.

## Reusable native arithmetic

Native numeric exposes two stream-ordered FP32 operations over contiguous
resident_vector buffers:

- **enqueue_elementwise_multiply(A, B, output, stream)** performs elementwise
  multiplication. Passing the same input twice computes its square.
- **enqueue_axpby(alpha, A, beta, B, output, stream)** computes
  alpha*A + beta*B, wrapping the existing affine device operation.

They allocate nothing, do not synchronize or change buffer generations, and
do not convert precision. The caller selects the device, retains all buffers
through stream completion, and supplies matching lengths and device metadata.
Inputs may be the same buffer; output storage must not overlap either input.
Zero-length valid bindings are no-ops. The API rejects unsupported
representations, invalid pointers, incompatible devices or streams, size
overflow, and output/input overlap using CUDA errors. These are general
arithmetic operations and remain separate from derivative APIs.

An in-progress shared FP64 device-elementwise surface has overlapping
multiply, square, and affine capability, with different context and alias
contracts. Source ownership is disjoint. Consolidation is deferred until the
owners coordinate the intended shared contract and rebuild against a stable
ABI; this probe has not benchmarked those new operations.

The relation path is FP32 for relation values, inputs, multiply,
accumulation, and output. The retained CSR candidate requires both FMA and
reassociation to be permitted by the operation descriptor. That permission
describes the candidate contract; it does not enable Tensor Core conversion.
The probe has no FP16 mode. Any future opportunistic FP16 path requires its
own numerical envelope and native fallback.

## File protocol

The executable reads little-endian raw arrays from the input directory; the
harness supplies shape and provenance separately. The CSR relation has M rows
and N source columns, and dense input fields have shape N x F:

| File | Type and element count |
| --- | --- |
| row_offsets.u32 | uint32, M + 1 |
| column_indices.u32 | uint32, E |
| weights.f32 | IEEE FP32, E |
| left.f32, right.f32 | IEEE FP32, N * F each, row-major |

Invoke it with --input, --output, --sources N, --destinations M, --features F,
--edges E, and optional --warmups K --repeats R. The --self-test option runs
a small analytic rectangular CSR case with an empty row, signed values, zero
and constant features, and feature width 131. The --help option prints the
CLI. The executable rejects malformed array sizes and CSR indices before
submitting the computation.

Each result is a raw row-major FP32 array of shape M x F, named FIELD.f32:
mean_left, mean_right, second_left, cross_second, second_right, variance_left,
variance_right, covariance, affine_second_left, and affine_cross_second.
metrics.json records dimensions, policy, GPU identity, timing scopes and
samples, memory counts, and the relation preparation report. The
orchestration harness writes input hashes, source revisions, reference
results, and acceptance limits in its own artifact directory; source
fixtures remain read-only.

## Build and run

From a Cellerator source worktree, the repository build helper records
configure/build commands, tool versions, source hashes, and generated binary
hashes in its build directory:

    python tests/native_numeric/build_probe.py \
      --build /tmp/moments-probe-20261006/ce-build \
      --baseplane-source /home/tumlinson/Baseplane \
      --cuda-root /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/cuda/12.9

The explicit Baseplane source path is needed when an isolated Cellerator
worktree has no sibling Baseplane checkout. The helper enables the optional
Semantic Spine examples and builds ceNativeNeighborhoodMoments; it does not
run GPU work. Its --baseplane-source argument sets CMake's
BASEPLANE_SOURCE_DIR.

Run correctness, Compute Sanitizer, and timing through the shared-host CUDA
controller, using a task-bound spec and the same CUDA toolkit for nvcc and
the sanitizer:

    python /home/tumlinson/.agents/skills/cuda/scripts/cuda_controller.py \
      run --spec /path/to/task-spec.json --json

The controller owns GPU admission and quiescence. Select one admitted device;
the executable uses device ordinal zero within the controller-provided
visibility. Do not invoke it directly for GPU qualification or timing.

## Measurement boundaries

resident_ms is a CUDA-event interval for one full composition, excluding the
separately recorded diagnostic iteration. resident_wall_ms brackets the
same work from host enqueue through stream synchronization. The diagnostic
iteration separates input products, the five CSR applications, and centered
and affine output work without adding synchronization between stages. Setup
metrics separately record raw input reading, resident-buffer allocation,
host-to-device event and wall times, relation preparation and value
publication, device-to-host materialization, raw output writes, and the
in-process interval. memory_bytes.tracked_device_allocations sums the exact
allocations owned by the example; process peak RSS and host input and output
byte counts are reported separately.

These measurements distinguish resident computation from setup and
materialization. Small biological slices establish functional agreement;
the seeded synthetic cases establish only synthetic performance. A
resident-composition ratio is not an end-to-end speedup claim. The scVelo
harness defines and writes numerical acceptance limits before running
comparisons;
failures are reported against those fixed limits.
