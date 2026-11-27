# Native source layout and cleanup

Place new and consolidated native code by responsibility. Give each numerical
operation one source home and one build owner that callers link. Use the
[contributor routing guide](routing.md) to trace existing targets and callers.
Current CMake targets and callers show which paths are in use.

| Home | Responsibility |
| --- | --- |
| `structure` | Logical identities, axes, ordering, and lookup indices. |
| `geometry/dependency` | Declared relationships and support. |
| `state` | Structured model coordinates, gather, and gradient pullback. |
| `geometry` | Packing policy and algorithms that choose row and feature groups. |
| `tensor` | Numerical encodings, addressing, and views over storage. |
| `layout` | Physical order, projections, and maps to and from logical order. |
| `memory` | Allocation and resource ownership. |
| `ops/<family>` | Native calculations, derivatives, and the provider code that performs them. |
| `execution/planning` | Legal operation and placement candidates, cost comparison, and reusable plan evidence. |
| `execution` | Preparation, program launch, profiling, tuning, and streaming, with explicit readiness and borrowed lifetimes. |
| `runtime` | Device sessions, streams, library handles, and fleet primitives. |
| `interop` | Boundaries to external storage formats and frameworks. |

Keep implementations under `src/` and mirror the owning path under
`include/Cellerator/` for public native interfaces. Keep private headers beside
the implementation. Place specialized kernels with their operation family.
An operation can have several backends under the same owner. Choose the backend
before repeated execution using its numerical contract and measured cost.

For row-masked block tiles and feature-major sparse tiles, `tensor` defines the
masks, offsets, value order, and borrowed views. `geometry` chooses the row and
feature groups. `layout` materializes the selected order and recovery maps into
supplied storage. Binding a validated packed image to a tensor view belongs
beside the image reader. It borrows the existing arrays in their current memory
placement. `interop` handles adapters to external formats and libraries.

## Standard sparse layouts on NVIDIA

Follow [NVIDIA's documented cuSPARSE storage layouts](https://docs.nvidia.com/cuda/archive/12.9.1/cusparse/storage-formats.html)
for standard formats where the operation permits it. Prepare compatible buffers
directly. cuSPARSE descriptors and custom kernels can borrow the same index and
value arrays. Keep descriptors private to the CUDA provider and keep logical
identity and ordering maps with their existing owners.

Prefer cuSPARSE for standard operations when the installed library supports the
required arithmetic and layout and its complete workload cost is competitive.
Use custom kernels for unsupported contracts or measured workload benefits.
Evaluate copies, packing, and value refresh along with kernel time.

CSR, COO, Blocked ELL, sliced ELL, and DIA views borrow index and value
arrays. The caller provisions the buffers and keeps them live through
execution. A packer can size the required buffers and fill supplied storage.
Keep physical layouts and custom kernels when changing ownership. Compare
complete workload costs before replacing a layout. Block-masked retained reads
also hold a published value generation through asynchronous use.

## Ordinary CSR multiplication

Ordinary CUDA CSR multiplication lives in `src/ops/matrix/csr_spmm.cu` and
`src/ops/matrix/csr_spmv.cu`. Both use the internal `Cellerator::csr_multiply`
target declared in [the matrix build](../../src/ops/matrix/CMakeLists.txt).
Use the [compressed views](../../include/Cellerator/tensor/base/compressed.hh)
for indices and values. The native kernels take borrowed storage and the
caller's stream. The relation caller owns the mapping from logical edges to
stored values.

Keep cuSPARSE descriptor provisioning and enqueue together at the provider
boundary. Reuse the descriptors and caller-provided workspace for compatible
calls. With stable addresses, the vendor enqueue should reduce to the library
call and its status check. Make address changes explicit. Keep device-specific
libraries private to the CUDA implementation. Compare the direct kernel and
vendor route on the same calculation, including setup, reordering, value
refresh, and peak live memory. The choice of source home is independent of the
measured backend choice.

## Consolidating an operation

Move the implementation together with its in-tree callers and build wiring.
Remove superseded entry points and bookkeeping once those callers use the
canonical owner. Keep independent references and source-bound evidence usable.
Check affected builds and tests and account for complete workload costs before
changing the preferred backend.

Follow the [protected C++ coding style](../../coding_style.md) for every native
path. Use plain borrowed views and direct functions, with resource owners for
actual lifetimes. Provision explicitly and reuse sufficient storage. Repeated
execution within capacity has zero allocations or growth, including host
bookkeeping. Keep copies, conversions, and synchronization visible at the call
site or in the operation contract.
