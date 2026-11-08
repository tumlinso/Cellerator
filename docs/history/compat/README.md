# Retired compatibility tree

This archive preserves the former `compat` tree, its five legacy forwarding
headers, and the original execution-plan contract test from the source revision
recorded in [manifest.json](manifest.json). Every snapshot is byte-for-byte
identical to its original Git blob. The added `.txt` suffix prevents these
historical CMake and C++ sources from entering current builds or test discovery.
The manifest records original paths, archive paths, Git blob IDs, SHA-256 hashes,
and canonical destinations for retained implementations.

The original [CP-Math evidence record](source/compat/cp_math_v1/MATH_V1_EVIDENCE.md.txt)
and [legacy preprocessing benchmark](source/compat/legacy_sparse/bench/preprocess_format_compare_bench.cu.txt)
retain their historical claims and dependencies. They are not instructions for
the current build, current API documentation, or new qualification evidence.
Restore the recorded source revision to reproduce the original build context.

Current numerical request validation, signatures and independent SpMM referees
live under `include/Cellerator/math/reference` and `src/math/reference`, linked
as `Cellerator::spmm_reference`. The retained request schema is a reference
contract; it does not replace the supported operation or execution APIs.

Borrowed CSR adaptation, BELL lowering and operand packing belong to
`include/Cellerator/layout/projection` and `src/layout/projection`. The existing
native-tile header remains `Cellerator/compute/projection/native_tile_view.hh`,
with its implementation in `src/layout/native_tile_view.cc`; its build owner
is now the physical-projection target.
`Cellerator::physical_projection` supplies host construction;
`Cellerator::packed_dense_operand` remains an optional CUDA target. The existing
namespace, schema numbers, enum values and packing algorithms are preserved.
Moving this source does not extend its historical numerical or performance claims.

Geometry libraries, tests and benchmarks are registered with the geometry,
test and benchmark owners. Four legacy `compute/math` headers forward to shipped
canonical headers. The unused execution-plan header is archived here and is no
longer installed. No current target compiles or links an archived source.

The installed-consumer fixture is
`tests/package/reference_consumer`. Host checks use `CELLERATOR_ENABLE_CUDA=OFF`;
device compilation, execution, sanitizers and measurements remain separate.
