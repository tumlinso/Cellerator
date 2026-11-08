# CUDA device data

`anndatapp.cuda` supplies optional CUDA allocation, views, and explicit data
movement. The shared view ABI lets other packages use their own allocations and
algorithms. It does not add device pointers to host `Array` or `AnnData`.

Build the component with a CUDA toolkit and NVCC 12 or newer:

```sh
cmake -S . -B build/cuda -G Ninja -DCMAKE_CXX_COMPILER=clang++-18 \
  -DANNDATAPP_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build/cuda
cmake --install build/cuda --prefix /path/to/anndatapp-install
```

For an installed component use `find_package(anndatapp CONFIG REQUIRED
COMPONENTS cuda)`, link `anndatapp::cuda`, and `import anndatapp.cuda;`. Requesting
only `anndatapp::anndatapp` does not discover or require CUDA. The device module
is not part of the `anndatapp` facade. NVCC and other ordinary C++ consumers can
include the narrow `<anndatapp/device_abi.h>` instead of importing a module; its
POD types are the same types exported by the CUDA module.

## Spans and views

`Span` describes a borrowed pointer, its accessible byte capacity, a CUDA
device ordinal, and `Access::read_only` or `Access::read_write`. Access is a
contract checked by these functions; the raw `void*` itself does not enforce
constness. The caller guarantees the pointer really addresses the stated device
and retains it through all submitted operations. No device pointer is exposed
as a host `Array` pointer.

`DenseView` combines a span with a byte `origin`, row and column counts, signed
row and column **element** strides, and a `ScalarType`. Negative strides and
read-only broadcast views are supported. A writable view must address distinct
logical elements. `SparseView` describes CSR or CSC with separate spans for
values, indices, and offsets; each has its own capacity and access permission.
Index and offset types are independent signed or unsigned integers of 8, 16,
32, or 64 bits. Offsets hold `major_extent + 1` entries and the other components
hold `nnz` entries. Duplicate coordinates are allowed.

`ScalarType` has explicit `scalar_type(DType)` and `dtype(ScalarType)` mappings;
its ABI values are independent of `DType` enum positions. FP16 is IEEE binary16
storage. Data supplied by another package can be used directly:

```cpp
#include <cstdint>
import anndatapp.cuda;

// external_ptr and its owner are provided by the caller.
void inspect(void* external_ptr, std::uint64_t capacity_bytes, int device,
             std::uint64_t cells, std::uint64_t genes, std::int64_t row_stride) {
    anndatapp::cuda::Span foreign{external_ptr, capacity_bytes, device,
                                anndatapp::cuda::Access::read_only};
    anndatapp::cuda::DenseView x{foreign, 0, cells, genes, row_stride, 1,
                               anndatapp::cuda::ScalarType::float16};
    anndatapp::cuda::validate(x);
}
```

`validate` checks metadata, overflow, alignment, extents, and access without
reading device contents or synchronizing. Sparse `validate(view, true)` requires
all three components writable; validate an individual span when only that
component is writable. `validate_sparse_contents` separately enqueues a device
check into a caller-owned writable `uint32` status: bit 0 means invalid offsets
and bit 1 invalid indices. It clears the status first and allows duplicates.

## Ownership, transfers, and conversion

`DeviceBuffer(bytes, device)` uses `cudaMalloc` and `cudaFree` by default.
`Allocator` supplies optional allocation/release callbacks and a retained
owner; `DeviceBuffer::adopt` invokes a supplied release callback, while
`DeviceBuffer::retain` holds an external owner without freeing its pointer.
Borrowed spans do not require either wrapper. Buffers are movable, not copyable.
Allocation and destruction are explicit costs; the default `cudaFree` may
synchronize. Callback contexts and native streams must remain valid until
release, and caller-managed ownership must outlive queued work.

A `Stream` consists of a native `cudaStream_t` represented as `void*` and a
device ordinal; `nullptr` selects the CUDA default stream. `copy_to_device`,
`copy_to_host`, `copy_device`, `zero`, `fill_bits`, `fill`, `cast`, and the
gather operations take an explicit stream. Device-to-device copies require the
same device and reject nonidentical overlapping ranges. Host source, destination, and staging
buffers must remain valid until completion; CUDA may block or stage pageable
host transfers. Use `synchronize(stream)` or caller-owned CUDA events to observe
completion. Operations report immediate CUDA errors; asynchronous errors can
surface at synchronization.

```cpp
#include <cstddef>
import anndatapp.cuda;

void upload_example() {
    float input[6] = {1, 2, 3, 4, 5, 6};
    int device = 0;
    anndatapp::cuda::Stream stream{nullptr, device};
    anndatapp::cuda::DeviceBuffer data(sizeof(input), device);
    anndatapp::cuda::copy_to_device(data.span(), 0, input, sizeof(input), stream);
    anndatapp::cuda::DenseView view{data.span(), 0, 2, 3, 3, 1,
                                  anndatapp::cuda::ScalarType::float32};
    anndatapp::cuda::validate(view, true);
    anndatapp::cuda::synchronize(stream);
}
```

`fill_bits` writes the low dtype-width bits unchanged. Numeric `fill` converts
from a double: floating-point conversion rounds to nearest; integer conversion
truncates toward zero and rejects nonfinite or out-of-range values; boolean
fill accepts only zero and one. `cast` supports the declared floating and
integer types. A same-dtype cast preserves bits. FP64 to FP16 converts directly
with round-to-nearest-even; a floating conversion may produce infinity on
overflow and need not preserve a NaN payload. A cast that can produce an
unrepresentable integer requires a caller-owned device `uint32` status: the
operation clears it, writes zero for invalid values, and sets bit 0.

## Bounded selected-data packing

`pack_host_selected` copies selected columns from a strided `HostDenseView`
into caller-provided row-major host storage with the same dtype. It allocates
nothing and preserves bits. A null index pointer requests all columns in order.
Callers explicitly transfer or cast that packed storage as needed; neither
operation hides a whole-input copy.

`gather_columns` packs a device dense view into another same-dtype dense view
using device-resident signed or unsigned integer column IDs. The caller supplies
output, IDs, status, and stream. The checked operation clears status, sets bit 0
for an invalid ID, and writes zero for that output column. It checks IDs even
when there are no rows. `gather_columns_trusted` omits ID checks when the caller
has already established every index is in range. The destination must not
overlap the source or indices; status must not overlap any of them. Packing performs no
implicit dtype conversion; call `cast` explicitly if needed.

These primitives operate on caller-supplied arrays and do not schedule backed
I/O, impose a persistent packed layout, or select a numerical algorithm. Backed
read/transfer contexts can fill reusable host batches before the explicit copy.
See [FP16 storage](fp16.md) and the [device validation receipt](../tests/results/device-data-cuda.json).
