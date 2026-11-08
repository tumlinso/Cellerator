module;
#include <anndatapp/device_abi.h>
#include <cstddef>
#include <cstdint>
#include <memory>

export module anndatapp.cuda;
export import anndatapp.core;

export namespace anndatapp::cuda {

using device::Access;
using device::DenseView;
using device::ScalarType;
using device::Span;
using device::SparseLayout;
using device::SparseView;

// native is a cudaStream_t represented without a CUDA header; nullptr denotes
// the default stream. The caller keeps it and all referenced buffers alive and
// orders their subsequent use until submitted work completes.
struct Stream { void* native = nullptr; int device = -1; };

// Custom callbacks use the declared device and native stream. release must not
// throw. keepalive retains any allocator state through release.
struct Allocator {
    void* context = nullptr;
    void* (*allocate)(void*, std::size_t, int, void*) = nullptr;
    void (*release)(void*, void*, std::size_t, int, void*) noexcept = nullptr;
    std::shared_ptr<void> keepalive;
};

namespace detail { struct BufferState; }
class DeviceBuffer {
public:
    DeviceBuffer() noexcept;
    explicit DeviceBuffer(std::size_t bytes, int device);
    DeviceBuffer(std::size_t bytes, int device, Allocator allocator, void* native_stream = nullptr);
    // Adopt invokes allocator.release at destruction; retain holds keepalive
    // but never frees its pointer. Both retain the source's access permission.
    static DeviceBuffer adopt(Span, Allocator, void* native_stream = nullptr);
    static DeviceBuffer retain(Span, std::shared_ptr<void> keepalive);
    ~DeviceBuffer();
    DeviceBuffer(DeviceBuffer&&) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* data() const noexcept;
    std::size_t size_bytes() const noexcept;
    int device() const noexcept;
    Span span(Access access = Access::read_write) const;
private:
    std::unique_ptr<detail::BufferState> state_;
};

// Checked enum conversions; ABI ScalarType values are deliberately independent
// of anndatapp::DType values. Unsupported/invalid values throw.
ScalarType scalar_type(DType);
DType dtype(ScalarType);
std::size_t scalar_size(ScalarType);

// These check metadata, spans and representable address ranges only. They do
// not dereference a device pointer or synchronize. A writable sparse check
// requires all three spans writable; individual spans can be checked instead.
// Sparse contents can be checked separately by an asynchronous operation.
void validate(Span, bool require_write = false);
void validate(DenseView, bool require_write = false);
void validate(SparseView, bool require_write = false);

// Caller supplies a writable uint32 device status. The operation clears it,
// then sets bit 0 for invalid offsets and bit 1 for invalid indices. Duplicate
// sparse coordinates are allowed. Status and input remain alive until stream
// completion; no host synchronization occurs.
void validate_sparse_contents(SparseView, Span device_status, Stream);

// Host operations below enqueue work; pageable host memory can cause the CUDA
// runtime to stage or block. Buffers and stream must remain valid until complete.
void copy_to_device(Span destination, std::size_t offset, const void* source,
                    std::size_t bytes, Stream);
void copy_to_host(void* destination, Span source, std::size_t offset,
                  std::size_t bytes, Stream);
void copy_device(Span destination, std::size_t destination_offset, Span source,
                 std::size_t source_offset, std::size_t bytes, Stream);
void zero(Span destination, std::size_t offset, std::size_t bytes, Stream);
void synchronize(Stream);

// fill_bits writes the low dtype-width bits unchanged at each logical element.
// fill converts a double to the destination type. Integer conversions truncate
// toward zero and reject nonfinite/out-of-range values; boolean fill accepts
// only 0 and 1. cast converts numeric types; for an invalid integer destination
// it writes zero and sets bit 0 in device_status. An absent status is permitted
// only when integer conversion cannot fail. Same-dtype cast preserves bits;
// floating conversion uses round-to-nearest and may produce infinity on overflow.
void fill_bits(DenseView destination, std::uint64_t bits, Stream);
void fill(DenseView destination, double value, Stream);
void cast(DenseView destination, DenseView source, Stream, Span device_status = {});

// Copy selected source columns to a same-dtype destination, preserving raw
// bits. columns stores destination.columns signed or unsigned integer indices.
// The checked form clears status and sets bit 0 for an invalid index, writing
// zero in its output column. The trusted form requires every index in range.
// The source, destination, columns, status, and stream must remain alive until
// stream completion; views and status must not overlap writable storage.
void gather_columns(DenseView destination, DenseView source, Span columns,
                    ScalarType index_type, Stream, Span device_status);
void gather_columns_trusted(DenseView destination, DenseView source, Span columns,
                            ScalarType index_type, Stream);

struct HostDenseView {
    const void* data = nullptr;
    std::uint64_t bytes = 0, origin = 0, rows = 0, columns = 0;
    std::int64_t row_stride = 0, column_stride = 0;
    ScalarType dtype = ScalarType::float32;
};
// Copies selected columns in caller-provided row-major host storage with the
// same dtype, preserving bits. `columns == nullptr` requests all columns in
// order and count must equal source.columns. No allocation or transfer occurs.
void pack_host_selected(HostDenseView source, const std::uint64_t* columns,
                        std::size_t count, void* destination, std::size_t capacity);

}
