#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

import anndatapp.cuda;

namespace {

using anndatapp::DType;
using namespace anndatapp::cuda;

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template<class F>
void require_throws(F&& operation, std::string_view message) {
    try {
        operation();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(std::string("expected exception: ") + std::string(message));
}

Span metadata_span(void* data, std::uint64_t bytes,
                   Access access = Access::read_write) {
    return {data, bytes, 0, access};
}

DenseView dense_view(Span values, std::uint64_t origin, std::uint64_t rows,
                     std::uint64_t columns, std::int64_t row_stride,
                     std::int64_t column_stride, ScalarType type) {
    return {values, origin, rows, columns, row_stride, column_stride, type};
}

void test_dtype_mapping() {
    const std::array<std::pair<DType, ScalarType>, 12> mappings{{
        {DType::boolean, ScalarType::boolean},
        {DType::int8, ScalarType::int8},
        {DType::uint8, ScalarType::uint8},
        {DType::int16, ScalarType::int16},
        {DType::uint16, ScalarType::uint16},
        {DType::int32, ScalarType::int32},
        {DType::uint32, ScalarType::uint32},
        {DType::int64, ScalarType::int64},
        {DType::uint64, ScalarType::uint64},
        {DType::float16, ScalarType::float16},
        {DType::float32, ScalarType::float32},
        {DType::float64, ScalarType::float64},
    }};
    const std::array<std::size_t, 12> widths{
        1, 1, 1, 2, 2, 4, 4, 8, 8, 2, 4, 8
    };
    for (std::size_t i = 0; i < mappings.size(); ++i) {
        require(scalar_type(mappings[i].first) == mappings[i].second,
                "DType maps to its independent CUDA ABI scalar type");
        require(dtype(mappings[i].second) == mappings[i].first,
                "CUDA ABI scalar type maps back to DType");
        require(scalar_size(mappings[i].second) == widths[i],
                "CUDA ABI scalar type has the declared element width");
    }
    require_throws([] { (void)scalar_type(static_cast<DType>(255)); },
                   "invalid DType mapping");
    require_throws([] { (void)dtype(static_cast<ScalarType>(0)); },
                   "invalid scalar type mapping");
    require_throws([] { (void)scalar_size(static_cast<ScalarType>(99)); },
                   "invalid scalar width");
}

void test_metadata_validation() {
    alignas(8) std::array<std::byte, 64> storage{};
    const auto rw = metadata_span(storage.data(), storage.size());

    // Starting at element four, the negative row stride reaches element zero;
    // the final row/column address is element five.
    const auto reverse = dense_view(rw, 4 * sizeof(float), 3, 2, -2, 1,
                                    ScalarType::float32);
    validate(reverse);

    auto negative_underflow = reverse;
    negative_underflow.origin = sizeof(float);
    require_throws([&] { validate(negative_underflow); },
                   "negative-stride view whose lowest address precedes the span");

    auto positive_overflow = dense_view(rw, 0, 3, 3, 8, 1,
                                        ScalarType::float32);
    require_throws([&] { validate(positive_overflow); },
                   "positive-stride view whose highest address exceeds the span");

    const auto zero_stride = dense_view(rw, 0, 2, 2, 0, 1,
                                        ScalarType::float32);
    require_throws([&] { validate(zero_stride, true); },
                   "writable dense view rejects a zero stride alias");
    const auto colliding_rows = dense_view(rw, 0, 2, 2, 1, 1,
                                           ScalarType::float32);
    require_throws([&] { validate(colliding_rows, true); },
                   "writable dense view rejects colliding row and column strides");

    auto invalid_origin = dense_view(rw, storage.size(), 1, 1, 1, 1,
                                     ScalarType::float32);
    require_throws([&] { validate(invalid_origin); },
                   "nonempty view with origin at the end of its span");

    Span read_only = rw;
    read_only.access = Access::read_only;
    validate(read_only);
    require_throws([&] { validate(read_only, true); },
                   "write validation rejects a read-only span");
    auto read_only_dense = dense_view(read_only, 0, 1, 1, 1, 1,
                                      ScalarType::float32);
    validate(read_only_dense);
    require_throws([&] { validate(read_only_dense, true); },
                   "write validation rejects a read-only dense view");

    require_throws(
        [] { validate(Span{nullptr, 4, 0, Access::read_write}); },
        "nonzero span capacity requires a pointer");
    require_throws(
        [] { validate(Span{reinterpret_cast<void*>(std::uintptr_t{1}), 4, -1,
                           Access::read_write}); },
        "span rejects a negative device ordinal");
    auto invalid_access = rw;
    invalid_access.access = static_cast<Access>(99);
    require_throws([&] { validate(invalid_access); },
                   "span rejects an unknown access value");

    // Empty views do not address storage, even when their strides would
    // overflow address arithmetic for a nonempty extent.
    const Span empty_span{nullptr, 0, 0, Access::read_only};
    const auto empty_dense = dense_view(
        empty_span, 0, 0, 3, INT64_MAX, INT64_MIN, ScalarType::float32);
    validate(empty_dense);

    std::array<float, 2> sparse_values{};
    std::array<std::int32_t, 2> sparse_indices{};
    std::array<std::int32_t, 3> csr_offsets{};
    SparseView csr{
        metadata_span(sparse_values.data(), sizeof(sparse_values)),
        metadata_span(sparse_indices.data(), sizeof(sparse_indices)),
        metadata_span(csr_offsets.data(), sizeof(csr_offsets)),
        2, 3, 2, ScalarType::float32, ScalarType::int32, ScalarType::int32,
        SparseLayout::csr
    };
    validate(csr);

    std::array<double, 1> csc_values{};
    std::array<std::int64_t, 1> csc_indices{};
    std::array<std::int64_t, 4> csc_offsets{};
    SparseView csc{
        metadata_span(csc_values.data(), sizeof(csc_values)),
        metadata_span(csc_indices.data(), sizeof(csc_indices)),
        metadata_span(csc_offsets.data(), sizeof(csc_offsets)),
        2, 3, 1, ScalarType::float64, ScalarType::int64, ScalarType::int64,
        SparseLayout::csc
    };
    validate(csc);

    auto short_offsets = csr;
    short_offsets.offsets.bytes = 2 * sizeof(std::int32_t);
    require_throws([&] { validate(short_offsets); },
                   "CSR offsets require rows plus one entries");

    auto short_values = csr;
    short_values.values.bytes = sizeof(float);
    require_throws([&] { validate(short_values); },
                   "sparse values capacity covers every nonzero");

    auto read_only_indices = csr;
    read_only_indices.indices.access = Access::read_only;
    validate(read_only_indices);
    require_throws([&] { validate(read_only_indices, true); },
                   "sparse write validation checks every component access");

    auto noninteger_indices = csr;
    noninteger_indices.index_type = ScalarType::float32;
    require_throws([&] { validate(noninteger_indices); },
                   "sparse indices reject floating-point element types");

    auto invalid_layout = csr;
    invalid_layout.layout = static_cast<SparseLayout>(0);
    require_throws([&] { validate(invalid_layout); },
                   "sparse metadata rejects an unknown layout");

    // A zero-row CSR still has its one-element offsets array, but no values or
    // indices. This keeps the empty shape distinct from missing metadata.
    std::array<std::int32_t, 1> empty_offsets{};
    SparseView empty_sparse{
        Span{nullptr, 0, 0, Access::read_only},
        Span{nullptr, 0, 0, Access::read_only},
        metadata_span(empty_offsets.data(), sizeof(empty_offsets), Access::read_only),
        0, 3, 0, ScalarType::float32, ScalarType::int32, ScalarType::int32,
        SparseLayout::csr
    };
    validate(empty_sparse);
}

void test_host_selected_packing() {
    // Float payloads are deliberately stored and compared as raw bits. This
    // covers signed zero and noncanonical NaN payloads without normalizing them.
    const std::array<std::uint32_t, 12> bits{
        0x80000000u, 0x7fc12345u, 0x00000001u, 0xff800000u,
        0x7f800001u, 0x3f800000u, 0xdeadbeefu, 0x00000000u,
        0x7fc54321u, 0xbf800000u, 0x00800000u, 0x12345678u
    };
    const HostDenseView reversed{
        bits.data(), sizeof(bits), 8 * sizeof(std::uint32_t), 3, 2,
        -3, 2, ScalarType::float32
    };
    const std::array<std::uint64_t, 2> selected{1, 0};
    std::array<std::uint32_t, 6> packed{};
    pack_host_selected(reversed, selected.data(), selected.size(), packed.data(),
                       sizeof(packed));
    const std::array<std::uint32_t, 6> expected{
        bits[10], bits[8], bits[7], bits[5], bits[4], bits[2]
    };
    require(std::memcmp(packed.data(), expected.data(), sizeof(packed)) == 0,
            "selected-column packing preserves exact bits through negative strides");

    std::array<std::uint32_t, 6> all_columns{};
    pack_host_selected(reversed, nullptr, 2, all_columns.data(),
                       sizeof(all_columns));
    const std::array<std::uint32_t, 6> expected_all{
        bits[8], bits[10], bits[5], bits[7], bits[2], bits[4]
    };
    require(std::memcmp(all_columns.data(), expected_all.data(),
                        sizeof(all_columns)) == 0,
            "null column selection packs all columns in their source order");

    require_throws(
        [&] {
            const std::array<std::uint64_t, 1> bad_column{2};
            pack_host_selected(reversed, bad_column.data(), bad_column.size(),
                               packed.data(), sizeof(packed));
        },
        "selected-column packing rejects an out-of-range column");
    require_throws(
        [&] {
            pack_host_selected(reversed, selected.data(), selected.size(),
                               packed.data(), sizeof(packed) - 1);
        },
        "selected-column packing rejects a short destination");
    require_throws(
        [&] {
            pack_host_selected(reversed, nullptr, 1, packed.data(),
                               sizeof(packed));
        },
        "all-column packing requires count to match source width");
    require_throws(
        [&] {
            auto bad_origin = reversed;
            bad_origin.origin = bad_origin.bytes;
            pack_host_selected(bad_origin, selected.data(), selected.size(),
                               packed.data(), sizeof(packed));
        },
        "selected-column packing rejects an out-of-range source origin");

    const HostDenseView empty{
        nullptr, 0, 0, 0, 3, 3, 1, ScalarType::float32
    };
    pack_host_selected(empty, nullptr, 3, nullptr, 0);
}

void test_host_only() {
    test_dtype_mapping();
    test_metadata_validation();
    test_host_selected_packing();
}

void cuda_check(cudaError_t status, std::string_view operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

struct StreamGuard {
    cudaStream_t value = nullptr;
    ~StreamGuard() {
        if (value) (void)cudaStreamDestroy(value);
    }
};

struct AdoptState { int releases = 0; };

void release_adopted(void* context, void* pointer, std::size_t, int,
                     void*) noexcept {
    auto* state = static_cast<AdoptState*>(context);
    ++state->releases;
    (void)cudaFree(pointer);
}

struct AllocatorState {
    int allocations = 0;
    int releases = 0;
    std::size_t allocated_bytes = 0;
    std::size_t released_bytes = 0;
    int allocated_device = -1;
    int released_device = -1;
    void* allocated_stream = nullptr;
    void* released_stream = nullptr;
};

void* allocate_foreign(void* context, std::size_t bytes, int device,
                       void* stream) {
    auto* state = static_cast<AllocatorState*>(context);
    ++state->allocations;
    state->allocated_bytes = bytes;
    state->allocated_device = device;
    state->allocated_stream = stream;
    void* pointer = nullptr;
    if (cudaMalloc(&pointer, bytes) != cudaSuccess) return nullptr;
    return pointer;
}

void release_foreign(void* context, void* pointer, std::size_t bytes,
                     int device, void* stream) noexcept {
    auto* state = static_cast<AllocatorState*>(context);
    ++state->releases;
    state->released_bytes = bytes;
    state->released_device = device;
    state->released_stream = stream;
    (void)cudaFree(pointer);
}

DenseView vector_view(Span span, std::size_t count, ScalarType dtype) {
    return dense_view(span, 0, 1, count, static_cast<std::int64_t>(count), 1,
                      dtype);
}

void test_gpu() {
    int device = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    StreamGuard native_stream;
    cuda_check(cudaStreamCreateWithFlags(&native_stream.value,
                                         cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
    const Stream stream{reinterpret_cast<void*>(native_stream.value), device};

    // Offset H2D, D2D, zero, and D2H operations are all ordered on this
    // explicit non-default stream.
    const std::array<float, 4> input{1.25f, -2.5f, 8.0f, 0.5f};
    std::array<float, 6> copied{};
    DeviceBuffer first(6 * sizeof(float), device);
    DeviceBuffer second(6 * sizeof(float), device);
    zero(second.span(), 0, second.size_bytes(), stream);
    copy_to_device(first.span(), sizeof(float), input.data(), sizeof(input), stream);
    zero(first.span(), 0, sizeof(float), stream);
    zero(first.span(), 5 * sizeof(float), sizeof(float), stream);
    copy_device(second.span(), sizeof(float), first.span(), sizeof(float),
                sizeof(input), stream);
    copy_to_host(copied.data(), second.span(), 0, sizeof(copied), stream);

    // Numeric floating fill, exact low-width bit fill, and a strided write.
    DeviceBuffer fill_buffer(4 * sizeof(float), device);
    fill(vector_view(fill_buffer.span(), 4, ScalarType::float32), 3.25, stream);
    std::array<float, 4> filled{};
    copy_to_host(filled.data(), fill_buffer.span(), 0, sizeof(filled), stream);

    DeviceBuffer bit_buffer(4 * sizeof(std::uint32_t), device);
    constexpr std::uint32_t bit_pattern = 0x7fc12345u;
    fill_bits(vector_view(bit_buffer.span(), 4, ScalarType::uint32),
              bit_pattern, stream);
    std::array<std::uint32_t, 4> bit_filled{};
    copy_to_host(bit_filled.data(), bit_buffer.span(), 0, sizeof(bit_filled),
                 stream);

    DeviceBuffer strided_buffer(8 * sizeof(float), device);
    const auto strided_all = vector_view(strided_buffer.span(), 8,
                                         ScalarType::float32);
    fill(strided_all, -9.0, stream);
    auto strided = dense_view(strided_buffer.span(), sizeof(float), 2, 2, 3, 1,
                              ScalarType::float32);
    fill(strided, 6.5, stream);
    std::array<float, 8> strided_result{};
    copy_to_host(strided_result.data(), strided_buffer.span(), 0,
                 sizeof(strided_result), stream);

    // Exactly representable values exercise the FP32 -> FP16 -> FP32 path.
    const std::array<float, 4> cast_input{1.5f, -2.25f, 0.5f, 65504.0f};
    std::array<float, 4> cast_output{};
    DeviceBuffer cast_source(sizeof(cast_input), device);
    DeviceBuffer cast_half(4 * sizeof(std::uint16_t), device);
    DeviceBuffer cast_result(sizeof(cast_output), device);
    copy_to_device(cast_source.span(), 0, cast_input.data(), sizeof(cast_input),
                   stream);
    cast(vector_view(cast_half.span(), 4, ScalarType::float16),
         vector_view(cast_source.span(), 4, ScalarType::float32), stream);
    cast(vector_view(cast_result.span(), 4, ScalarType::float32),
         vector_view(cast_half.span(), 4, ScalarType::float16), stream);
    copy_to_host(cast_output.data(), cast_result.span(), 0, sizeof(cast_output),
                 stream);

    // A value just above a binary16 midpoint distinguishes direct FP64->FP16
    // round-to-nearest from an FP32 intermediate.
    const double double_rounding_input = 1.00048828125 + 0x1p-25;
    std::uint16_t double_rounding_output = 0;
    DeviceBuffer double_source(sizeof(double_rounding_input), device);
    DeviceBuffer double_half(sizeof(double_rounding_output), device);
    copy_to_device(double_source.span(), 0, &double_rounding_input,
                   sizeof(double_rounding_input), stream);
    cast(vector_view(double_half.span(), 1, ScalarType::float16),
         vector_view(double_source.span(), 1, ScalarType::float64), stream);
    copy_to_host(&double_rounding_output, double_half.span(), 0,
                 sizeof(double_rounding_output), stream);

    // Signed-to-unsigned conversion reports the negative input and keeps the
    // adjacent value above the exact-integer range of FP64 intact.
    const std::array<std::int64_t, 2> integer_cast_input{
        -1, 9007199254740993LL
    };
    std::array<std::uint64_t, 2> integer_cast_output{};
    std::uint32_t integer_cast_status = 0xffffffffu;
    DeviceBuffer integer_source(sizeof(integer_cast_input), device);
    DeviceBuffer integer_result(sizeof(integer_cast_output), device);
    DeviceBuffer integer_status(sizeof(integer_cast_status), device);
    copy_to_device(integer_source.span(), 0, integer_cast_input.data(),
                   sizeof(integer_cast_input), stream);
    cast(vector_view(integer_result.span(), 2, ScalarType::uint64),
         vector_view(integer_source.span(), 2, ScalarType::int64), stream,
         integer_status.span());
    copy_to_host(integer_cast_output.data(), integer_result.span(), 0,
                 sizeof(integer_cast_output), stream);
    copy_to_host(&integer_cast_status, integer_status.span(), 0,
                 sizeof(integer_cast_status), stream);

    // Retain an allocator-owned allocation through a foreign owner token.
    int retained_release_count = 0;
    void* retained_pointer = nullptr;
    constexpr std::size_t retained_bytes = 4 * sizeof(std::uint32_t);
    cuda_check(cudaMalloc(&retained_pointer, retained_bytes), "cudaMalloc retained");
    {
        std::shared_ptr<void> owner(retained_pointer, [&](void* pointer) {
            ++retained_release_count;
            (void)cudaFree(pointer);
        });
        auto retained = DeviceBuffer::retain(
            Span{retained_pointer, retained_bytes, device, Access::read_write},
            owner);
        owner.reset();
        require(retained_release_count == 0,
                "retained foreign owner stays alive while DeviceBuffer exists");
        const std::array<std::uint32_t, 4> retained_input{
            17u, 29u, 41u, 53u
        };
        std::array<std::uint32_t, 4> retained_output{};
        copy_to_device(retained.span(), 0, retained_input.data(),
                       sizeof(retained_input), stream);
        copy_to_host(retained_output.data(), retained.span(), 0,
                     sizeof(retained_output), stream);
        synchronize(stream);
        require(retained_output == retained_input,
                "retained foreign buffer participates in stream transfers");
    }
    require(retained_release_count == 1,
            "retained foreign owner releases after the buffer is destroyed");

    // Adopt invokes the supplied release callback exactly once.
    AdoptState adopted_state;
    void* adopted_pointer = nullptr;
    constexpr std::size_t adopted_bytes = sizeof(std::uint32_t);
    cuda_check(cudaMalloc(&adopted_pointer, adopted_bytes), "cudaMalloc adopted");
    {
        Allocator allocator;
        allocator.context = &adopted_state;
        allocator.release = &release_adopted;
        allocator.keepalive = std::make_shared<int>(7);
        auto adopted = DeviceBuffer::adopt(
            Span{adopted_pointer, adopted_bytes, device, Access::read_write},
            allocator, native_stream.value);
        const std::uint32_t adopted_value = 73;
        std::uint32_t adopted_result = 0;
        copy_to_device(adopted.span(), 0, &adopted_value, sizeof(adopted_value),
                       stream);
        copy_to_host(&adopted_result, adopted.span(), 0, sizeof(adopted_result),
                     stream);
        synchronize(stream);
        require(adopted_result == adopted_value,
                "adopted foreign buffer participates in stream transfers");
    }
    require(adopted_state.releases == 1,
            "adopted foreign allocation invokes its release callback once");

    // The paired allocator callbacks receive the same extent, device, and
    // stream, and release only after queued work has completed.
    AllocatorState allocator_state;
    Allocator custom_allocator;
    custom_allocator.context = &allocator_state;
    custom_allocator.allocate = &allocate_foreign;
    custom_allocator.release = &release_foreign;
    custom_allocator.keepalive = std::make_shared<int>(11);
    const std::uint32_t custom_value = 101;
    std::uint32_t custom_result = 0;
    constexpr std::size_t custom_bytes = sizeof(custom_value);
    {
        DeviceBuffer custom(custom_bytes, device, custom_allocator,
                            native_stream.value);
        copy_to_device(custom.span(), 0, &custom_value, sizeof(custom_value),
                       stream);
        copy_to_host(&custom_result, custom.span(), 0, sizeof(custom_result),
                     stream);
        synchronize(stream);
        require(custom_result == custom_value,
                "custom allocator buffer participates in explicit-stream transfers");
    }
    require(allocator_state.allocations == 1 && allocator_state.releases == 1 &&
            allocator_state.allocated_bytes == custom_bytes &&
            allocator_state.released_bytes == custom_bytes &&
            allocator_state.allocated_device == device &&
            allocator_state.released_device == device &&
            allocator_state.allocated_stream == native_stream.value &&
            allocator_state.released_stream == native_stream.value,
            "custom allocator callbacks preserve allocation context and stream");

    // Device sparse validation reports both malformed offsets and indices.
    const std::array<float, 2> sparse_values{4.0f, 5.0f};
    const std::array<std::int32_t, 2> sparse_indices{0, 3};
    const std::array<std::int32_t, 3> sparse_offsets{0, 2, 1};
    std::uint32_t sparse_status = 0xffffffffu;
    DeviceBuffer values_buffer(sizeof(sparse_values), device);
    DeviceBuffer indices_buffer(sizeof(sparse_indices), device);
    DeviceBuffer offsets_buffer(sizeof(sparse_offsets), device);
    DeviceBuffer status_buffer(sizeof(sparse_status), device);
    copy_to_device(values_buffer.span(), 0, sparse_values.data(),
                   sizeof(sparse_values), stream);
    copy_to_device(indices_buffer.span(), 0, sparse_indices.data(),
                   sizeof(sparse_indices), stream);
    copy_to_device(offsets_buffer.span(), 0, sparse_offsets.data(),
                   sizeof(sparse_offsets), stream);
    SparseView sparse{
        values_buffer.span(), indices_buffer.span(), offsets_buffer.span(),
        2, 3, 2, ScalarType::float32, ScalarType::int32, ScalarType::int32,
        SparseLayout::csr
    };
    validate(sparse);
    validate_sparse_contents(
        sparse, status_buffer.span(), stream);
    copy_to_host(&sparse_status, status_buffer.span(), 0,
                 sizeof(sparse_status), stream);

    // CSC permits duplicate row coordinates, and values, indices, and offsets
    // use independently declared widths.
    const std::array<float, 2> csc_values{3.0f, -4.0f};
    const std::array<std::int64_t, 2> csc_indices{1, 1};
    const std::array<std::uint32_t, 3> csc_offsets{0, 2, 2};
    std::uint32_t csc_status = 0xffffffffu;
    DeviceBuffer csc_values_buffer(sizeof(csc_values), device);
    DeviceBuffer csc_indices_buffer(sizeof(csc_indices), device);
    DeviceBuffer csc_offsets_buffer(sizeof(csc_offsets), device);
    DeviceBuffer csc_status_buffer(sizeof(csc_status), device);
    copy_to_device(csc_values_buffer.span(), 0, csc_values.data(),
                   sizeof(csc_values), stream);
    copy_to_device(csc_indices_buffer.span(), 0, csc_indices.data(),
                   sizeof(csc_indices), stream);
    copy_to_device(csc_offsets_buffer.span(), 0, csc_offsets.data(),
                   sizeof(csc_offsets), stream);
    copy_to_device(csc_status_buffer.span(), 0, &csc_status,
                   sizeof(csc_status), stream);
    SparseView csc{
        csc_values_buffer.span(), csc_indices_buffer.span(),
        csc_offsets_buffer.span(), 3, 2, 2, ScalarType::float32,
        ScalarType::int64, ScalarType::uint32, SparseLayout::csc
    };
    validate(csc);
    validate_sparse_contents(csc, csc_status_buffer.span(), stream);
    copy_to_host(&csc_status, csc_status_buffer.span(), 0, sizeof(csc_status),
                 stream);

    // A checked gather still validates selected indices for an empty matrix:
    // there are no output values to write, but the selection remains an input.
    const std::int32_t empty_gather_index = 7;
    std::uint32_t empty_gather_status = 0xffffffffu;
    DeviceBuffer empty_gather_source(0, device);
    DeviceBuffer empty_gather_destination(0, device);
    DeviceBuffer empty_gather_columns(sizeof(empty_gather_index), device);
    DeviceBuffer empty_gather_status_buffer(sizeof(empty_gather_status), device);
    copy_to_device(empty_gather_columns.span(), 0, &empty_gather_index,
                   sizeof(empty_gather_index), stream);
    copy_to_device(empty_gather_status_buffer.span(), 0, &empty_gather_status,
                   sizeof(empty_gather_status), stream);
    const auto empty_gather_source_view = dense_view(
        empty_gather_source.span(Access::read_only), 0, 0, 2, 2, 1,
        ScalarType::float32);
    const auto empty_gather_destination_view = dense_view(
        empty_gather_destination.span(), 0, 0, 1, 1, 1,
        ScalarType::float32);
    gather_columns(empty_gather_destination_view, empty_gather_source_view,
                   empty_gather_columns.span(Access::read_only),
                   ScalarType::int32, stream,
                   empty_gather_status_buffer.span());
    copy_to_host(&empty_gather_status, empty_gather_status_buffer.span(), 0,
                 sizeof(empty_gather_status), stream);

    synchronize(stream);

    require(copied[0] == 0.0f && copied[1] == input[0] &&
            copied[2] == input[1] && copied[3] == input[2] &&
            copied[4] == input[3] && copied[5] == 0.0f,
            "explicit-stream H2D, D2D, zero, and D2H preserve order and offsets");
    for (const auto value : filled)
        require(value == 3.25f, "floating fill writes the requested value");
    for (const auto value : bit_filled)
        require(value == bit_pattern, "bit fill writes exact element bits");
    const std::array<float, 8> expected_strided{
        -9.0f, 6.5f, 6.5f, -9.0f, 6.5f, 6.5f, -9.0f, -9.0f
    };
    require(strided_result == expected_strided,
            "strided dense fill updates only selected logical elements");
    require(cast_output == cast_input,
            "FP32 to FP16 to FP32 cast preserves exactly representable values");
    require(double_rounding_output == 0x3c01u,
            "FP64 to FP16 conversion rounds directly at a half-way counterexample");
    require(integer_cast_output[0] == 0 &&
            integer_cast_output[1] == 9007199254740993ull,
            "integer cast zeroes the invalid value and preserves an exact large integer");
    require(integer_cast_status == 1u,
            "integer cast sets its status for a negative-to-unsigned value");
    require(sparse_status == 3u,
            "sparse validation sets offset and index error bits");
    require(csc_status == 0u,
            "CSC sparse validation accepts duplicate indices and mixed widths");
    require(empty_gather_status == 1u,
            "checked gather reports an invalid selection for zero output rows");

    // Gather preserves payload bits, follows source column order exactly, and
    // supports a negative row stride in the source metadata.
    const std::array<std::uint32_t, 12> gather_source_bits{
        0x80000000u, 0x7fc12345u, 0x00000001u, 0xff800000u,
        0x7f800001u, 0x3f800000u, 0xdeadbeefu, 0x00000000u,
        0x7fc54321u, 0xbf800000u, 0x00800000u, 0x12345678u
    };
    const std::array<std::int32_t, 2> gather_columns_valid{2, 0};
    const std::array<std::int32_t, 2> gather_columns_invalid{1, 9};
    std::array<std::uint32_t, 6> gather_checked_output{};
    std::array<std::uint32_t, 6> gather_trusted_output{};
    std::array<std::uint32_t, 6> gather_invalid_output{};
    std::uint32_t gather_valid_status = 0xffffffffu;
    std::uint32_t gather_invalid_status = 0xffffffffu;
    DeviceBuffer gather_source(sizeof(gather_source_bits), device);
    DeviceBuffer gather_checked(sizeof(gather_checked_output), device);
    DeviceBuffer gather_trusted(sizeof(gather_trusted_output), device);
    DeviceBuffer gather_invalid(sizeof(gather_invalid_output), device);
    DeviceBuffer gather_valid_indices(sizeof(gather_columns_valid), device);
    DeviceBuffer gather_invalid_indices(sizeof(gather_columns_invalid), device);
    DeviceBuffer gather_valid_status_buffer(sizeof(gather_valid_status), device);
    DeviceBuffer gather_invalid_status_buffer(sizeof(gather_invalid_status), device);
    copy_to_device(gather_source.span(), 0, gather_source_bits.data(),
                   sizeof(gather_source_bits), stream);
    copy_to_device(gather_valid_indices.span(), 0, gather_columns_valid.data(),
                   sizeof(gather_columns_valid), stream);
    copy_to_device(gather_invalid_indices.span(), 0, gather_columns_invalid.data(),
                   sizeof(gather_columns_invalid), stream);
    copy_to_device(gather_valid_status_buffer.span(), 0, &gather_valid_status,
                   sizeof(gather_valid_status), stream);
    copy_to_device(gather_invalid_status_buffer.span(), 0, &gather_invalid_status,
                   sizeof(gather_invalid_status), stream);
    const auto gather_source_view = dense_view(
        gather_source.span(Access::read_only), 8 * sizeof(std::uint32_t),
        3, 3, -4, 1, ScalarType::float32);
    const auto gather_checked_view = dense_view(
        gather_checked.span(), 0, 3, 2, 2, 1, ScalarType::float32);
    const auto gather_trusted_view = dense_view(
        gather_trusted.span(), 0, 3, 2, 2, 1, ScalarType::float32);
    const auto gather_invalid_view = dense_view(
        gather_invalid.span(), 0, 3, 2, 2, 1, ScalarType::float32);
    gather_columns(
        gather_checked_view, gather_source_view,
        gather_valid_indices.span(Access::read_only), ScalarType::int32,
        stream, gather_valid_status_buffer.span());
    gather_columns_trusted(
        gather_trusted_view, gather_source_view,
        gather_valid_indices.span(Access::read_only), ScalarType::int32, stream);
    gather_columns(
        gather_invalid_view, gather_source_view,
        gather_invalid_indices.span(Access::read_only), ScalarType::int32,
        stream, gather_invalid_status_buffer.span());
    copy_to_host(gather_checked_output.data(), gather_checked.span(), 0,
                 sizeof(gather_checked_output), stream);
    copy_to_host(gather_trusted_output.data(), gather_trusted.span(), 0,
                 sizeof(gather_trusted_output), stream);
    copy_to_host(gather_invalid_output.data(), gather_invalid.span(), 0,
                 sizeof(gather_invalid_output), stream);
    copy_to_host(&gather_valid_status, gather_valid_status_buffer.span(), 0,
                 sizeof(gather_valid_status), stream);
    copy_to_host(&gather_invalid_status, gather_invalid_status_buffer.span(), 0,
                 sizeof(gather_invalid_status), stream);
    synchronize(stream);

    const std::array<std::uint32_t, 6> gather_expected{
        gather_source_bits[10], gather_source_bits[8],
        gather_source_bits[6], gather_source_bits[4],
        gather_source_bits[2], gather_source_bits[0]
    };
    const std::array<std::uint32_t, 6> gather_invalid_expected{
        gather_source_bits[9], 0u, gather_source_bits[5], 0u,
        gather_source_bits[1], 0u
    };
    require(gather_checked_output == gather_expected,
            "checked gather preserves selected bit patterns and selection order");
    require(gather_trusted_output == gather_expected,
            "trusted gather accepts valid indices with negative source strides");
    require(gather_valid_status == 0u,
            "checked gather clears status for valid indices");
    require(gather_invalid_output == gather_invalid_expected,
            "checked gather zeroes the output column for an invalid index");
    require(gather_invalid_status == 1u,
            "checked gather sets status bit zero for an invalid index");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            test_host_only();
            std::cout << "anndatapp CUDA metadata and host packing tests passed\n";
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--gpu") {
            test_host_only();
            test_gpu();
            std::cout << "anndatapp CUDA host and GPU tests passed\n";
            return 0;
        }
        throw std::runtime_error("usage: test_cuda [--gpu]");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
