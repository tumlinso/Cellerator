#include <CellShard/access.hh>

#if CELLSHARD_ENABLE_CUDA
#include <CellShard/formats/compressed_device_view.cuh>
#include <CellShard/formats/triplet.cuh>
#include <CellShard/formats/triplet_device_view.cuh>
#endif

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace {

#if CELLSHARD_ENABLE_CUDA
namespace formats = ::cellshard;
namespace types = ::cellshard::types;

static_assert(offsetof(formats::device::CompressedView, major_ptr)
    < offsetof(formats::device::CompressedView, minor_idx));
static_assert(offsetof(formats::device::TripletView, rowIdx)
    < offsetof(formats::device::TripletView, colIdx));

void require(bool condition, const char *message) {
    if (condition) return;
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

bool close_half(__half value, float expected) {
    return __builtin_fabsf(__half2float(value) - expected) < 1.0e-3f;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return ((value + alignment - 1u) / alignment) * alignment;
}

void check_dense() {
    formats::dense matrix{};
    formats::init(&matrix, 2u, 3u);
    require(formats::payload_elements(&matrix) == 6u, "dense element count changed");
    require(formats::payload_bytes(&matrix) == 6u * sizeof(types::storage_value_t), "dense payload byte count changed");
    require(formats::bytes(&matrix) == sizeof(matrix) + 6u * sizeof(types::storage_value_t), "dense descriptor byte count changed");
    require(formats::allocate(&matrix) != 0, "dense allocation failed");
    require(matrix.storage == matrix.val, "dense value pointer no longer begins at owned storage");
    *formats::at(&matrix, 1u, 2u) = __float2half(7.0f);
    require(close_half(*formats::at(&matrix, 1u, 2u), 7.0f), "dense row-major mapping changed");
    require(formats::at(&matrix, 2u, 0u) == nullptr, "dense bounds check changed");
    formats::clear(&matrix);

    __half columns[8]{};
    formats::attach(&matrix, 3u, 2u, columns, formats::dense_col_major, 4u);
    columns[1u * 4u + 2u] = __float2half(9.0f);
    require(formats::offset(&matrix, 2u, 1u) == 6u, "dense column-major offset changed");
    require(close_half(*formats::at(&matrix, 2u, 1u), 9.0f), "dense column-major mapping changed");
}

void check_compressed() {
    formats::compressed matrix{};
    formats::init(&matrix, 3u, 4u, 3u, formats::compressed_by_row);
    require(formats::allocate(&matrix) != 0, "compressed allocation failed");
    require(matrix.storage == matrix.majorPtr, "compressed major pointer base changed");
    const std::size_t ptr_bytes = 4u * sizeof(types::ptr_t);
    const std::size_t minor_offset = align_up(ptr_bytes, alignof(types::idx_t));
    const std::size_t value_offset = align_up(minor_offset + 3u * sizeof(types::idx_t), alignof(types::storage_value_t));
    require(reinterpret_cast<unsigned char *>(matrix.minorIdx) == static_cast<unsigned char *>(matrix.storage) + minor_offset,
            "compressed minor-index offset changed");
    require(reinterpret_cast<unsigned char *>(matrix.val) == static_cast<unsigned char *>(matrix.storage) + value_offset,
            "compressed value offset changed");
    const types::ptr_t row_ptr[] = {0u, 2u, 2u, 3u};
    const types::idx_t columns[] = {0u, 3u, 2u};
    std::memcpy(matrix.majorPtr, row_ptr, sizeof(row_ptr));
    std::memcpy(matrix.minorIdx, columns, sizeof(columns));
    matrix.val[0] = __float2half(1.0f);
    matrix.val[1] = __float2half(3.0f);
    matrix.val[2] = __float2half(6.0f);
    require(formats::bytes(&matrix) == sizeof(matrix) + 4u * sizeof(types::ptr_t)
                + 3u * sizeof(types::idx_t) + 3u * sizeof(types::storage_value_t),
            "compressed descriptor/payload byte count changed");
    require(formats::major_dim(&matrix) == 3u && formats::minor_dim(&matrix) == 4u, "compressed axes changed");
    require(close_half(*formats::at(&matrix, 0u, 3u), 3.0f), "compressed row lookup changed");
    require(formats::at(&matrix, 1u, 0u) == nullptr, "compressed absent lookup changed");
    formats::clear(&matrix);

    formats::init(&matrix, 3u, 4u, 2u, formats::compressed_by_col);
    require(formats::allocate(&matrix) != 0, "CSC allocation failed");
    const types::ptr_t col_ptr[] = {0u, 0u, 1u, 1u, 2u};
    const types::idx_t rows[] = {2u, 0u};
    std::memcpy(matrix.majorPtr, col_ptr, sizeof(col_ptr));
    std::memcpy(matrix.minorIdx, rows, sizeof(rows));
    matrix.val[0] = __float2half(8.0f); matrix.val[1] = __float2half(9.0f);
    require(formats::major_dim(&matrix) == 4u && formats::minor_dim(&matrix) == 3u, "CSC axes changed");
    require(close_half(*formats::at(&matrix, 2u, 1u), 8.0f), "CSC lookup changed");
    require(close_half(*formats::at(&matrix, 0u, 3u), 9.0f), "CSC second lookup changed");
    formats::clear(&matrix);
}

void check_coo() {
    formats::triplet matrix{};
    formats::init(&matrix, 2u, 3u, 2u);
    require(formats::allocate(&matrix) != 0, "COO allocation failed");
    require(matrix.storage == matrix.rowIdx, "COO row-index base changed");
    require(formats::bytes(&matrix) == sizeof(matrix) + 4u * sizeof(types::idx_t)
                + 2u * sizeof(types::storage_value_t),
            "COO descriptor/payload byte count changed");
    matrix.rowIdx[0] = 1u;
    matrix.colIdx[0] = 2u;
    matrix.val[0] = __float2half(5.0f);
    matrix.rowIdx[1] = 0u;
    matrix.colIdx[1] = 1u;
    matrix.val[1] = __float2half(4.0f);
    require(close_half(*formats::at(&matrix, 1u, 2u), 5.0f), "COO coordinate lookup changed");
    require(formats::at(&matrix, 1u, 1u) == nullptr, "COO absent lookup changed");
    formats::clear(&matrix);
}


#endif

struct fake_external_archive {
    std::uint64_t cells;
    std::uint64_t features;
    std::uint64_t nnz;
};

struct fake_archive_binding {
    const fake_external_archive *archive;
};

struct fake_pack_binding {
    std::uint32_t selected_format;
    std::uint64_t build_calls;
};

struct incomplete_binding {};

} // namespace

namespace cellshard {
namespace access {

template<>
struct archive_adapter<fake_archive_binding> {
    static archive_descriptor describe(const adapter_view<fake_archive_binding> &view) {
        return archive_descriptor{
            view.binding->archive->cells,
            view.binding->archive->features,
            view.binding->archive->nnz,
            42u,
            types::value_f32,
            disk_format_none,
            dataset_execution_format_unknown,
            capability_archive_to_pack
        };
    }
};

template<>
struct pack_adapter<fake_pack_binding> {
    static pack_descriptor describe(const adapter_view<fake_pack_binding> &view) {
        return pack_descriptor{0u, 0u, 0u, types::value_f32, view.binding->selected_format, 0u, capability_pack_write};
    }
};

template<>
struct archive_to_pack<fake_archive_binding, fake_pack_binding, passthrough_pack_policy> {
    static pack_build_result build(
        const adapter_view<fake_archive_binding> &archive,
        const adapter_view<fake_pack_binding> &pack,
        const pack_build_request &request) {
        ++pack.binding->build_calls;
        pack.binding->selected_format = request.requested_execution_format;
        const archive_descriptor desc = describe_archive(archive);
        return pack_build_result{
            desc.cell_count,
            desc.feature_count,
            desc.nnz,
            0u,
            request.requested_execution_format,
            1u
        };
    }
};

} // namespace access
} // namespace cellshard

int main() {
#if CELLSHARD_ENABLE_CUDA
    check_dense();
    check_compressed();
    check_coo();
#endif

    using namespace cellshard;
    using namespace cellshard::access;

    static_assert(is_archive_adapter<fake_archive_binding>::value, "fake archive binding should satisfy archive contract");
    static_assert(is_pack_adapter<fake_pack_binding>::value, "fake pack binding should satisfy pack contract");
    static_assert(is_archive_to_pack_adapter<fake_archive_binding, fake_pack_binding>::value,
                  "fake archive-to-pack override should be visible");
    static_assert(!is_archive_adapter<incomplete_binding>::value, "missing describe must fail the archive contract");

    fake_external_archive fake_archive{11u, 7u, 19u};
    fake_archive_binding fake_archive_binding_value{&fake_archive};
    fake_pack_binding fake_pack_binding_value{dataset_execution_format_unknown, 0u};
    auto fake_archive_view = make_adapter_view(fake_archive_binding_value);
    auto fake_pack_view = make_adapter_view(fake_pack_binding_value);
    const pack_build_result fake_result = build_pack(
        fake_archive_view,
        fake_pack_view,
        pack_build_request{
            dataset_generation_ref{1u, 2u, 3u, 4u},
            dataset_partition_ref{5u, 6u},
            cell_selection_view{},
            feature_selection_view{},
            mutable_byte_span{},
            mutable_byte_span{},
            dataset_execution_format_dense,
            0u,
            nullptr
        });
    assert(fake_result.status == 1u);
    assert(fake_result.cell_count == 11u);
    assert(fake_pack_binding_value.build_calls == 1u);
    assert(fake_pack_binding_value.selected_format == dataset_execution_format_dense);

    types::storage_value_t dense_values[6] = {
        __float2half(1.0f), __float2half(2.0f), __float2half(3.0f),
        __float2half(4.0f), __float2half(5.0f), __float2half(6.0f)
    };
    ::cellshard::dense dense_matrix{};
    attach(&dense_matrix, 2u, 3u, dense_values, ::cellshard::dense_row_major);
    dense_fallback_binding dense_binding{&dense_matrix, nullptr, 9u};
    auto dense_view = make_adapter_view(dense_binding);
    const archive_descriptor dense_desc = describe_archive(dense_view);
    assert(dense_desc.cell_count == 2u);
    assert(dense_desc.feature_count == 3u);
    assert(dense_desc.nnz == 6u);
    assert(dense_desc.archive_format == disk_format_dense);
    assert(has_capability(dense_desc.capabilities, capability_cell_span_copy));

    const cell_span dense_cell_spans[] = {cell_span{1u, 2u}};
    const feature_span dense_feature_spans[] = {feature_span{1u, 3u}};
    types::storage_value_t dense_out[2] = {};
    const copy_preflight dense_need = preflight_cell_spans(
        dense_view,
        make_cell_spans(dense_cell_spans, 1u),
        make_cell_spans(dense_feature_spans, 1u));
    assert(dense_need.cell_count == 1u);
    assert(dense_need.feature_count == 2u);
    assert(dense_need.output_bytes == sizeof(dense_out));

    const pack_build_result dense_copy = build_pack(
        dense_view,
        dense_view,
        pack_build_request{
            dataset_generation_ref{},
            dataset_partition_ref{},
            make_cell_spans(dense_cell_spans, 1u),
            make_cell_spans(dense_feature_spans, 1u),
            mutable_byte_span{dense_out, sizeof(dense_out), types::value_code<types::storage_value_t>::code},
            mutable_byte_span{},
            dataset_execution_format_dense,
            0u,
            nullptr
        });
    assert(dense_copy.status == 1u);
    assert(__half2float(dense_out[0]) == 5.0f);
    assert(__half2float(dense_out[1]) == 6.0f);

    types::ptr_t ptrs[] = {0u, 1u, 1u, 2u};
    types::idx_t idx[] = {2u, 0u};
    types::storage_value_t vals[] = {__float2half(7.0f), __float2half(8.0f)};
    ::cellshard::compressed compressed_matrix{};
    init(&compressed_matrix, 3u, 4u, 2u, ::cellshard::compressed_by_row);
    compressed_matrix.majorPtr = ptrs;
    compressed_matrix.minorIdx = idx;
    compressed_matrix.val = vals;
    compressed_fallback_binding compressed_binding{&compressed_matrix, nullptr, 10u};
    auto compressed_view = make_adapter_view(compressed_binding);
    const archive_descriptor compressed_desc = describe_archive(compressed_view);
    assert(compressed_desc.cell_count == 3u);
    assert(compressed_desc.feature_count == 4u);
    assert(compressed_desc.nnz == 2u);
    assert(compressed_desc.archive_format == disk_format_compressed);

    const fallback_payload_view compressed_payload = contiguous_payload(compressed_view);
    assert(compressed_payload.major_ptr.bytes == sizeof(ptrs));
    assert(compressed_payload.minor_idx.bytes == sizeof(idx));
    assert(compressed_payload.values.bytes == sizeof(vals));

    const pack_descriptor compressed_pack = describe_pack(compressed_view);
    assert(compressed_pack.execution_format == dataset_execution_format_compressed);
    assert(compressed_pack.nnz == 2u);

    return 0;
}
