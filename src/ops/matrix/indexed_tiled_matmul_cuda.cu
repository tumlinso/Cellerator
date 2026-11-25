#include <Cellerator/ops/matrix/indexed_tiled_matmul_cuda.cuh>
#include <Cellerator/memory/tiled/Panel.hh>
#include <mma.h>

namespace cellerator::ops::matrix {
namespace {

namespace wmma = nvcuda::wmma;
using HalfTile = memory::Tile<__half>;
constexpr unsigned extent = HalfTile::rows;
static_assert(HalfTile::columns == extent && sizeof(HalfTile) == HalfTile::bytes);
static_assert(sizeof(IndexedTilePair) == 4);
static_assert(memory::panel_rows == extent && memory::panel_tile_columns == extent);
static_assert((memory::panel_tile_columns*sizeof(__half))%alignof(HalfTile) == 0);

__device__ __forceinline__ const __half *feature_tile(const HalfTile *tiles, size_t slot) {
    return tiles[slot].data();
}

__device__ __forceinline__ const __half *feature_tile(const __half *panels, size_t slot) {
    const auto *panel = panels + (slot/memory::panel_tiles)*memory::panel_elements;
    return panel + (slot%memory::panel_tiles)*extent;
}

template<class Order, unsigned feature_stride, class Feature, class Item>
__global__ void multiply_indexed_tiles(const HalfTile *weights, const Feature *features,
    const uint32_t *offsets, const Item *items, const uint32_t *panel_bases,
    unsigned groups, size_t feature_blocks, float *output, unsigned output_stride) {
    const size_t m = blockIdx.x / groups, n = blockIdx.x % groups;
    const uint32_t begin = offsets[n], end = offsets[n + 1];
    constexpr bool implicit_slots = __is_same(Item, uint16_t);
    const size_t first_slot = implicit_slots && begin != end
        ? size_t(panel_bases[n])*memory::panel_tiles : 0;
    wmma::fragment<wmma::matrix_a, extent, extent, extent, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, extent, extent, extent, __half, Order> b;
    wmma::fragment<wmma::accumulator, extent, extent, extent, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (uint32_t i = begin; i < end; ++i) {
        uint16_t block;
        size_t slot;
        if constexpr (implicit_slots) {
            block = items[i];
            slot = first_slot + (i - begin);
        } else {
            const IndexedTilePair p = items[i];
            block = p.feature_block;
            slot = p.physical_slot;
        }
        wmma::load_matrix_sync(a, weights[m*feature_blocks + block].data(), extent);
        wmma::load_matrix_sync(b, feature_tile(features, slot), feature_stride);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(output + m*extent*output_stride + n*extent, c, output_stride,
                            wmma::mem_row_major);
}

bool valid_shape(TiledMatmulShape shape) noexcept {
    constexpr size_t max_tiles = SIZE_MAX / sizeof(HalfTile);
    const size_t m = shape.output_tiles, n = shape.cell_tiles, k = shape.feature_tiles;
    return m && n && k && k <= size_t(UINT16_MAX) + 1 && m <= max_tiles/k
        && n <= UINT32_MAX/extent && m <= SIZE_MAX/n;
}

bool on_device(memory::placement where, int device) noexcept {
    return where.kind == memory::domain::device && where.device_ordinal == device;
}

bool aligned(const void *data, size_t alignment) noexcept {
    return data && reinterpret_cast<uintptr_t>(data) % alignment == 0;
}

template<class Item>
bool valid_worklist_extent(TiledMatmulShape shape, memory::const_array_view<uint32_t> offsets,
                           memory::const_array_view<Item> items) noexcept {
    return offsets.count == shape.cell_tiles + 1 && items.count <= UINT32_MAX
        && aligned(offsets.data, alignof(uint32_t))
        && (!items.count || aligned(items.data, alignof(Item)));
}

template<class Item>
bool valid_host_offsets(TiledMatmulShape shape, memory::const_array_view<uint32_t> offsets,
                        memory::const_array_view<Item> items) noexcept {
    if (!valid_shape(shape) || !valid_worklist_extent(shape, offsets, items)
        || offsets.where.kind == memory::domain::device
        || (items.count && items.where.kind == memory::domain::device))
        return false;
    if (offsets.data[0] || offsets.data[shape.cell_tiles] != items.count) return false;
    for (size_t n = 0; n < shape.cell_tiles; ++n)
        if (offsets.data[n + 1] < offsets.data[n]) return false;
    return true;
}

bool valid_panel_bases(TiledMatmulShape shape, memory::const_array_view<uint32_t> bases) noexcept {
    return bases.count == shape.cell_tiles && aligned(bases.data, alignof(uint32_t));
}

constexpr size_t max_panels = SIZE_MAX / (memory::panel_elements*sizeof(__half));

template<class Order, unsigned feature_stride, class Feature, class Item>
cudaError_t enqueue_indexed(TiledMatmulShape shape, int device,
    memory::const_array_view<HalfTile> weights, memory::const_array_view<Feature> features,
    memory::const_array_view<uint32_t> offsets, memory::const_array_view<Item> items,
    memory::const_array_view<uint32_t> panel_bases, memory::matrix_view<float> output,
    cudaStream_t stream) noexcept {
    const auto [m, n, k] = shape;
    const size_t rows = m*extent, columns = n*extent, stride = output.row_stride;
    if (device < 0 || !valid_worklist_extent(shape, offsets, items)
        || !on_device(offsets.where, device) || !on_device(output.where, device)
        || !aligned(output.data, alignof(HalfTile)) || output.rows != rows || output.columns != columns
        || output.column_stride != 1 || stride < columns || stride > UINT32_MAX || stride % 4
        || rows - 1 > (SIZE_MAX / sizeof(float) - columns)/stride)
        return cudaErrorInvalidValue;
    if constexpr (__is_same(Feature, __half)) {
        if (features.count%memory::panel_elements || features.count/memory::panel_elements > max_panels)
            return cudaErrorInvalidValue;
    }
    if constexpr (__is_same(Item, uint16_t)) {
        if (!valid_panel_bases(shape, panel_bases) || !on_device(panel_bases.where, device))
            return cudaErrorInvalidValue;
    }
    if (items.count && (!on_device(items.where, device) || !on_device(weights.where, device)
        || !on_device(features.where, device) || weights.count < m*k || !features.count
        || !aligned(weights.data, alignof(HalfTile)) || !aligned(features.data, alignof(HalfTile))))
        return cudaErrorInvalidValue;

    multiply_indexed_tiles<Order, feature_stride><<<unsigned(m*n), 32, 0, stream>>>(
        weights.data, features.data, offsets.data, items.data, panel_bases.data, unsigned(n), k,
        output.data, unsigned(stride));
    return cudaPeekAtLastError();
}

} // namespace

cudaError_t validate_indexed_tile_worklist(TiledMatmulShape shape,
    IndexedTileWorklist work, size_t physical_slots) noexcept {
    if (!valid_host_offsets(shape, work.group_offsets, work.pairs))
        return cudaErrorInvalidValue;

    for (size_t i = 0; i < work.pairs.count; ++i) {
        const IndexedTilePair p = work.pairs.data[i];
        if (p.feature_block >= shape.feature_tiles || p.physical_slot >= physical_slots)
            return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

cudaError_t validate_indexed_panel_worklist(TiledMatmulShape shape,
    IndexedTileWorklist work, size_t panel_count) noexcept {
    if (panel_count > max_panels) return cudaErrorInvalidValue;
    return validate_indexed_tile_worklist(shape, work, panel_count*memory::panel_tiles);
}

cudaError_t validate_indexed_panel_worklist(TiledMatmulShape shape,
    PackedPanelWorklist work, size_t panel_count) noexcept {
    if (panel_count > max_panels || !valid_host_offsets(shape, work.group_offsets, work.feature_blocks)
        || !valid_panel_bases(shape, work.group_panel_bases)
        || work.group_panel_bases.where.kind == memory::domain::device)
        return cudaErrorInvalidValue;

    const auto *offsets = work.group_offsets.data;
    for (size_t n = 0; n < shape.cell_tiles; ++n) {
        const size_t count = offsets[n + 1] - offsets[n], base = work.group_panel_bases.data[n];
        const size_t panels = count/memory::panel_tiles + (count%memory::panel_tiles != 0);
        if (base > panel_count || panels > panel_count - base) return cudaErrorInvalidValue;
    }
    for (size_t i = 0; i < work.feature_blocks.count; ++i)
        if (work.feature_blocks.data[i] >= shape.feature_tiles) return cudaErrorInvalidValue;
    return cudaSuccess;
}

cudaError_t prepare_indexed_tiled_matmul(int device_ordinal, TiledMatmulShape shape,
    IndexedTiledMatmulPlan &plan) noexcept {
    plan = {};
    if (device_ordinal < 0 || !valid_shape(shape)) return cudaErrorInvalidValue;
    cudaDeviceProp device{};
    const cudaError_t status = cudaGetDeviceProperties(&device, device_ordinal);
    if (status != cudaSuccess) return status;
    if (device.major != 7 || device.minor != 0) return cudaErrorNotSupported;
    if (shape.output_tiles*shape.cell_tiles > size_t(device.maxGridSize[0]))
        return cudaErrorInvalidValue;
    plan.shape_ = shape;
    plan.device_ordinal_ = device_ordinal;
    return cudaSuccess;
}

cudaError_t enqueue_indexed_tiled_matmul(const IndexedTiledMatmulPlan &plan,
    memory::const_array_view<HalfTile> weights, memory::const_array_view<HalfTile> features,
    IndexedTileWorklist work, memory::matrix_view<float> output, cudaStream_t stream) noexcept {
    return enqueue_indexed<wmma::row_major, extent>(plan.shape_, plan.device_ordinal_,
        weights, features, work.group_offsets, work.pairs, {}, output, stream);
}

cudaError_t enqueue_indexed_panel_matmul(const IndexedTiledMatmulPlan &plan,
    memory::const_array_view<HalfTile> weights, memory::const_array_view<__half> panels,
    IndexedTileWorklist work, memory::matrix_view<float> output, cudaStream_t stream) noexcept {
    return enqueue_indexed<wmma::col_major, memory::panel_columns>(plan.shape_, plan.device_ordinal_,
        weights, panels, work.group_offsets, work.pairs, {}, output, stream);
}

cudaError_t enqueue_indexed_panel_matmul(const IndexedTiledMatmulPlan &plan,
    memory::const_array_view<HalfTile> weights, memory::const_array_view<__half> panels,
    PackedPanelWorklist work, memory::matrix_view<float> output, cudaStream_t stream) noexcept {
    return enqueue_indexed<wmma::col_major, memory::panel_columns>(plan.shape_, plan.device_ordinal_,
        weights, panels, work.group_offsets, work.feature_blocks, work.group_panel_bases, output, stream);
}

} // namespace cellerator::ops::matrix
