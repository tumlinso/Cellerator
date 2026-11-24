#include <Cellerator/ops/matrix/tiled_matmul_cuda.cuh>
#include <mma.h>
#include <stdint.h>

namespace cellerator::ops::matrix {
namespace {

namespace wmma = nvcuda::wmma;
using HalfTile = memory::Tile<__half>;
constexpr unsigned extent = HalfTile::rows;
static_assert(HalfTile::columns == extent && sizeof(HalfTile) == HalfTile::bytes);

__global__ void multiply_tiles(const HalfTile *weights, const HalfTile *features,
                               size_t cell_tiles, size_t feature_tiles,
                               float *output, unsigned output_stride) {
    const size_t m = blockIdx.x / cell_tiles, n = blockIdx.x % cell_tiles;
    wmma::fragment<wmma::matrix_a, extent, extent, extent, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, extent, extent, extent, __half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, extent, extent, extent, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (size_t k = 0; k < feature_tiles; ++k) {
        wmma::load_matrix_sync(a, weights[m*feature_tiles + k].data(), extent);
        wmma::load_matrix_sync(b, features[k*cell_tiles + n].data(), extent);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(output + m*extent*output_stride + n*extent,
                            c, output_stride, wmma::mem_row_major);
}

bool on_device(memory::placement where, int device_ordinal) noexcept {
    return where.kind == memory::domain::device && where.device_ordinal == device_ordinal;
}

bool tile_aligned(const void *data) noexcept {
    return data && reinterpret_cast<uintptr_t>(data) % alignof(HalfTile) == 0;
}

} // namespace

cudaError_t prepare_tiled_matmul(int device_ordinal, TiledMatmulShape shape,
                                 TiledMatmulPlan &plan) noexcept {
    plan = {};
    const size_t m = shape.output_tiles, n = shape.cell_tiles, k = shape.feature_tiles;
    constexpr size_t max_tiles = SIZE_MAX / sizeof(HalfTile);
    if (!m || !n || !k || device_ordinal < 0 || k > max_tiles / m || k > max_tiles / n
        || n > UINT32_MAX / extent || m > SIZE_MAX / n)
        return cudaErrorInvalidValue;

    cudaDeviceProp device{};
    const cudaError_t status = cudaGetDeviceProperties(&device, device_ordinal);
    if (status != cudaSuccess) return status;
    if (device.major != 7 || device.minor != 0) return cudaErrorNotSupported;
    if (m*n > size_t(device.maxGridSize[0])) return cudaErrorInvalidValue;

    plan.shape_ = shape;
    plan.device_ordinal_ = device_ordinal;
    return cudaSuccess;
}

cudaError_t enqueue_tiled_matmul(
    const TiledMatmulPlan &plan, memory::const_array_view<HalfTile> weights,
    memory::const_array_view<HalfTile> features, memory::matrix_view<float> output,
    cudaStream_t stream) noexcept {
    const int device = plan.device_ordinal_;
    const auto [m, n, k] = plan.shape_;
    const size_t rows = m*extent, columns = n*extent;
    const size_t stride = output.row_stride;
    if (device < 0 || weights.count < m*k || features.count < k*n
        || !tile_aligned(weights.data) || !tile_aligned(features.data) || !tile_aligned(output.data)
        || !on_device(weights.where, device) || !on_device(features.where, device)
        || !on_device(output.where, device) || output.rows != rows || output.columns != columns
        || output.column_stride != 1 || stride < columns || stride > UINT32_MAX || stride % 4
        || rows - 1 > (SIZE_MAX / sizeof(float) - columns) / stride)
        return cudaErrorInvalidValue;

    multiply_tiles<<<unsigned(m*n), 32, 0, stream>>>(weights.data, features.data, n, k,
                                                    output.data, unsigned(stride));
    return cudaPeekAtLastError();
}

} // namespace cellerator::ops::matrix
