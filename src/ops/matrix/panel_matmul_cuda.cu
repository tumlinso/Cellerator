#include <Cellerator/ops/matrix/panel_matmul_cuda.cuh>
#include <Cellerator/memory/tiled/Panel.hh>
#include <mma.h>
#include <stdint.h>

namespace cellerator::ops::matrix {
namespace {

namespace wmma = nvcuda::wmma;
constexpr unsigned extent = memory::panel_rows, tiles_per_panel = memory::panel_tiles;
constexpr unsigned panel_width = memory::panel_columns;
constexpr unsigned tile_values = extent*extent, panel_values = extent*panel_width;
constexpr unsigned warp_threads = 32, shared_warps = 4;

using Accumulator = wmma::fragment<wmma::accumulator, extent, extent, extent, float>;

__device__ __forceinline__ void accumulate_panel(
    Accumulator &sum, const __half *weights, const __half *features) {
    wmma::fragment<wmma::matrix_a, extent, extent, extent, __half, wmma::row_major> w;
    wmma::fragment<wmma::matrix_b, extent, extent, extent, __half, wmma::col_major> x;
#pragma unroll
    for (unsigned tile = 0; tile < tiles_per_panel; ++tile) {
        wmma::load_matrix_sync(w, weights + tile*tile_values, extent);
        wmma::load_matrix_sync(x, features + tile*extent, panel_width);
        wmma::mma_sync(sum, w, x, sum);
    }
}

__device__ __forceinline__ void project_direct(const __half *weights, const __half *features,
    size_t panels, float *output, unsigned stride) {
    Accumulator sum;
    wmma::fill_fragment(sum, 0.0f);
    for (size_t panel = 0; panel < panels; ++panel) {
        accumulate_panel(sum, weights, features);
        weights += panel_values;
        features += panel_values;
    }
    wmma::store_matrix_sync(output, sum, stride, wmma::mem_row_major);
}

template<unsigned warps>
__global__ void multiply_panels(const __half *weights, const __half *features,
    size_t panels, size_t output_tiles, unsigned bundles, float *output, unsigned stride) {
    const unsigned bundle = warps == 1 ? 0 : blockIdx.x % bundles;
    const unsigned group = warps == 1 ? blockIdx.x : blockIdx.x / bundles;
    const unsigned warp = threadIdx.x / warp_threads;
    const size_t first = size_t(bundle)*warps, tile = first + warp;
    const bool active = tile < output_tiles;
    const __half *x = panels ? features + size_t(group)*panels*panel_values : nullptr;
    const __half *w = panels && active ? weights + tile*panels*panel_values : nullptr;
    float *y = active ? output + tile*extent*stride + size_t(group)*extent : nullptr;

    if constexpr (warps == 1) {
        project_direct(w, x, panels, y, stride);
    } else {
        if (first + 1 == output_tiles) {
            if (warp == 0) project_direct(w, x, panels, y, stride);
            return;
        }

        __shared__ __align__(32) __half shared_panel[panel_values];
        Accumulator sum;
        wmma::fill_fragment(sum, 0.0f);
        for (size_t panel = 0; panel < panels; ++panel) {
            // Each warp copies two adjacent halves per lane, one 128-byte row.
#pragma unroll
            for (unsigned i = threadIdx.x; i < panel_values/2; i += warps*warp_threads) {
                uint32_t pair;
                const void *source = x + 2*i;
                void *destination = shared_panel + 2*i;
                __builtin_memcpy(&pair, __builtin_assume_aligned(source, 4), sizeof(pair));
                __builtin_memcpy(__builtin_assume_aligned(destination, 4), &pair, sizeof(pair));
            }
            __syncthreads();
            if (active) {
                accumulate_panel(sum, w, shared_panel);
                w += panel_values;
            }
            x += panel_values;
            __syncthreads();
        }
        if (active) wmma::store_matrix_sync(y, sum, stride, wmma::mem_row_major);
    }
}

bool resident_aligned(const void *data, memory::placement where, int device) noexcept {
    return data && reinterpret_cast<uintptr_t>(data) % 32 == 0
        && where.kind == memory::domain::device && where.device_ordinal == device;
}

} // namespace

cudaError_t prepare_panel_matmul(int device, PanelMatmulShape shape,
                                PanelMatmulPlan &plan) noexcept {
    plan = {};
    const auto [groups, panels, output_tiles] = shape;
    constexpr size_t max_panels = SIZE_MAX / (panel_values*sizeof(__half));
    if (!groups || !output_tiles || device < 0 || groups > UINT32_MAX/extent
        || output_tiles > SIZE_MAX/extent || panels > max_panels/groups
        || panels > max_panels/output_tiles)
        return cudaErrorInvalidValue;

    cudaDeviceProp properties{};
    const cudaError_t status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) return status;
    if (properties.major != 7 || properties.minor != 0) return cudaErrorNotSupported;
    const size_t bundles = (output_tiles + shared_warps - 1)/shared_warps;
    if (bundles > size_t(properties.maxGridSize[0])/groups) return cudaErrorInvalidValue;

    plan.shape_ = shape;
    plan.output_bundles_ = unsigned(bundles);
    plan.device_ = device;
    return cudaSuccess;
}

cudaError_t enqueue_panel_matmul(
    const PanelMatmulPlan &plan, memory::const_array_view<__half> weights,
    memory::const_array_view<__half> features, memory::matrix_view<float> output,
    cudaStream_t stream) noexcept {
    const auto [groups, panels, output_tiles] = plan.shape_;
    const size_t rows = output_tiles*extent, columns = groups*extent, stride = output.row_stride;
    const int device = plan.device_;
    if (device < 0 || !resident_aligned(output.data, output.where, device)
        || output.rows != rows || output.columns != columns || output.column_stride != 1
        || stride < columns || stride > UINT32_MAX || stride % 4
        || rows - 1 > (SIZE_MAX / sizeof(float) - columns)/stride)
        return cudaErrorInvalidValue;
    if (panels && (weights.count < output_tiles*panels*panel_values
        || features.count < groups*panels*panel_values
        || !resident_aligned(weights.data, weights.where, device)
        || !resident_aligned(features.data, features.where, device)))
        return cudaErrorInvalidValue;

    const unsigned bundles = plan.output_bundles_, grid = unsigned(groups*bundles);
    if (output_tiles == 1)
        multiply_panels<1><<<grid, warp_threads, 0, stream>>>(
            weights.data, features.data, panels, output_tiles, bundles, output.data, unsigned(stride));
    else
        multiply_panels<shared_warps><<<grid, shared_warps*warp_threads, 0, stream>>>(
            weights.data, features.data, panels, output_tiles, bundles, output.data, unsigned(stride));
    return cudaPeekAtLastError();
}

} // namespace cellerator::ops::matrix
