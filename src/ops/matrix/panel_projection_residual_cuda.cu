#include <Cellerator/ops/matrix/panel_projection_cuda.cuh>

namespace cellerator::ops::matrix {
namespace {

constexpr unsigned block_cells = 128;

__global__ void accumulate_residual(const uint32_t *offsets, const uint16_t *features,
    const __half *values, const __half *weights, size_t outputs, size_t weight_stride,
    float *output, size_t cells, size_t output_stride) {
    const size_t cell = size_t(blockIdx.x)*block_cells + threadIdx.x;
    if (cell >= cells) return;
    const uint32_t begin = offsets[cell], end = offsets[cell + 1];
    if (begin == end) return;

    for (size_t out = blockIdx.y; out < outputs; out += gridDim.y) {
        const __half *w = weights + out*weight_stride;
        float *y = output + out*output_stride + cell;
        float sum = *y;
        for (uint32_t i = begin; i < end; ++i)
            sum = __fmaf_rn(__half2float(w[features[i]]), __half2float(values[i]), sum);
        *y = sum;
    }
}

bool on_device(memory::placement where, int device) noexcept {
    return where.kind == memory::domain::device && where.device_ordinal == device;
}

cudaError_t residual_grid(PanelProjectionResidualView residual,
    memory::const_matrix_view<__half> weights, memory::matrix_view<float> output,
    dim3 &grid) noexcept {
    constexpr size_t max_features = size_t(UINT16_MAX) + 1;
    constexpr size_t max_weights = SIZE_MAX / sizeof(__half), max_output = SIZE_MAX / sizeof(float);
    const int device = output.where.device_ordinal;
    const size_t cells = output.columns, outputs = weights.rows;
    const size_t nw = weights.columns, ldw = weights.row_stride, ldy = output.row_stride;
    const size_t nnz = residual.values.count;
    if (device < 0 || output.where.kind != memory::domain::device
        || !on_device(weights.where, device) || !on_device(residual.offsets.where, device)
        || !output.data || !weights.data || !residual.offsets.data
        || !cells || !outputs || outputs > output.rows || !nw || nw > max_features
        || weights.column_stride != 1 || output.column_stride != 1
        || ldw < nw || ldy < cells || cells > max_output
        || outputs - 1 > (max_weights - nw)/ldw
        || output.rows - 1 > (max_output - cells)/ldy
        || cells + 1 != residual.offsets.count || nnz != residual.features.count
        || nnz > UINT32_MAX)
        return cudaErrorInvalidValue;
    if (!nnz) return cudaSuccess;
    if (!residual.features.data || !residual.values.data
        || !on_device(residual.features.where, device) || !on_device(residual.values.where, device))
        return cudaErrorInvalidValue;

    const size_t blocks = 1 + (cells - 1)/block_cells;
    if (blocks > INT32_MAX) return cudaErrorInvalidValue;
    grid = dim3(unsigned(blocks), unsigned(outputs < UINT16_MAX ? outputs : UINT16_MAX));
    return cudaSuccess;
}

cudaError_t launch_residual(dim3 grid, PanelProjectionResidualView residual,
    memory::const_matrix_view<__half> weights, memory::matrix_view<float> output,
    cudaStream_t stream) noexcept {
    accumulate_residual<<<grid, block_cells, 0, stream>>>(residual.offsets.data,
        residual.features.data, residual.values.data, weights.data, weights.rows,
        weights.row_stride, output.data, output.columns, output.row_stride);
    return cudaPeekAtLastError();
}

template<class Project>
cudaError_t project_with_residual(Project project, PanelProjectionResidualView residual,
    memory::const_matrix_view<__half> weights, memory::matrix_view<float> output,
    cudaStream_t stream) noexcept {
    if (!residual.values.count && !residual.features.count) return project();
    dim3 grid;
    cudaError_t status = residual_grid(residual, weights, output, grid);
    if (status != cudaSuccess) return status;
    status = project();
    return status == cudaSuccess ? launch_residual(grid, residual, weights, output, stream) : status;
}

} // namespace

cudaError_t enqueue_panel_projection_residual(PanelProjectionResidualView residual,
    memory::const_matrix_view<__half> weights, memory::matrix_view<float> output,
    cudaStream_t stream) noexcept {
    dim3 grid;
    const cudaError_t status = residual_grid(residual, weights, output, grid);
    if (status != cudaSuccess || !residual.values.count) return status;
    return launch_residual(grid, residual, weights, output, stream);
}

cudaError_t enqueue_panel_projection(const PanelMatmulPlan &plan,
    memory::const_array_view<__half> packed_weights, memory::const_array_view<__half> features,
    PanelProjectionResidualView residual, memory::const_matrix_view<__half> canonical_weights,
    memory::matrix_view<float> output, cudaStream_t stream) noexcept {
    return project_with_residual([&] {
        return enqueue_panel_matmul(plan, packed_weights, features, output, stream);
    }, residual, canonical_weights, output, stream);
}

cudaError_t enqueue_indexed_panel_projection(const IndexedTiledMatmulPlan &plan,
    memory::const_array_view<memory::Tile<__half>> weights, memory::const_array_view<__half> panels,
    PackedPanelWorklist work, PanelProjectionResidualView residual,
    memory::const_matrix_view<__half> canonical_weights, memory::matrix_view<float> output,
    cudaStream_t stream) noexcept {
    return project_with_residual([&] {
        return enqueue_indexed_panel_matmul(plan, weights, panels, work, output, stream);
    }, residual, canonical_weights, output, stream);
}

cudaError_t enqueue_indexed_panel_projection(const IndexedTiledMatmulPlan &plan,
    memory::const_array_view<memory::Tile<__half>> weights, memory::const_array_view<__half> panels,
    IndexedTileWorklist work, PanelProjectionResidualView residual,
    memory::const_matrix_view<__half> canonical_weights, memory::matrix_view<float> output,
    cudaStream_t stream) noexcept {
    return project_with_residual([&] {
        return enqueue_indexed_panel_matmul(plan, weights, panels, work, output, stream);
    }, residual, canonical_weights, output, stream);
}

} // namespace cellerator::ops::matrix
