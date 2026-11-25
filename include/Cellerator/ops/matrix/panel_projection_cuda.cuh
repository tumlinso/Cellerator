#pragma once

#include <Cellerator/ops/matrix/indexed_tiled_matmul_cuda.cuh>
#include <Cellerator/ops/matrix/panel_matmul_cuda.cuh>
#include <Cellerator/ops/matrix/panel_projection_residual_cuda.cuh>

namespace cellerator::ops::matrix {

// Project activations and weights packed in the same prepared panel order.
// The optional residual uses the same physical cell rows and canonical
// weights. An empty residual may be passed as {}. With finite operands this
// computes the selected CSR rows' projection, with FP32 accumulation.
// Output columns are in prepared cell order, including any padded tail.
// The caller restores logical row identity at the output boundary.
cudaError_t enqueue_panel_projection(const PanelMatmulPlan &plan,
    memory::const_array_view<__half> packed_weights, memory::const_array_view<__half> features,
    PanelProjectionResidualView residual, memory::const_matrix_view<__half> canonical_weights,
    memory::matrix_view<float> output, cudaStream_t stream) noexcept;

// Project the same resident FP16 panels with a different active block list
// for each 16-cell group. Weights use [output tile][logical feature block].
// Packed lists follow local slice order; pairs can select arbitrary slices
// in the panel window. Validate the worklist before upload. The residual
// uses canonical feature IDs and must exclude blocks covered by that group.
cudaError_t enqueue_indexed_panel_projection(const IndexedTiledMatmulPlan &plan,
    memory::const_array_view<memory::Tile<__half>> weights, memory::const_array_view<__half> panels,
    PackedPanelWorklist work, PanelProjectionResidualView residual,
    memory::const_matrix_view<__half> canonical_weights, memory::matrix_view<float> output,
    cudaStream_t stream) noexcept;
cudaError_t enqueue_indexed_panel_projection(const IndexedTiledMatmulPlan &plan,
    memory::const_array_view<memory::Tile<__half>> weights, memory::const_array_view<__half> panels,
    IndexedTileWorklist work, PanelProjectionResidualView residual,
    memory::const_matrix_view<__half> canonical_weights, memory::matrix_view<float> output,
    cudaStream_t stream) noexcept;

} // namespace cellerator::ops::matrix
