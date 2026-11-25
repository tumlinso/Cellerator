#pragma once

#include <Cellerator/memory/view.hh>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

namespace cellerator::ops::matrix {

struct PanelProjectionResidualView {
    memory::const_array_view<uint32_t> offsets;
    memory::const_array_view<uint16_t> features;
    memory::const_array_view<__half> values;
};

// Add W * residual to the existing FP32 output. W is canonical row-major
// [output, feature] and output is row-major [output, physical cell]. Padded
// output rows are left alone. Padded cell rows have no residual entries.
// Offsets have output.columns + 1 elements and end at values.count. The
// prepared CSR is validated before upload and retained through stream use.
// All nonempty views and the stream belong to the current device. The output
// must be disjoint from the inputs and ready on the supplied stream.
cudaError_t enqueue_panel_projection_residual(PanelProjectionResidualView residual,
    memory::const_matrix_view<__half> weights, memory::matrix_view<float> output,
    cudaStream_t stream) noexcept;

} // namespace cellerator::ops::matrix
