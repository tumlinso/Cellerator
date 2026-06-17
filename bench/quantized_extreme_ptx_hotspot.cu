#include <Cellerator/quantized/extreme_backend.cuh>

#include <cstdint>
#include <cuda_runtime.h>

namespace msq = ::cellerator::quantized;
namespace msqe = ::cellerator::quantized::extreme_backend;

extern "C" __global__ __launch_bounds__(256, 3)
void quantized_extreme_hotspot_kernel(
    const float *values,
    const float *scales,
    const float *offsets,
    const int *columns,
    std::uint32_t count,
    unsigned char *codes,
    float *reconstructed) {
    const std::uint32_t idx = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= count) {
        return;
    }

    const msq::per_gene_affine<float> metadata = msq::make_per_gene_affine(scales, offsets);
    const msq::per_gene_affine<float>::row_cache row_cache = metadata.prepare_row(0);
    const int column = columns[idx];
    const unsigned int code = msqe::quantize_code_sm70_extreme<8, float, msq::per_gene_affine<float>>(
        values[idx],
        metadata,
        row_cache,
        column);

    codes[idx] = static_cast<unsigned char>(code);
    reconstructed[idx] = msqe::dequantize_code_sm70_extreme<8, float, msq::per_gene_affine<float>>(
        code,
        metadata,
        row_cache,
        column);
}
