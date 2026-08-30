// Include the implementation so this independent validation can exercise the
// production epilogue kernel directly across widths and launch tails. The
// linked production value-pack, N64, and residual entry points still satisfy
// the hybrid implementation's external definitions.
#include "../../../src/compute/architecture/providers/nvidia/sm70/relation_apply_hybrid.cu"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sm70 = cellerator::compute::architecture::providers::nvidia::sm70;

namespace {

void require_cuda(cudaError_t status) {
    assert(status == cudaSuccess);
}

void verify_epilogue(std::uint32_t count, float alpha, float beta,
    bool provide_prior) {
    std::vector<float> accumulation(count);
    std::vector<float> prior(count);
    std::vector<float> output(count, -1000.0f);
    for (std::uint32_t index = 0u; index < count; ++index) {
        accumulation[index] = static_cast<float>(index % 17u) - 4.0f;
        prior[index] = static_cast<float>((index * 3u) % 11u) + 0.25f;
    }

    float *device_accumulation = nullptr;
    float *device_prior = nullptr;
    float *device_output = nullptr;
    require_cuda(cudaMalloc(&device_accumulation, count * sizeof(float)));
    require_cuda(cudaMalloc(&device_prior, count * sizeof(float)));
    require_cuda(cudaMalloc(&device_output, count * sizeof(float)));
    require_cuda(cudaMemcpy(device_accumulation, accumulation.data(),
        count * sizeof(float), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_prior, prior.data(), count * sizeof(float),
        cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_output, output.data(), count * sizeof(float),
        cudaMemcpyHostToDevice));

    constexpr std::uint32_t block_size = 256u;
    const std::uint32_t grid_size =
        (count + block_size - 1u) / block_size;
    sm70::relation_apply_epilogue_v1<<<grid_size, block_size>>>(
        device_accumulation, provide_prior ? device_prior : nullptr,
        device_output, count, alpha, beta);
    require_cuda(cudaGetLastError());
    require_cuda(cudaMemcpy(output.data(), device_output,
        count * sizeof(float), cudaMemcpyDeviceToHost));
    for (std::uint32_t index = 0u; index < count; ++index) {
        const float expected = alpha * accumulation[index]
            + beta * (provide_prior ? prior[index] : 0.0f);
        assert(std::fabs(output[index] - expected) < 1.0e-6f);
    }

    require_cuda(cudaFree(device_output));
    require_cuda(cudaFree(device_prior));
    require_cuda(cudaFree(device_accumulation));
}

void verify_residual_width(std::uint32_t width) {
    constexpr std::uint32_t rows = 3u;
    constexpr std::uint32_t sources = 4u;
    const std::uint32_t row_offsets[] = {0u, 1u, 3u, 4u};
    const std::uint32_t columns[] = {3u, 0u, 2u, 1u};
    const __half values[] = {
        __float2half(2.0f), __float2half(-1.0f),
        __float2half(0.5f), __float2half(3.0f)};
    std::vector<__half> rhs(static_cast<std::size_t>(sources) * width);
    std::vector<float> accumulation(static_cast<std::size_t>(rows) * width);
    for (std::uint32_t source = 0u; source < sources; ++source)
        for (std::uint32_t column = 0u; column < width; ++column)
            rhs[static_cast<std::size_t>(source) * width + column] =
                __float2half(static_cast<float>(source * 5u + column % 5u));
    for (std::uint32_t row = 0u; row < rows; ++row)
        for (std::uint32_t column = 0u; column < width; ++column)
            accumulation[static_cast<std::size_t>(row) * width + column] =
                static_cast<float>(100u * row + column);

    std::uint32_t *device_row_offsets = nullptr;
    std::uint32_t *device_columns = nullptr;
    __half *device_values = nullptr;
    __half *device_rhs = nullptr;
    float *device_accumulation = nullptr;
    require_cuda(cudaMalloc(&device_row_offsets, sizeof(row_offsets)));
    require_cuda(cudaMalloc(&device_columns, sizeof(columns)));
    require_cuda(cudaMalloc(&device_values, sizeof(values)));
    require_cuda(cudaMalloc(&device_rhs, rhs.size() * sizeof(__half)));
    require_cuda(cudaMalloc(&device_accumulation,
        accumulation.size() * sizeof(float)));
    require_cuda(cudaMemcpy(device_row_offsets, row_offsets,
        sizeof(row_offsets), cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_columns, columns, sizeof(columns),
        cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_values, values, sizeof(values),
        cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_rhs, rhs.data(), rhs.size() * sizeof(__half),
        cudaMemcpyHostToDevice));
    require_cuda(cudaMemcpy(device_accumulation, accumulation.data(),
        accumulation.size() * sizeof(float), cudaMemcpyHostToDevice));

    sm70::residual_apply_request_v1 request{};
    request.row_offsets = device_row_offsets;
    request.row_count = rows;
    request.column_indices = device_columns;
    request.edge_count = 4u;
    request.edge_values = device_values;
    request.dense_rhs = device_rhs;
    request.source_count = sources;
    request.dense_width = width;
    request.accumulation = device_accumulation;
    assert(sm70::enqueue_row_owned_residual_v1(request)
        == sm70::residual_apply_status_v1::success);
    require_cuda(cudaMemcpy(accumulation.data(), device_accumulation,
        accumulation.size() * sizeof(float), cudaMemcpyDeviceToHost));

    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t column = 0u; column < width; ++column) {
            float residual = 0.0f;
            for (std::uint32_t edge = row_offsets[row];
                 edge < row_offsets[row + 1u]; ++edge) {
                residual += __half2float(values[edge]) * __half2float(
                    rhs[static_cast<std::size_t>(columns[edge]) * width
                        + column]);
            }
            const float expected = static_cast<float>(100u * row + column)
                + residual;
            assert(std::fabs(
                accumulation[static_cast<std::size_t>(row) * width + column]
                - expected) < 1.0e-5f);
        }
    }

    require_cuda(cudaFree(device_accumulation));
    require_cuda(cudaFree(device_rhs));
    require_cuda(cudaFree(device_values));
    require_cuda(cudaFree(device_columns));
    require_cuda(cudaFree(device_row_offsets));
}

} // namespace

int main() {
    constexpr std::array<std::uint32_t, 12u> counts = {
        1u, 15u, 16u, 17u, 31u, 32u, 63u, 64u, 65u, 80u, 255u, 257u};
    for (std::uint32_t count : counts) {
        verify_epilogue(count, 2.0f, 0.5f, true);
        verify_epilogue(count, -0.75f, 0.0f, false);
        verify_epilogue(count, 0.0f, 1.0f, true);
    }

    constexpr std::array<std::uint32_t, 8u> widths = {
        1u, 15u, 16u, 17u, 32u, 64u, 65u, 80u};
    for (std::uint32_t width : widths) verify_residual_width(width);
    return 0;
}
