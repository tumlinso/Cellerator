#include <Cellerator/compute/operators/sparse/row_transforms.cuh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

template<class T>
T *device_copy(const T *host, std::size_t count) {
    T *device = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&device), count * sizeof(T)) != cudaSuccess) return nullptr;
    if (cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
    return device;
}

bool close(__half got, float expected) {
    return std::fabs(__half2float(got) - expected) < 2.0e-3f;
}

bool run_blocked(const float *row_sums, const std::uint8_t *row_mask) {
    unsigned int columns[] = {0u, 1u};
    __half values[] = {__float2half(1.0f), __float2half(0.0f), __float2half(3.0f), __float2half(0.0f),
                       __float2half(0.0f), __float2half(2.0f), __float2half(0.0f), __float2half(4.0f)};
    auto *d_columns = device_copy(columns, 2u);
    auto *d_values = device_copy(values, 8u);
    cellerator::matrix::device::blocked_ell_view view{2u, 4u, 4u, 2u, 4u, d_columns, d_values};
    cellerator::compute::sparse::masked_row_scale_log1p_params params{row_sums, row_mask, 4.0f};
    const bool launched = cellerator::compute::sparse::masked_row_scale_log1p_inplace(&view, &params);
    cudaMemcpy(values, d_values, sizeof(values), cudaMemcpyDeviceToHost);
    cudaFree(d_values); cudaFree(d_columns);
    return launched && close(values[0], std::log1p(1.0f)) && close(values[2], std::log1p(3.0f))
        && close(values[5], 2.0f) && close(values[7], 4.0f);
}

bool run_sliced(const float *row_sums, const std::uint8_t *row_mask) {
    unsigned int offsets[] = {0u};
    unsigned int widths[] = {2u};
    unsigned int slots[] = {0u};
    unsigned int columns[] = {0u, 2u, 1u, 3u};
    __half values[] = {__float2half(1.0f), __float2half(3.0f), __float2half(2.0f), __float2half(4.0f)};
    auto *d_offsets = device_copy(offsets, 1u); auto *d_widths = device_copy(widths, 1u);
    auto *d_slots = device_copy(slots, 1u); auto *d_columns = device_copy(columns, 4u);
    auto *d_values = device_copy(values, 4u);
    cellerator::matrix::device::sliced_ell_view view{2u, 4u, 4u, 1u, 2u, d_offsets, d_widths, d_slots, d_columns, d_values};
    cellerator::compute::sparse::masked_row_scale_log1p_params params{row_sums, row_mask, 4.0f};
    const bool launched = cellerator::compute::sparse::masked_row_scale_log1p_inplace(&view, &params);
    cudaMemcpy(values, d_values, sizeof(values), cudaMemcpyDeviceToHost);
    cudaFree(d_values); cudaFree(d_columns); cudaFree(d_slots); cudaFree(d_widths); cudaFree(d_offsets);
    return launched && close(values[0], std::log1p(1.0f)) && close(values[1], std::log1p(3.0f))
        && close(values[2], 2.0f) && close(values[3], 4.0f);
}

bool run_compressed(const float *row_sums, const std::uint8_t *row_mask) {
    unsigned int pointers[] = {0u, 2u, 4u};
    unsigned int columns[] = {0u, 2u, 1u, 3u};
    __half values[] = {__float2half(1.0f), __float2half(3.0f), __float2half(2.0f), __float2half(4.0f)};
    auto *d_pointers = device_copy(pointers, 3u); auto *d_columns = device_copy(columns, 4u);
    auto *d_values = device_copy(values, 4u);
    cellerator::matrix::device::compressed_view view{2u, 4u, 4u,
        cellerator::matrix::device::compressed_by_row, d_pointers, d_columns, d_values};
    cellerator::compute::sparse::masked_row_scale_log1p_params params{row_sums, row_mask, 4.0f};
    const bool launched = cellerator::compute::sparse::masked_row_scale_log1p_inplace(&view, &params);
    cudaMemcpy(values, d_values, sizeof(values), cudaMemcpyDeviceToHost);
    cudaFree(d_values); cudaFree(d_columns); cudaFree(d_pointers);
    return launched && close(values[0], std::log1p(1.0f)) && close(values[1], std::log1p(3.0f))
        && close(values[2], 2.0f) && close(values[3], 4.0f);
}

} // namespace

int main() {
    const float sums[] = {4.0f, 6.0f};
    const std::uint8_t mask[] = {1u, 0u};
    auto *d_sums = device_copy(sums, 2u);
    auto *d_mask = device_copy(mask, 2u);
    const bool ok = d_sums != nullptr && d_mask != nullptr
        && run_blocked(d_sums, d_mask) && run_sliced(d_sums, d_mask) && run_compressed(d_sums, d_mask);
    cudaFree(d_mask); cudaFree(d_sums);
    if (!ok) std::fprintf(stderr, "generic masked row-scale/log1p validation failed\n");
    return ok ? 0 : 1;
}
