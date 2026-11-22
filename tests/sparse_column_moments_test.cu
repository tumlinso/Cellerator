#include <Cellerator/compute/operators/sparse/column_moments.cuh>

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
    if (cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(device);
        return nullptr;
    }
    return device;
}

struct output_buffers {
    float *sum = nullptr;
    float *squares = nullptr;
    float *support = nullptr;
    float *active = nullptr;
};

struct expected_moments {
    float sum[4];
    float squares[4];
    float support[4];
    float active;
};

constexpr expected_moments first_row{
    {1.0f, 0.0f, 3.0f, 0.0f}, {1.0f, 0.0f, 9.0f, 0.0f}, {1.0f, 0.0f, 1.0f, 0.0f}, 1.0f};

bool check_output(const output_buffers &buffers, const expected_moments &expected) {
    float sum[4], squares[4], support[4], active = 0.0f;
    if (cudaMemcpy(sum, buffers.sum, sizeof(sum), cudaMemcpyDeviceToHost) != cudaSuccess
        || cudaMemcpy(squares, buffers.squares, sizeof(squares), cudaMemcpyDeviceToHost) != cudaSuccess
        || cudaMemcpy(support, buffers.support, sizeof(support), cudaMemcpyDeviceToHost) != cudaSuccess
        || cudaMemcpy(&active, buffers.active, sizeof(active), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    for (unsigned int i = 0; i < 4u; ++i) {
        if (!(std::fabs(sum[i] - expected.sum[i]) < 1.0e-5f)
            || !(std::fabs(squares[i] - expected.squares[i]) < 1.0e-5f)
            || support[i] != expected.support[i]) return false;
    }
    return active == expected.active;
}

template<class View>
bool run(const View &view, const std::uint8_t *mask, const expected_moments &expected = first_row) {
    output_buffers buffers;
    cudaMalloc(reinterpret_cast<void **>(&buffers.sum), 4u * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&buffers.squares), 4u * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&buffers.support), 4u * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&buffers.active), sizeof(float));
    cellerator::compute::sparse::column_moments_view output{4u, buffers.sum, buffers.squares, buffers.support, buffers.active};
    const bool ok = buffers.active != nullptr && cellerator::compute::sparse::zero_column_moments(&output)
        && cellerator::compute::sparse::accumulate_column_moments(&view, mask, &output)
        && cudaDeviceSynchronize() == cudaSuccess && check_output(buffers, expected);
    cudaFree(buffers.active); cudaFree(buffers.support); cudaFree(buffers.squares); cudaFree(buffers.sum);
    return ok;
}

bool run_nonuniform_sliced(const std::uint8_t *mask) {
    namespace base = cellerator::tensor::base;
    const unsigned int offsets[] = {0u, 1u, 3u}, widths[] = {3u, 2u, 1u}, slots[] = {0u, 3u, 7u};
    const unsigned int columns[] = {0u, base::sliced_ell_invalid_col, 2u, 1u, 3u, 2u, 4u, 0u};
    const __half values[] = {__float2half(1.0f), __float2half(7.0f), __float2half(3.0f), __float2half(2.0f),
                             __float2half(4.0f), __float2half(4.0f), __float2half(11.0f), __float2half(8.0f)};
    auto *d_offsets = device_copy(offsets, 3u);
    auto *d_widths = device_copy(widths, 3u);
    auto *d_slots = device_copy(slots, 3u);
    auto *d_columns = device_copy(columns, 8u);
    auto *d_values = device_copy(values, 8u);
    const base::device::SlicedEllView view{4u, 4u, 6u, 3u, 0u, d_offsets, d_widths, d_slots, d_columns, d_values};
    const expected_moments expected{
        {9.0f, 0.0f, 7.0f, 0.0f}, {65.0f, 0.0f, 25.0f, 0.0f}, {2.0f, 0.0f, 2.0f, 0.0f}, 3.0f};
    const bool ok = d_offsets && d_widths && d_slots && d_columns && d_values
        && run(view, mask, expected);
    cudaFree(d_values); cudaFree(d_columns); cudaFree(d_slots);
    cudaFree(d_widths); cudaFree(d_offsets);
    return ok;
}

template<class Index>
bool run_compressed(const std::uint8_t *mask, const __half *values) {
    namespace base = cellerator::tensor::base;
    const Index pointers[] = {0, 2, 4}, columns[] = {0, 2, 1, 3};
    auto *d_pointers = device_copy(pointers, 3u);
    auto *d_columns = device_copy(columns, 4u);
    auto *d_values = device_copy(values, 4u);
    const base::CompressedView<Index, const void> view{
        {2u, 4u, 4u, base::CompressedAxis::row, d_pointers, d_columns}, d_values};
    const bool ok = d_pointers != nullptr && d_columns != nullptr && d_values != nullptr
        && run(view, mask);
    cudaFree(d_values); cudaFree(d_columns); cudaFree(d_pointers);
    return ok;
}

} // namespace

int main() {
    const std::uint8_t mask[] = {1u, 0u, 1u, 1u};
    auto *d_mask = device_copy(mask, 4u);

    unsigned int blocked_columns[] = {0u, 1u};
    __half blocked_values[] = {__float2half(1.0f), __float2half(0.0f), __float2half(3.0f), __float2half(0.0f),
                               __float2half(0.0f), __float2half(2.0f), __float2half(0.0f), __float2half(4.0f)};
    auto *d_blocked_columns = device_copy(blocked_columns, 2u); auto *d_blocked_values = device_copy(blocked_values, 8u);
    cellerator::tensor::base::device::BlockedEllView blocked{2u, 4u, 4u, 2u, 4u, d_blocked_columns, d_blocked_values};

    unsigned int offsets[] = {0u}, widths[] = {2u}, slots[] = {0u}, columns[] = {0u, 2u, 1u, 3u};
    __half values[] = {__float2half(1.0f), __float2half(3.0f), __float2half(2.0f), __float2half(4.0f)};
    auto *d_offsets = device_copy(offsets, 1u); auto *d_widths = device_copy(widths, 1u);
    auto *d_slots = device_copy(slots, 1u); auto *d_columns = device_copy(columns, 4u); auto *d_values = device_copy(values, 4u);
    cellerator::tensor::base::device::SlicedEllView sliced{2u, 4u, 4u, 1u, 2u, d_offsets, d_widths, d_slots, d_columns, d_values};

    const bool ok = d_mask != nullptr && run(blocked, d_mask) && run(sliced, d_mask)
        && run_nonuniform_sliced(d_mask)
        && run_compressed<uint32_t>(d_mask, values) && run_compressed<int32_t>(d_mask, values)
        && run_compressed<int64_t>(d_mask, values);
    cudaFree(d_values); cudaFree(d_columns); cudaFree(d_slots); cudaFree(d_widths); cudaFree(d_offsets);
    cudaFree(d_blocked_values); cudaFree(d_blocked_columns); cudaFree(d_mask);
    if (!ok) std::fprintf(stderr, "generic sparse column-moments validation failed\n");
    return ok ? 0 : 1;
}
