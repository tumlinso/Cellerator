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
    if (cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(device);
        return nullptr;
    }
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
    cellerator::tensor::base::device::BlockedEllView view{2u, 4u, 4u, 2u, 4u, d_columns, d_values};
    cellerator::compute::sparse::masked_row_scale_log1p_params params{row_sums, row_mask, 4.0f};
    const bool launched = cellerator::compute::sparse::masked_row_scale_log1p_inplace(&view, &params);
    cudaMemcpy(values, d_values, sizeof(values), cudaMemcpyDeviceToHost);
    cudaFree(d_values); cudaFree(d_columns);
    return launched && close(values[0], std::log1p(1.0f)) && close(values[2], std::log1p(3.0f))
        && close(values[5], 2.0f) && close(values[7], 4.0f);
}

template<std::size_t Count>
bool check_sliced(cellerator::tensor::base::device::SlicedEllView view,
                  const float (&expected)[Count], const float *row_sums,
                  const std::uint8_t *row_mask) {
    view.slice_row_offsets = device_copy(view.slice_row_offsets, view.slice_count);
    view.slice_widths = device_copy(view.slice_widths, view.slice_count);
    view.slice_slot_offsets = device_copy(view.slice_slot_offsets, view.slice_count);
    view.col_idx = device_copy(view.col_idx, Count);
    view.val = device_copy(view.val, Count);
    cellerator::compute::sparse::masked_row_scale_log1p_params params{row_sums, row_mask, 4.0f};
    __half actual[Count];
    const bool copied = view.slice_row_offsets && view.slice_widths && view.slice_slot_offsets
        && view.col_idx && view.val
        && cellerator::compute::sparse::masked_row_scale_log1p_inplace(&view, &params)
        && cudaMemcpy(actual, view.val, sizeof(actual), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(view.val); cudaFree(view.col_idx); cudaFree(view.slice_slot_offsets);
    cudaFree(view.slice_widths); cudaFree(view.slice_row_offsets);
    if (!copied) return false;
    for (std::size_t i = 0; i < Count; ++i) {
        if (!close(actual[i], expected[i])) return false;
    }
    return true;
}

bool run_sliced(const float *row_sums, const std::uint8_t *row_mask) {
    unsigned int offsets[] = {0u}, widths[] = {2u}, slots[] = {0u}, columns[] = {0u, 2u, 1u, 3u};
    __half values[] = {__float2half(1.0f), __float2half(3.0f), __float2half(2.0f), __float2half(4.0f)};
    const float expected[] = {std::log1p(1.0f), std::log1p(3.0f), 2.0f, 4.0f};
    return check_sliced({2u, 4u, 4u, 1u, 2u, offsets, widths, slots, columns, values},
        expected, row_sums, row_mask);
}

bool run_nonuniform_sliced(const float *row_sums, const std::uint8_t *row_mask) {
    namespace base = cellerator::tensor::base;
    unsigned int offsets[] = {0u, 1u, 3u}, widths[] = {3u, 2u, 1u}, slots[] = {0u, 3u, 7u};
    // Nonzero padding and a masked row must survive without a write.
    unsigned int columns[] = {0u, base::sliced_ell_invalid_col, 2u, 1u, 3u, 2u, 4u, 0u};
    __half values[] = {__float2half(1.0f), __float2half(7.0f), __float2half(3.0f), __float2half(2.0f),
                       __float2half(4.0f), __float2half(4.0f), __float2half(11.0f), __float2half(8.0f)};
    const float expected[] = {std::log1p(1.0f), 7.0f, std::log1p(3.0f), 2.0f,
                              4.0f, std::log1p(4.0f), 11.0f, std::log1p(4.0f)};
    return check_sliced({4u, 4u, 6u, 3u, 0u, offsets, widths, slots, columns, values},
        expected, row_sums, row_mask);
}

template<class Index>
bool run_compressed(const float *row_sums, const std::uint8_t *row_mask) {
    namespace base = cellerator::tensor::base;
    const Index pointers[] = {0, 2, 4}, columns[] = {0, 2, 1, 3};
    __half values[] = {__float2half(1.0f), __float2half(3.0f), __float2half(2.0f), __float2half(4.0f)};
    auto *d_pointers = device_copy(pointers, 3u); auto *d_columns = device_copy(columns, 4u);
    auto *d_values = device_copy(values, 4u);
    const base::CompressedView<Index, void> view{
        {2u, 4u, 4u, base::CompressedAxis::row, d_pointers, d_columns}, d_values};
    cellerator::compute::sparse::masked_row_scale_log1p_params params{row_sums, row_mask, 4.0f};
    const bool transformed = cellerator::compute::sparse::masked_row_scale_log1p_inplace(&view, &params)
        && cudaMemcpy(values, d_values, sizeof(values), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(d_values); cudaFree(d_columns); cudaFree(d_pointers);
    return transformed && close(values[0], std::log1p(1.0f)) && close(values[1], std::log1p(3.0f))
        && close(values[2], 2.0f) && close(values[3], 4.0f);
}

} // namespace

int main() {
    const float sums[] = {4.0f, 6.0f, 4.0f, 8.0f};
    const std::uint8_t mask[] = {1u, 0u, 1u, 1u};
    auto *d_sums = device_copy(sums, 4u);
    auto *d_mask = device_copy(mask, 4u);
    const bool ok = d_sums != nullptr && d_mask != nullptr
        && run_blocked(d_sums, d_mask) && run_sliced(d_sums, d_mask)
        && run_nonuniform_sliced(d_sums, d_mask)
        && run_compressed<uint32_t>(d_sums, d_mask) && run_compressed<int32_t>(d_sums, d_mask)
        && run_compressed<int64_t>(d_sums, d_mask);
    cudaFree(d_mask); cudaFree(d_sums);
    if (!ok) std::fprintf(stderr, "generic masked row-scale/log1p validation failed\n");
    return ok ? 0 : 1;
}
