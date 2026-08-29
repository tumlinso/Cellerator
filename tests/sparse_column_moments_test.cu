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
    if (cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) return nullptr;
    return device;
}

struct output_buffers {
    float *sum = nullptr;
    float *squares = nullptr;
    float *support = nullptr;
    float *active = nullptr;
};

bool check_output(output_buffers &buffers) {
    float sum[4], squares[4], support[4], active = 0.0f;
    cudaMemcpy(sum, buffers.sum, sizeof(sum), cudaMemcpyDeviceToHost);
    cudaMemcpy(squares, buffers.squares, sizeof(squares), cudaMemcpyDeviceToHost);
    cudaMemcpy(support, buffers.support, sizeof(support), cudaMemcpyDeviceToHost);
    cudaMemcpy(&active, buffers.active, sizeof(active), cudaMemcpyDeviceToHost);
    return std::fabs(sum[0] - 1.0f) < 1.0e-5f && std::fabs(sum[2] - 3.0f) < 1.0e-5f
        && std::fabs(squares[0] - 1.0f) < 1.0e-5f && std::fabs(squares[2] - 9.0f) < 1.0e-5f
        && support[0] == 1.0f && support[2] == 1.0f && sum[1] == 0.0f && sum[3] == 0.0f
        && active == 1.0f;
}

template<class View>
bool run(const View &view, const std::uint8_t *mask) {
    output_buffers buffers;
    cudaMalloc(reinterpret_cast<void **>(&buffers.sum), 4u * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&buffers.squares), 4u * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&buffers.support), 4u * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&buffers.active), sizeof(float));
    cellerator::compute::sparse::column_moments_view output{4u, buffers.sum, buffers.squares, buffers.support, buffers.active};
    const bool ok = cellerator::compute::sparse::zero_column_moments(&output)
        && cellerator::compute::sparse::accumulate_column_moments(&view, mask, &output)
        && cudaDeviceSynchronize() == cudaSuccess && check_output(buffers);
    cudaFree(buffers.active); cudaFree(buffers.support); cudaFree(buffers.squares); cudaFree(buffers.sum);
    return ok;
}

} // namespace

int main() {
    const std::uint8_t mask[] = {1u, 0u};
    auto *d_mask = device_copy(mask, 2u);

    unsigned int blocked_columns[] = {0u, 1u};
    __half blocked_values[] = {__float2half(1.0f), __float2half(0.0f), __float2half(3.0f), __float2half(0.0f),
                               __float2half(0.0f), __float2half(2.0f), __float2half(0.0f), __float2half(4.0f)};
    auto *d_blocked_columns = device_copy(blocked_columns, 2u); auto *d_blocked_values = device_copy(blocked_values, 8u);
    cellerator::matrix::device::blocked_ell_view blocked{2u, 4u, 4u, 2u, 4u, d_blocked_columns, d_blocked_values};

    unsigned int offsets[] = {0u}, widths[] = {2u}, slots[] = {0u}, columns[] = {0u, 2u, 1u, 3u};
    __half values[] = {__float2half(1.0f), __float2half(3.0f), __float2half(2.0f), __float2half(4.0f)};
    auto *d_offsets = device_copy(offsets, 1u); auto *d_widths = device_copy(widths, 1u);
    auto *d_slots = device_copy(slots, 1u); auto *d_columns = device_copy(columns, 4u); auto *d_values = device_copy(values, 4u);
    cellerator::matrix::device::sliced_ell_view sliced{2u, 4u, 4u, 1u, 2u, d_offsets, d_widths, d_slots, d_columns, d_values};

    unsigned int pointers[] = {0u, 2u, 4u};
    auto *d_pointers = device_copy(pointers, 3u); auto *d_csr_columns = device_copy(columns, 4u); auto *d_csr_values = device_copy(values, 4u);
    cellerator::matrix::device::compressed_view compressed{2u, 4u, 4u,
        cellerator::matrix::device::compressed_by_row, d_pointers, d_csr_columns, d_csr_values};

    const bool ok = d_mask != nullptr && run(blocked, d_mask) && run(sliced, d_mask) && run(compressed, d_mask);
    cudaFree(d_csr_values); cudaFree(d_csr_columns); cudaFree(d_pointers);
    cudaFree(d_values); cudaFree(d_columns); cudaFree(d_slots); cudaFree(d_widths); cudaFree(d_offsets);
    cudaFree(d_blocked_values); cudaFree(d_blocked_columns); cudaFree(d_mask);
    if (!ok) std::fprintf(stderr, "generic sparse column-moments validation failed\n");
    return ok ? 0 : 1;
}
