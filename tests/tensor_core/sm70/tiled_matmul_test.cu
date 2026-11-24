#include <Cellerator/memory/cuda/resources.hh>
#include <Cellerator/ops/matrix/tiled_matmul_cuda.cuh>

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

namespace {

namespace matrix = cellerator::ops::matrix;
namespace memory = cellerator::memory;
namespace gpu = memory::cuda;
using HalfTile = memory::Tile<__half>;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

int find_volta() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return -1;
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp properties{};
        gpu::check(cudaGetDeviceProperties(&properties, device), "read device properties");
        if (properties.major == 7 && properties.minor == 0) return device;
    }
    return -1;
}

void check_multiply(int device) {
    constexpr size_t extent = HalfTile::rows;
    constexpr float sentinel = -1234.0f;
    constexpr matrix::TiledMatmulShape large{2, 3, 4}, small{1, 1, 1};
    const gpu::DeviceGuard guard(device);
    const gpu::Stream stream(device);
    const memory::placement where{memory::domain::device, int16_t(device)};
    std::vector<HalfTile> weights(large.output_tiles*large.feature_tiles);
    std::vector<HalfTile> features(large.feature_tiles*large.cell_tiles);
    std::vector<__half> dense_w(weights.size()*extent*extent);
    std::vector<__half> dense_x(features.size()*extent*extent);
    std::vector<float> output(large.output_tiles*extent*(large.cell_tiles*extent + 8));
    gpu::DeviceBuffer weights_gpu(weights.size()*sizeof(HalfTile), device);
    gpu::DeviceBuffer features_gpu(features.size()*sizeof(HalfTile), device);
    gpu::DeviceBuffer output_gpu(output.size()*sizeof(float), device);
    const memory::const_array_view<HalfTile> w{static_cast<const HalfTile *>(weights_gpu.data()),
        weights.size(), where};
    const memory::const_array_view<HalfTile> x{static_cast<const HalfTile *>(features_gpu.data()),
        features.size(), where};

    matrix::TiledMatmulPlan large_plan, small_plan;
    gpu::check(matrix::prepare_tiled_matmul(device, large, large_plan), "prepare rectangular multiply");
    gpu::check(matrix::prepare_tiled_matmul(device, small, small_plan), "prepare one-tile multiply");

    auto run = [&](matrix::TiledMatmulShape shape, const matrix::TiledMatmulPlan &plan, int phase) {
        const size_t m = shape.output_tiles*extent, n = shape.cell_tiles*extent;
        const size_t k = shape.feature_tiles*extent, stride = n + 8;
        const memory::matrix_view<float> y{static_cast<float *>(output_gpu.data()), m, n, stride, where};
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < k; ++j) {
                const auto value = __float2half(float(int((i*7 + j*3 + phase) % 17) - 8) * 0.125f);
                dense_w[i*k + j] = value;
                weights[(i/extent)*shape.feature_tiles + j/extent].data()[(i%extent)*extent + j%extent] = value;
            }
        for (size_t i = 0; i < k; ++i)
            for (size_t j = 0; j < n; ++j) {
                const auto value = __float2half(float(int((i*11 + j*13 + 3*phase) % 19) - 9) * 0.0625f);
                dense_x[i*n + j] = value;
                features[(i/extent)*shape.cell_tiles + j/extent].data()[(i%extent)*extent + j%extent] = value;
            }

        const size_t weight_count = shape.output_tiles*shape.feature_tiles;
        const size_t feature_count = shape.feature_tiles*shape.cell_tiles;
        for (float &value : output) value = sentinel;
        weights_gpu.upload_async(weights.data(), weight_count*sizeof(HalfTile), stream);
        features_gpu.upload_async(features.data(), feature_count*sizeof(HalfTile), stream);
        output_gpu.upload_async(output.data(), output.size()*sizeof(float), stream);

        auto short_weights = w;
        short_weights.count = weight_count - 1;
        require(matrix::enqueue_tiled_matmul(plan, short_weights, x, y, stream.get())
            == cudaErrorInvalidValue, "accepted short weight tile array");
        auto misaligned = y;
        ++misaligned.data;
        require(matrix::enqueue_tiled_matmul(plan, w, x, misaligned, stream.get())
            == cudaErrorInvalidValue, "accepted misaligned output");
        auto bad_stride = y;
        ++bad_stride.row_stride;
        require(matrix::enqueue_tiled_matmul(plan, w, x, bad_stride, stream.get())
            == cudaErrorInvalidValue, "accepted invalid output stride");

        gpu::check(matrix::enqueue_tiled_matmul(plan, w, x, y, stream.get()), "enqueue tiled multiply");
        output_gpu.download_async(output.data(), output.size()*sizeof(float), stream);
        stream.synchronize();
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double expected = 0;
                for (size_t p = 0; p < k; ++p)
                    expected += double(__half2float(dense_w[i*k + p])) * __half2float(dense_x[p*n + j]);
                require(output[i*stride + j] == expected, "tiled multiply disagrees with FP64 reference");
            }
            for (size_t j = n; j < stride; ++j)
                require(output[i*stride + j] == sentinel, "wrote to output row padding");
        }
        for (size_t i = m*stride; i < output.size(); ++i)
            require(output[i] == sentinel, "wrote outside the output view");
    };

    run(large, large_plan, 0);
    run(large, large_plan, 1);
    run(small, small_plan, 2);

    require(matrix::prepare_tiled_matmul(device, {1, 1, 0}, small_plan) == cudaErrorInvalidValue,
        "accepted an empty feature axis");
    const memory::matrix_view<float> small_output{static_cast<float *>(output_gpu.data()),
        extent, extent, extent + 8, where};
    require(matrix::enqueue_tiled_matmul(small_plan, w, x, small_output, stream.get())
        == cudaErrorInvalidValue, "accepted a plan after failed preparation");
}

} // namespace

int main() try {
    const int device = find_volta();
    if (device < 0) {
        std::puts("Volta (sm_70) required");
        return 77;
    }
    check_multiply(device);
} catch (const std::exception &error) {
    std::fprintf(stderr, "tiled matmul test: %s\n", error.what());
    return 1;
}
