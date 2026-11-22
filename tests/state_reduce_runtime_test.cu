#include <Cellerator/examples/models/state_reduce/stateReduce.hh>
#include <Cellerator/memory/cuda/shared_buffer.hh>
#include "model_runtime_fixtures.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace sr = ::cellerator::models::state_reduce;
namespace runtime = ::cellerator::runtime;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

bool close_value(float lhs, float rhs, float tol = 1.0e-4f) {
    return std::fabs(lhs - rhs) <= tol;
}

std::vector<float> download_embeddings(const ::cellerator::memory::cuda::SharedDeviceBuffer<float> &latent,
                                       const runtime::execution_context &ctx,
                                       std::size_t rows,
                                       std::size_t cols) {
    std::vector<float> host(rows * cols, 0.0f);
    auto stream = ::cellerator::memory::cuda::Stream::borrow(ctx.stream, ctx.device);
    latent.download_async(host.data(), host.size(), stream);
    stream.synchronize();
    return host;
}

float row_distance_sq(const std::vector<float> &embedding,
                      std::size_t rows,
                      std::size_t cols,
                      std::size_t lhs,
                      std::size_t rhs) {
    float accum = 0.0f;
    for (std::size_t col = 0; col < cols; ++col) {
        const float diff = embedding[lhs * cols + col] - embedding[rhs * cols + col];
        accum += diff * diff;
    }
    return accum;
}

void run_case(const sr::StateReduceBatchView &batch,
              sr::StateReduceBackend backend,
              bool require_graph_geometry) {
    sr::StateReduceModel model{};
    sr::StateReduceModelConfig config;
    config.input_genes = 16u;
    config.latent_dim = 16u;
    config.factor_dim = 16u;
    config.backend = backend;
    sr::init(&model, config);

    const sr::StateReduceTrainMetrics initial = sr::evaluate(&model, batch);
    require(std::isfinite(initial.total), "initial state_reduce loss must be finite");
    sr::StateReduceTrainMetrics step_metrics{};
    for (int i = 0; i < 6; ++i) step_metrics = sr::train_step(&model, batch);
    const sr::StateReduceTrainMetrics final = sr::evaluate(&model, batch);
    require(std::isfinite(step_metrics.total), "train step loss must be finite");
    require(std::isfinite(final.total), "final state_reduce loss must be finite");
    require(final.total <= initial.total, "state_reduce loss should not increase after training");

    if (require_graph_geometry) {
        ::cellerator::memory::cuda::SharedDeviceBuffer<float> latent = sr::infer_embeddings(&model, batch);
        const std::vector<float> embedding = download_embeddings(latent, model.ctx, batch.rows, config.latent_dim);
        const float same_pair = row_distance_sq(embedding, batch.rows, config.latent_dim, 0u, 1u);
        const float cross_pair = row_distance_sq(embedding, batch.rows, config.latent_dim, 0u, 8u);
        require(same_pair < cross_pair, "graph-linked rows should embed closer than cross-state rows");
    }

    sr::clear(&model);
}

} // namespace

int main() {
    int device_count = 0;
    require(cudaGetDeviceCount(&device_count) == cudaSuccess, "cudaGetDeviceCount failed");
    require(device_count > 0, "stateReduceRuntimeTest requires at least one visible CUDA device");

    constexpr unsigned int rows = 16u;
    constexpr unsigned int cols = 16u;
    std::vector<float> dense(rows * cols, 0.0f);
    for (unsigned int row = 0; row < rows; ++row) {
        const bool low_state = row < rows / 2u;
        for (unsigned int col = 0; col < cols; ++col) {
            const bool active = low_state ? (col < cols / 2u) : (col >= cols / 2u);
            dense[static_cast<std::size_t>(row) * cols + col] = active
                ? 1.0f + 0.02f * static_cast<float>(row & 1u)
                : 0.05f * static_cast<float>((row + col) % 3u == 0u);
        }
    }

    model_test::DenseEllFixture<rows, cols> host_ell(dense.data());

    std::vector<std::uint32_t> graph_src, graph_dst;
    std::vector<float> graph_weight;
    for (std::uint32_t row = 0u; row < rows; row += 2u) {
        graph_src.push_back(row);
        graph_dst.push_back(row + 1u);
        graph_weight.push_back(1.0f);
    }

    int device = 0;
    require(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice failed");
    auto d_src = ::cellerator::memory::cuda::SharedDeviceBuffer<std::uint32_t>::allocate(graph_src.size(), device);
    auto d_dst = ::cellerator::memory::cuda::SharedDeviceBuffer<std::uint32_t>::allocate(graph_dst.size(), device);
    auto d_weight = ::cellerator::memory::cuda::SharedDeviceBuffer<float>::allocate(graph_weight.size(), device);
    ::cellerator::memory::cuda::Stream upload_stream(device);
    d_src.upload_async(graph_src.data(), graph_src.size(), upload_stream);
    d_dst.upload_async(graph_dst.data(), graph_dst.size(), upload_stream);
    d_weight.upload_async(graph_weight.data(), graph_weight.size(), upload_stream);
    upload_stream.synchronize();

    model_test::BlockedEllUpload blocked_upload = model_test::upload_blocked_ell(host_ell.blocked(), upload_stream);
    model_test::SlicedEllUpload sliced_upload = model_test::upload_sliced_ell(host_ell.sliced(), upload_stream);

    const sr::StateReduceGraphView graph{
        static_cast<std::uint32_t>(graph_src.size()),
        d_src.data(),
        d_dst.data(),
        d_weight.data()
    };

    const sr::StateReduceBatchView blocked_batch =
        sr::make_state_reduce_blocked_ell_batch(blocked_upload.view, graph);
    const sr::StateReduceBatchView sliced_batch =
        sr::make_state_reduce_sliced_ell_batch(sliced_upload.view, graph);

    run_case(blocked_batch, sr::StateReduceBackend::wmma_fused, true);
    run_case(blocked_batch, sr::StateReduceBackend::cusparse_heavy, false);
    run_case(sliced_batch, sr::StateReduceBackend::wmma_fused, false);

    return 0;
}
