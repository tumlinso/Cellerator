#include "models/developmental_time_trajectory/developmentalTimeTrajectory.hh"
#include <Cellerator/memory/cuda/shared_buffer.hh>
#include "model_runtime_fixtures.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace dtt = ::cellerator::models::developmental_time_trajectory;
namespace runtime = ::cellerator::runtime;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

float download_first(const ::cellerator::memory::cuda::SharedDeviceBuffer<float> &buf,
                    const runtime::execution_context &ctx) {
    float value = 0.0f;
    auto stream = ::cellerator::memory::cuda::Stream::borrow(ctx.stream, ctx.device);
    buf.download_async(&value, 1u, stream);
    stream.synchronize();
    return value;
}

void run_case(const dtt::DevelopmentalTimeTrajectoryBatchView &batch) {
    dtt::DevelopmentalTimeTrajectoryModel model{};
    dtt::DevelopmentalTimeTrajectoryModelConfig config;
    config.input_genes = 16u;
    config.stem_dim = 16u;
    config.hidden_dim = 16u;
    dtt::init(&model, config);
    const dtt::DevelopmentalTimeTrajectoryMetrics initial = dtt::evaluate(&model, batch);
    require(std::isfinite(initial.total), "initial trajectory loss must be finite");
    dtt::DevelopmentalTimeTrajectoryMetrics step_metrics{};
    for (int i = 0; i < 10; ++i) step_metrics = dtt::train_step(&model, batch);
    const dtt::DevelopmentalTimeTrajectoryMetrics final = dtt::evaluate(&model, batch);
    require(std::isfinite(step_metrics.total), "train step trajectory loss must be finite");
    require(std::isfinite(final.total), "final trajectory loss must be finite");
    require(final.total <= initial.total, "trajectory loss should not increase after training");

    ::cellerator::memory::cuda::SharedDeviceBuffer<float> predicted = dtt::infer_time(&model, batch);
    require(std::isfinite(download_first(predicted, model.ctx)), "trajectory inference should be finite");
    dtt::clear(&model);
}

} // namespace

int main() {
    int device_count = 0;
    require(cudaGetDeviceCount(&device_count) == cudaSuccess, "cudaGetDeviceCount failed");
    require(device_count > 0, "developmentalTimeTrajectoryRuntimeTest requires a CUDA device");

    constexpr unsigned int rows = 16u;
    constexpr unsigned int cols = 16u;
    std::vector<float> dense(rows * cols, 0.0f), target(rows, 0.0f);
    std::vector<std::uint32_t> graph_src, graph_dst;
    std::vector<float> graph_weight;
    for (unsigned int row = 0; row < rows; ++row) {
        target[row] = 0.05f * static_cast<float>(row);
        for (unsigned int col = 0; col < cols; ++col) {
            const bool active = col / 4u == row / 4u;
            dense[static_cast<std::size_t>(row) * cols + col] = active ? 1.0f + 0.01f * static_cast<float>(row) : 0.02f;
        }
        if (row + 1u < rows) {
            graph_src.push_back(row);
            graph_dst.push_back(row + 1u);
            graph_weight.push_back(1.0f);
        }
    }

    model_test::DenseEllFixture<rows, cols> host_ell(dense.data());

    int device = 0;
    require(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice failed");
    auto d_target = ::cellerator::memory::cuda::SharedDeviceBuffer<float>::allocate(target.size(), device);
    auto d_src = ::cellerator::memory::cuda::SharedDeviceBuffer<std::uint32_t>::allocate(graph_src.size(), device);
    auto d_dst = ::cellerator::memory::cuda::SharedDeviceBuffer<std::uint32_t>::allocate(graph_dst.size(), device);
    auto d_weight = ::cellerator::memory::cuda::SharedDeviceBuffer<float>::allocate(graph_weight.size(), device);
    ::cellerator::memory::cuda::Stream upload_stream(device);
    d_target.upload_async(target.data(), target.size(), upload_stream);
    d_src.upload_async(graph_src.data(), graph_src.size(), upload_stream);
    d_dst.upload_async(graph_dst.data(), graph_dst.size(), upload_stream);
    d_weight.upload_async(graph_weight.data(), graph_weight.size(), upload_stream);
    upload_stream.synchronize();

    model_test::BlockedEllUpload blocked_upload = model_test::upload_blocked_ell(host_ell.blocked(), upload_stream);
    model_test::SlicedEllUpload sliced_upload = model_test::upload_sliced_ell(host_ell.sliced(), upload_stream);

    const dtt::DevelopmentalTimeGraphView graph{
        static_cast<std::uint32_t>(graph_src.size()),
        d_src.data(),
        d_dst.data(),
        d_weight.data()
    };

    const dtt::DevelopmentalTimeTrajectoryBatchView blocked_batch =
        dtt::make_developmental_time_trajectory_blocked_ell_batch(blocked_upload.view, d_target.data(), graph);
    const dtt::DevelopmentalTimeTrajectoryBatchView sliced_batch =
        dtt::make_developmental_time_trajectory_sliced_ell_batch(sliced_upload.view, d_target.data(), graph);

    run_case(blocked_batch);
    run_case(sliced_batch);

    return 0;
}
