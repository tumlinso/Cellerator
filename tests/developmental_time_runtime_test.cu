#include "models/developmental_time/developmentalTime.hh"
#include <Cellerator/memory/cuda/shared_buffer.hh>
#include "model_runtime_fixtures.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace dt = ::cellerator::models::developmental_time;
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

void run_case(const dt::DevelopmentalTimeBatchView &batch, dt::DevelopmentalTimeBackend backend) {
    dt::DevelopmentalTimeModel model{};
    dt::DevelopmentalTimeModelConfig config;
    config.input_genes = 16u;
    config.stem_dim = 16u;
    config.hidden_dim = 16u;
    config.backend = backend;
    dt::init(&model, config);
    const dt::DevelopmentalTimeMetrics initial = dt::evaluate(&model, batch);
    require(std::isfinite(initial.total), "initial developmental_time loss must be finite");
    dt::DevelopmentalTimeMetrics step_metrics{};
    for (int i = 0; i < 8; ++i) step_metrics = dt::train_step(&model, batch);
    const dt::DevelopmentalTimeMetrics final = dt::evaluate(&model, batch);
    require(std::isfinite(step_metrics.total), "train step developmental_time loss must be finite");
    require(std::isfinite(final.total), "final developmental_time loss must be finite");
    require(final.total <= initial.total, "developmental_time loss should not increase after training");

    ::cellerator::memory::cuda::SharedDeviceBuffer<float> predicted = dt::infer_time(&model, batch);
    require(std::isfinite(download_first(predicted, model.ctx)), "developmental_time inference should be finite");
    dt::clear(&model);
}

} // namespace

int main() {
    int device_count = 0;
    require(cudaGetDeviceCount(&device_count) == cudaSuccess, "cudaGetDeviceCount failed");
    require(device_count > 0, "developmentalTimeRuntimeTest requires a CUDA device");

    constexpr unsigned int rows = 16u;
    constexpr unsigned int cols = 16u;
    std::vector<float> dense(rows * cols, 0.0f), target(rows, 0.0f);
    for (unsigned int row = 0; row < rows; ++row) {
        const bool early = row < rows / 2u;
        target[row] = early ? 0.15f + 0.01f * static_cast<float>(row) : 0.75f + 0.01f * static_cast<float>(row - rows / 2u);
        for (unsigned int col = 0; col < cols; ++col) {
            const bool active = early ? (col < cols / 2u) : (col >= cols / 2u);
            dense[static_cast<std::size_t>(row) * cols + col] = active ? 1.0f + 0.02f * static_cast<float>(row & 1u) : 0.05f;
        }
    }

    model_test::DenseEllFixture<rows, cols> host_ell(dense.data());

    int device = 0;
    require(cudaGetDevice(&device) == cudaSuccess, "cudaGetDevice failed");
    auto d_target = ::cellerator::memory::cuda::SharedDeviceBuffer<float>::allocate(target.size(), device);
    ::cellerator::memory::cuda::Stream upload_stream(device);
    d_target.upload_async(target.data(), target.size(), upload_stream);
    upload_stream.synchronize();

    model_test::BlockedEllUpload blocked_upload = model_test::upload_blocked_ell(host_ell.blocked(), upload_stream);
    model_test::SlicedEllUpload sliced_upload = model_test::upload_sliced_ell(host_ell.sliced(), upload_stream);

    const dt::DevelopmentalTimeBatchView blocked_batch =
        dt::make_developmental_time_blocked_ell_batch(blocked_upload.view, d_target.data());
    const dt::DevelopmentalTimeBatchView sliced_batch =
        dt::make_developmental_time_sliced_ell_batch(sliced_upload.view, d_target.data());

    run_case(blocked_batch, dt::DevelopmentalTimeBackend::tensor_cusparse);
    run_case(blocked_batch, dt::DevelopmentalTimeBackend::custom_wmma);
    run_case(sliced_batch, dt::DevelopmentalTimeBackend::tensor_cusparse);

    return 0;
}
