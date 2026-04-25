#include <Cellerator/models/developmental_time_trajectory.hh>

#include "../extern/CellShard/include/CellShard/formats/blocked_ell.cuh"
#include "../extern/CellShard/include/CellShard/formats/sliced_ell.cuh"
#include "../extern/CellShard/include/CellShard/runtime/device/sharded_device.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace dtt = ::cellerator::models::developmental_time_trajectory;
namespace autograd = ::cellerator::compute::autograd;
namespace css = ::cellshard::sparse;
namespace csv = ::cellshard::device;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

bool populate_blocked_ell(css::blocked_ell *matrix, const std::vector<float> &dense, unsigned int rows, unsigned int cols) {
    css::init(matrix, rows, cols, rows * cols, 1u, cols);
    if (!css::allocate(matrix)) return false;
    for (unsigned int row = 0; row < rows; ++row) {
        for (unsigned int col = 0; col < cols; ++col) {
            matrix->blockColIdx[static_cast<std::size_t>(row) * cols + col] = col;
            matrix->val[static_cast<std::size_t>(row) * cols + col] = __float2half_rn(dense[static_cast<std::size_t>(row) * cols + col]);
        }
    }
    return true;
}

bool populate_sliced_ell(css::sliced_ell *matrix, const std::vector<float> &dense, unsigned int rows, unsigned int cols) {
    std::vector<unsigned int> slice_offsets(rows + 1u, 0u), slice_widths(rows, cols);
    for (unsigned int row = 0; row <= rows; ++row) slice_offsets[row] = row;
    css::init(matrix, rows, cols, rows * cols);
    if (!css::allocate(matrix, rows, slice_offsets.data(), slice_widths.data())) return false;
    for (unsigned int row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * cols;
        for (unsigned int col = 0; col < cols; ++col) {
            matrix->col_idx[base + col] = col;
            matrix->val[base + col] = __float2half_rn(dense[base + col]);
        }
    }
    return true;
}

csv::blocked_ell_view make_blocked_view(const css::blocked_ell &host, const csv::partition_record<css::blocked_ell> &record) {
    csv::blocked_ell_view out{};
    out.rows = host.rows;
    out.cols = host.cols;
    out.nnz = host.nnz;
    out.block_size = host.block_size;
    out.ell_cols = host.ell_cols;
    out.blockColIdx = static_cast<unsigned int *>(record.a0);
    out.val = static_cast<__half *>(record.a1);
    return out;
}

csv::sliced_ell_view make_sliced_view(const css::sliced_ell &host, const csv::partition_record<css::sliced_ell> &record) {
    csv::sliced_ell_view out{};
    const std::size_t offsets_bytes = static_cast<std::size_t>(host.slice_count + 1u) * sizeof(unsigned int);
    const std::size_t widths_offset = csv::align_up_bytes(offsets_bytes, alignof(unsigned int));
    const std::size_t widths_bytes = static_cast<std::size_t>(host.slice_count) * sizeof(unsigned int);
    const std::size_t slot_offsets_offset = csv::align_up_bytes(widths_offset + widths_bytes, alignof(unsigned int));
    out.rows = host.rows;
    out.cols = host.cols;
    out.nnz = host.nnz;
    out.slice_count = host.slice_count;
    out.slice_rows = css::uniform_slice_rows(&host);
    out.slice_row_offsets = static_cast<unsigned int *>(record.a0);
    out.slice_widths = static_cast<unsigned int *>(record.a1);
    out.slice_slot_offsets = host.slice_count != 0u ? reinterpret_cast<unsigned int *>(static_cast<char *>(record.storage) + slot_offsets_offset) : nullptr;
    out.col_idx = static_cast<unsigned int *>(record.a2);
    out.val = static_cast<__half *>(record.a3);
    return out;
}

float download_first(const autograd::device_buffer<float> &buf) {
    float value = 0.0f;
    autograd::download_device_buffer(buf, &value, 1u);
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

    autograd::device_buffer<float> predicted = dtt::infer_time(&model, batch);
    require(std::isfinite(download_first(predicted)), "trajectory inference should be finite");
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

    css::blocked_ell blocked{};
    css::sliced_ell sliced{};
    require(populate_blocked_ell(&blocked, dense, rows, cols), "populate_blocked_ell failed");
    require(populate_sliced_ell(&sliced, dense, rows, cols), "populate_sliced_ell failed");

    csv::partition_record<css::blocked_ell> blocked_record{};
    csv::partition_record<css::sliced_ell> sliced_record{};
    require(csv::upload(&blocked, &blocked_record) == cudaSuccess, "blocked upload failed");
    require(csv::upload(&sliced, &sliced_record) == cudaSuccess, "sliced upload failed");

    autograd::device_buffer<float> d_target = autograd::allocate_device_buffer<float>(target.size());
    autograd::device_buffer<std::uint32_t> d_src = autograd::allocate_device_buffer<std::uint32_t>(graph_src.size());
    autograd::device_buffer<std::uint32_t> d_dst = autograd::allocate_device_buffer<std::uint32_t>(graph_dst.size());
    autograd::device_buffer<float> d_weight = autograd::allocate_device_buffer<float>(graph_weight.size());
    autograd::upload_device_buffer(&d_target, target.data(), target.size());
    autograd::upload_device_buffer(&d_src, graph_src.data(), graph_src.size());
    autograd::upload_device_buffer(&d_dst, graph_dst.data(), graph_dst.size());
    autograd::upload_device_buffer(&d_weight, graph_weight.data(), graph_weight.size());

    const dtt::DevelopmentalTimeGraphView graph{
        static_cast<std::uint32_t>(graph_src.size()),
        d_src.data,
        d_dst.data,
        d_weight.data
    };

    const dtt::DevelopmentalTimeTrajectoryBatchView blocked_batch =
        dtt::make_developmental_time_trajectory_blocked_ell_batch(make_blocked_view(blocked, blocked_record), d_target.data, graph);
    const dtt::DevelopmentalTimeTrajectoryBatchView sliced_batch =
        dtt::make_developmental_time_trajectory_sliced_ell_batch(make_sliced_view(sliced, sliced_record), d_target.data, graph);

    run_case(blocked_batch);
    run_case(sliced_batch);

    require(csv::release(&blocked_record) == cudaSuccess, "release blocked record failed");
    require(csv::release(&sliced_record) == cudaSuccess, "release sliced record failed");
    css::clear(&blocked);
    css::clear(&sliced);
    return 0;
}
