#include <Cellerator/models/state_reduce.hh>

#include "../extern/CellShard/include/CellShard/formats/blocked_ell.cuh"
#include "../extern/CellShard/include/CellShard/formats/sliced_ell.cuh"
#include "../extern/CellShard/include/CellShard/runtime/device/sharded_device.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace sr = ::cellerator::models::state_reduce;
namespace autograd = ::cellerator::compute::autograd;
namespace cs = ::cellshard;
namespace css = ::cellshard::sparse;
namespace csv = ::cellshard::device;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

bool close_value(float lhs, float rhs, float tol = 1.0e-4f) {
    return std::fabs(lhs - rhs) <= tol;
}

bool populate_blocked_ell(css::blocked_ell *matrix,
                          const std::vector<float> &dense,
                          unsigned int rows,
                          unsigned int cols) {
    css::init(matrix, rows, cols, rows * cols, 1u, cols);
    if (!css::allocate(matrix)) return false;
    for (unsigned int row = 0; row < rows; ++row) {
        for (unsigned int col = 0; col < cols; ++col) {
            matrix->blockColIdx[static_cast<std::size_t>(row) * cols + col] = col;
            matrix->val[static_cast<std::size_t>(row) * cols + col] =
                __float2half_rn(dense[static_cast<std::size_t>(row) * cols + col]);
        }
    }
    return true;
}

bool populate_sliced_ell(css::sliced_ell *matrix,
                         const std::vector<float> &dense,
                         unsigned int rows,
                         unsigned int cols) {
    std::vector<unsigned int> slice_offsets(rows + 1u, 0u);
    std::vector<unsigned int> slice_widths(rows, cols);
    for (unsigned int row = 0; row <= rows; ++row) slice_offsets[row] = row;
    css::init(matrix, rows, cols, rows * cols);
    if (!css::allocate(matrix, rows, slice_offsets.data(), slice_widths.data())) return false;
    for (unsigned int row = 0; row < rows; ++row) {
        const std::size_t slot_base = static_cast<std::size_t>(row) * cols;
        for (unsigned int col = 0; col < cols; ++col) {
            matrix->col_idx[slot_base + col] = col;
            matrix->val[slot_base + col] = __float2half_rn(dense[slot_base + col]);
        }
    }
    return true;
}

csv::blocked_ell_view make_blocked_view(const css::blocked_ell &host,
                                        const csv::partition_record<css::blocked_ell> &record) {
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

csv::sliced_ell_view make_sliced_view(const css::sliced_ell &host,
                                      const csv::partition_record<css::sliced_ell> &record) {
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
    out.slice_slot_offsets = host.slice_count != 0u
        ? reinterpret_cast<unsigned int *>(static_cast<char *>(record.storage) + slot_offsets_offset)
        : nullptr;
    out.col_idx = static_cast<unsigned int *>(record.a2);
    out.val = static_cast<__half *>(record.a3);
    return out;
}

std::vector<float> download_embeddings(const autograd::device_buffer<float> &latent,
                                       std::size_t rows,
                                       std::size_t cols) {
    std::vector<float> host(rows * cols, 0.0f);
    autograd::download_device_buffer(latent, host.data(), host.size());
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
        autograd::device_buffer<float> latent = sr::infer_embeddings(&model, batch);
        const std::vector<float> embedding = download_embeddings(latent, batch.rows, config.latent_dim);
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

    css::blocked_ell blocked{};
    css::sliced_ell sliced{};
    require(populate_blocked_ell(&blocked, dense, rows, cols), "populate_blocked_ell failed");
    require(populate_sliced_ell(&sliced, dense, rows, cols), "populate_sliced_ell failed");

    csv::partition_record<css::blocked_ell> blocked_record{};
    csv::partition_record<css::sliced_ell> sliced_record{};
    require(csv::upload(&blocked, &blocked_record) == cudaSuccess, "blocked upload failed");
    require(csv::upload(&sliced, &sliced_record) == cudaSuccess, "sliced upload failed");

    std::vector<std::uint32_t> graph_src, graph_dst;
    std::vector<float> graph_weight;
    for (std::uint32_t row = 0u; row < rows; row += 2u) {
        graph_src.push_back(row);
        graph_dst.push_back(row + 1u);
        graph_weight.push_back(1.0f);
    }

    autograd::device_buffer<std::uint32_t> d_src = autograd::allocate_device_buffer<std::uint32_t>(graph_src.size());
    autograd::device_buffer<std::uint32_t> d_dst = autograd::allocate_device_buffer<std::uint32_t>(graph_dst.size());
    autograd::device_buffer<float> d_weight = autograd::allocate_device_buffer<float>(graph_weight.size());
    autograd::upload_device_buffer(&d_src, graph_src.data(), graph_src.size());
    autograd::upload_device_buffer(&d_dst, graph_dst.data(), graph_dst.size());
    autograd::upload_device_buffer(&d_weight, graph_weight.data(), graph_weight.size());

    const sr::StateReduceGraphView graph{
        static_cast<std::uint32_t>(graph_src.size()),
        d_src.data,
        d_dst.data,
        d_weight.data
    };

    const sr::StateReduceBatchView blocked_batch =
        sr::make_state_reduce_blocked_ell_batch(make_blocked_view(blocked, blocked_record), graph);
    const sr::StateReduceBatchView sliced_batch =
        sr::make_state_reduce_sliced_ell_batch(make_sliced_view(sliced, sliced_record), graph);

    run_case(blocked_batch, sr::StateReduceBackend::wmma_fused, true);
    run_case(blocked_batch, sr::StateReduceBackend::cusparse_heavy, false);
    run_case(sliced_batch, sr::StateReduceBackend::wmma_fused, false);

    require(csv::release(&blocked_record) == cudaSuccess, "release blocked record failed");
    require(csv::release(&sliced_record) == cudaSuccess, "release sliced record failed");
    css::clear(&blocked);
    css::clear(&sliced);
    return 0;
}
