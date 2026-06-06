#include <Cellerator/compute/model_ops.hh>
#include <Cellerator/models/dense_reduce.hh>

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace dr = ::cellerator::models::dense_reduce;
namespace ops = ::cellerator::compute::model_ops;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

bool close_scalar(const torch::Tensor &lhs, const torch::Tensor &rhs, double atol = 1.0e-5, double rtol = 1.0e-5) {
    return torch::allclose(lhs.detach().to(torch::kCPU), rhs.detach().to(torch::kCPU), rtol, atol);
}

bool close_tensor(const torch::Tensor &lhs, const torch::Tensor &rhs, double atol = 1.0e-5, double rtol = 1.0e-5) {
    return torch::allclose(lhs.detach().to(torch::kCPU), rhs.detach().to(torch::kCPU), rtol, atol);
}

void check_dense_reduce_cuda_matches_cpu() {
    dr::DenseReduceLossConfig config;
    config.recon_weight = 1.0;
    config.nonzero_recon_weight = 3.0;
    config.local_time_window = 0.08;
    config.far_time_window = 0.30;
    config.local_weight = 0.4;
    config.far_weight = 0.7;
    config.margin = 0.6;
    config.max_sampled_pairs = 64;

    const torch::Tensor features = torch::tensor(
        {
            {0.0f, 1.0f, 0.5f},
            {0.5f, 0.0f, 1.5f},
            {1.0f, 0.5f, 0.0f},
            {1.5f, 1.0f, 0.5f},
        },
        torch::TensorOptions().dtype(torch::kFloat32));
    const torch::Tensor developmental_time = torch::tensor(
        {0.10f, 0.16f, 0.62f, 0.95f},
        torch::TensorOptions().dtype(torch::kFloat32));
    const torch::Tensor cell_indices = torch::tensor(
        {10, 11, 12, 13},
        torch::TensorOptions().dtype(torch::kInt64));
    const torch::Tensor time_buckets = torch::tensor(
        {0, 0, 1, 2},
        torch::TensorOptions().dtype(torch::kInt64));

    torch::manual_seed(0);
    torch::Tensor reconstruction_cpu = torch::tensor(
        {
            {0.1f, 0.9f, 0.6f},
            {0.6f, 0.1f, 1.4f},
            {0.9f, 0.6f, 0.1f},
            {1.4f, 0.8f, 0.6f},
        },
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);
    torch::Tensor latent_cpu = torch::tensor(
        {
            {1.0f, 0.0f},
            {0.8f, 0.6f},
            {-0.4f, 0.9f},
            {-0.8f, 0.5f},
        },
        torch::TensorOptions().dtype(torch::kFloat32)).clone().set_requires_grad(true);

    dr::DenseReduceOutput output_cpu{
        reconstruction_cpu,
        latent_cpu,
        latent_cpu
    };
    dr::DenseReduceBatch batch_cpu{
        features,
        developmental_time,
        time_buckets,
        cell_indices
    };
    dr::DenseReduceLoss loss_cpu = dr::compute_dense_reduce_loss(output_cpu, batch_cpu, config);
    loss_cpu.total.backward();
    const torch::Tensor grad_cpu = latent_cpu.grad().detach().clone();

    torch::Tensor reconstruction_cuda = reconstruction_cpu.detach().clone().to(torch::kCUDA).set_requires_grad(true);
    torch::Tensor latent_cuda = latent_cpu.detach().clone().to(torch::kCUDA).set_requires_grad(true);
    dr::DenseReduceOutput output_cuda{
        reconstruction_cuda,
        latent_cuda,
        latent_cuda
    };
    dr::DenseReduceBatch batch_cuda{
        features.to(torch::kCUDA),
        developmental_time.to(torch::kCUDA),
        time_buckets.to(torch::kCUDA),
        cell_indices.to(torch::kCUDA)
    };
    dr::DenseReduceLoss loss_cuda = dr::compute_dense_reduce_loss(output_cuda, batch_cuda, config);
    loss_cuda.total.backward();
    const torch::Tensor grad_cuda = latent_cuda.grad().detach().to(torch::kCPU);

    require(close_scalar(loss_cpu.local_smoothness, loss_cuda.local_smoothness, 1.0e-5, 1.0e-5),
            "dense_reduce local loss mismatch");
    require(close_scalar(loss_cpu.far_separation, loss_cuda.far_separation, 1.0e-5, 1.0e-5),
            "dense_reduce far loss mismatch");
    require(close_scalar(loss_cpu.total, loss_cuda.total, 1.0e-5, 1.0e-5),
            "dense_reduce total loss mismatch");
    require(close_tensor(grad_cpu, grad_cuda, 2.0e-5, 2.0e-5),
            "dense_reduce latent gradient mismatch");
}

void check_weighted_future_target_cuda() {
    const torch::Tensor reference_dense = torch::tensor(
        {
            {1.0f, 10.0f},
            {2.0f, 20.0f},
            {4.0f, 40.0f},
        },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    const torch::Tensor neighbor_rows = torch::tensor(
        {
            {0, 1, -1},
            {2, -1, -1},
        },
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    const torch::Tensor neighbor_weights = torch::tensor(
        {
            {0.25f, 0.75f, 0.0f},
            {1.00f, 0.0f, 0.0f},
        },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    const torch::Tensor target = ops::weighted_future_target(reference_dense, neighbor_rows, neighbor_weights).to(torch::kCPU);
    const torch::Tensor expected = torch::tensor(
        {
            {1.75f, 17.50f},
            {4.00f, 40.00f},
        },
        torch::TensorOptions().dtype(torch::kFloat32));
    require(close_tensor(target, expected, 1.0e-5, 1.0e-5), "weighted future target mismatch");
}

} // namespace

int main() {
    require(torch::cuda::is_available(), "modelCustomOpsTest requires CUDA");

    torch::manual_seed(0);
    check_dense_reduce_cuda_matches_cpu();
    check_weighted_future_target_cuda();
    return 0;
}
