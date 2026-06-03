#include <Cellerator/models/dense_reduce.hh>

#include <torch/torch.h>

int main() {
    using namespace cellerator::models::dense_reduce;

    SparseDenseReduceConfig model_config;
    model_config.input_genes = 5;
    model_config.hidden_dim = 8;
    model_config.bottleneck_dim = 6;
    model_config.latent_dim = 3;
    model_config.dropout = 0.0;
    model_config.corruption_rate = 0.1;

    DenseReduceModel model(model_config);

    torch::Tensor crow = torch::tensor({0, 2, 4, 6}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor col = torch::tensor({0, 3, 1, 4, 0, 2}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor values = torch::tensor(
        {1.0f, 0.5f, 2.0f, 1.5f, 1.25f, 0.75f},
        torch::TensorOptions().dtype(torch::kFloat16));
    torch::Tensor features = torch::sparse_csr_tensor(
        crow,
        col,
        values,
        {3, 5},
        torch::TensorOptions().dtype(torch::kFloat16));

    DenseReduceBatch batch{
        features,
        torch::tensor({0.1f, 0.14f, 0.9f}, torch::TensorOptions().dtype(torch::kFloat32)),
        torch::tensor({0, 0, 1}, torch::TensorOptions().dtype(torch::kInt64)),
        torch::tensor({10, 11, 12}, torch::TensorOptions().dtype(torch::kInt64))
    };

    DenseReduceLossConfig loss_config;
    loss_config.local_time_window = 0.1;
    loss_config.far_time_window = 0.4;
    loss_config.max_sampled_pairs = 8;

    DenseReduceOutput output = model->forward(features);
    torch::Tensor encoded = model->encode(features);
    DenseReduceLoss loss = compute_dense_reduce_loss(output, batch, loss_config);

    DenseReduceTrainConfig train_config;
    train_config.clip_gradients = false;
    torch::optim::AdamW optimizer = make_dense_reduce_optimizer(model, train_config);
    DenseReduceTrainStep train_step = train_dense_reduce_step(model, optimizer, batch, loss_config, train_config);
    DenseReduceTrainStep eval_step = evaluate_dense_reduce_step(model, batch, loss_config);

    const bool ok = output.reconstruction.sizes() == torch::IntArrayRef({3, 5})
        && output.latent_raw.sizes() == torch::IntArrayRef({3, 3})
        && encoded.sizes() == torch::IntArrayRef({3, 3})
        && loss.total.defined()
        && torch::isfinite(train_step.loss.total).item<bool>()
        && torch::isfinite(eval_step.loss.total).item<bool>();
    return ok ? 0 : 1;
}
