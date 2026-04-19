#include <Cellerator/models/developmental_time.hh>

#include <torch/torch.h>

int main() {
    using namespace cellerator::models::developmental_time;

    SparseTimeEncoderConfig encoder_config;
    encoder_config.input_genes = 4;
    encoder_config.hidden_dim = 8;
    encoder_config.proj_dim = 4;

    DevelopmentalTimeHeadConfig head_config;
    head_config.input_dim = encoder_config.proj_dim;
    head_config.hidden_dim = 4;
    head_config.num_time_bins = 3;

    DevelopmentalTimeModel model(encoder_config, head_config);
    torch::Tensor crow = torch::tensor({0, 2, 3}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor col = torch::tensor({0, 3, 1}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor values = torch::tensor({1.0f, 2.0f, 3.0f}, torch::TensorOptions().dtype(torch::kFloat16));
    torch::Tensor features = torch::sparse_csr_tensor(
        crow,
        col,
        values,
        {2, 4},
        torch::TensorOptions().dtype(torch::kFloat16));

    DevelopmentalStageOutput output = model->forward(features);
    TimeBatch batch{
        features,
        torch::tensor({3.0f, 5.0f}, torch::TensorOptions().dtype(torch::kFloat32)),
        torch::tensor({0, 2}, torch::TensorOptions().dtype(torch::kInt64)),
        torch::tensor({10, 11}, torch::TensorOptions().dtype(torch::kInt64))
    };
    DevelopmentalTimeLoss loss = compute_developmental_time_loss(output, batch);
    return (output.predicted_time.numel() == 2 && output.time_bin_logits.size(1) == 3 && loss.total.defined()) ? 0 : 1;
}
