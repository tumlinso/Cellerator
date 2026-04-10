#include "../src/models/developmental_time/developmentalTime.hh"

#include <torch/torch.h>

int main() {
    using namespace cellerator::models::developmental_time;

    SparseTimeEncoderConfig encoder_config;
    encoder_config.input_genes = 4;
    encoder_config.hidden_dim = 8;
    encoder_config.proj_dim = 4;

    DevelopmentalStageHeadConfig head_config;
    head_config.input_dim = encoder_config.proj_dim;
    head_config.hidden_dim = 4;

    DevelopmentalStageModel model(encoder_config, head_config);
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
        torch::tensor({0, 1}, torch::TensorOptions().dtype(torch::kInt64)),
        torch::tensor({10, 11}, torch::TensorOptions().dtype(torch::kInt64))
    };
    DevelopmentalStageLoss loss = compute_developmental_stage_loss(output, batch);
    return (output.stage.numel() == 2 && loss.total.defined()) ? 0 : 1;
}
