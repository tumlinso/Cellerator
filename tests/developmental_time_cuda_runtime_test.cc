#include "../src/models/developmental_time/developmentalTime.hh"
#include "../src/models/developmental_time_cuda/developmentalTimeCuda.hh"

#include <torch/torch.h>

#include <stdexcept>

namespace dt = ::cellerator::models::developmental_time;
namespace dtc = ::cellerator::models::developmental_time_cuda;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    require(torch::cuda::is_available(), "developmentalTimeCudaRuntimeTest requires CUDA");

    torch::Tensor crow = torch::tensor({0, 2, 4, 5}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor col = torch::tensor({0, 3, 1, 2, 0}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor values = torch::tensor({1.0f, 2.0f, 3.0f, 1.5f, 0.5f}, torch::TensorOptions().dtype(torch::kFloat16));
    torch::Tensor features = torch::sparse_csr_tensor(
        crow,
        col,
        values,
        {3, 4},
        torch::TensorOptions().dtype(torch::kFloat16)).to(torch::kCUDA);

    dt::TimeBatch batch{
        features,
        torch::tensor({0.1f, 0.5f, 0.9f}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)),
        torch::tensor({0, 1, 2}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)),
        torch::tensor({10, 11, 12}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA))
    };

    dtc::DevelopmentalTimeCudaConfig config;
    config.input_genes = 4;
    config.stem_dim = 8;
    config.hidden_dim = 6;
    config.num_time_bins = 3;
    dtc::DevelopmentalTimeCudaModel model(config);

    dtc::DevelopmentalTimeCudaOutput output = dtc::forward(model, batch.features);
    require(output.predicted_time.numel() == 3, "predicted_time size mismatch");
    require(output.time_bin_logits.size(0) == 3, "time_bin_logits rows mismatch");
    require(output.time_bin_logits.size(1) == 3, "time_bin_logits cols mismatch");

    dtc::DevelopmentalTimeCudaLoss loss = dtc::compute_loss(output, batch);
    require(loss.total.defined(), "loss total must be defined");

    dtc::DevelopmentalTimeCudaTrainStep step = dtc::train_step(model, batch);
    require(step.output.predicted_time.numel() == 3, "train_step predicted_time size mismatch");
    require(step.loss.total.defined(), "train_step loss total must be defined");
    return 0;
}
