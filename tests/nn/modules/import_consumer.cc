#ifdef NDEBUG
#undef NDEBUG
#endif
#include <torch/torch.h>

import cellerator.nn;

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nn = cellerator::nn;
namespace bio = cellerator::biology;

std::size_t header_owner_slots(const nn::Parameters& owner);
bool header_tensor_defined(const at::Tensor& value);
bool module_only_surface_works();

#ifdef CELLERATOR_NN_MODULES_TEST_CUDA
std::size_t cuda_header_owner_slots(const nn::Parameters& owner);
at::Device cuda_header_parameter_device(const nn::Parameters& owner);
bool cuda_header_tensor_defined(const at::Tensor& value);
#endif

// A complete user-defined native model remains an ordinary C++ type.
struct SmallFeatureModel {
    nn::Linear hidden;
    nn::Linear readout;

    at::Tensor operator()(const at::Tensor& input, nn::Run run = {}) const {
        return readout(at::relu(hidden(input, run)), run);
    }
};

int main() {
    assert(module_only_surface_works());

    nn::Parameters parameters(
        at::Device(at::DeviceType::CPU), at::ScalarType::Float);
    auto group = parameters.group();

    const auto genes = bio::FeatureBinding<std::string>::from_ids(
        {"gene-A", "gene-B", "gene-C"}, {2, 0, 1});
    const auto observations = bio::OccurrenceCoordinates<std::string>::from_ids(
        {"gene-B", "gene-A", "gene-C", "gene-B"}, {0, 4});
    const auto aligned = genes.map_occurrences(observations);
    assert((aligned.model_indices() == std::vector<std::int64_t>{0, 2, 1, 0}));

    nn::FeatureEncoder encoder(
        group, nn::FeatureEncoderOptions{3, 3, std::nullopt, std::nullopt});
    auto ids = at::tensor(aligned.model_indices(),
        at::TensorOptions().dtype(at::ScalarType::Long));
    nn::FeatureInput feature_input{ids, std::nullopt, std::nullopt, std::nullopt};
    auto encoded = encoder(feature_input);

    SmallFeatureModel model{
        nn::Linear(group, 3, 4),
        nn::Linear(group, 4, 2)};
    auto targets = at::tensor(std::vector<float>{
        0.2f, 0.8f, 0.4f, 0.6f, 0.7f, 0.1f, 0.9f, 0.3f})
        .reshape({4, 2});

    torch::optim::SGD optimizer(
        parameters.torch_parameters(), torch::optim::SGDOptions(0.05));
    optimizer.zero_grad();
    const auto before = model.hidden.weight().clone();
    auto predictions = model(encoded, nn::Run{true});
    auto loss = at::mse_loss(predictions, targets);
    loss.backward();
    assert(model.hidden.weight().grad().defined());
    assert(header_owner_slots(parameters) == parameters.size());
    assert(header_tensor_defined(predictions));
    optimizer.step();
    assert(!at::equal(before, model.hidden.weight()));

#ifdef CELLERATOR_NN_MODULES_TEST_CUDA
    assert(cuda_header_owner_slots(parameters) == parameters.size());
    assert(cuda_header_parameter_device(parameters).type() == at::DeviceType::CPU);
    assert(cuda_header_tensor_defined(predictions));
#endif

#ifdef CELLERATOR_NN_HAS_CUDA
    // Resolve an existing CUDA NN entry point through the installed exported
    // target without calling it. This checks linkage while requiring no GPU.
    using CudaOperation = decltype(&nn::ops::grouped_linear_cuda);
    volatile CudaOperation cuda_operation = &nn::ops::grouped_linear_cuda;
    if (cuda_operation == nullptr) {
        return 2;
    }
#endif

    return 0;
}
