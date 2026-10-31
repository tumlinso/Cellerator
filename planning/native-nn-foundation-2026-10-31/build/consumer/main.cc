#include <Cellerator/nn.hh>

#include <torch/optim/sgd.h>

#include <iostream>

int main() {
    namespace nn = cellerator::nn;

    nn::Parameters parameters(at::Device(at::kCPU), at::kFloat);
    auto group = parameters.group();
    nn::Linear layer(group, 2, 1);

    const auto input = at::ones({4, 2},
        at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    const auto loss = layer(input).square().mean();
    loss.backward();

    auto optimizer_parameters = parameters.torch_parameters();
    if (optimizer_parameters.size() != 2 || !layer.weight().grad().defined()) {
        return 1;
    }

    const auto previous_weight = layer.weight().detach().clone();
    torch::optim::SGD optimizer(
        optimizer_parameters, torch::optim::SGDOptions(1e-2));
    optimizer.step();
    if (at::allclose(previous_weight, layer.weight())) {
        return 2;
    }

    std::cout << "installed Cellerator NN consumer passed\n";
    return 0;
}
