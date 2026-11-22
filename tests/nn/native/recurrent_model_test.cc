#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/nn/layers.hh>

#include <ATen/ATen.h>

#include <cassert>

namespace nn = cellerator::nn;

struct RecurrentState {
    at::Tensor value;
};

struct RecurrentModel {
    nn::Linear transition;

    at::Tensor operator()(RecurrentState& state, const at::Tensor& input) const {
        state.value = transition(state.value) + input;
        return state.value;
    }
};

int main() {
    const auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCPU);
    auto initial = at::full({1}, 2.0, options).set_requires_grad(true);
    auto weight = at::full({1, 1}, 3.0, options).set_requires_grad(true);
    const auto first_input = at::full({1}, 5.0, options);
    const auto second_input = at::full({1}, 7.0, options);

    nn::Parameters parameters(at::Device(at::kCPU), at::kDouble);
    auto group = parameters.group();
    const auto weight_id = group.add(weight);
    const RecurrentModel model{nn::Linear(group, nn::LinearParameters(group, weight_id))};
    RecurrentState state{initial};

    const auto first = model(state, first_input);
    const auto retained_state = state.value;
    assert(first.item<double>() == 11.0);
    assert(retained_state.data_ptr() == first.data_ptr());
    assert(state.value.data_ptr() != initial.data_ptr());

    const auto second = model(state, second_input);
    assert(second.item<double>() == 40.0);
    assert(state.value.data_ptr() == second.data_ptr());
    assert(second.data_ptr() != first.data_ptr());
    assert(first.item<double>() == 11.0);
    assert(retained_state.item<double>() == 11.0);

    (first + second).sum().backward();
    assert(parameters.get(weight_id).grad().item<double>() == 19.0);
    assert(initial.grad().item<double>() == 12.0);
}
