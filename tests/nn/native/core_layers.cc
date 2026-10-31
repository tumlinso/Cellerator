#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/nn/layers.hh>

#include <ATen/ATen.h>

#include <cassert>
#include <stdexcept>
#include <vector>

using namespace cellerator::nn;

int main() {
    Parameters owner(at::Device(at::kCPU), at::kFloat);
    auto group = owner.group();
    Linear projection(group, 3, 2);

    auto input = at::tensor(std::vector<float>{0.5f, -1.0f, 2.0f},
        at::TensorOptions().dtype(at::kFloat)).reshape({1, 3});
    input.requires_grad_(true);
    auto output = projection(input);
    assert(output.sizes().vec() == (std::vector<std::int64_t>{1, 2}));
    output.square().sum().backward();
    assert(input.grad().defined());
    assert(projection.weight().grad().defined());
    assert(projection.bias().grad().defined());
    assert(owner.get(projection.parameters().weight_id).unsafeGetTensorImpl()
           == projection.weight().unsafeGetTensorImpl());
    assert(owner.torch_parameters().size() == 2);

    Linear no_bias(group, 3, 4, false);
    assert(!no_bias.bias().defined());
    assert(no_bias(input, Run{true}).sizes().vec()
           == (std::vector<std::int64_t>{1, 4}));

    // A tied weight handle and the functional API both use the exact owner leaf.
    const auto tied_weight = group.tie(projection.parameters().weight_id);
    LinearParameters tied(group, tied_weight);
    assert(tied.weight.unsafeGetTensorImpl()
           == projection.weight().unsafeGetTensorImpl());
    assert(at::allclose(linear(input, tied),
                        at::linear(input, projection.weight())));
    assert(owner.torch_parameters().size() == 3); // Two layer biases + one shared weight.

    Embedding embedding(group, 7, 5);
    auto ids = at::tensor(std::vector<std::int64_t>{1, 4, 1},
        at::TensorOptions().dtype(at::kLong)).reshape({1, 3});
    auto embedded = embedding(ids);
    assert(embedded.sizes().vec() == (std::vector<std::int64_t>{1, 3, 5}));
    embedded.sum().backward();
    assert(embedding.weight().grad().defined());
    assert(embedding.weight().grad().select(0, 1).sum().item<float>() == 10.0f);

    Identity identity;
    assert(identity(17) == 17);
    assert(identity(output).unsafeGetTensorImpl() == output.unsafeGetTensorImpl());

    bool rejected = false;
    try { (void)projection(at::ones({2, 4})); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    return 0;
}
