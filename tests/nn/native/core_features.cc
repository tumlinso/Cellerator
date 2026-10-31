#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/nn/features.hh>

#include <ATen/ATen.h>

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace cellerator::nn;

int main() {
    Parameters owner(at::Device(at::kCPU), at::kFloat);
    auto group = owner.group();
    FeatureEncoder encoder(group, FeatureEncoderOptions{8, 6, 1, std::nullopt});

    auto ids = at::tensor(std::vector<std::int64_t>{1, 3, 5},
        at::TensorOptions().dtype(at::kLong)).reshape({1, 3});
    auto observed = at::tensor(std::vector<std::int64_t>{1, 0, 1},
        at::TensorOptions().dtype(at::kLong)).to(at::kBool).reshape({1, 3});
    auto first_values = at::tensor(std::vector<float>{2.0f,
        std::numeric_limits<float>::quiet_NaN(), -1.0f}).reshape({1, 3});
    FeatureInput first{ids, first_values, observed, std::nullopt};
    auto first_encoded = encoder(first);

    auto second_values = at::tensor(std::vector<float>{2.0f, 1.0e30f, -1.0f})
        .reshape({1, 3});
    FeatureInput second{ids, second_values, observed, std::nullopt};
    auto second_encoded = encoder(second);
    assert(at::allclose(first_encoded, second_encoded));

    auto grad_values = at::tensor(std::vector<float>{2.0f,
        std::numeric_limits<float>::quiet_NaN(), -1.0f},
        at::TensorOptions().dtype(at::kFloat)).reshape({1, 3});
    grad_values.requires_grad_(true);
    FeatureInput grad_input{ids, grad_values, observed, std::nullopt};
    auto grad_result = encoder(grad_input);
    grad_result.square().sum().backward();
    assert(grad_values.grad().defined());
    assert(grad_values.grad()[0][1].item<float>() == 0.0f);
    assert(!at::isnan(grad_values.grad()).any().item<bool>());
    assert(encoder.parameters().feature_embedding.grad().defined());
    assert(encoder.parameters().value_projection.grad().defined());

    auto zero = at::zeros({1, 1});
    auto one_id = at::tensor(std::vector<std::int64_t>{2},
        at::TensorOptions().dtype(at::kLong)).reshape({1, 1});
    auto measured = encoder(FeatureInput{
        one_id, zero, at::ones_like(one_id, at::TensorOptions().dtype(at::kBool)),
        std::nullopt});
    auto missing = encoder(FeatureInput{
        one_id, zero, at::zeros_like(one_id, at::TensorOptions().dtype(at::kBool)),
        std::nullopt});
    assert(!at::equal(measured, missing));

    FeatureEncoderOptions multi_options{6, 5, 2, 3};
    FeatureEncoder multi(group, multi_options);
    auto multi_ids = at::tensor(std::vector<std::int32_t>{4, 1},
        at::TensorOptions().dtype(at::kInt));
    auto values = at::tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f})
        .reshape({2, 2});
    values.requires_grad_(true);
    auto attributes = at::tensor(std::vector<float>{0.2f, 0.4f, 0.6f,
        0.1f, 0.3f, 0.5f}).reshape({2, 3});
    attributes.requires_grad_(true);
    auto multi_result = multi(FeatureInput{multi_ids, values, std::nullopt, attributes});
    assert(multi_result.sizes().vec() == (std::vector<std::int64_t>{2, 5}));
    multi_result.sum().backward();
    assert(values.grad().defined() && attributes.grad().defined());
    assert(multi.parameters().attribute_projection.grad().defined());

    FeatureEncoder identity_only(group, FeatureEncoderOptions{6, 4, std::nullopt,
                                                              std::nullopt});
    assert(identity_only(FeatureInput{multi_ids, std::nullopt, std::nullopt,
                                      std::nullopt}).sizes().vec()
           == (std::vector<std::int64_t>{2, 4}));

    bool rejected = false;
    try {
        (void)identity_only(FeatureInput{
            multi_ids, at::ones({2}), std::nullopt, std::nullopt});
    } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);

    rejected = false;
    try {
        (void)encoder(FeatureInput{
            ids, first_values, at::ones({1, 2}, at::TensorOptions().dtype(at::kBool)),
            std::nullopt});
    } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    return 0;
}
