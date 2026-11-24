#include <Cellerator/nn/ops.hh>

#include <torch/torch.h>
#include <torch/csrc/autograd/autograd.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace nn_ops = cellerator::nn::ops;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void near(const at::Tensor& actual, const at::Tensor& expected,
          const char* message) {
    require(actual.sizes() == expected.sizes(), message);
    require(actual.scalar_type() == at::kFloat
                && expected.scalar_type() == at::kFloat,
            message);
    if (actual.numel() == 0) return;
    const auto error = (actual - expected).abs().max().item<float>();
    if (!(error <= 1e-6f)) throw std::runtime_error(message);
}

at::Tensor noncontiguous_float(const std::initializer_list<float>& values) {
    const auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    auto result = at::empty_strided(
        {static_cast<std::int64_t>(values.size())}, {2}, options);
    {
        at::NoGradGuard no_grad;
        result.copy_(at::tensor(values, at::TensorOptions().dtype(at::kFloat)));
    }
    return result.set_requires_grad(true);
}

at::Tensor noncontiguous_indices(
    const std::initializer_list<std::int64_t>& values) {
    auto result = at::empty_strided(
        {static_cast<std::int64_t>(values.size())}, {2},
        at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    result.copy_(at::tensor(values, at::TensorOptions().dtype(at::kLong)));
    return result;
}

template<class F>
void expect_torch_error(F&& f, const char* message) {
    bool rejected = false;
    try {
        f();
    } catch (const c10::Error&) {
        rejected = true;
    }
    require(rejected, message);
}

void compare_forward_and_vjp() {
    auto x = noncontiguous_float({0.0f, 2.0f, 3.0f});
    auto coefficient = noncontiguous_float({2.0f, 3.0f, 4.0f, 5.0f});
    auto lhs = noncontiguous_indices({0, 0, 1, 2});
    auto rhs = noncontiguous_indices({0, 1, 0, 2});

    require(!x.is_contiguous() && !coefficient.is_contiguous()
                && !lhs.is_contiguous() && !rhs.is_contiguous(),
            "fixture tensors must exercise contiguous-copy handling");
    const auto output = nn_ops::product2_cpu(x, coefficient, lhs, rhs);
    const auto reference = coefficient * x.index_select(0, lhs)
                           * x.index_select(0, rhs);
    require(output.requires_grad(), "Product2 output must attach to differentiable inputs");
    require(reference.requires_grad(), "ATen reference must track differentiable inputs");
    near(output, reference, "Product2 forward differs from ATen composition");
    near(output, at::tensor({0.0f, 0.0f, 0.0f, 45.0f}),
         "Product2 forward fixture changed");

    const auto native_grads = torch::autograd::grad(
        {output.sum()}, {x, coefficient});
    const auto reference_grads = torch::autograd::grad(
        {reference.sum()}, {x, coefficient});
    require(native_grads.size() == 2 && reference_grads.size() == 2,
            "Product2 must return gradients for both differentiable inputs");
    near(native_grads[0], reference_grads[0],
         "Product2 input VJP differs from ATen composition");
    near(native_grads[1], reference_grads[1],
         "Product2 coefficient VJP differs from ATen composition");
    near(native_grads[0], at::tensor({14.0f, 0.0f, 30.0f}),
         "Product2 repeated-index accumulation changed");
    near(native_grads[1], at::tensor({0.0f, 0.0f, 0.0f, 9.0f}),
         "Product2 coefficient gradient changed");
}

void supports_empty_products() {
    auto x = at::tensor({1.0f, 2.0f}).set_requires_grad(true);
    auto coefficient = at::empty(
        {0}, at::TensorOptions().dtype(at::kFloat)).set_requires_grad(true);
    auto lhs = at::empty({0}, at::TensorOptions().dtype(at::kLong));
    auto rhs = at::empty({0}, at::TensorOptions().dtype(at::kLong));
    const auto output = nn_ops::product2_cpu(x, coefficient, lhs, rhs);
    require(output.numel() == 0, "empty Product2 must return an empty vector");
    const auto gradients = torch::autograd::grad({output.sum()}, {x, coefficient});
    near(gradients[0], at::zeros_like(x), "empty Product2 input VJP must be zero");
    near(gradients[1], at::zeros_like(coefficient),
         "empty Product2 coefficient VJP must be zero");
}

void saved_originals_enforce_versions() {
    {
        auto x = noncontiguous_float({1.0f, 2.0f, 3.0f});
        auto coefficient = at::tensor({2.0f},
            at::TensorOptions().dtype(at::kFloat)).set_requires_grad(true);
        auto lhs = at::tensor({0}, at::TensorOptions().dtype(at::kLong));
        auto rhs = at::tensor({1}, at::TensorOptions().dtype(at::kLong));
        const auto output = nn_ops::product2_cpu(x, coefficient, lhs, rhs);
        {
            at::NoGradGuard no_grad;
            x.add_(1.0f);
        }
        expect_torch_error([&] { output.sum().backward(); },
                           "Product2 must version-check the original noncontiguous input");
    }

    {
        auto x = at::tensor({1.0f, 2.0f},
            at::TensorOptions().dtype(at::kFloat)).set_requires_grad(true);
        auto coefficient = noncontiguous_float({2.0f});
        auto lhs = at::tensor({0}, at::TensorOptions().dtype(at::kLong));
        auto rhs = at::tensor({1}, at::TensorOptions().dtype(at::kLong));
        const auto output = nn_ops::product2_cpu(x, coefficient, lhs, rhs);
        {
            at::NoGradGuard no_grad;
            coefficient.mul_(2.0f);
        }
        expect_torch_error([&] { output.sum().backward(); },
                           "Product2 must version-check the original coefficient");
    }

    {
        auto x = at::tensor({1.0f, 2.0f},
            at::TensorOptions().dtype(at::kFloat)).set_requires_grad(true);
        auto coefficient = at::tensor({2.0f},
            at::TensorOptions().dtype(at::kFloat)).set_requires_grad(true);
        auto lhs = noncontiguous_indices({0});
        auto rhs = at::tensor({1}, at::TensorOptions().dtype(at::kLong));
        const auto output = nn_ops::product2_cpu(x, coefficient, lhs, rhs);
        {
            at::NoGradGuard no_grad;
            lhs.add_(1);
        }
        expect_torch_error([&] { output.sum().backward(); },
                           "Product2 must version-check original index tensors");
    }
}

void rejects_higher_order_gradients() {
    auto x = at::tensor({1.0f, 2.0f},
        at::TensorOptions().dtype(at::kFloat)).set_requires_grad(true);
    auto coefficient = at::tensor({2.0f},
        at::TensorOptions().dtype(at::kFloat)).set_requires_grad(true);
    auto lhs = at::tensor({0}, at::TensorOptions().dtype(at::kLong));
    auto rhs = at::tensor({1}, at::TensorOptions().dtype(at::kLong));
    const auto output = nn_ops::product2_cpu(x, coefficient, lhs, rhs);
    expect_torch_error([&] {
        (void)torch::autograd::grad({output.sum()}, {x}, {}, true, true);
    }, "Product2 must reject create_graph and higher-order differentiation");
}

void rejects_invalid_contracts() {
    auto x = at::tensor({1.0, 2.0}, at::TensorOptions().dtype(at::kDouble));
    auto coefficient = at::tensor({2.0f}, at::TensorOptions().dtype(at::kFloat));
    auto lhs = at::tensor({0}, at::TensorOptions().dtype(at::kLong));
    auto rhs = at::tensor({1}, at::TensorOptions().dtype(at::kLong));
    expect_torch_error([&] {
        (void)nn_ops::product2_cpu(x, coefficient, lhs, rhs);
    }, "Product2 must reject non-FP32 values");

    auto float_x = at::tensor({1.0f, 2.0f}, at::TensorOptions().dtype(at::kFloat));
    auto bad_rhs = at::tensor({2}, at::TensorOptions().dtype(at::kLong));
    expect_torch_error([&] {
        (void)nn_ops::product2_cpu(float_x, coefficient, lhs, bad_rhs);
    }, "Product2 must reject indices outside the input extent");
}

}  // namespace

int main() {
    try {
        compare_forward_and_vjp();
        supports_empty_products();
        saved_originals_enforce_versions();
        rejects_higher_order_gradients();
        rejects_invalid_contracts();
        std::cout << "native Product2 CPU autograd forward/VJP, noncontiguous "
                     "versions, repeated/zero indices, empty products and "
                     "first-order boundary PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "native Product2 CPU autograd test FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
