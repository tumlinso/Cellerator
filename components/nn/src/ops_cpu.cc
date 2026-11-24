#include <Cellerator/nn/ops.hh>
#include <Cellerator/compute/operation/product2/product2.hh>

#include <torch/csrc/autograd/custom_function.h>

#include <cstdint>
#include <memory>
#include <string>

namespace cellerator::nn::ops {
namespace {

namespace product2 = ::cellerator::compute::product2;

void validate_product2_inputs(const at::Tensor& x,
                              const at::Tensor& coefficient,
                              const at::Tensor& lhs,
                              const at::Tensor& rhs) {
    TORCH_CHECK(x.defined() && coefficient.defined()
                    && lhs.defined() && rhs.defined(),
                "Product2 inputs must be defined");
    TORCH_CHECK(x.device().is_cpu() && coefficient.device().is_cpu()
                    && lhs.device().is_cpu() && rhs.device().is_cpu(),
                "product2_cpu requires CPU tensors");
    TORCH_CHECK(x.scalar_type() == at::kFloat
                    && coefficient.scalar_type() == at::kFloat,
                "product2_cpu requires float32 values");
    TORCH_CHECK(lhs.scalar_type() == at::kLong
                    && rhs.scalar_type() == at::kLong,
                "Product2 indices must use int64");
    TORCH_CHECK(x.layout() == at::kStrided
                    && coefficient.layout() == at::kStrided
                    && lhs.layout() == at::kStrided
                    && rhs.layout() == at::kStrided,
                "Product2 requires strided tensors");
    TORCH_CHECK(x.dim() == 1 && coefficient.dim() == 1
                    && lhs.dim() == 1 && rhs.dim() == 1,
                "Product2 expects one-dimensional values and indices");
    TORCH_CHECK(coefficient.numel() == lhs.numel()
                    && lhs.numel() == rhs.numel(),
                "Product2 coefficient and index extents must match");
    TORCH_CHECK(!lhs.requires_grad() && !rhs.requires_grad(),
                "Product2 indices cannot require gradients");
}

void set_generations(ce_product2_binding& binding) {
    binding.expected_structure_generation = 0;
    binding.current_structure_generation = 0;
    binding.expected_value_generation = 0;
    binding.current_value_generation = 0;
    binding.expected_parameter_generation = 0;
    binding.current_parameter_generation = 0;
}

std::unique_ptr<product2::prepared_owner> make_owner(
    const at::Tensor& x, const at::Tensor& coefficient,
    const at::Tensor& lhs, const at::Tensor& rhs) {
    // prepared_owner copies the topology during construction, so its lifetime
    // is independent of these local contiguous index copies.
    try {
        return std::make_unique<product2::prepared_owner>(product2::descriptor{
            static_cast<std::uint64_t>(x.numel()),
            static_cast<std::uint64_t>(coefficient.numel()),
            0,
            lhs.data_ptr<std::int64_t>(),
            rhs.data_ptr<std::int64_t>(),
            static_cast<std::uint64_t>(lhs.numel()),
            static_cast<std::uint64_t>(rhs.numel())});
    } catch (const std::exception& error) {
        TORCH_CHECK(false, "invalid Product2 topology: ", error.what());
    }
}

class Product2CPUAutograd final
    : public torch::autograd::Function<Product2CPUAutograd> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* context,
                              at::Tensor x,
                              at::Tensor coefficient,
                              at::Tensor lhs,
                              at::Tensor rhs) {
        validate_product2_inputs(x, coefficient, lhs, rhs);

        // Keep the caller's original tensors in the autograd context. This
        // lets get_saved_variables() enforce version checks even when the
        // native call uses one of the contiguous copies below.
        context->save_for_backward({x, coefficient, lhs, rhs});

        auto x_contiguous = x.contiguous();
        auto coefficient_contiguous = coefficient.contiguous();
        auto lhs_contiguous = lhs.contiguous();
        auto rhs_contiguous = rhs.contiguous();
        auto output = at::empty(
            {coefficient.numel()}, x.options().requires_grad(false));

        auto owner = make_owner(x_contiguous, coefficient_contiguous,
                                lhs_contiguous, rhs_contiguous);
        ce_product2_binding binding{};
        binding.x = x_contiguous.data_ptr<float>();
        binding.x_count = static_cast<std::uint64_t>(x_contiguous.numel());
        binding.k = coefficient_contiguous.data_ptr<float>();
        binding.k_count = static_cast<std::uint64_t>(coefficient_contiguous.numel());
        binding.y = output.data_ptr<float>();
        binding.y_count = static_cast<std::uint64_t>(output.numel());
        set_generations(binding);

        const auto status = owner->execute(binding, product2::operation::forward);
        TORCH_CHECK(status == CE_PRODUCT2_SUCCESS,
                    "native Product2 CPU forward failed with status ",
                    static_cast<int>(status));
        return output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* context,
        torch::autograd::variable_list output_gradients) {
        TORCH_CHECK(!torch::autograd::GradMode::is_enabled(),
                    "product2_cpu supports first-order gradients only");
        TORCH_CHECK(output_gradients.size() == 1
                        && output_gradients.front().defined(),
                    "Product2 expects one defined output gradient");

        const auto saved = context->get_saved_variables();
        TORCH_CHECK(saved.size() == 4,
                    "Product2 autograd context is incomplete");
        const auto& x = saved[0];
        const auto& coefficient = saved[1];
        const auto& lhs = saved[2];
        const auto& rhs = saved[3];
        validate_product2_inputs(x, coefficient, lhs, rhs);

        auto x_contiguous = x.contiguous();
        auto coefficient_contiguous = coefficient.contiguous();
        auto lhs_contiguous = lhs.contiguous();
        auto rhs_contiguous = rhs.contiguous();
        auto gradient_contiguous = output_gradients.front().contiguous();
        TORCH_CHECK(gradient_contiguous.device().is_cpu()
                        && gradient_contiguous.scalar_type() == at::kFloat
                        && gradient_contiguous.dim() == 1
                        && gradient_contiguous.numel() == coefficient.numel(),
                    "Product2 output gradient must be a matching CPU float32 vector");

        auto input_gradient = at::empty(
            {x.numel()}, x.options().requires_grad(false));
        auto coefficient_gradient = at::empty(
            {coefficient.numel()}, coefficient.options().requires_grad(false));
        auto owner = make_owner(x_contiguous, coefficient_contiguous,
                                lhs_contiguous, rhs_contiguous);

        ce_product2_binding binding{};
        binding.x = x_contiguous.data_ptr<float>();
        binding.x_count = static_cast<std::uint64_t>(x_contiguous.numel());
        binding.k = coefficient_contiguous.data_ptr<float>();
        binding.k_count = static_cast<std::uint64_t>(coefficient_contiguous.numel());
        binding.g = gradient_contiguous.data_ptr<float>();
        binding.g_count = static_cast<std::uint64_t>(gradient_contiguous.numel());
        binding.gx = input_gradient.data_ptr<float>();
        binding.gx_count = static_cast<std::uint64_t>(input_gradient.numel());
        binding.gk = coefficient_gradient.data_ptr<float>();
        binding.gk_count = static_cast<std::uint64_t>(coefficient_gradient.numel());
        set_generations(binding);

        const auto status = owner->execute(binding, product2::operation::vjp);
        TORCH_CHECK(status == CE_PRODUCT2_SUCCESS,
                    "native Product2 CPU VJP failed with status ",
                    static_cast<int>(status));
        return {input_gradient, coefficient_gradient, at::Tensor(), at::Tensor()};
    }
};

}  // namespace

at::Tensor product2_cpu(const at::Tensor& x, const at::Tensor& coefficient,
                        const at::Tensor& lhs, const at::Tensor& rhs) {
    return Product2CPUAutograd::apply(x, coefficient, lhs, rhs);
}

}  // namespace cellerator::nn::ops
