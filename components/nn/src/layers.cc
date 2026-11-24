#include <Cellerator/nn/layers.hh>

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/nn/init.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace cellerator::nn {
namespace {

void require_positive(std::int64_t value, const char* name) {
    if (value <= 0)
        throw std::invalid_argument(std::string(name) + " must be positive");
}

void validate_linear_parameters(const at::Tensor& input,
                                const LinearParameters& parameters) {
    if (!input.defined()) throw std::invalid_argument("linear input is undefined");
    if (input.dim() < 1)
        throw std::invalid_argument("linear input must have at least one dimension");
    if (!parameters.weight.defined() || parameters.weight.dim() != 2)
        throw std::invalid_argument("linear weight must be a defined rank-two tensor");
    if (input.size(-1) != parameters.weight.size(1))
        throw std::invalid_argument("linear input width does not match weight");
    if (input.device() != parameters.weight.device())
        throw std::invalid_argument("linear input and weight must share a device");
    if (input.scalar_type() != parameters.weight.scalar_type())
        throw std::invalid_argument("linear input and weight must share a dtype");
    if (parameters.bias.defined()) {
        if (parameters.bias.dim() != 1
            || parameters.bias.size(0) != parameters.weight.size(0))
            throw std::invalid_argument("linear bias must match the output width");
        if (parameters.bias.device() != parameters.weight.device())
            throw std::invalid_argument("linear bias and weight must share a device");
        if (parameters.bias.scalar_type() != parameters.weight.scalar_type())
            throw std::invalid_argument("linear bias and weight must share a dtype");
    }
}

} // namespace

LinearParameters::LinearParameters(ParameterGroup group,
                                   std::int64_t input_features,
                                   std::int64_t output_features,
                                   bool with_bias)
    : group_(group) {
    require_positive(input_features, "input_features");
    require_positive(output_features, "output_features");

    weight_id = group.empty({output_features, input_features});
    weight = group.get(weight_id);
    {
        at::NoGradGuard no_grad;
        // This matches the conventional Torch Linear reset bound while using
        // the library's shared Torch initialization implementation.
        torch::nn::init::kaiming_uniform_(weight, std::sqrt(5.0));
    }

    if (with_bias) {
        bias_id = group.empty({output_features});
        bias = group.get(bias_id);
        const double bound = 1.0 / std::sqrt(static_cast<double>(input_features));
        at::NoGradGuard no_grad;
        torch::nn::init::uniform_(bias, -bound, bound);
    }
}

LinearParameters::LinearParameters(at::Tensor borrowed_weight,
                                   at::Tensor borrowed_bias)
    : weight(std::move(borrowed_weight)), bias(std::move(borrowed_bias)) {}

LinearParameters::LinearParameters(ParameterGroup group, ParameterId weight_slot,
                                   std::optional<ParameterId> bias_slot)
    : weight_id(weight_slot), group_(group) {
    weight = group.get(weight_slot);
    if (bias_slot) {
        bias_id = *bias_slot;
        bias = group.get(*bias_slot);
    }
}

at::Tensor linear(const at::Tensor& input,
                  const LinearParameters& parameters) {
    validate_linear_parameters(input, parameters);
    std::optional<at::Tensor> bias;
    if (parameters.bias.defined()) bias = parameters.bias;
    return at::linear(input, parameters.weight, bias);
}

Linear::Linear(ParameterGroup group, std::int64_t input_features,
               std::int64_t output_features, bool with_bias)
    : group_(group),
      parameters_(group, input_features, output_features, with_bias) {}

Linear::Linear(ParameterGroup group, LinearParameters parameters)
    : group_(group), parameters_(std::move(parameters)) {}

at::Tensor Linear::operator()(const at::Tensor& input, Run) const {
    return linear(input, parameters_);
}

Embedding::Embedding(ParameterGroup group, std::int64_t count,
                     std::int64_t width)
    : group_(group) {
    require_positive(count, "count");
    require_positive(width, "width");
    weight_id_ = group_.normal({count, width}, 1.0);
    weight_ = group_.get(weight_id_);
}

at::Tensor Embedding::operator()(const at::Tensor& ids, Run) const {
    if (!ids.defined()) throw std::invalid_argument("embedding IDs are undefined");
    if (!(ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong))
        throw std::invalid_argument("embedding IDs must have dtype int32 or int64");
    if (ids.device() != weight_.device())
        throw std::invalid_argument("embedding IDs and weight must share a device");
    return at::embedding(weight_, ids.to(at::kLong), -1, false, false);
}

} // namespace cellerator::nn
