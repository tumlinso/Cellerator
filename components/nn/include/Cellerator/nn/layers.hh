#pragma once

#include <Cellerator/nn/parameters.hh>
#include <Cellerator/nn/types.hh>

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <optional>

namespace cellerator::nn {

// Borrowed tensor parameters for the conventional affine operation. Undefined
// bias means that the operation has no bias. The slot IDs describe parameters
// registered through a ParameterGroup; they are invalid for arbitrary borrowed
// tensors constructed directly by a caller.
struct LinearParameters {
    at::Tensor weight;
    at::Tensor bias;
    ParameterId weight_id = invalid_parameter_id;
    ParameterId bias_id = invalid_parameter_id;

    LinearParameters() = default;
    LinearParameters(ParameterGroup group, std::int64_t input_features,
                     std::int64_t output_features, bool with_bias = true);
    LinearParameters(at::Tensor borrowed_weight, at::Tensor borrowed_bias = {});
    LinearParameters(ParameterGroup group, ParameterId weight,
                     std::optional<ParameterId> bias = std::nullopt);

private:
    ParameterGroup group_;
};

at::Tensor linear(const at::Tensor& input, const LinearParameters& parameters);

class Linear {
public:
    Linear(ParameterGroup group, std::int64_t input_features,
           std::int64_t output_features, bool bias = true);
    Linear(ParameterGroup group, LinearParameters parameters);

    at::Tensor operator()(const at::Tensor& input, Run = {}) const;
    const LinearParameters& parameters() const noexcept { return parameters_; }
    const at::Tensor& weight() const noexcept { return parameters_.weight; }
    const at::Tensor& bias() const noexcept { return parameters_.bias; }

private:
    ParameterGroup group_;
    LinearParameters parameters_;
};

class Embedding {
public:
    Embedding(ParameterGroup group, std::int64_t count, std::int64_t width);

    at::Tensor operator()(const at::Tensor& ids, Run = {}) const;
    const at::Tensor& weight() const noexcept { return weight_; }
    ParameterId weight_id() const noexcept { return weight_id_; }

private:
    ParameterGroup group_;
    ParameterId weight_id_ = invalid_parameter_id;
    at::Tensor weight_;
};

} // namespace cellerator::nn
