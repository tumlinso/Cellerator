#pragma once

#include <Cellerator/nn/parameters.hh>
#include <Cellerator/nn/types.hh>

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <optional>

namespace cellerator::nn {

// Tensors aligned to one feature-ID axis or a rectangular set of feature axes.
struct FeatureInput {
    at::Tensor ids;
    std::optional<at::Tensor> values;
    std::optional<at::Tensor> observed;
    std::optional<at::Tensor> attributes;
};

struct FeatureEncoderOptions {
    std::int64_t vocabulary_size = 0;
    std::int64_t width = 0;
    std::optional<std::int64_t> value_size = 1;
    std::optional<std::int64_t> attribute_size;
};

// Borrowed tensor parameters for the functional feature encoder. Optional
// projection weights are undefined when their option is absent. Tables and
// weights may be views/copies of the same leaf tensors held by Parameters.
struct FeatureEncoderParameters {
    at::Tensor feature_embedding;
    at::Tensor status_embedding;
    at::Tensor value_projection;
    at::Tensor attribute_projection;
    ParameterId feature_embedding_id = invalid_parameter_id;
    ParameterId status_embedding_id = invalid_parameter_id;
    ParameterId value_projection_id = invalid_parameter_id;
    ParameterId attribute_projection_id = invalid_parameter_id;
};

at::Tensor encode_features(const FeatureEncoderOptions& options,
                           const FeatureEncoderParameters& parameters,
                           const FeatureInput& input);

class FeatureEncoder {
public:
    FeatureEncoder(ParameterGroup group, FeatureEncoderOptions options);

    at::Tensor operator()(const FeatureInput& input, Run = {}) const;
    const FeatureEncoderOptions& options() const noexcept { return options_; }
    const FeatureEncoderParameters& parameters() const noexcept { return parameters_; }

private:
    ParameterGroup group_;
    FeatureEncoderOptions options_;
    FeatureEncoderParameters parameters_;
};

} // namespace cellerator::nn
