#include <Cellerator/nn/features.hh>

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/nn/init.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace cellerator::nn {
namespace {

void require_positive(std::int64_t value, const char* name) {
    if (value <= 0)
        throw std::invalid_argument(std::string(name) + " must be positive");
}

void validate_options(const FeatureEncoderOptions& options) {
    require_positive(options.vocabulary_size, "vocabulary_size");
    require_positive(options.width, "width");
    if (options.value_size) require_positive(*options.value_size, "value_size");
    if (options.attribute_size)
        require_positive(*options.attribute_size, "attribute_size");
}

std::vector<std::int64_t> feature_shape(const at::Tensor& ids) {
    if (!ids.defined()) throw std::invalid_argument("feature IDs are undefined");
    if (ids.dim() < 1)
        throw std::invalid_argument("feature IDs must have at least one dimension");
    if (!(ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong))
        throw std::invalid_argument("feature IDs must have dtype int32 or int64");
    return ids.sizes().vec();
}

void require_same_device_dtype(const at::Tensor& tensor,
                               const at::Tensor& reference,
                               const char* name) {
    if (tensor.device() != reference.device())
        throw std::invalid_argument(std::string(name) + " must share the feature device");
    if (tensor.scalar_type() != reference.scalar_type())
        throw std::invalid_argument(std::string(name) + " must share the parameter dtype");
}

at::Tensor make_projection(ParameterGroup& group, std::int64_t input,
                           std::int64_t output, ParameterId& slot) {
    slot = group.empty({output, input});
    auto weight = group.get(slot);
    at::NoGradGuard no_grad;
    torch::nn::init::kaiming_uniform_(weight, std::sqrt(5.0));
    return weight;
}

} // namespace

at::Tensor encode_features(const FeatureEncoderOptions& options,
                           const FeatureEncoderParameters& parameters,
                           const FeatureInput& input) {
    validate_options(options);
    const auto ids_shape = feature_shape(input.ids);
    const auto& ids = input.ids;

    if (!parameters.feature_embedding.defined()
        || parameters.feature_embedding.dim() != 2
        || parameters.feature_embedding.size(0) != options.vocabulary_size
        || parameters.feature_embedding.size(1) != options.width)
        throw std::invalid_argument("feature embedding shape does not match options");
    if (!parameters.status_embedding.defined()
        || parameters.status_embedding.dim() != 2
        || parameters.status_embedding.size(0) != 2
        || parameters.status_embedding.size(1) != options.width)
        throw std::invalid_argument("status embedding must have shape [2, width]");
    if (parameters.status_embedding.device() != parameters.feature_embedding.device()
        || parameters.status_embedding.scalar_type()
            != parameters.feature_embedding.scalar_type())
        throw std::invalid_argument("feature and status embeddings must share device and dtype");
    if (ids.device() != parameters.feature_embedding.device())
        throw std::invalid_argument("feature IDs and parameters must share a device");

    if (options.value_size) {
        if (!parameters.value_projection.defined()
            || parameters.value_projection.dim() != 2
            || parameters.value_projection.size(0) != options.width
            || parameters.value_projection.size(1) != *options.value_size)
            throw std::invalid_argument("value projection shape does not match options");
        require_same_device_dtype(parameters.value_projection,
                                  parameters.feature_embedding, "value projection");
    } else if (parameters.value_projection.defined()) {
        throw std::invalid_argument("value projection is present while value_size is absent");
    }
    if (options.attribute_size) {
        if (!parameters.attribute_projection.defined()
            || parameters.attribute_projection.dim() != 2
            || parameters.attribute_projection.size(0) != options.width
            || parameters.attribute_projection.size(1) != *options.attribute_size)
            throw std::invalid_argument("attribute projection shape does not match options");
        require_same_device_dtype(parameters.attribute_projection,
                                  parameters.feature_embedding, "attribute projection");
    } else if (parameters.attribute_projection.defined()) {
        throw std::invalid_argument(
            "attribute projection is present while attribute_size is absent");
    }

    std::optional<at::Tensor> observed = input.observed;
    if (observed) {
        if (!observed->defined() || observed->scalar_type() != at::kBool)
            throw std::invalid_argument("observed must be a Boolean tensor");
        if (observed->sizes().vec() != ids_shape)
            throw std::invalid_argument("observed must have the same shape as feature IDs");
        if (observed->device() != ids.device())
            throw std::invalid_argument("observed and feature IDs must share a device");
    } else {
        observed = input.values
            ? at::ones_like(ids, ids.options().dtype(at::kBool))
            : at::zeros_like(ids, ids.options().dtype(at::kBool));
    }

    auto encoded = at::embedding(parameters.feature_embedding,
                                 ids.to(at::kLong), -1, false, false);
    const auto status_ids = observed->to(at::kLong);
    encoded = encoded + at::embedding(parameters.status_embedding,
                                      status_ids, -1, false, false);

    if (input.values) {
        const auto& values = *input.values;
        if (!values.defined()) throw std::invalid_argument("values are undefined");
        if (!options.value_size)
            throw std::invalid_argument("values were supplied but value_size is absent");
        require_same_device_dtype(values, parameters.feature_embedding, "values");

        auto projected_values = values;
        if (*options.value_size == 1 && values.sizes().vec() == ids_shape)
            projected_values = values.unsqueeze(-1);
        auto expected_shape = ids_shape;
        expected_shape.push_back(*options.value_size);
        if (projected_values.sizes().vec() != expected_shape)
            throw std::invalid_argument(
                "values must align with feature IDs and end in value_size");

        // Replacing hidden values before projection prevents masked NaNs and
        // withheld targets from affecting either results or input gradients.
        const auto visible_values = at::where(
            observed->unsqueeze(-1), projected_values,
            at::zeros_like(projected_values));
        encoded = encoded + at::linear(visible_values,
                                       parameters.value_projection,
                                       std::optional<at::Tensor>{});
    }

    if (input.attributes) {
        const auto& attributes = *input.attributes;
        if (!attributes.defined())
            throw std::invalid_argument("attributes are undefined");
        if (!options.attribute_size)
            throw std::invalid_argument(
                "attributes were supplied but attribute_size is absent");
        require_same_device_dtype(attributes, parameters.feature_embedding,
                                  "attributes");
        auto expected_shape = ids_shape;
        expected_shape.push_back(*options.attribute_size);
        if (attributes.sizes().vec() != expected_shape)
            throw std::invalid_argument(
                "attributes must align with feature IDs and end in attribute_size");
        encoded = encoded + at::linear(attributes,
                                       parameters.attribute_projection,
                                       std::optional<at::Tensor>{});
    }
    return encoded;
}

FeatureEncoder::FeatureEncoder(ParameterGroup group,
                               FeatureEncoderOptions options)
    : group_(group), options_(std::move(options)) {
    validate_options(options_);

    parameters_.feature_embedding_id = group_.normal(
        {options_.vocabulary_size, options_.width}, 1.0);
    parameters_.feature_embedding = group_.get(parameters_.feature_embedding_id);
    parameters_.status_embedding_id = group_.normal({2, options_.width}, 1.0);
    parameters_.status_embedding = group_.get(parameters_.status_embedding_id);
    if (options_.value_size)
        parameters_.value_projection = make_projection(
            group_, *options_.value_size, options_.width,
            parameters_.value_projection_id);
    if (options_.attribute_size)
        parameters_.attribute_projection = make_projection(
            group_, *options_.attribute_size, options_.width,
            parameters_.attribute_projection_id);
}

at::Tensor FeatureEncoder::operator()(const FeatureInput& input, Run) const {
    return encode_features(options_, parameters_, input);
}

} // namespace cellerator::nn
