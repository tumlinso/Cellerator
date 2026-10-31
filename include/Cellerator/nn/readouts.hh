#pragma once

#include <ATen/ATen.h>
#include <Cellerator/biology/feature_binding.hh>
#include <Cellerator/nn/attention.hh>
#include <Cellerator/nn/layers.hh>
#include <Cellerator/nn/query.hh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cellerator::nn {

void validate_feature_readout_context(const at::Tensor& context,
                                      const at::Tensor& target_embedding,
                                      std::int64_t width);

// A biological feature readout which answers only the requested target set.
// Target identities select distinct model embedding rows through the
// PreparedAlignment; output columns are not inferred from model indices.
template<class FeatureId>
class FeatureReadout {
public:
    FeatureReadout(ParameterGroup group,
                   biology::FeatureBinding<FeatureId> binding,
                   std::int64_t width,
                   std::int64_t output_size = 1,
                   std::int64_t heads = 1,
                   double dropout = 0.0)
        : group_(std::move(group)), binding_(std::move(binding)),
          width_(width), output_size_(output_size),
          target_embedding_(group_, target_capacity(binding_), width_),
          interactions_(group_, attention_options(width_, heads, dropout)),
          projection_(group_, width_, output_size_) {
        if (width_ <= 0 || output_size_ <= 0 || heads <= 0) {
            throw std::invalid_argument("FeatureReadout dimensions must be positive");
        }
        binding_.validate_model_capacity(target_embedding_.weight().size(0));
    }

    at::Tensor operator()(const at::Tensor& context, Run run = {}) const {
        return project(context, binding_.model_indices(), run);
    }

    template<class ObservationId>
    at::Tensor query(
        const at::Tensor& context,
        const biology::PreparedAlignment<FeatureId>& targets,
        const ObservationCoordinates<ObservationId>& coordinates,
        Run run = {}) const {
        (void)coordinates; // observation labels are descriptive, not numerical input
        binding_.validate_alignment(targets);
        return project(context, targets.model_indices(), run);
    }

    const biology::FeatureBinding<FeatureId>& binding() const noexcept {
        return binding_;
    }
    const at::Tensor& target_embeddings() const noexcept {
        return target_embedding_.weight();
    }
    const AttentionOptions& attention_options() const noexcept {
        return interactions_.options();
    }
    std::int64_t width() const noexcept { return width_; }
    std::int64_t output_size() const noexcept { return output_size_; }

private:
    static std::int64_t target_capacity(
        const biology::FeatureBinding<FeatureId>& binding) {
        if (binding.model_capacity()) return *binding.model_capacity();
        std::int64_t capacity = 0;
        for (const auto index : binding.model_indices()) {
            if (index == std::numeric_limits<std::int64_t>::max()) {
                throw std::overflow_error("feature model index cannot be represented as a capacity");
            }
            capacity = std::max(capacity, index + 1);
        }
        if (capacity <= 0) {
            throw std::invalid_argument("FeatureReadout binding needs at least one target feature");
        }
        return capacity;
    }

    static AttentionOptions attention_options(std::int64_t width,
                                              std::int64_t heads,
                                              double dropout) {
        AttentionOptions options;
        options.width = width;
        options.heads = heads;
        options.dropout = dropout;
        return options;
    }

    at::Tensor project(const at::Tensor& context,
                       const std::vector<std::int64_t>& target_indices,
                       Run run) const {
        validate_feature_readout_context(context, target_embedding_.weight(), width_);
        auto source = context.dim() == 2 ? context.unsqueeze(1) : context;
        auto indices = at::tensor(target_indices,
            at::TensorOptions().dtype(at::kLong).device(context.device()));
        auto targets = target_embedding_(indices, run);
        targets = targets.unsqueeze(0).expand(
            {source.size(0), static_cast<std::int64_t>(target_indices.size()), width_});
        auto attended = interactions_.cross(source, targets, AttentionLayout{}, run);
        auto output = projection_(attended, run);
        return output_size_ == 1 ? output.squeeze(-1) : output;
    }

    ParameterGroup group_;
    biology::FeatureBinding<FeatureId> binding_;
    std::int64_t width_;
    std::int64_t output_size_;
    Embedding target_embedding_;
    RelationAttention interactions_;
    Linear projection_;
};

} // namespace cellerator::nn
