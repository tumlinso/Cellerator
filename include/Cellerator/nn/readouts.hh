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
          width_(checked_width(width, output_size, heads, dropout)), output_size_(output_size),
          target_embedding_(group_, target_capacity(binding_), width_),
          all_targets_(prepare_targets(binding_, binding_.canonical_ids(),
                                       target_embedding_.weight().device())),
          interactions_(group_, attention_options(width_, heads, dropout)),
          projection_(group_, width_, output_size_) {
        binding_.validate_model_capacity(target_embedding_.weight().size(0));
    }

    at::Tensor operator()(const at::Tensor& context, Run run = {}) const {
        return project(context, all_targets_.model_index_tensor(), all_targets_.device(), run);
    }

    template<class ObservationId>
    at::Tensor query(
        const at::Tensor& context,
        const PreparedTargets<FeatureId>& targets,
        const ObservationCoordinates<ObservationId>& coordinates,
        Run run = {}) const {
        (void)coordinates; // observation labels are descriptive, not numerical input
        binding_.validate_alignment(targets.alignment());
        return project(context, targets.model_index_tensor(), targets.device(), run);
    }

    // The Torch-free alignment overload is kept for boundary and CPU callers.
    // It materializes a temporary device index tensor for each call; repeated
    // queries should use prepare_targets(binding, ids, device) instead.
    template<class ObservationId>
    at::Tensor query(
        const at::Tensor& context,
        const biology::PreparedAlignment<FeatureId>& targets,
        const ObservationCoordinates<ObservationId>& coordinates,
        Run run = {}) const {
        (void)coordinates;
        binding_.validate_alignment(targets);
        auto indices = at::tensor(targets.model_indices(),
            at::TensorOptions().dtype(at::kLong).device(context.device()));
        return project(context, indices, context.device(), run);
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
    static std::int64_t checked_width(std::int64_t width,
                                     std::int64_t output_size,
                                     std::int64_t heads,
                                     double dropout) {
        if (width <= 0 || output_size <= 0 || heads <= 0) {
            throw std::invalid_argument("FeatureReadout dimensions must be positive");
        }
        if (width % heads != 0) {
            throw std::invalid_argument(
                "FeatureReadout width must be divisible by attention heads");
        }
        if (!(dropout >= 0.0 && dropout < 1.0)) {
            throw std::invalid_argument("FeatureReadout dropout must be in [0,1)");
        }
        return width;
    }

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
        if (!(dropout >= 0.0 && dropout < 1.0)) {
            throw std::invalid_argument("FeatureReadout dropout must be in [0,1)");
        }
        AttentionOptions options;
        options.width = width;
        options.heads = heads;
        options.dropout = dropout;
        return options;
    }

    at::Tensor project(const at::Tensor& context, const at::Tensor& target_indices,
                       at::Device target_device, Run run) const {
        validate_feature_readout_context(context, target_embedding_.weight(), width_);
        if (target_device != context.device() || target_indices.device() != context.device()) {
            throw std::invalid_argument(
                "prepared target indices and FeatureReadout context must share a device");
        }
        auto source = context.dim() == 2 ? context.unsqueeze(1) : context;
        auto targets = target_embedding_(target_indices, run);
        targets = targets.unsqueeze(0).expand(
            {source.size(0), target_indices.numel(), width_});
        auto attended = interactions_.cross(source, targets, AttentionLayout{}, run);
        auto output = projection_(attended, run);
        return output_size_ == 1 ? output.squeeze(-1) : output;
    }

    ParameterGroup group_;
    biology::FeatureBinding<FeatureId> binding_;
    std::int64_t width_;
    std::int64_t output_size_;
    Embedding target_embedding_;
    PreparedTargets<FeatureId> all_targets_;
    RelationAttention interactions_;
    Linear projection_;
};

} // namespace cellerator::nn
