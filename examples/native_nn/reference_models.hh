#pragma once

#include <Cellerator/biology/coordinates.hh>
#include <Cellerator/biology/feature_binding.hh>
#include <Cellerator/nn/attention.hh>
#include <Cellerator/nn/features.hh>
#include <Cellerator/nn/layers.hh>
#include <Cellerator/nn/parameters.hh>
#include <Cellerator/nn/readouts.hh>
#include <Cellerator/nn/relations.hh>
#include <Cellerator/nn/supervision.hh>
#include <Cellerator/nn/types.hh>

#include <ATen/ATen.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Small C++ reference compositions over native Cellerator building blocks.
// These ordinary structs demonstrate one possible modeling style; they do
// not prescribe a scientific architecture or objective.
namespace cellerator::examples::native_nn {

namespace nn = cellerator::nn;
namespace biology = cellerator::biology;
using FeatureId = std::string;
using FeatureBinding = biology::FeatureBinding<FeatureId>;

inline std::int64_t model_capacity(const FeatureBinding& binding) {
    if (binding.model_capacity()) return *binding.model_capacity();
    std::int64_t capacity = 0;
    for (const auto index : binding.model_indices()) {
        if (index == std::numeric_limits<std::int64_t>::max()) {
            throw std::overflow_error("feature model index cannot be represented as a capacity");
        }
        capacity = std::max(capacity, index + 1);
    }
    if (capacity <= 0) throw std::invalid_argument("feature binding cannot be empty");
    return capacity;
}

inline at::Tensor index_tensor(const std::vector<std::int64_t>& indices,
                               const at::Tensor& reference) {
    return at::tensor(indices,
        at::TensorOptions().dtype(at::kLong).device(reference.device()));
}

inline at::Tensor make_row_ids(const std::vector<std::int64_t>& model_indices,
                               std::int64_t rows, const at::Tensor& reference) {
    auto ids = index_tensor(model_indices, reference);
    return ids.unsqueeze(0).expand({rows, static_cast<std::int64_t>(model_indices.size())});
}

inline std::vector<std::int64_t> identity_offsets(std::int64_t rows) {
    std::vector<std::int64_t> offsets(static_cast<std::size_t>(rows + 1));
    for (std::int64_t row = 0; row <= rows; ++row) {
        offsets[static_cast<std::size_t>(row)] = row;
    }
    return offsets;
}

inline std::vector<std::int64_t> repeated_group_offsets(
    std::int64_t groups, std::int64_t rows_per_group) {
    std::vector<std::int64_t> offsets;
    offsets.reserve(static_cast<std::size_t>(groups + 1));
    offsets.push_back(0);
    for (std::int64_t group = 0; group < groups; ++group) {
        offsets.push_back((group + 1) * rows_per_group);
    }
    return offsets;
}

inline std::vector<std::int64_t> checked_offsets(
    const std::vector<std::int64_t>& offsets, std::int64_t extent,
    const char* label, bool require_group = true) {
    if (offsets.empty() || offsets.front() != 0 || offsets.back() != extent
        || (require_group && offsets.size() < 2)) {
        throw std::invalid_argument(std::string(label)
            + " must start at zero, end at the extent, and declare a group");
    }
    for (std::size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] < offsets[i - 1]) {
            throw std::invalid_argument(std::string(label) + " must be nondecreasing");
        }
    }
    return offsets;
}

struct ExpressionBatch {
    // Feature IDs label the final axis of values and observed, in source order.
    std::vector<FeatureId> feature_ids;
    at::Tensor values;
    at::Tensor observed;
    std::vector<std::string> sample_ids;
};

struct ExpressionResult {
    at::Tensor expression;       // [samples, features], in feature_ids order
    at::Tensor features;         // [samples, features, width]
    at::Tensor pooled;           // [samples, width], for caller-selected views
    std::vector<FeatureId> feature_ids;
    std::vector<std::string> sample_ids;
};

struct ExpressionReferenceModel {
    FeatureBinding feature_binding;
    FeatureBinding atac_binding;
    nn::Parameters parameters;
    nn::ParameterGroup base_group;
    nn::ParameterGroup atac_group;
    nn::FeatureEncoder encoder;
    nn::RelationAttention interactions;
    nn::Linear expression_projection;
    nn::FeatureReadout<FeatureId> atac_readout;
    std::int64_t width;

    ExpressionReferenceModel(FeatureBinding feature_binding_,
                             FeatureBinding atac_binding_,
                             std::int64_t width_,
                             std::int64_t heads = 1,
                             at::Device device = at::Device(at::kCPU),
                             at::ScalarType dtype = at::kFloat)
        : feature_binding(std::move(feature_binding_)),
          atac_binding(std::move(atac_binding_)),
          parameters(device, dtype),
          base_group(parameters.group()),
          atac_group(parameters.group()),
          encoder(base_group, nn::FeatureEncoderOptions{
              model_capacity(feature_binding), width_, 1, std::nullopt}),
          interactions(base_group, nn::AttentionOptions{
              .width = width_, .heads = heads}),
          expression_projection(base_group, width_, 1),
          atac_readout(atac_group, atac_binding, width_, 1, heads),
          width(width_) {
        if (width <= 0 || heads <= 0) {
            throw std::invalid_argument("expression width and heads must be positive");
        }
        feature_binding.validate_model_capacity(model_capacity(feature_binding));
    }

    ExpressionResult operator()(const ExpressionBatch& batch,
                                nn::Run run = {}) const {
        if (!batch.values.defined() || batch.values.dim() != 2
            || !batch.values.is_floating_point()) {
            throw std::invalid_argument("expression values must be a rank-two floating tensor");
        }
        if (!batch.observed.defined() || batch.observed.scalar_type() != at::kBool
            || batch.observed.sizes() != batch.values.sizes()
            || batch.observed.device() != batch.values.device()) {
            throw std::invalid_argument("observed must be a matching Boolean tensor");
        }
        if (batch.values.size(1) != static_cast<std::int64_t>(batch.feature_ids.size())
            || batch.values.size(0) != static_cast<std::int64_t>(batch.sample_ids.size())
            || batch.values.size(0) <= 0 || batch.values.size(1) <= 0) {
            throw std::invalid_argument("expression identities must label every input row and feature");
        }

        const auto aligned = feature_binding.prepare(
            batch.feature_ids, feature_binding.canonical_ids());
        auto columns = index_tensor(aligned.source_positions(), batch.values);
        auto values = batch.values.index_select(-1, columns);
        auto observed = batch.observed.index_select(-1, columns);
        auto ids = make_row_ids(aligned.model_indices(), values.size(0), values);
        auto encoded = encoder(nn::FeatureInput{ids, values, observed, std::nullopt}, run);

        // Keep the learned relation update explicit in the composition.
        auto features = encoded + interactions(encoded, nn::AttentionLayout{}, run);
        auto expression = expression_projection(features, run).squeeze(-1);
        auto pooled = nn::mean_by_group(
            features.reshape({features.size(0) * features.size(1), width}),
            repeated_group_offsets(features.size(0), features.size(1)));
        return {std::move(expression), std::move(features), std::move(pooled),
                feature_binding.canonical_ids(), batch.sample_ids};
    }
};

// The start marker and repeated biological event IDs occupy separate
// occurrence identities. Model rows are held only in the numeric input tensor.
struct HistoryTokenIdentity {
    bool start = false;
    FeatureId event_id;

    static HistoryTokenIdentity start_token() { return {true, {}}; }
    static HistoryTokenIdentity event(FeatureId id) { return {false, std::move(id)}; }

    friend bool operator==(const HistoryTokenIdentity&, const HistoryTokenIdentity&) = default;
};

struct EventBatch {
    // Packed biological event labels; repeats are distinct ordered occurrences.
    std::vector<FeatureId> event_ids;
    std::vector<std::int64_t> event_offsets;
    std::vector<std::string> sample_ids;
    std::uint64_t structure_generation = 0;
};

struct EventResult {
    at::Tensor logits;                  // [event occurrences, classes]
    at::Tensor history_features;        // [samples, width], includes START for empty histories
    at::Tensor context_model_ids;       // numeric encoder rows, including START
    at::Tensor target_class_ids;        // categorical target columns, shifted from inputs
    std::vector<FeatureId> target_ids;
    std::vector<std::int64_t> target_positions;
    std::vector<std::int64_t> context_offsets;
    biology::OccurrenceCoordinates<HistoryTokenIdentity> context_order;
    nn::ObservationCoordinates<std::string> event_coordinates;
};

struct EventHistoryReferenceModel {
    FeatureBinding event_binding;
    FeatureBinding atac_binding;
    nn::Parameters parameters;
    nn::ParameterGroup base_group;
    nn::ParameterGroup atac_group;
    nn::FeatureEncoder encoder;
    nn::Embedding order_embedding;
    nn::RelationAttention interactions;
    nn::Linear event_projection;
    nn::FeatureReadout<FeatureId> atac_readout;
    std::int64_t width;
    std::int64_t max_history;
    std::int64_t start_model_id;

    EventHistoryReferenceModel(FeatureBinding event_binding_,
                               FeatureBinding atac_binding_,
                               std::int64_t width_,
                               std::int64_t max_history_,
                               std::int64_t heads = 1,
                               at::Device device = at::Device(at::kCPU),
                               at::ScalarType dtype = at::kFloat)
        : event_binding(std::move(event_binding_)),
          atac_binding(std::move(atac_binding_)),
          parameters(device, dtype),
          base_group(parameters.group()),
          atac_group(parameters.group()),
          encoder(base_group, nn::FeatureEncoderOptions{
              event_capacity(event_binding), width_, std::nullopt, std::nullopt}),
          order_embedding(base_group, max_history_, width_),
          interactions(base_group, nn::AttentionOptions{
              .width = width_, .heads = heads, .causal = true}),
          event_projection(base_group, width_,
              static_cast<std::int64_t>(event_binding.canonical_ids().size())),
          atac_readout(atac_group, atac_binding, width_, 1, heads),
          width(width_), max_history(max_history_),
          start_model_id(event_capacity(event_binding) - 1) {
        if (width <= 0 || max_history <= 0 || heads <= 0) {
            throw std::invalid_argument("event width, max_history, and heads must be positive");
        }
        // event_capacity reserves its final row for START; known and optional
        // unknown feature rows remain in the declared binding capacity.
        const auto highest = *std::max_element(
            event_binding.model_indices().begin(), event_binding.model_indices().end());
        if (start_model_id <= highest) {
            throw std::invalid_argument("event model capacity must reserve one START row");
        }
    }

    static std::int64_t event_capacity(const FeatureBinding& binding) {
        const auto feature_capacity = model_capacity(binding);
        if (feature_capacity == std::numeric_limits<std::int64_t>::max()) {
            throw std::overflow_error("event capacity cannot reserve a START row");
        }
        // One final model row is reserved for the explicit computational START.
        return feature_capacity + 1;
    }

    EventResult operator()(const EventBatch& batch, nn::Run run = {}) const {
        const auto offsets = checked_offsets(
            batch.event_offsets, static_cast<std::int64_t>(batch.event_ids.size()),
            "event_offsets");
        if (offsets.size() != batch.sample_ids.size() + 1) {
            throw std::invalid_argument("event_offsets and sample_ids must declare the same histories");
        }

        const auto mapped = event_binding.map_occurrences(batch.event_ids);
        if (!mapped.dropped_ids().empty()) {
            throw std::invalid_argument("event occurrences cannot be dropped by their binding");
        }
        const auto& model_ids = mapped.model_indices();
        std::vector<std::int64_t> context_ids;
        std::vector<std::int64_t> context_offsets{0};
        std::vector<std::int64_t> order_positions;
        std::vector<std::int64_t> target_positions;
        std::vector<std::int64_t> target_classes;
        std::vector<HistoryTokenIdentity> context_identities;

        for (std::size_t group = 0; group + 1 < offsets.size(); ++group) {
            const auto left = offsets[group];
            const auto right = offsets[group + 1];
            const auto length = right - left;
            const auto context_begin = static_cast<std::int64_t>(context_ids.size());
            context_ids.push_back(start_model_id);
            context_identities.push_back(HistoryTokenIdentity::start_token());
            order_positions.push_back(0);
            for (std::int64_t position = 0; position < length; ++position) {
                const auto label = batch.event_ids[static_cast<std::size_t>(left + position)];
                const auto found = std::find(event_binding.canonical_ids().begin(),
                                             event_binding.canonical_ids().end(), label);
                if (found == event_binding.canonical_ids().end()) {
                    throw std::out_of_range("event label has no output class in its feature binding");
                }
                target_classes.push_back(static_cast<std::int64_t>(
                    found - event_binding.canonical_ids().begin()));
                target_positions.push_back(context_begin + position);
                if (position + 1 < length) {
                    const auto event_index = static_cast<std::size_t>(left + position);
                    context_ids.push_back(model_ids[event_index]);
                    context_identities.push_back(HistoryTokenIdentity::event(label));
                    order_positions.push_back(position + 1);
                }
            }
            context_offsets.push_back(static_cast<std::int64_t>(context_ids.size()));
        }

        if (!order_positions.empty()
            && *std::max_element(order_positions.begin(), order_positions.end()) >= max_history) {
            throw std::invalid_argument("event context exceeds max_history");
        }
        auto context_order = biology::OccurrenceCoordinates<HistoryTokenIdentity>::from_ids(
            std::move(context_identities), context_offsets, {}, batch.structure_generation);
        auto context_tensor = at::tensor(context_ids,
            at::TensorOptions().dtype(at::kLong).device(parameters.device()));
        auto observed = at::ones_like(context_tensor, context_tensor.options().dtype(at::kBool));
        auto encoded = encoder(nn::FeatureInput{
            context_tensor, std::nullopt, observed, std::nullopt}, run);
        auto positions = at::tensor(order_positions,
            at::TensorOptions().dtype(at::kLong).device(parameters.device()));
        encoded = encoded + order_embedding(positions, run);

        // Declare dense within-history support once for this reference call;
        // causal() retains only each occurrence's prefix, including itself.
        nn::CsrSupport support;
        support.row_ptr.reserve(context_ids.size() + 1);
        support.row_ptr.push_back(0);
        for (std::size_t group = 0; group + 1 < context_offsets.size(); ++group) {
            const auto left = context_offsets[group];
            const auto right = context_offsets[group + 1];
            for (auto target = left; target < right; ++target) {
                for (auto source = left; source < right; ++source) {
                    support.col_indices.push_back(source);
                }
                support.row_ptr.push_back(
                    static_cast<std::int64_t>(support.col_indices.size()));
            }
        }
        auto full_group_relation = nn::PreparedRelation::from_csr(
            std::move(support), context_order.order_token(), context_order.order_token(),
            batch.structure_generation, parameters.device());
        auto causal_relation = full_group_relation.causal();
        auto features = encoded + interactions(encoded, causal_relation, run);

        auto target_index = at::tensor(target_positions,
            at::TensorOptions().dtype(at::kLong).device(parameters.device()));
        auto target_features = features.index_select(0, target_index);
        auto logits = event_projection(target_features, run);
        auto target_class_tensor = at::tensor(target_classes,
            at::TensorOptions().dtype(at::kLong).device(parameters.device()));
        auto histories = nn::mean_by_group(features, context_offsets);
        return {std::move(logits), std::move(histories), std::move(context_tensor),
                std::move(target_class_tensor), batch.event_ids,
                std::move(target_positions), context_offsets, std::move(context_order),
                nn::ObservationCoordinates<std::string>{batch.sample_ids, offsets}};
    }
};

inline std::optional<at::Tensor> masked_expression_mse(
    const at::Tensor& prediction, const at::Tensor& target,
    const at::Tensor& target_mask) {
    if (!prediction.defined() || !target.defined() || !target_mask.defined()
        || prediction.sizes() != target.sizes()
        || target_mask.sizes() != target.sizes()
        || target_mask.scalar_type() != at::kBool
        || prediction.device() != target.device()
        || target_mask.device() != target.device()) {
        throw std::invalid_argument("prediction, target, and Boolean mask must align");
    }
    if (target_mask.numel() == 0 || !target_mask.any().item<bool>()) return std::nullopt;
    const auto selected_prediction = prediction.masked_select(target_mask);
    const auto selected_target = target.masked_select(target_mask);
    return (selected_prediction - selected_target).square().mean();
}

inline std::optional<at::Tensor> event_next_event_cross_entropy(
    const at::Tensor& logits, const at::Tensor& target_classes) {
    if (!logits.defined() || !target_classes.defined() || logits.dim() != 2
        || target_classes.dim() != 1 || logits.size(0) != target_classes.size(0)
        || target_classes.scalar_type() != at::kLong) {
        throw std::invalid_argument("event logits and shifted class targets must align");
    }
    if (target_classes.numel() == 0) return std::nullopt;
    return at::cross_entropy_loss(logits, target_classes);
}

} // namespace cellerator::examples::native_nn
