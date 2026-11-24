#include "reference_models.hh"

#include <ATen/Parallel.h>
#include <torch/optim/adam.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ex = cellerator::examples::native_nn;
namespace nn = cellerator::nn;

namespace {

ex::FeatureBinding make_binding(std::vector<ex::FeatureId> ids,
                                std::vector<std::int64_t> model_ids,
                                std::string domain,
                                std::int64_t capacity) {
    cellerator::biology::FeatureBindingOptions options;
    options.domain = std::move(domain);
    options.model_capacity = capacity;
    return ex::FeatureBinding::from_ids(std::move(ids), std::move(model_ids),
                                         std::move(options));
}

ex::ExpressionBatch expression_fixture() {
    const std::vector<float> canonical_values{
        0.10f, 0.20f, 0.30f, 0.40f, 0.50f,
        0.80f, 0.70f, 0.60f, 0.50f, 0.40f,
        0.20f, 0.60f, 0.30f, 0.70f, 0.50f,
        0.90f, 0.10f, 0.80f, 0.20f, 0.70f,
    };
    auto canonical = at::tensor(canonical_values, at::TensorOptions().dtype(at::kFloat))
                        .reshape({4, 5});
    // Source order is C, A, E, B, D; model rows and canonical IDs are separate.
    auto source_columns = at::tensor({2, 0, 4, 1, 3},
        at::TensorOptions().dtype(at::kLong));
    auto values = canonical.index_select(1, source_columns);
    auto observed = at::ones({4, 5}, at::TensorOptions().dtype(at::kBool));
    observed.select(1, 1).fill_(false);
    observed.select(1, 4).fill_(false);
    return {
        {"gene-C", "gene-A", "gene-E", "gene-B", "gene-D"},
        std::move(values), std::move(observed),
        {"expr-cell-0", "expr-cell-1", "expr-cell-2", "expr-cell-3"}
    };
}

at::Tensor aligned_expression_targets(const ex::ExpressionReferenceModel& model,
                                      const ex::ExpressionBatch& batch) {
    const auto alignment = model.feature_binding.prepare(
        batch.feature_ids, model.feature_binding.canonical_ids());
    auto columns = ex::index_tensor(alignment.source_positions(), batch.values);
    return batch.values.index_select(-1, columns);
}

at::Tensor aligned_expression_mask(const ex::ExpressionReferenceModel& model,
                                   const ex::ExpressionBatch& batch) {
    const auto alignment = model.feature_binding.prepare(
        batch.feature_ids, model.feature_binding.canonical_ids());
    auto columns = ex::index_tensor(alignment.source_positions(), batch.values);
    return batch.observed.logical_not().index_select(-1, columns);
}

std::optional<at::Tensor> expression_loss(
    const ex::ExpressionReferenceModel& model, const ex::ExpressionBatch& batch,
    bool training) {
    auto result = model(batch, nn::Run{training});
    return ex::masked_expression_mse(result.expression,
        aligned_expression_targets(model, batch), aligned_expression_mask(model, batch));
}

at::Tensor expression_atac_prediction(ex::ExpressionReferenceModel& model,
                                      const ex::ExpressionBatch& batch,
                                      const nn::PreparedTargets<ex::FeatureId>& targets,
                                      bool training) {
    const auto rows = static_cast<std::int64_t>(batch.sample_ids.size());
    nn::ObservationCoordinates<std::string> coordinates{
        batch.sample_ids, ex::identity_offsets(rows)};
    auto view = nn::readout(model.atac_readout, &ex::ExpressionResult::pooled,
                            model.atac_binding);
    auto request = nn::query(view, targets, coordinates);
    auto prediction = nn::predict(model, batch, nn::Run{training}, request);
    return prediction.values;
}

std::pair<double, double> train_expression(ex::ExpressionReferenceModel& model,
                                           const ex::ExpressionBatch& batch,
                                           std::int64_t steps) {
    torch::optim::Adam optimizer(
        model.parameters.torch_parameters(), torch::optim::AdamOptions(0.025));
    const auto initial_loss = expression_loss(model, batch, false);
    if (!initial_loss) throw std::runtime_error("expression fixture has no masked targets");
    const double initial = initial_loss->detach().item<double>();
    for (std::int64_t step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        const auto loss = expression_loss(model, batch, true);
        if (!loss) throw std::runtime_error("expression supervision disappeared");
        loss->backward();
        optimizer.step();
    }
    const auto final_loss = expression_loss(model, batch, false);
    if (!final_loss) throw std::runtime_error("expression supervision disappeared");
    return {initial, final_loss->detach().item<double>()};
}

double expression_atac_loss(ex::ExpressionReferenceModel& model,
                            const ex::ExpressionBatch& batch,
                            const nn::PreparedTargets<ex::FeatureId>& prepared_targets,
                            const at::Tensor& targets,
                            bool training) {
    const auto logits = expression_atac_prediction(
        model, batch, prepared_targets, training);
    return at::binary_cross_entropy_with_logits(logits, targets).item<double>();
}

std::pair<double, double> train_expression_atac(
    ex::ExpressionReferenceModel& model, const ex::ExpressionBatch& batch,
    const nn::PreparedTargets<ex::FeatureId>& prepared_targets,
    const at::Tensor& targets, bool train_base, std::int64_t steps) {
    model.base_group.set_trainable(train_base);
    torch::optim::Adam optimizer(
        model.parameters.torch_parameters(), torch::optim::AdamOptions(0.02));
    const double initial = expression_atac_loss(
        model, batch, prepared_targets, targets, false);
    for (std::int64_t step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        const auto logits = expression_atac_prediction(
            model, batch, prepared_targets, true);
        const auto loss = at::binary_cross_entropy_with_logits(logits, targets);
        loss.backward();
        optimizer.step();
    }
    return {initial, expression_atac_loss(
        model, batch, prepared_targets, targets, false)};
}

ex::EventBatch event_fixture() {
    // Leading, consecutive, interleaved, and trailing empty histories are kept.
    return {
        {"event-A", "event-B", "event-A", "event-B",
         "event-B", "event-A", "event-B", "event-A"},
        {0, 0, 0, 4, 4, 8, 8},
        {"history-0", "history-1", "history-2", "history-3",
         "history-4", "history-5"},
        7
    };
}

struct EventPrediction {
    at::Tensor logits;
    at::Tensor targets;
};

EventPrediction event_prediction(ex::EventHistoryReferenceModel& model,
                                  const ex::EventBatch& batch,
                                  const nn::PreparedTargets<ex::FeatureId>& prepared_targets,
                                  bool training) {
    auto view = nn::readout(nn::Identity{}, &ex::EventResult::logits,
                            model.event_binding);
    nn::ObservationCoordinates<std::string> coordinates{
        batch.sample_ids, batch.event_offsets};
    auto request = nn::query(view, prepared_targets, coordinates);
    auto prediction = nn::predict(model, batch, nn::Run{training}, request);
    auto class_ids = nn::categorical_target_columns(prediction, batch.event_ids);
    return {std::move(prediction.values), std::move(class_ids)};
}

at::Tensor event_atac_prediction(ex::EventHistoryReferenceModel& model,
                                 const ex::EventBatch& batch,
                                 const nn::PreparedTargets<ex::FeatureId>& targets,
                                 bool training) {
    const auto groups = static_cast<std::int64_t>(batch.sample_ids.size());
    nn::ObservationCoordinates<std::string> coordinates{
        batch.sample_ids, ex::identity_offsets(groups)};
    auto view = nn::readout(model.atac_readout, &ex::EventResult::history_features,
                            model.atac_binding);
    auto request = nn::query(view, targets, coordinates);
    auto prediction = nn::predict(model, batch, nn::Run{training}, request);
    return prediction.values;
}

std::optional<double> event_loss(ex::EventHistoryReferenceModel& model,
                                 const ex::EventBatch& batch,
                                 const nn::PreparedTargets<ex::FeatureId>& prepared_targets,
                                 bool training) {
    auto output = event_prediction(model, batch, prepared_targets, training);
    auto loss = ex::event_next_event_cross_entropy(output.logits, output.targets);
    if (!loss) return std::nullopt;
    return loss->item<double>();
}

std::pair<double, double> train_events(ex::EventHistoryReferenceModel& model,
                                       const ex::EventBatch& batch,
                                       const nn::PreparedTargets<ex::FeatureId>& prepared_targets,
                                       std::int64_t steps) {
    torch::optim::Adam optimizer(
        model.parameters.torch_parameters(), torch::optim::AdamOptions(0.02));
    const auto initial_loss = event_loss(model, batch, prepared_targets, false);
    if (!initial_loss) throw std::runtime_error("event fixture has no next-event targets");
    for (std::int64_t step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        auto output = event_prediction(model, batch, prepared_targets, true);
        const auto loss = ex::event_next_event_cross_entropy(output.logits, output.targets);
        if (!loss) throw std::runtime_error("event supervision disappeared");
        loss->backward();
        optimizer.step();
    }
    const auto final_loss = event_loss(model, batch, prepared_targets, false);
    if (!final_loss) throw std::runtime_error("event supervision disappeared");
    return {*initial_loss, *final_loss};
}

double event_atac_loss(ex::EventHistoryReferenceModel& model,
                       const ex::EventBatch& batch,
                       const nn::PreparedTargets<ex::FeatureId>& prepared_targets,
                       const at::Tensor& targets, bool training) {
    return at::binary_cross_entropy_with_logits(
        event_atac_prediction(model, batch, prepared_targets, training), targets).item<double>();
}

std::pair<double, double> train_event_atac(
    ex::EventHistoryReferenceModel& model, const ex::EventBatch& batch,
    const nn::PreparedTargets<ex::FeatureId>& prepared_targets,
    const at::Tensor& targets, bool train_base, std::int64_t steps) {
    model.base_group.set_trainable(train_base);
    torch::optim::Adam optimizer(
        model.parameters.torch_parameters(), torch::optim::AdamOptions(0.02));
    const double initial = event_atac_loss(
        model, batch, prepared_targets, targets, false);
    for (std::int64_t step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        const auto logits = event_atac_prediction(
            model, batch, prepared_targets, true);
        const auto loss = at::binary_cross_entropy_with_logits(logits, targets);
        loss.backward();
        optimizer.step();
    }
    return {initial, event_atac_loss(
        model, batch, prepared_targets, targets, false)};
}

void report(const char* name, const std::pair<double, double>& values) {
    if (!std::isfinite(values.first) || !std::isfinite(values.second)) {
        throw std::runtime_error(std::string(name) + " produced a non-finite loss");
    }
    std::cout << name << ": initial=" << values.first
              << " final=" << values.second << '\n';
}

void require_decreased(const char* name, const std::pair<double, double>& values) {
    if (!(values.second < values.first)) {
        throw std::runtime_error(std::string(name) + " did not decrease on the fixture");
    }
}

} // namespace

int main() {
    at::manual_seed(20261031);
    at::set_num_threads(1);

    auto gene_binding = make_binding(
        {"gene-A", "gene-B", "gene-C", "gene-D", "gene-E"},
        {2, 7, 11, 15, 19}, "gene", 21);
    auto peak_binding = make_binding(
        {"peak-A", "peak-B", "peak-C"}, {1, 4, 9}, "peak", 11);
    auto expression_batch = expression_fixture();
    ex::ExpressionReferenceModel expression_model(
        gene_binding, peak_binding, 12, 3);
    const auto expression_peak_targets = nn::prepare_targets(
        expression_model.atac_binding, expression_model.atac_binding.canonical_ids(),
        expression_model.parameters.device());
    const auto expression_metrics = train_expression(
        expression_model, expression_batch, 70);
    require_decreased("expression masked MSE", expression_metrics);
    const auto masked_target_count = aligned_expression_mask(
        expression_model, expression_batch).sum().item<std::int64_t>();
    if (masked_target_count != 8) {
        throw std::runtime_error("expression fixture lost its selected masked targets");
    }

    const auto expression_atac_targets = at::tensor(
        {1.f, 0.f, 1.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f, 0.f, 0.f, 0.f},
        at::TensorOptions().dtype(at::kFloat)).reshape({4, 3});
    const auto expression_atac_frozen = train_expression_atac(
        expression_model, expression_batch, expression_peak_targets,
        expression_atac_targets, false, 55);
    const auto expression_atac_joint = train_expression_atac(
        expression_model, expression_batch, expression_peak_targets,
        expression_atac_targets, true, 55);

    auto event_binding = make_binding(
        {"event-A", "event-B"}, {5, 12}, "event", 14);
    auto event_batch = event_fixture();
    ex::EventHistoryReferenceModel event_model(
        event_binding, peak_binding, 12, 16, 3);
    const auto event_class_targets = nn::prepare_targets(
        event_model.event_binding, event_model.event_binding.canonical_ids(),
        event_model.parameters.device());
    const auto history_peak_targets = nn::prepare_targets(
        event_model.atac_binding, event_model.atac_binding.canonical_ids(),
        event_model.parameters.device());
    const auto event_metrics = train_events(
        event_model, event_batch, event_class_targets, 75);
    require_decreased("event next-event cross entropy", event_metrics);
    const auto event_reference = event_model(event_batch);
    if (event_reference.logits.size(0)
            != static_cast<std::int64_t>(event_batch.event_ids.size())
        || event_reference.event_coordinates.row_offsets != event_batch.event_offsets) {
        throw std::runtime_error("event labels or packed order were lost during teacher forcing");
    }
    std::int64_t empty_histories = 0;
    for (std::size_t group = 0; group + 1 < event_batch.event_offsets.size(); ++group) {
        if (event_batch.event_offsets[group] == event_batch.event_offsets[group + 1]) {
            ++empty_histories;
        }
    }
    ex::EventBatch all_empty{{}, {0, 0, 0}, {"empty-history-0", "empty-history-1"}, 8};
    const auto empty_result = event_model(all_empty);
    if (ex::event_next_event_cross_entropy(
            empty_result.logits, empty_result.target_class_ids)) {
        throw std::runtime_error("all-empty histories must not produce a target loss");
    }

    const auto event_atac_targets = at::tensor(
        {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 1.f,
         0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f},
        at::TensorOptions().dtype(at::kFloat)).reshape({6, 3});
    const auto event_atac_frozen = train_event_atac(
        event_model, event_batch, history_peak_targets, event_atac_targets, false, 55);
    const auto event_atac_joint = train_event_atac(
        event_model, event_batch, history_peak_targets, event_atac_targets, true, 55);

    report("expression_masked_mse", expression_metrics);
    report("expression_atac_frozen_base_bce", expression_atac_frozen);
    report("expression_atac_joint_base_bce", expression_atac_joint);
    report("event_next_event_ce", event_metrics);
    report("history_atac_frozen_base_bce", event_atac_frozen);
    report("history_atac_joint_base_bce", event_atac_joint);
    std::cout << "expression_masked_targets=" << masked_target_count
              << " event_occurrences=" << event_reference.logits.size(0)
              << " event_empty_histories=" << empty_histories
              << " all_empty_loss=none\n";
    return 0;
}
