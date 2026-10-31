#ifdef NDEBUG
#undef NDEBUG
#endif

#include <Cellerator/nn/query.hh>
#include "../../../examples/native_nn/reference_models.hh"

#include <ATen/ATen.h>
#include <torch/optim/sgd.h>

#include <cassert>
#include <cmath>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ex = cellerator::examples::native_nn;
namespace nn = cellerator::nn;

namespace {

ex::FeatureBinding binding(std::vector<ex::FeatureId> ids,
                           std::vector<std::int64_t> model_ids,
                           const char* domain, std::int64_t capacity) {
    cellerator::biology::FeatureBindingOptions options;
    options.domain = domain;
    options.model_capacity = capacity;
    return ex::FeatureBinding::from_ids(std::move(ids), std::move(model_ids),
                                         std::move(options));
}

ex::ExpressionBatch expression_batch() {
    auto values = at::tensor({
        0.3f, 0.1f, 0.2f,
        0.7f, 0.9f, 0.8f,
    }, at::TensorOptions().dtype(at::kFloat)).reshape({2, 3});
    auto observed = at::ones({2, 3}, at::TensorOptions().dtype(at::kBool));
    observed.select(1, 0).fill_(false);
    return {{"gene-C", "gene-A", "gene-B"}, std::move(values), std::move(observed),
            {"cell-0", "cell-1"}};
}

ex::EventBatch event_batch() {
    return {
        {"event-A", "event-B", "event-C", "event-A", "event-A", "event-B"},
        {0, 0, 3, 3, 3, 6},
        {"history-empty-0", "history-abc", "history-empty-1",
         "history-empty-2", "history-repeated"},
        41
    };
}

void assert_close(const at::Tensor& a, const at::Tensor& b, double tolerance = 1e-6) {
    assert(a.sizes() == b.sizes());
    assert(at::allclose(a, b, tolerance, tolerance));
}

struct CountedExpressionModel {
    ex::ExpressionReferenceModel* model;
    int* calls;

    ex::ExpressionResult operator()(const ex::ExpressionBatch& input,
                                    nn::Run run) const {
        ++*calls;
        return (*model)(input, run);
    }
};

struct CountedEventModel {
    ex::EventHistoryReferenceModel* model;
    int* calls;

    ex::EventResult operator()(const ex::EventBatch& input, nn::Run run) const {
        ++*calls;
        return (*model)(input, run);
    }
};

using ParameterSnapshots = std::vector<std::pair<nn::ParameterId, at::Tensor>>;

ParameterSnapshots snapshot_group(nn::Parameters& parameters, std::uint64_t group_id) {
    ParameterSnapshots result;
    for (const auto& slot : parameters.layout()) {
        if (slot.group_id == group_id) {
            result.emplace_back(slot.id, parameters.get(slot.id).detach().clone());
        }
    }
    assert(!result.empty());
    return result;
}

void assert_group_unchanged(nn::Parameters& parameters,
                            const ParameterSnapshots& snapshots) {
    for (const auto& [id, original] : snapshots) {
        assert(at::equal(parameters.get(id), original));
    }
}

void assert_group_changed(nn::Parameters& parameters,
                          const ParameterSnapshots& snapshots) {
    bool changed = false;
    for (const auto& [id, original] : snapshots) {
        changed = changed || !at::equal(parameters.get(id), original);
    }
    assert(changed);
}

void expression_binding_masking_and_queries() {
    at::manual_seed(91);
    auto genes = binding({"gene-A", "gene-B", "gene-C"}, {2, 6, 11}, "gene", 14);
    auto peaks = binding({"peak-A", "peak-B"}, {1, 5}, "peak", 8);
    ex::ExpressionReferenceModel model(genes, peaks, 8, 2);
    auto batch = expression_batch();

    const auto canonical_result = model(batch);
    auto reordered = batch;
    const auto reorder_columns = at::tensor({2, 0, 1},
        at::TensorOptions().dtype(at::kLong));
    reordered.feature_ids = {"gene-B", "gene-C", "gene-A"};
    reordered.values = batch.values.index_select(1, reorder_columns);
    reordered.observed = batch.observed.index_select(1, reorder_columns);
    const auto reordered_result = model(reordered);
    assert(canonical_result.feature_ids == genes.canonical_ids());
    assert_close(canonical_result.expression, reordered_result.expression);

    auto changed_hidden = batch;
    changed_hidden.values = batch.values.clone();
    changed_hidden.values.select(1, 0).fill_(1000.0);
    assert_close(model(batch).expression, model(changed_hidden).expression);

    auto target = at::tensor({0.4f, 0.1f, 0.2f, 0.8f, 0.9f, 0.7f},
        at::TensorOptions().dtype(at::kFloat)).reshape({2, 3});
    auto selected = at::zeros_like(target, at::TensorOptions().dtype(at::kBool));
    selected.select(1, 0).fill_(true);
    auto pred = model(batch).expression;
    const auto loss = ex::masked_expression_mse(pred, target, selected);
    assert(loss && std::isfinite(loss->item<double>()));
    auto unselected_changed = target.clone();
    unselected_changed.select(1, 1).fill_(12345.0);
    const auto same_loss = ex::masked_expression_mse(pred, unselected_changed, selected);
    assert(same_loss);
    assert(std::abs(loss->item<double>() - same_loss->item<double>()) < 1e-8);
    auto empty_mask = at::zeros_like(selected);
    assert(!ex::masked_expression_mse(pred, target, empty_mask));

    const auto rows = static_cast<std::int64_t>(batch.sample_ids.size());
    nn::ObservationCoordinates<std::string> coordinates{
        batch.sample_ids, ex::identity_offsets(rows)};
    auto expression_view = nn::readout(nn::Identity{}, &ex::ExpressionResult::expression,
                                       model.feature_binding);
    auto expression_targets = nn::prepare_targets(
        model.feature_binding, model.feature_binding.canonical_ids(), batch.values.device());
    auto expression_request = nn::query(expression_view, expression_targets, coordinates);
    auto atac_view = nn::readout(model.atac_readout, &ex::ExpressionResult::pooled,
                                 model.atac_binding);
    auto atac_targets = nn::prepare_targets(
        model.atac_binding, model.atac_binding.canonical_ids(), batch.values.device());
    auto atac_request = nn::query(atac_view, atac_targets, coordinates);
    int calls = 0;
    auto both = nn::predict(CountedExpressionModel{&model, &calls}, batch, nn::Run{},
                            expression_request, atac_request);
    static_assert(std::tuple_size_v<decltype(both)> == 2);
    assert(calls == 1);
    assert(std::get<0>(both).values.sizes().vec() == std::vector<std::int64_t>({2, 3}));
    assert(std::get<1>(both).values.sizes().vec() == std::vector<std::int64_t>({2, 2}));

    // Freezing a base owner's parameters still leaves the forward differentiable
    // with respect to visible input values through a shared ATAC readout.
    at::manual_seed(92);
    ex::ExpressionReferenceModel frozen_model(genes, peaks, 8, 2);
    frozen_model.base_group.set_trainable(false);
    auto gradient_batch = batch;
    gradient_batch.values = batch.values.clone().set_requires_grad(true);
    auto gradient_view = nn::readout(frozen_model.atac_readout,
        &ex::ExpressionResult::pooled, frozen_model.atac_binding);
    auto gradient_targets = nn::prepare_targets(
        frozen_model.atac_binding, frozen_model.atac_binding.canonical_ids(),
        gradient_batch.values.device());
    auto gradient_request = nn::query(gradient_view, gradient_targets, coordinates);
    const auto frozen_base = snapshot_group(
        frozen_model.parameters, frozen_model.base_group.id());
    const auto frozen_head = snapshot_group(
        frozen_model.parameters, frozen_model.atac_group.id());
    torch::optim::SGD frozen_optimizer(frozen_model.parameters.torch_parameters(),
                                      torch::optim::SGDOptions(0.01));
    auto atac = nn::predict(frozen_model, gradient_batch, gradient_request);
    atac.values.square().sum().backward();
    assert(gradient_batch.values.grad().defined());
    assert(gradient_batch.values.grad().abs().sum().item<double>() > 0.0);
    auto hidden_gradient = gradient_batch.values.grad().select(1, 0);
    assert(hidden_gradient.abs().sum().item<double>() == 0.0);
    bool readout_has_gradient = false;
    for (const auto& slot : frozen_model.parameters.layout()) {
        const auto parameter = frozen_model.parameters.get(slot.id);
        if (slot.group_id == frozen_model.base_group.id()) {
            assert(!parameter.grad().defined());
        } else if (slot.group_id == frozen_model.atac_group.id()) {
            readout_has_gradient = readout_has_gradient || parameter.grad().defined();
        }
    }
    assert(readout_has_gradient);
    frozen_optimizer.step();
    assert_group_unchanged(frozen_model.parameters, frozen_base);
    assert_group_changed(frozen_model.parameters, frozen_head);

    frozen_model.base_group.set_trainable(true);
    frozen_optimizer.zero_grad();
    const auto joint_base = snapshot_group(
        frozen_model.parameters, frozen_model.base_group.id());
    const auto joint_head = snapshot_group(
        frozen_model.parameters, frozen_model.atac_group.id());
    auto joint_prediction = nn::predict(frozen_model, gradient_batch, gradient_request);
    joint_prediction.values.square().sum().backward();
    frozen_optimizer.step();
    assert_group_changed(frozen_model.parameters, joint_base);
    assert_group_changed(frozen_model.parameters, joint_head);
    bool base_has_gradient = false;
    for (const auto& slot : frozen_model.parameters.layout()) {
        if (slot.group_id == frozen_model.base_group.id()) {
            base_has_gradient = base_has_gradient
                || frozen_model.parameters.get(slot.id).grad().defined();
        }
    }
    assert(base_has_gradient);
}

void event_shift_order_groups_and_queries() {
    at::manual_seed(101);
    auto events = binding({"event-A", "event-B", "event-C"},
                          {2, 7, 11}, "event", 14);
    auto peaks = binding({"peak-A", "peak-B"}, {1, 5}, "peak", 8);
    ex::EventHistoryReferenceModel model(events, peaks, 8, 12, 2);
    const auto batch = event_batch();
    const auto result = model(batch);

    assert(result.target_ids == batch.event_ids);
    assert((result.context_offsets == std::vector<std::int64_t>{0, 1, 4, 5, 6, 9}));
    assert((result.target_positions == std::vector<std::int64_t>{1, 2, 3, 6, 7, 8}));
    assert(result.logits.size(0) == 6);
    const std::vector<std::int64_t> expected_target_classes{0, 1, 2, 0, 0, 1};
    for (std::size_t i = 0; i < expected_target_classes.size(); ++i) {
        assert(result.target_class_ids[i].item<std::int64_t>()
            == expected_target_classes[i]);
    }
    assert(result.event_coordinates.row_offsets == batch.event_offsets);
    assert(result.context_order.identities().size() == 9);
    const auto& context_ids = result.context_order.identities();
    // History A,B,C is teacher-forced as [START,A,B] with labels [A,B,C].
    assert(context_ids[1].start);
    assert(context_ids[2] == ex::HistoryTokenIdentity::event("event-A"));
    assert(context_ids[3] == ex::HistoryTokenIdentity::event("event-B"));
    assert(result.target_ids[1] == "event-B");
    assert(context_ids[2].event_id != result.target_ids[1]);
    // The final history has two distinct ordered occurrences of event-A.
    assert(context_ids[7] == ex::HistoryTokenIdentity::event("event-A"));
    assert(context_ids[8] == ex::HistoryTokenIdentity::event("event-A"));

    ex::EventBatch all_empty{{}, {0, 0, 0}, {"empty-0", "empty-1"}, 42};
    const auto empty_result = model(all_empty);
    assert(empty_result.logits.sizes().vec() == std::vector<std::int64_t>({0, 3}));
    assert(empty_result.context_offsets == std::vector<std::int64_t>({0, 1, 2}));
    assert(!ex::event_next_event_cross_entropy(
        empty_result.logits, empty_result.target_class_ids));

    // A later teacher-forced context token cannot affect predictions at earlier
    // positions in the same declared history.
    ex::EventBatch causal_a{{"event-A", "event-B", "event-C"},
        {0, 3}, {"one"}, 43};
    ex::EventBatch causal_b{{"event-A", "event-C", "event-C"},
        {0, 3}, {"one"}, 43};
    const auto predictions_a = model(causal_a).logits;
    const auto predictions_b = model(causal_b).logits;
    assert_close(predictions_a.narrow(0, 0, 2), predictions_b.narrow(0, 0, 2));

    // The last event is a target only; it is outside the shifted input prefix.
    ex::EventBatch label_shift{{"event-A", "event-B", "event-A"},
        {0, 3}, {"one"}, 43};
    const auto shifted_predictions = model(label_shift).logits;
    assert_close(predictions_a, shifted_predictions);

    // Reordering packed histories reorders their predictions without cross-group influence.
    ex::EventBatch ordered{{"event-A", "event-B", "event-C", "event-A"},
        {0, 2, 4}, {"left", "right"}, 44};
    ex::EventBatch swapped{{"event-C", "event-A", "event-A", "event-B"},
        {0, 2, 4}, {"right", "left"}, 44};
    const auto ordered_logits = model(ordered).logits;
    const auto swapped_logits = model(swapped).logits;
    assert_close(ordered_logits.narrow(0, 0, 2), swapped_logits.narrow(0, 2, 2));
    assert_close(ordered_logits.narrow(0, 2, 2), swapped_logits.narrow(0, 0, 2));

    auto event_view = nn::readout(nn::Identity{}, &ex::EventResult::logits,
                                  model.event_binding);
    auto event_targets = nn::prepare_targets(
        model.event_binding, model.event_binding.canonical_ids(), at::Device(at::kCPU));
    nn::ObservationCoordinates<std::string> event_coordinates{
        batch.sample_ids, batch.event_offsets};
    auto event_request = nn::query(event_view, event_targets, event_coordinates);
    const auto groups = static_cast<std::int64_t>(batch.sample_ids.size());
    nn::ObservationCoordinates<std::string> history_coordinates{
        batch.sample_ids, ex::identity_offsets(groups)};
    auto atac_view = nn::readout(model.atac_readout, &ex::EventResult::history_features,
                                 model.atac_binding);
    auto atac_targets = nn::prepare_targets(
        model.atac_binding, model.atac_binding.canonical_ids(), at::Device(at::kCPU));
    auto atac_request = nn::query(atac_view, atac_targets, history_coordinates);
    int calls = 0;
    auto both = nn::predict(CountedEventModel{&model, &calls}, batch, nn::Run{},
                            event_request, atac_request);
    static_assert(std::tuple_size_v<decltype(both)> == 2);
    assert(calls == 1);
    assert(std::get<0>(both).values.sizes().vec() == std::vector<std::int64_t>({6, 3}));
    assert(std::get<1>(both).values.sizes().vec() == std::vector<std::int64_t>({5, 2}));

    model.base_group.set_trainable(false);
    auto frozen_view = nn::readout(model.atac_readout,
        &ex::EventResult::history_features, model.atac_binding);
    auto frozen_request = nn::query(frozen_view, atac_targets, history_coordinates);
    const auto frozen_base = snapshot_group(model.parameters, model.base_group.id());
    const auto frozen_head = snapshot_group(model.parameters, model.atac_group.id());
    torch::optim::SGD frozen_optimizer(
        model.parameters.torch_parameters(), torch::optim::SGDOptions(0.01));
    auto frozen_prediction = nn::predict(model, batch, frozen_request);
    frozen_prediction.values.square().sum().backward();
    bool readout_has_gradient = false;
    for (const auto& slot : model.parameters.layout()) {
        const auto parameter = model.parameters.get(slot.id);
        if (slot.group_id == model.base_group.id()) {
            assert(!parameter.grad().defined());
        } else if (slot.group_id == model.atac_group.id()) {
            readout_has_gradient = readout_has_gradient || parameter.grad().defined();
        }
    }
    assert(readout_has_gradient);
    frozen_optimizer.step();
    assert_group_unchanged(model.parameters, frozen_base);
    assert_group_changed(model.parameters, frozen_head);

    model.base_group.set_trainable(true);
    frozen_optimizer.zero_grad();
    const auto joint_base = snapshot_group(model.parameters, model.base_group.id());
    const auto joint_head = snapshot_group(model.parameters, model.atac_group.id());
    auto joint_prediction = nn::predict(model, batch, frozen_request);
    joint_prediction.values.square().sum().backward();
    frozen_optimizer.step();
    assert_group_changed(model.parameters, joint_base);
    assert_group_changed(model.parameters, joint_head);
    bool base_has_gradient = false;
    for (const auto& slot : model.parameters.layout()) {
        if (slot.group_id == model.base_group.id()) {
            base_has_gradient = base_has_gradient
                || model.parameters.get(slot.id).grad().defined();
        }
    }
    assert(base_has_gradient);
}

} // namespace

int main() {
    expression_binding_masking_and_queries();
    event_shift_order_groups_and_queries();
    return 0;
}
