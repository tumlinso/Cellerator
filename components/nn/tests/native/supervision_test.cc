#ifdef NDEBUG
#undef NDEBUG
#endif
#include <ATen/ATen.h>
#include <Cellerator/biology/feature_binding.hh>
#include <Cellerator/nn/supervision.hh>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace bio = cellerator::biology;
namespace nn = cellerator::nn;

at::Tensor floats(std::vector<float> values, std::vector<std::int64_t> shape,
                  bool requires_grad = false) {
    auto result = at::tensor(values, at::TensorOptions().dtype(at::kFloat)).reshape(shape);
    return result.set_requires_grad(requires_grad);
}

int main() {
    auto binding = bio::FeatureBinding<std::string>::from_ids(
        {"gene-a", "gene-b", "gene-c"}, {8, 3, 5});
    auto alignment = nn::prepare_targets(
        binding, std::vector<std::string>{"gene-c", "gene-a"}, at::Device(at::kCPU));
    nn::ObservationCoordinates<std::string> prediction_coordinates{
        {"cell-b", "cell-a"}, {}};
    nn::ObservationCoordinates<std::string> target_coordinates{
        {"cell-a", "cell-b"}, {}};
    at::Tensor prediction_values = floats({10, 20, 30, 40}, {2, 2}, true);
    using Payload = nn::QueryPayload<std::string,
                                      nn::ObservationCoordinates<std::string>>;
    const std::vector<std::string> requested_ids{"gene-c", "gene-a"};
    nn::Prediction<at::Tensor, Payload> prediction{
        prediction_values, Payload{std::span<const std::string>(requested_ids),
                                   std::cref(prediction_coordinates), {}}};
    at::Tensor targets = floats({1, 2, 3, 4, 5, 6}, {2, 3}, true);

    auto [aligned_predictions, aligned_targets] = nn::align_prediction_targets(
        prediction, targets, alignment, target_coordinates);
    assert(at::allclose(aligned_predictions, prediction_values));
    assert(at::allclose(aligned_targets, floats({6, 4, 3, 1}, {2, 2})));
    (aligned_predictions.sum() + aligned_targets.sum()).backward();
    assert(prediction_values.grad().defined() && targets.grad().defined());

    nn::ObservationCoordinates<std::string> repeated_predictions{
        {"cell-x", "cell-x"}, {}};
    nn::ObservationCoordinates<std::string> repeated_targets{
        {"cell-x", "cell-x"}, {}};
    const std::vector<std::string> one_feature_ids{"gene-a"};
    nn::Prediction<at::Tensor, Payload> repeated_prediction{
        at::zeros({2, 1}), Payload{std::span<const std::string>(one_feature_ids),
                                   std::cref(repeated_predictions), {}}};
    auto one_feature = binding.prepare(binding.canonical_ids(), {"gene-a"});
    at::Tensor repeated_values = floats({1, 2, 3, 4, 5, 6}, {2, 3});
    bool ambiguous = false;
    try {
        (void)nn::align_prediction_targets(repeated_prediction, repeated_values,
                                           one_feature, repeated_targets);
    } catch (const std::invalid_argument&) {
        ambiguous = true;
    }
    assert(ambiguous);
    auto [mapped_predictions, mapped_targets] = nn::align_prediction_targets(
        repeated_prediction, repeated_values, one_feature, repeated_targets,
        std::vector<std::int64_t>{1, 0});
    assert(mapped_predictions.sizes() == at::IntArrayRef({2, 1}));
    assert(at::allclose(mapped_targets.squeeze(-1), floats({4, 1}, {2})));

    auto middle_feature = nn::prepare_targets(
        binding, std::vector<std::string>{"gene-b"}, at::Device(at::kCPU));
    nn::ObservationCoordinates<std::string> packed_predictions{
        {"window-a", "window-b"}, {0, 1, 3}};
    nn::ObservationCoordinates<std::string> packed_targets{
        {"window-b", "window-a"}, {0, 2, 3}};
    const std::vector<std::string> middle_feature_ids{"gene-b"};
    nn::Prediction<at::Tensor, Payload> packed_middle_prediction{
        at::zeros({3, 1}), Payload{std::span<const std::string>(middle_feature_ids),
                                  std::cref(packed_predictions), {}}};
    auto [packed_values, packed_aligned] = nn::align_prediction_targets(
        packed_middle_prediction, floats({1, 10, 100, 2, 20, 200, 3, 30, 300}, {3, 3}),
        middle_feature, packed_targets);
    assert(packed_values.sizes() == at::IntArrayRef({3, 1}));
    assert(at::allclose(packed_aligned.squeeze(-1), floats({30, 10, 20}, {3})));

    auto class_columns = nn::categorical_target_columns(
        prediction, std::vector<std::string>{"gene-a", "gene-c", "gene-a"});
    assert(at::equal(class_columns,
        at::tensor(std::vector<std::int64_t>{1, 0, 1},
                   at::TensorOptions().dtype(at::kLong))));
    bool unknown_class = false;
    try {
        (void)nn::categorical_target_columns(
            prediction, std::vector<std::string>{"gene-z"});
    } catch (const std::out_of_range&) {
        unknown_class = true;
    }
    assert(unknown_class);

    at::Tensor losses = floats({1, 3, 5, 7, 10, 14}, {6}, true);
    nn::ObservationCoordinates<std::string> windows{
        {"a-window-1", "a-window-2", "b-window-1"}, {0, 2, 4, 6}};
    auto [cell_mean, cell_count] = nn::reduce_prediction_loss(
        losses, nn::SupervisionUnit::cell, windows,
        std::vector<std::string>{"cell-a", "cell-a", "cell-b"});
    assert(std::abs(cell_mean.item<float>() - 8.0f) < 1e-6f);
    assert(cell_count.item<std::int64_t>() == 2);
    cell_mean.backward();
    assert(losses.grad().defined() && at::all(losses.grad() != 0).item<bool>());

    auto masked_nan_losses = floats(
        {std::numeric_limits<float>::quiet_NaN(), 3, 5, 7, 10, 14}, {6}, true);
    auto empty_mask = at::zeros_like(masked_nan_losses,
                                     masked_nan_losses.options().dtype(at::kBool));
    auto [empty_mean, empty_count] = nn::reduce_prediction_loss(
        masked_nan_losses, nn::SupervisionUnit::observation, windows, empty_mask);
    assert(empty_mean.item<float>() == 0.0f);
    assert(empty_count.item<std::int64_t>() == 0);
    empty_mean.backward();
    assert(at::isfinite(masked_nan_losses.grad()).all().item<bool>());

    at::Tensor no_losses = at::empty({0},
        at::TensorOptions().dtype(at::kFloat)).set_requires_grad(true);
    nn::ObservationCoordinates<std::string> empty_window{{"empty-window"}, {0, 0}};
    auto [no_loss_mean, no_loss_count] = nn::reduce_prediction_loss(
        no_losses, nn::SupervisionUnit::cell, empty_window,
        std::vector<std::string>{"empty-cell"});
    assert(no_loss_mean.item<float>() == 0.0f && no_loss_count.item<std::int64_t>() == 0);
    no_loss_mean.backward();
    assert(no_losses.grad().defined() && no_losses.grad().sizes() == no_losses.sizes());

    at::Tensor rows = floats({1, 2, 3, 4, 5, 6}, {3, 2}, true);
    const std::vector<std::int64_t> offsets{0, 2, 2, 3};
    auto means = nn::mean_by_group(rows, offsets);
    auto lasts = nn::last_by_group(rows, offsets);
    assert(at::allclose(means, floats({2, 3, 0, 0, 5, 6}, {3, 2})));
    assert(at::allclose(lasts, floats({3, 4, 0, 0, 5, 6}, {3, 2})));
}
