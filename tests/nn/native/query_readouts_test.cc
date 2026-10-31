#ifdef NDEBUG
#undef NDEBUG
#endif
#include <ATen/ATen.h>
#include <Cellerator/biology/feature_binding.hh>
#include <Cellerator/nn/parameters.hh>
#include <Cellerator/nn/query.hh>
#include <Cellerator/nn/readouts.hh>

#include <cassert>
#include <array>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace bio = cellerator::biology;
namespace nn = cellerator::nn;

struct MoveOnlyBaseResult {
    at::Tensor regulatory;
    std::unique_ptr<int> marker;

    MoveOnlyBaseResult(at::Tensor values, int mark)
        : regulatory(std::move(values)), marker(std::make_unique<int>(mark)) {}
    MoveOnlyBaseResult(MoveOnlyBaseResult&&) noexcept = default;
    MoveOnlyBaseResult& operator=(MoveOnlyBaseResult&&) noexcept = default;
    MoveOnlyBaseResult(const MoveOnlyBaseResult&) = delete;
    MoveOnlyBaseResult& operator=(const MoveOnlyBaseResult&) = delete;
};

struct BaseModel {
    int* calls;
    MoveOnlyBaseResult operator()(const at::Tensor& input, nn::Run run) const {
        ++*calls;
        return {input * (run.training ? 3.0 : 2.0), *calls};
    }
};

struct FrozenBaseModel {
    nn::Linear* encoder;
    int* calls;
    MoveOnlyBaseResult operator()(const at::Tensor& input, nn::Run run) const {
        ++*calls;
        return {(*encoder)(input, run), *calls};
    }
};

struct SelectedHead {
    int* query_calls;
    int* full_calls;

    template<class Targets>
    at::Tensor query(const at::Tensor& source,
                     const Targets& targets,
                     const nn::ObservationCoordinates<std::string>&,
                     nn::Run) const {
        ++*query_calls;
        if constexpr (requires { targets.model_index_tensor(); }) {
            return source.index_select(-1, targets.model_index_tensor());
        } else {
            auto columns = at::tensor(targets.model_indices(),
                at::TensorOptions().dtype(at::kLong).device(source.device()));
            return source.index_select(-1, columns);
        }
    }

    at::Tensor operator()(const at::Tensor& source, nn::Run) const {
        ++*full_calls;
        return source;
    }
};

struct FullHead {
    int* calls;
    at::Tensor operator()(const at::Tensor& source, nn::Run) const {
        ++*calls;
        return source.slice(-1, 0, 3);
    }
};

struct UnrequestedHead {
    int* calls;
    at::Tensor operator()(const at::Tensor& source, nn::Run) const {
        ++*calls;
        return source;
    }
};

struct BorrowedViewBinding {
    std::vector<int> ids{11};
    const std::vector<int>& canonical_ids() const noexcept { return ids; }
};

struct BorrowedViewTargets {
    std::vector<int> ids{11};
    std::vector<std::int64_t> rows{0};
    const std::vector<int>& canonical_ids() const noexcept { return ids; }
    const std::vector<int>& source_ids() const noexcept { return ids; }
    const std::vector<std::int64_t>& model_indices() const noexcept { return rows; }
    const std::vector<std::int64_t>& output_columns() const noexcept { return rows; }
};

struct OwnedVectorResult {
    std::vector<int> data;
    std::unique_ptr<int> marker;
    OwnedVectorResult(std::vector<int> values, int mark)
        : data(std::move(values)), marker(std::make_unique<int>(mark)) {}
    OwnedVectorResult(OwnedVectorResult&&) noexcept = default;
    OwnedVectorResult& operator=(OwnedVectorResult&&) noexcept = default;
    OwnedVectorResult(const OwnedVectorResult&) = delete;
    OwnedVectorResult& operator=(const OwnedVectorResult&) = delete;
};

struct OwnedVectorModel {
    OwnedVectorResult operator()(int) const { return {{41, 42}, 1}; }
};

struct SpanHead {
    template<class Source, class Targets, class Coordinates>
    std::span<const int> query(const Source& source, const Targets&,
                               const Coordinates&) const {
        return {source.data(), static_cast<std::size_t>(source.size())};
    }
};

struct FullVectorOutput {
    std::vector<int> data;
    template<class Columns>
    std::span<const int> gather_feature_columns(const Columns&) && {
        return data;
    }
};

struct FullVectorHead {
    FullVectorOutput operator()(const std::vector<int>& source) const {
        return {source};
    }
};

struct ArraySelector {
    template<class Result, class Input>
    std::array<int, 2> operator()(const Result&, const Input&) const {
        return {51, 52};
    }
};

struct InputSpanModel {
    std::span<const int> operator()(const std::vector<int>& input) const {
        return input;
    }
};

struct CopyingVectorConditioner {
    std::vector<int> operator()(const std::vector<int>& input, int) const {
        return input;
    }
};

struct BorrowingVectorConditioner {
    const std::vector<int>& operator()(const std::vector<int>& input, int) const {
        return input;
    }
};

struct ReturningSpanConditioner {
    std::span<const int> operator()(const std::vector<int>& input, int) const {
        return input;
    }
};

struct SpanInputModel {
    std::span<const int> operator()(std::span<const int> input) const {
        return input;
    }
};

struct ModelSpan {
    std::vector<int> data;
    std::span<const int> operator()(int) { return data; }
};

struct IdentitySpanSelector {
    std::span<const int> operator()(const std::span<const int>& values) const {
        return values;
    }
};

int main() {
    bio::FeatureBindingOptions binding_options;
    binding_options.domain = "gene";
    binding_options.model_capacity = 10;
    auto binding = bio::FeatureBinding<std::string>::from_ids(
        {"gene-a", "gene-b", "gene-c"}, {9, 2, 7}, binding_options);
    auto prepared_requested = nn::prepare_targets(
        binding, std::vector<std::string>{"gene-c", "gene-a"}, at::Device(at::kCPU));
    auto prepared_first_only = nn::prepare_targets(
        binding, std::vector<std::string>{"gene-a"}, at::Device(at::kCPU));

    int base_calls = 0;
    int selected_queries = 0;
    int selected_full = 0;
    int full_calls = 0;
    int unrequested_calls = 0;
    BaseModel model{&base_calls};
    SelectedHead selected_head{&selected_queries, &selected_full};
    FullHead full_head{&full_calls};
    UnrequestedHead unrequested_head{&unrequested_calls};
    auto selected_view = nn::readout(selected_head, &MoveOnlyBaseResult::regulatory, binding);
    auto full_view = nn::readout(full_head, &MoveOnlyBaseResult::regulatory, binding);
    auto ignored_view = nn::readout(unrequested_head, &MoveOnlyBaseResult::regulatory, binding);
    (void)ignored_view;
    nn::ObservationCoordinates<std::string> observations{{"cell-b", "cell-a"}, {}};
    nn::ObservationCoordinates<std::string> one_observation{{"cell-b", "cell-a"}, {}};
    auto selected_request = nn::query(selected_view, prepared_requested, observations);
    auto full_request = nn::query(full_view, prepared_requested, one_observation);

    auto input = at::arange(20, at::TensorOptions().dtype(at::kFloat))
        .reshape({2, 10}).set_requires_grad(true);
    auto outputs = nn::predict(model, input, nn::Run{}, selected_request, full_request);
    static_assert(std::tuple_size_v<decltype(outputs)> == 2);
    assert(std::get<0>(outputs).payload.lifetime_owner.get() ==
           std::get<1>(outputs).payload.lifetime_owner.get());
    assert(base_calls == 1);
    assert(selected_queries == 1 && selected_full == 0);
    assert(full_calls == 1 && unrequested_calls == 0);
    // The query-capable head indexes model rows {7,9}; the ordinary head gathers
    // requested output columns {2,0} after computing its complete output.
    at::Tensor expected_selected = (input * 2.0).index_select(
        -1, at::tensor({7, 9}, at::TensorOptions().dtype(at::kLong)));
    at::Tensor expected_full = (input * 2.0).slice(-1, 0, 3).index_select(
        -1, at::tensor({2, 0}, at::TensorOptions().dtype(at::kLong)));
    assert(at::allclose(std::get<0>(outputs).values, expected_selected));
    assert(at::allclose(std::get<1>(outputs).values, expected_full));
    (std::get<0>(outputs).values.sum() + std::get<1>(outputs).values.sum()).backward();
    assert(input.grad().defined() && at::any(input.grad() != 0).item<bool>());

    auto training = nn::predict(model, input.detach(), nn::Run{true},
                                nn::query(selected_view, prepared_first_only, observations));
    assert(base_calls == 2);
    assert(training.values.sizes() == at::IntArrayRef({2, 1}));
    assert(training.payload.lifetime_owner);
    assert(training.payload.feature_ids.size() == 1 &&
           training.payload.feature_ids.front() == "gene-a");
    assert(at::allclose(training.values,
        (input.detach() * 3.0).index_select(
            -1, at::tensor({9}, at::TensorOptions().dtype(at::kLong)))));

    // Explicit conditioning changes the numerical input seen by the base.
    auto conditioner = [](const at::Tensor& source, const at::Tensor& condition) {
        return source + condition;
    };
    auto condition = at::ones({}, input.options());
    auto invocation = nn::conditioned(model, input, conditioner, condition);
    auto conditioned_output = nn::predict(invocation, nn::Run{},
        nn::query(selected_view, prepared_first_only, observations));
    assert(base_calls == 3);
    auto expected_conditioned = ((input + condition) * 2.0).index_select(
        -1, at::tensor({9}, at::TensorOptions().dtype(at::kLong)));
    assert(at::allclose(conditioned_output.values, expected_conditioned));

    auto retained_metadata = nn::predict(model, input.detach(), nn::Run{},
        nn::query(selected_view,
            nn::prepare_targets(binding, std::vector<std::string>{"gene-a"},
                                at::Device(at::kCPU)),
            nn::ObservationCoordinates<std::string>{{"obs-a", "obs-b"}, {}}));
    assert(retained_metadata.payload.lifetime_owner);
    assert(retained_metadata.payload.coordinates.get().sample_ids.front() == "obs-a");

    // Batched query predictions retain one move-only base result when a head
    // returns a primitive view into one of its members.
    BorrowedViewBinding borrowed_binding;
    BorrowedViewTargets borrowed_targets;
    nn::ObservationCoordinates<int> borrowed_coordinates{{7}, {}};
    auto borrowed_view = nn::readout(
        SpanHead{}, &OwnedVectorResult::data, borrowed_binding);
    auto borrowed_prediction = nn::predict(OwnedVectorModel{}, 0, nn::Run{},
        nn::query(borrowed_view, borrowed_targets, borrowed_coordinates));
    assert(borrowed_prediction.values.size() == 2);
    assert(borrowed_prediction.values[0] == 41 && borrowed_prediction.values[1] == 42);

    // Full user-defined head results must also outlive a gathered borrowed
    // payload; the full result is retained in the existing typed frame.
    auto full_vector_view = nn::readout(
        FullVectorHead{}, &OwnedVectorResult::data, borrowed_binding);
    auto gathered_prediction = nn::predict(OwnedVectorModel{}, 0, nn::Run{},
        nn::query(full_vector_view, borrowed_targets, borrowed_coordinates));
    assert(gathered_prediction.values.size() == 2);
    assert(gathered_prediction.values[0] == 41 && gathered_prediction.values[1] == 42);

    // Selector-produced array values are retained beside the base result so
    // a decoder may safely return a span into that value.
    auto array_view = nn::readout(SpanHead{}, ArraySelector{}, borrowed_binding);
    auto array_prediction = nn::predict(OwnedVectorModel{}, 0, nn::Run{},
        nn::query(array_view, borrowed_targets, borrowed_coordinates));
    assert(array_prediction.values.size() == 2);
    assert(array_prediction.values[0] == 51 && array_prediction.values[1] == 52);

    // Rvalue inputs and models are held in the same frame when the base result
    // borrows from them.
    auto input_view = nn::readout(SpanHead{}, IdentitySpanSelector{}, borrowed_binding);
    auto temporary_input_prediction = nn::predict(InputSpanModel{},
        std::vector<int>{61, 62}, nn::Run{},
        nn::query(input_view, borrowed_targets, borrowed_coordinates));
    assert(temporary_input_prediction.values[0] == 61 &&
           temporary_input_prediction.values[1] == 62);

    auto temporary_model_prediction = nn::predict(
        ModelSpan{{71, 72}}, 0, nn::Run{},
        nn::query(input_view, borrowed_targets, borrowed_coordinates));
    assert(temporary_model_prediction.values[0] == 71 &&
           temporary_model_prediction.values[1] == 72);

    auto conditioned_value_prediction = [&] {
        auto vector_invocation = nn::conditioned(
            InputSpanModel{}, std::vector<int>{81, 82},
            CopyingVectorConditioner{}, 0);
        return nn::predict(vector_invocation, nn::Run{},
            nn::query(input_view, borrowed_targets, borrowed_coordinates));
    }();
    assert(conditioned_value_prediction.values[0] == 81 &&
           conditioned_value_prediction.values[1] == 82);

    auto conditioned_borrow_prediction = nn::predict(
        nn::conditioned(InputSpanModel{}, std::vector<int>{91, 92},
                        BorrowingVectorConditioner{}, 0),
        nn::Run{}, nn::query(input_view, borrowed_targets, borrowed_coordinates));
    assert(conditioned_borrow_prediction.values[0] == 91 &&
           conditioned_borrow_prediction.values[1] == 92);

    // A conditioner may return a view by value. The rvalue invocation itself
    // remains retained because the returned span still borrows its input.
    auto conditioned_span_prediction = nn::predict(
        nn::conditioned(SpanInputModel{}, std::vector<int>{101, 102},
                        ReturningSpanConditioner{}, 0),
        nn::Run{}, nn::query(input_view, borrowed_targets, borrowed_coordinates));
    assert(conditioned_span_prediction.values[0] == 101 &&
           conditioned_span_prediction.values[1] == 102);

    nn::Parameters frozen_parameters(at::Device(at::kCPU), at::kFloat);
    auto frozen_group = frozen_parameters.group();
    frozen_group.set_trainable(false);
    nn::Linear frozen_encoder(frozen_group, 10, 10, false);
    assert(!frozen_encoder.weight().requires_grad());
    int frozen_calls = 0;
    FrozenBaseModel frozen_model{&frozen_encoder, &frozen_calls};
    auto differentiable_input = at::ones({1, 10},
        at::TensorOptions().dtype(at::kFloat)).set_requires_grad(true);
    auto frozen_result = nn::predict(frozen_model, differentiable_input, nn::Run{},
        nn::query(selected_view, prepared_first_only, observations));
    frozen_result.values.sum().backward();
    assert(frozen_calls == 1);
    assert(differentiable_input.grad().defined());
    assert(!frozen_encoder.weight().requires_grad());

    // The standard feature readout composes a target embedding, cross-attention,
    // and a conventional output projection using the same explicit parameter group.
    nn::Parameters parameters(at::Device(at::kCPU), at::kFloat);
    auto group = parameters.group();
    nn::FeatureReadout<std::string> feature_readout(group, binding, 2, 1, 1);
    auto context = at::randn({2, 3, 2}, at::TensorOptions().dtype(at::kFloat))
        .set_requires_grad(true);
    auto all_targets = binding.prepare(binding.canonical_ids());
    auto selected_target = binding.prepare(binding.canonical_ids(), {"gene-b"});
    auto all_values = feature_readout(context);
    auto selected_values = feature_readout.query(context, selected_target, observations);
    auto device_targets = nn::prepare_targets(
        binding, std::vector<std::string>{"gene-b"}, at::Device(at::kCPU));
    auto prepared_values = feature_readout.query(context, device_targets, observations);
    assert(all_values.sizes() == at::IntArrayRef({2, 3}));
    assert(selected_values.sizes() == at::IntArrayRef({2, 1}));
    assert(at::allclose(selected_values, prepared_values));
    (all_values.sum() + selected_values.sum() + prepared_values.sum()).backward();
    assert(context.grad().defined());
}
