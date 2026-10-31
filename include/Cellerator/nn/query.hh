#pragma once

#include <ATen/ATen.h>
#include <Cellerator/biology/feature_binding.hh>
#include <Cellerator/nn/types.hh>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace cellerator::nn {

// Row coordinates describe the observations represented by a packed prediction.
// They are descriptive keys, never numerical conditioning by implication.
template<class ObservationId>
struct ObservationCoordinates {
    std::vector<ObservationId> sample_ids;
    std::vector<std::int64_t> row_offsets;

    friend bool operator==(const ObservationCoordinates&, const ObservationCoordinates&) = default;
};

// Boundary metadata is optional. The Values type remains exactly the head's
// result type, so ordinary native models can return any type they need.
template<class Values, class Payload>
struct Prediction {
    Values values;
    Payload payload;
};

template<class FeatureId, class Coordinates>
struct QueryPayload {
    // These views borrow the prepared target and coordinate declarations.
    // For the batched `predict` path, lifetime_owner retains one typed frame
    // containing the base result and any owned model/input/request/selected
    // source values. Lvalue dependencies remain caller-owned; temporary query
    // requests are held by the frame. The direct Readout path has no owner.
    std::span<const FeatureId> feature_ids;
    std::reference_wrapper<const Coordinates> coordinates;
    std::shared_ptr<const void> lifetime_owner;
};

// A host-prepared target alignment plus immutable device indices. Building a
// PreparedTargets is an explicit boundary operation; repeated readout calls
// reuse its index buffers instead of rebuilding them from host IDs.
template<class FeatureId>
class PreparedTargets {
public:
    using identity_type = FeatureId;

    const biology::PreparedAlignment<FeatureId>& alignment() const noexcept {
        return alignment_;
    }
    const std::vector<FeatureId>& canonical_ids() const noexcept {
        return alignment_.canonical_ids();
    }
    const std::vector<FeatureId>& source_ids() const noexcept {
        return alignment_.source_ids();
    }
    const std::vector<std::int64_t>& model_indices() const noexcept {
        return alignment_.model_indices();
    }
    const std::vector<std::int64_t>& output_columns() const noexcept {
        return alignment_.output_columns();
    }
    const at::Tensor& model_index_tensor() const noexcept { return model_indices_; }
    const at::Tensor& output_column_tensor() const noexcept { return output_columns_; }
    at::Device device() const noexcept { return device_; }

private:
    PreparedTargets(biology::PreparedAlignment<FeatureId> alignment,
                    at::Tensor model_indices, at::Tensor output_columns,
                    at::Device device)
        : alignment_(std::move(alignment)), model_indices_(std::move(model_indices)),
          output_columns_(std::move(output_columns)), device_(device) {}

    biology::PreparedAlignment<FeatureId> alignment_;
    at::Tensor model_indices_;
    at::Tensor output_columns_;
    at::Device device_ = at::Device(at::kCPU);

    template<class Id>
    friend PreparedTargets<Id> prepare_targets(
        const biology::FeatureBinding<Id>&, const std::vector<Id>&, at::Device);
};

template<class FeatureId>
PreparedTargets<FeatureId> prepare_targets(
    const biology::FeatureBinding<FeatureId>& binding,
    const std::vector<FeatureId>& requested_ids,
    at::Device device) {
    auto alignment = binding.prepare(binding.canonical_ids(), requested_ids);
    const auto options = at::TensorOptions().dtype(at::kLong).device(device);
    auto model_indices = at::tensor(alignment.model_indices(), options);
    auto output_columns = at::tensor(alignment.output_columns(), options);
    const auto realized_device = model_indices.device();
    if (output_columns.device() != realized_device) {
        throw std::logic_error("prepared target index tensors landed on different devices");
    }
    return PreparedTargets<FeatureId>(std::move(alignment),
        std::move(model_indices), std::move(output_columns), realized_device);
}

namespace detail {

template<class T> struct is_reference_wrapper : std::false_type {};
template<class T> struct is_reference_wrapper<std::reference_wrapper<T>> : std::true_type {};

template<class T>
using held_t = std::conditional_t<
    std::is_lvalue_reference_v<T>,
    std::reference_wrapper<std::remove_reference_t<T>>,
    std::remove_cvref_t<T>>;

template<class T>
auto hold(T&& value) -> held_t<T&&> {
    if constexpr (std::is_lvalue_reference_v<T&&>) {
        return std::ref(value);
    } else {
        return std::forward<T>(value);
    }
}

template<class T>
decltype(auto) unhold(T&& value) {
    if constexpr (is_reference_wrapper<std::remove_cvref_t<T>>::value) {
        return value.get();
    } else {
        return std::forward<T>(value);
    }
}

// Forward owned components as rvalues while borrowed lvalues remain lvalues.
// PredictionFrame uses this for its model, preserving callable overload
// selection while keeping owned model storage at a stable address.
template<class T>
decltype(auto) forward_held(T& value) {
    if constexpr (is_reference_wrapper<std::remove_cvref_t<T>>::value) {
        return value.get();
    } else {
        return std::move(value);
    }
}

template<class Callable, class... Args>
decltype(auto) invoke_with_run(Callable&& callable, Run run, Args&&... args) {
    if constexpr (std::is_invocable_v<Callable, Args..., Run>) {
        return std::invoke(std::forward<Callable>(callable),
                           std::forward<Args>(args)..., run);
    } else {
        return std::invoke(std::forward<Callable>(callable),
                           std::forward<Args>(args)...);
    }
}

template<class Model, class Input>
decltype(auto) invoke_model(Model&& model, Input&& input, Run run) {
    if constexpr (std::is_invocable_v<Model, Input, Run>) {
        return std::invoke(std::forward<Model>(model), std::forward<Input>(input), run);
    } else {
        return std::invoke(std::forward<Model>(model), std::forward<Input>(input));
    }
}

template<class Conditioner, class Input, class Condition>
decltype(auto) invoke_conditioner(Conditioner&& conditioner, Input&& input,
                                  Condition&& condition, Run run) {
    if constexpr (std::is_invocable_v<Conditioner, Input, Condition, Run>) {
        return std::invoke(std::forward<Conditioner>(conditioner),
                           std::forward<Input>(input),
                           std::forward<Condition>(condition), run);
    } else {
        return std::invoke(std::forward<Conditioner>(conditioner),
                           std::forward<Input>(input),
                           std::forward<Condition>(condition));
    }
}

template<class Selector, class Result, class Input>
decltype(auto) select_source(Selector&& selector, Result&& result, Input&& input) {
    if constexpr (std::is_invocable_v<Selector, Result, Input>) {
        return std::invoke(std::forward<Selector>(selector),
                           std::forward<Result>(result), std::forward<Input>(input));
    } else {
        return std::invoke(std::forward<Selector>(selector), std::forward<Result>(result));
    }
}

template<class Binding, class Targets>
void validate_readout_targets(const Binding& binding, const Targets& targets) {
    if constexpr (requires { binding.validate_alignment(targets.alignment()); }) {
        binding.validate_alignment(targets.alignment());
    } else if constexpr (requires { binding.validate_alignment(targets); }) {
        binding.validate_alignment(targets);
    } else if constexpr (requires { binding.canonical_ids(); targets.source_ids(); }) {
        if (binding.canonical_ids() != targets.source_ids()) {
            throw std::invalid_argument(
                "prepared target source order does not match the readout binding");
        }
    }
}

inline at::Tensor index_tensor(const std::vector<std::int64_t>& indices,
                               const at::Tensor& values) {
    return at::tensor(indices,
        at::TensorOptions().dtype(at::kLong).device(values.device()));
}

template<class Values, class Targets>
decltype(auto) gather_full_output(Values&& values, const Targets& targets) {
    if constexpr (std::is_same_v<std::remove_cvref_t<Values>, at::Tensor>) {
        const auto& source_ids = targets.source_ids();
        if (values.dim() == 0 || values.size(-1) != static_cast<std::int64_t>(source_ids.size())) {
            throw std::invalid_argument(
                "full readout output axis does not match the prepared source order");
        }
        if constexpr (requires { targets.output_column_tensor(); targets.device(); }) {
            if (targets.device() != values.device()) {
                throw std::invalid_argument(
                    "prepared target output columns are on a different device than readout values");
            }
            return values.index_select(-1, targets.output_column_tensor());
        } else {
            // Host-only alignments stay available; repeated queries should
            // prepare device indices once with prepare_targets().
            return values.index_select(-1, index_tensor(targets.output_columns(), values));
        }
    } else if constexpr (requires { targets.gather(std::forward<Values>(values), -1); }) {
        return targets.gather(std::forward<Values>(values), -1);
    } else if constexpr (requires {
        std::forward<Values>(values).gather_feature_columns(targets.output_columns());
    }) {
        return std::forward<Values>(values).gather_feature_columns(targets.output_columns());
    } else {
        static_assert(std::is_same_v<std::remove_cvref_t<Values>, void>,
            "a full-head query must return a gatherable value; provide a query-capable head otherwise");
    }
}

struct NoFullOutput {};

template<class Decoder, class Source>
auto full_output_storage() {
    if constexpr (std::is_invocable_v<Decoder, Source, Run> ||
                  std::is_invocable_v<Decoder, Source>) {
        using full_type = decltype(invoke_with_run(
            std::declval<Decoder>(), Run{}, std::declval<Source>()));
        if constexpr (!std::is_same_v<std::remove_cvref_t<full_type>, at::Tensor>) {
            return std::optional<held_t<full_type>>{};
        } else {
            return NoFullOutput{};
        }
    } else {
        return NoFullOutput{};
    }
}

template<class Decoder, class Source, class Targets, class Coordinates, class RetainFull>
decltype(auto) invoke_head(Decoder&& decoder, Source&& source,
                           const Targets& targets, const Coordinates& coordinates,
                           Run run, RetainFull&& retain_full) {
    if constexpr (requires {
        decoder.query(source, targets, coordinates, run);
    }) {
        return decoder.query(std::forward<Source>(source), targets, coordinates, run);
    } else if constexpr (requires {
        decoder.query(source, targets, coordinates);
    }) {
        return decoder.query(std::forward<Source>(source), targets, coordinates);
    } else if constexpr (requires {
        decoder.query(source, targets, run);
    }) {
        return decoder.query(std::forward<Source>(source), targets, run);
    } else if constexpr (requires {
        decoder.query(source, targets);
    }) {
        return decoder.query(std::forward<Source>(source), targets);
    } else if constexpr (requires {
        decoder.query(source, targets.model_indices(), run);
    }) {
        return decoder.query(std::forward<Source>(source), targets.model_indices(), run);
    } else if constexpr (requires {
        decoder.query(source, targets.model_indices());
    }) {
        return decoder.query(std::forward<Source>(source), targets.model_indices());
    } else {
        decltype(auto) full = invoke_with_run(std::forward<Decoder>(decoder), run,
                                              std::forward<Source>(source));
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(full)>, at::Tensor>) {
            return gather_full_output(std::forward<decltype(full)>(full), targets);
        } else {
            // A user-defined gather can return a view into the full result.
            // Keep that result at a stable address in the same batch frame.
            return gather_full_output(
                retain_full(std::forward<decltype(full)>(full)), targets);
        }
    }
}

template<class View, class Source, class Targets, class Coordinates, class RetainFull>
auto evaluate_selected_request(View& view, Source&& source,
                               const Targets& targets, const Coordinates& coordinates,
                               Run run, std::shared_ptr<const void> owner,
                               RetainFull&& retain_full) {
    const auto& binding = unhold(view.binding_holder());
    validate_readout_targets(binding, targets);
    auto&& decoder = unhold(view.decoder_holder());
    decltype(auto) raw = invoke_head(decoder, std::forward<decltype(source)>(source),
                                     targets, coordinates, run,
                                     std::forward<RetainFull>(retain_full));
    using values_type = std::remove_cvref_t<decltype(raw)>;
    using ids_type = typename std::remove_cvref_t<
        decltype(targets.canonical_ids())>::value_type;
    using coordinates_type = std::remove_cvref_t<Coordinates>;
    using payload_type = QueryPayload<ids_type, coordinates_type>;
    return Prediction<values_type, payload_type>{
        std::forward<decltype(raw)>(raw),
        payload_type{std::span<const ids_type>(targets.canonical_ids()),
                     std::cref(coordinates), std::move(owner)}
    };
}

template<class Request>
concept query_request = requires(Request& request) {
    request.view_holder();
    request.targets_holder();
    request.coordinates_holder();
};

} // namespace detail

// A readout is a local composition: decoder, source selector, and biological
// binding stay explicit. Lvalue arguments are borrowed; rvalues are owned by
// the composite. A member pointer is an ordinary selector via std::invoke.
template<class DecoderHolder, class SelectorHolder, class BindingHolder>
class Readout {
public:
    Readout(DecoderHolder decoder, SelectorHolder selector, BindingHolder binding)
        : decoder_(std::move(decoder)), selector_(std::move(selector)),
          binding_(std::move(binding)) {}

    DecoderHolder& decoder_holder() noexcept { return decoder_; }
    const DecoderHolder& decoder_holder() const noexcept { return decoder_; }
    SelectorHolder& selector_holder() noexcept { return selector_; }
    const SelectorHolder& selector_holder() const noexcept { return selector_; }
    BindingHolder& binding_holder() noexcept { return binding_; }
    const BindingHolder& binding_holder() const noexcept { return binding_; }

    // This direct decoder call does not allocate a prediction owner. Any view
    // returned by the decoder must obey the caller's source/dependency lifetime.
    template<class Result, class Input>
    decltype(auto) operator()(Result&& result, Input&& input, Run run = {}) const {
        auto&& selected = detail::select_source(
            detail::unhold(selector_), std::forward<Result>(result), std::forward<Input>(input));
        return detail::invoke_with_run(detail::unhold(decoder_), run,
                                       std::forward<decltype(selected)>(selected));
    }

    template<class Result>
    decltype(auto) operator()(Result&& result, Run run = {}) const {
        auto&& selected = std::invoke(detail::unhold(selector_), std::forward<Result>(result));
        return detail::invoke_with_run(detail::unhold(decoder_), run,
                                       std::forward<decltype(selected)>(selected));
    }

private:
    DecoderHolder decoder_;
    SelectorHolder selector_;
    BindingHolder binding_;
};

template<class Decoder, class Selector, class Binding>
auto readout(Decoder&& decoder, Selector&& selector, Binding&& binding) {
    return Readout<detail::held_t<Decoder&&>, detail::held_t<Selector&&>,
                   detail::held_t<Binding&&>>(
        detail::hold(std::forward<Decoder>(decoder)),
        detail::hold(std::forward<Selector>(selector)),
        detail::hold(std::forward<Binding>(binding)));
}

template<class ViewHolder, class TargetsHolder, class CoordinatesHolder>
class QueryRequest {
public:
    QueryRequest(ViewHolder view, TargetsHolder targets, CoordinatesHolder coordinates)
        : view_(std::move(view)), targets_(std::move(targets)),
          coordinates_(std::move(coordinates)) {}

    ViewHolder& view_holder() noexcept { return view_; }
    const ViewHolder& view_holder() const noexcept { return view_; }
    TargetsHolder& targets_holder() noexcept { return targets_; }
    const TargetsHolder& targets_holder() const noexcept { return targets_; }
    CoordinatesHolder& coordinates_holder() noexcept { return coordinates_; }
    const CoordinatesHolder& coordinates_holder() const noexcept { return coordinates_; }

private:
    ViewHolder view_;
    TargetsHolder targets_;
    CoordinatesHolder coordinates_;
};

// Construct a typed request. This call performs no model or head work. Each
// lvalue argument is borrowed and each rvalue is owned by the request. A
// prediction made from a temporary request retains that request, so its owned
// targets and coordinates stay alive; any borrowed lvalue dependencies remain
// the caller's responsibility.
template<class View, class Targets, class Coordinates>
auto query(View&& view, Targets&& targets, Coordinates&& coordinates) {
    return QueryRequest<detail::held_t<View&&>, detail::held_t<Targets&&>,
                        detail::held_t<Coordinates&&>>(
        detail::hold(std::forward<View>(view)),
        detail::hold(std::forward<Targets>(targets)),
        detail::hold(std::forward<Coordinates>(coordinates)));
}

namespace detail {

template<class RequestHolder>
using request_ref_t = decltype(unhold(std::declval<RequestHolder&>()));

template<class RequestHolder>
using request_view_ref_t = decltype(unhold(
    std::declval<request_ref_t<RequestHolder>>().view_holder()));

template<class RequestHolder>
using request_selector_ref_t = decltype(unhold(
    std::declval<request_view_ref_t<RequestHolder>>().selector_holder()));

template<class RequestHolder>
using request_decoder_ref_t = decltype(unhold(
    std::declval<request_view_ref_t<RequestHolder>>().decoder_holder()));

template<class Holder>
using unheld_ref_t = decltype(unhold(std::declval<Holder&>()));

template<class RequestHolder, class ResultHolder, class InputHolder>
using selected_source_t = decltype(select_source(
    std::declval<request_selector_ref_t<RequestHolder>>(),
    std::declval<unheld_ref_t<ResultHolder>>(),
    std::declval<unheld_ref_t<InputHolder>>()));

template<class ModelHolder, class InputHolder>
using model_result_t = decltype(invoke_model(
    forward_held(std::declval<ModelHolder&>()),
    unhold(std::declval<InputHolder&>()), Run{}));

template<class RequestHolder, class ResultHolder, class InputHolder>
using selected_source_holder_t = held_t<selected_source_t<
    RequestHolder, ResultHolder, InputHolder>>;

template<class Holder>
decltype(auto) forward_source_holder(Holder& holder) {
    return forward_held(holder);
}

// One typed owner stores every lifetime participating in a batched query:
// owned model/input, the single base result, request values, and selector
// values. Every returned prediction shares the same lifetime token.
template<class ModelHolder, class InputHolder, class... RequestHolders>
class PredictionFrame {
    using raw_result_type = model_result_t<ModelHolder, InputHolder>;
    using result_holder_type = held_t<raw_result_type>;
    using source_holders_type = std::tuple<std::optional<selected_source_holder_t<
        RequestHolders, result_holder_type, InputHolder>>...>;
    using full_output_holders_type = std::tuple<decltype(full_output_storage<
        request_decoder_ref_t<RequestHolders>,
        decltype(forward_held(std::declval<selected_source_holder_t<
            RequestHolders, result_holder_type, InputHolder>&>()))>())...>;

public:
    PredictionFrame(ModelHolder model, InputHolder input,
                    std::tuple<RequestHolders...> requests,
                    std::shared_ptr<const void> lifetime_anchor = {})
        : lifetime_anchor_(std::move(lifetime_anchor)), model_(std::move(model)),
          input_(std::move(input)), requests_(std::move(requests)) {}

    void evaluate_base(Run run) {
        decltype(auto) raw_result = invoke_model(
            forward_held(model_), unhold(input_), run);
        result_.emplace(hold(std::forward<decltype(raw_result)>(raw_result)));
    }

    template<std::size_t I>
    auto evaluate_one(Run run, std::shared_ptr<const void> lifetime_owner) {
        capture_source<I>();
        auto&& request = unhold(std::get<I>(requests_));
        auto&& view = unhold(request.view_holder());
        auto&& targets = unhold(request.targets_holder());
        auto&& coordinates = unhold(request.coordinates_holder());
        auto&& source = forward_source_holder(*std::get<I>(sources_));
        auto retain_full = [this](auto&& full) -> decltype(auto) {
            auto& storage = std::get<I>(full_outputs_);
            if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(storage)>,
                                          NoFullOutput>) {
                storage.emplace(hold(std::forward<decltype(full)>(full)));
                return forward_held(*storage);
            } else {
                static_assert(std::is_same_v<decltype(full), void>,
                    "only non-Tensor full-head results require retained storage");
            }
        };
        return evaluate_selected_request(view, std::forward<decltype(source)>(source),
            targets, coordinates, run, std::move(lifetime_owner), retain_full);
    }

    template<std::size_t I = 0>
    auto evaluate_in_order(Run run,
                           const std::shared_ptr<const void>& lifetime_owner) {
        auto current = evaluate_one<I>(run, lifetime_owner);
        if constexpr (I + 1 == sizeof...(RequestHolders)) {
            return std::tuple{std::move(current)};
        } else {
            return std::tuple_cat(std::tuple{std::move(current)},
                                  evaluate_in_order<I + 1>(run, lifetime_owner));
        }
    }

private:
    template<std::size_t I>
    void capture_source() {
        auto&& request = unhold(std::get<I>(requests_));
        auto&& view = unhold(request.view_holder());
        auto&& selector = unhold(view.selector_holder());
        decltype(auto) selected = select_source(selector, unhold(*result_),
                                               unhold(input_));
        std::get<I>(sources_).emplace(
            hold(std::forward<decltype(selected)>(selected)));
    }

    // Declared first so any borrowed frame members are destroyed before the
    // upstream conditioned invocation that may back them.
    std::shared_ptr<const void> lifetime_anchor_;
    ModelHolder model_;
    InputHolder input_;
    std::optional<result_holder_type> result_;
    std::tuple<RequestHolders...> requests_;
    source_holders_type sources_;
    full_output_holders_type full_outputs_;
};

template<class ModelHolder, class InputHolder, class... RequestHolders>
auto make_prediction_frame(ModelHolder model, InputHolder input,
                           std::tuple<RequestHolders...> requests,
                           std::shared_ptr<const void> lifetime_anchor = {}) {
    using frame_type = PredictionFrame<ModelHolder, InputHolder, RequestHolders...>;
    return std::make_shared<frame_type>(std::move(model), std::move(input),
                                        std::move(requests),
                                        std::move(lifetime_anchor));
}

template<class Model, class Input, class... Requests>
auto predict_with_lifetime_anchor(std::shared_ptr<const void> lifetime_anchor,
                                 Model&& model, Input&& input, Run run,
                                 Requests&&... requests) {
    if constexpr (sizeof...(Requests) == 0) {
        return invoke_model(std::forward<Model>(model), input, run);
    } else {
        auto frame = make_prediction_frame(
            hold(std::forward<Model>(model)), hold(std::forward<Input>(input)),
            std::tuple<held_t<Requests&&>...>{
                hold(std::forward<Requests>(requests))...},
            std::move(lifetime_anchor));
        frame->evaluate_base(run);
        std::shared_ptr<const void> owner = frame;
        auto outputs = frame->template evaluate_in_order<0>(run, owner);
        if constexpr (sizeof...(Requests) == 1) {
            return std::get<0>(std::move(outputs));
        } else {
            return outputs;
        }
    }
}

} // namespace detail

// Invoke the base exactly once, then run each concrete query request against
// that shared result. The safe query path uses one shared typed lifetime frame
// per batched invocation, so predictions may return borrowed views into the
// model, input, base result, requests, or selector-produced sources. Lvalue
// dependencies remain borrowed and must outlive the predictions. One request
// returns its Prediction directly; multiple requests return a fixed tuple.
// This costs one shared frame allocation per batch, not one owner per result.
template<class Model, class Input, class... Requests>
auto predict(Model&& model, Input&& input, Run run, Requests&&... requests) {
    return detail::predict_with_lifetime_anchor({}, std::forward<Model>(model),
        std::forward<Input>(input), run, std::forward<Requests>(requests)...);
}

template<class Model, class Input, class... Requests>
requires (sizeof...(Requests) > 0 && (detail::query_request<Requests> && ...))
auto predict(Model&& model, Input&& input, Requests&&... requests) {
    return predict(std::forward<Model>(model), std::forward<Input>(input), Run{},
                   std::forward<Requests>(requests)...);
}

template<class Model, class Input, class... Requests>
requires (sizeof...(Requests) > 0 && (detail::query_request<Requests> && ...))
auto predict_query(Model&& model, Input&& input, Requests&&... requests) {
    return predict(std::forward<Model>(model), std::forward<Input>(input), Run{},
                   std::forward<Requests>(requests)...);
}

template<class Model, class Input, class Conditioner, class Condition>
class ConditionedInvocation {
public:
    ConditionedInvocation(Model model, Input input, Conditioner conditioner, Condition condition)
        : model_(std::move(model)), input_(std::move(input)),
          conditioner_(std::move(conditioner)), condition_(std::move(condition)) {}

    Model& model() noexcept { return model_; }
    Input& input() noexcept { return input_; }
    Conditioner& conditioner() noexcept { return conditioner_; }
    Condition& condition() noexcept { return condition_; }

private:
    Model model_;
    Input input_;
    Conditioner conditioner_;
    Condition condition_;
};

template<class Model, class Input, class Conditioner, class Condition>
auto conditioned(Model&& model, Input&& input, Conditioner&& conditioner, Condition&& condition) {
    return ConditionedInvocation<detail::held_t<Model&&>, detail::held_t<Input&&>,
                                 detail::held_t<Conditioner&&>, detail::held_t<Condition>>(
        detail::hold(std::forward<Model>(model)), detail::hold(std::forward<Input>(input)),
        detail::hold(std::forward<Conditioner>(conditioner)),
        detail::hold(std::forward<Condition>(condition)));
}

namespace detail {
template<class Model, class Input, class Conditioner, class Condition, class... Requests>
auto predict_conditioned(ConditionedInvocation<Model, Input, Conditioner, Condition>& invocation,
                         Run run, Requests&&... requests) {
    auto&& model = unhold(invocation.model());
    auto&& input = unhold(invocation.input());
    auto&& conditioner = unhold(invocation.conditioner());
    auto&& condition = unhold(invocation.condition());
    decltype(auto) actual_input = invoke_conditioner(conditioner, input, condition, run);
    // If conditioning creates a value, forward it into predict so the batched
    // prediction frame owns it. A returned reference remains an explicit
    // borrow of the conditioner/input's lifetime.
    return predict(model, std::forward<decltype(actual_input)>(actual_input), run,
                   std::forward<Requests>(requests)...);
}

template<class Invocation, class... Requests>
auto predict_conditioned_owned(std::shared_ptr<Invocation> lifetime_anchor,
                               Run run, Requests&&... requests) {
    auto& invocation = *lifetime_anchor;
    auto&& model_holder = invocation.model();
    auto&& input = unhold(invocation.input());
    auto&& conditioner = unhold(invocation.conditioner());
    auto&& condition = unhold(invocation.condition());
    decltype(auto) actual_input = invoke_conditioner(conditioner, input, condition, run);
    return predict_with_lifetime_anchor(std::move(lifetime_anchor),
        forward_held(model_holder),
        std::forward<decltype(actual_input)>(actual_input), run,
        std::forward<Requests>(requests)...);
}
} // namespace detail

template<class Model, class Input, class Conditioner, class Condition, class... Requests>
auto predict(ConditionedInvocation<Model, Input, Conditioner, Condition>& invocation,
             Run run, Requests&&... requests) {
    return detail::predict_conditioned(invocation, run,
                                       std::forward<Requests>(requests)...);
}

template<class Model, class Input, class Conditioner, class Condition, class... Requests>
auto predict(ConditionedInvocation<Model, Input, Conditioner, Condition>&& invocation,
             Run run, Requests&&... requests) {
    using invocation_type = ConditionedInvocation<Model, Input, Conditioner, Condition>;
    auto owner = std::make_shared<invocation_type>(std::move(invocation));
    return detail::predict_conditioned_owned(std::move(owner), run,
        std::forward<Requests>(requests)...);
}

} // namespace cellerator::nn
