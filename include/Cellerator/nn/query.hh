#pragma once

#include <ATen/ATen.h>
#include <Cellerator/nn/types.hh>

#include <cstdint>
#include <functional>
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

template<class FeatureIds, class Coordinates>
struct QueryPayload {
    FeatureIds feature_ids;
    Coordinates coordinates;
};

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
    if constexpr (requires { binding.canonical_ids(); targets.source_ids(); }) {
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
        const auto& columns = targets.output_columns();
        if (values.dim() == 0 || values.size(-1) != static_cast<std::int64_t>(source_ids.size())) {
            throw std::invalid_argument(
                "full readout output axis does not match the prepared source order");
        }
        return values.index_select(-1, index_tensor(columns, values));
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

template<class Decoder, class Source, class Targets, class Coordinates>
decltype(auto) invoke_head(Decoder&& decoder, Source&& source,
                           const Targets& targets, const Coordinates& coordinates,
                           Run run) {
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
        auto full = invoke_with_run(std::forward<Decoder>(decoder), run,
                                    std::forward<Source>(source));
        return gather_full_output(std::move(full), targets);
    }
}

template<class View, class BaseResult, class Input, class Targets, class Coordinates>
auto evaluate_request(View& view, BaseResult&& base_result, Input&& input,
                      const Targets& targets, const Coordinates& coordinates, Run run) {
    const auto& binding = unhold(view.binding_holder());
    validate_readout_targets(binding, targets);
    auto&& selector = unhold(view.selector_holder());
    auto&& decoder = unhold(view.decoder_holder());
    decltype(auto) source = select_source(selector,
        std::forward<BaseResult>(base_result), std::forward<Input>(input));
    decltype(auto) raw = invoke_head(decoder, std::forward<decltype(source)>(source),
                                     targets, coordinates, run);
    using values_type = std::remove_cvref_t<decltype(raw)>;
    using ids_type = std::remove_cvref_t<decltype(targets.canonical_ids())>;
    using coordinates_type = std::remove_cvref_t<Coordinates>;
    using payload_type = QueryPayload<ids_type, coordinates_type>;
    return Prediction<values_type, payload_type>{
        std::forward<decltype(raw)>(raw),
        payload_type{targets.canonical_ids(), coordinates}
    };
}

template<class Request, class BaseResult, class Input>
auto evaluate_request(Request& request, BaseResult&& base_result, Input&& input, Run run) {
    auto&& view = unhold(request.view_holder());
    auto&& targets = unhold(request.targets_holder());
    auto&& coordinates = unhold(request.coordinates_holder());
    return evaluate_request(view, std::forward<BaseResult>(base_result),
                            std::forward<Input>(input), targets, coordinates, run);
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

// Construct a typed request. This call performs no model or head work.
template<class View, class Targets, class Coordinates>
auto query(View&& view, Targets&& targets, Coordinates&& coordinates) {
    return QueryRequest<detail::held_t<View&&>, detail::held_t<Targets&&>,
                        detail::held_t<Coordinates&&>>(
        detail::hold(std::forward<View>(view)),
        detail::hold(std::forward<Targets>(targets)),
        detail::hold(std::forward<Coordinates>(coordinates)));
}

// Invoke the base exactly once, then run each concrete query request against
// that shared result. One request returns its Prediction directly; multiple
// requests return a fixed tuple in request order. No dynamic registry is used.
template<class Model, class Input, class... Requests>
auto predict(Model&& model, Input&& input, Run run, Requests&&... requests) {
    auto request_tuple = std::forward_as_tuple(std::forward<Requests>(requests)...);
    decltype(auto) base_result = detail::invoke_model(std::forward<Model>(model), input, run);
    if constexpr (sizeof...(Requests) == 0) {
        return base_result;
    } else {
        auto outputs = std::apply([&](auto&... request) {
            return std::tuple{detail::evaluate_request(
                request, base_result, input, run)...};
        }, request_tuple);
        if constexpr (sizeof...(Requests) == 1) {
            return std::get<0>(std::move(outputs));
        } else {
            return outputs;
        }
    }
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
    decltype(auto) actual_input = [&]() -> decltype(auto) {
        if constexpr (std::is_invocable_v<decltype(conditioner), decltype(input),
                                          decltype(condition), Run>) {
            return std::invoke(conditioner, input, condition, run);
        } else {
            return std::invoke(conditioner, input, condition);
        }
    }();
    return predict(model, actual_input, run, std::forward<Requests>(requests)...);
}
} // namespace detail

template<class Model, class Input, class Conditioner, class Condition, class... Requests>
auto predict(ConditionedInvocation<Model, Input, Conditioner, Condition>& invocation,
             Run run, Requests&&... requests) {
    return detail::predict_conditioned(invocation, run,
                                       std::forward<Requests>(requests)...);
}

} // namespace cellerator::nn
