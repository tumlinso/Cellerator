#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace cellerator::modeling {

// Borrowed synchronous invocation data. No biological schema, tensor layout,
// RNG algorithm or metadata representation is prescribed. The pointed-to
// objects must outlive the callback; callbacks must not retain these pointers.
template<class Parameters = std::monostate, class Inputs = std::monostate,
         class Random = std::monostate, class Metadata = std::monostate>
struct StepContext {
    using parameters_type = Parameters;
    using inputs_type = Inputs;
    using random_type = Random;
    using metadata_type = Metadata;

    double time = 0.0;
    double dt = 1.0;
    const Inputs* inputs = nullptr;
    const Parameters* parameters = nullptr;
    Random* rng = nullptr;
    const Metadata* metadata = nullptr;
};

// The state and event/effect containers are caller types. In particular State
// may be sparse, ragged, an external resource handle, or a move-only object.
// An absent result time means context.time + context.dt when adopted by Cell.
template<class State, class Events = std::vector<std::monostate>,
         class Effects = std::vector<std::monostate>, class Metadata = std::monostate>
struct StepResult {
    State state;
    Events events{};
    Effects effects{};
    std::optional<double> time = std::nullopt;
    Metadata metadata{};
};

struct IdentityObservation {
    template<class State, class Context>
    const State& operator()(const State& state, const Context&) const noexcept { return state; }
};

namespace detail {
inline void validate_step_clock(double time, double dt) {
    if (!std::isfinite(time) || !std::isfinite(dt))
        throw std::invalid_argument("step time and dt must be finite");
}
inline void validate_result_clock(const std::optional<double>& time) {
    if (time && !std::isfinite(*time)) throw std::invalid_argument("result time must be finite");
}
} // namespace detail

// Shared behavior and parameter ownership, independent of individual state.
// Dynamics is invoked as (const State&, const Context&), returning StepResult
// (or a compatible result with state/time fields), or simply a new State.
// Callbacks are const-invocable. Thread safety of captures/external resources
// and parameter replacement remains the caller's responsibility.
template<class State, class Context, class Dynamics, class Observation = IdentityObservation>
class Model final {
public:
    using state_type = State;
    using context_type = Context;
    using parameters_type = typename Context::parameters_type;
    using raw_result_type = std::decay_t<std::invoke_result_t<const Dynamics&, const State&, const Context&>>;
    using result_type = std::conditional_t<std::is_same_v<raw_result_type, State>,
                                            StepResult<State>, raw_result_type>;
    static_assert(std::is_same_v<std::decay_t<decltype(std::declval<result_type>().state)>, State>,
                  "model result.state must have the declared State type");

    Model(Dynamics dynamics, std::shared_ptr<const parameters_type> parameters = {},
          Observation observation = {})
        : dynamics_(std::move(dynamics)), observation_(std::move(observation)), parameters_(std::move(parameters)) {}

    result_type step(const State& state, Context context = {}) const {
        detail::validate_step_clock(context.time, context.dt);
        const auto keepalive = parameters_;
        if (!context.parameters) context.parameters = keepalive.get();
        if constexpr (std::is_same_v<raw_result_type, State>) {
            return result_type{std::invoke(dynamics_, state, std::as_const(context))};
        } else {
            auto result = std::invoke(dynamics_, state, std::as_const(context));
            detail::validate_result_clock(result.time);
            return result;
        }
    }

    decltype(auto) observe(const State& state, Context context = {}) const {
        detail::validate_step_clock(context.time, context.dt);
        const auto keepalive = parameters_;
        if (!context.parameters) context.parameters = keepalive.get();
        return std::invoke(observation_, state, std::as_const(context));
    }

    const std::shared_ptr<const parameters_type>& parameters() const noexcept { return parameters_; }
    // Replaces the shared parameter owner for subsequent calls. A context can
    // instead override parameters for exactly one invocation by supplying its
    // own non-null pointer. Neither route mutates individual cell state.
    void set_parameters(std::shared_ptr<const parameters_type> parameters) { parameters_ = std::move(parameters); }

private:
    Dynamics dynamics_;
    Observation observation_;
    std::shared_ptr<const parameters_type> parameters_;
};

template<class State, class Context = StepContext<>, class Dynamics, class Observation = IdentityObservation>
auto make_model(Dynamics dynamics,
                std::shared_ptr<const typename Context::parameters_type> parameters = {},
                Observation observation = {}) {
    using model_type = Model<State, Context, Dynamics, Observation>;
    return std::make_shared<model_type>(std::move(dynamics), std::move(parameters), std::move(observation));
}

// A persistent identity and independently owned state attached to shared model
// behavior. advance returns a shared immutable result snapshot. The Cell
// aliases that result's state, avoiding a copy even for move-only State; keeping
// the result handle pins the previous state after subsequent transitions.
// Different cells are not automatically synchronized or scheduled.
template<class ModelType, class Identity = std::uint64_t>
class Cell final {
public:
    using model_type = ModelType;
    using state_type = typename ModelType::state_type;
    using context_type = typename ModelType::context_type;
    using result_type = typename ModelType::result_type;
    using identity_type = Identity;

    Cell(std::shared_ptr<const ModelType> model, state_type state, Identity identity, double time = 0.0)
        : model_(std::move(model)), state_(std::make_shared<const state_type>(std::move(state))),
          identity_(std::move(identity)), time_(time) {
        if (!model_) throw std::invalid_argument("cell model cannot be null");
        detail::validate_step_clock(time_, 0.0);
    }
    Cell(const Cell&) = delete;
    Cell& operator=(const Cell&) = delete;
    Cell(Cell&&) = default;
    Cell& operator=(Cell&&) = default;

    const Identity& id() const noexcept { return identity_; }
    const state_type& state() const noexcept { return *state_; }
    std::shared_ptr<const state_type> state_handle() const noexcept { return state_; }
    const std::shared_ptr<const ModelType>& model() const noexcept { return model_; }
    double time() const noexcept { return time_; }
    std::uint64_t value_generation() const noexcept { return value_generation_; }

    std::shared_ptr<const result_type> advance(double dt = 1.0) {
        context_type context;
        context.dt = dt;
        return advance(std::move(context));
    }

    // Cell owns its clock, so supplied context.time is replaced with its
    // current time. All other context fields, including dt, remain explicit.
    std::shared_ptr<const result_type> advance(context_type context) {
        context.time = time_;
        if (value_generation_ == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("cell value generation overflow");
        const double dt = context.dt;
        auto result = std::make_shared<result_type>(model_->step(*state_, std::move(context)));
        const auto next_time = result->time ? *result->time : time_ + dt;
        detail::validate_result_clock(next_time);
        result->time = next_time;
        // All callback, clock and allocation checks precede adoption. Callbacks
        // that mutate external handles still own their own rollback semantics.
        auto next_state = std::shared_ptr<const state_type>(result, &result->state);
        state_ = std::move(next_state);
        time_ = next_time;
        ++value_generation_;
        return result;
    }

    decltype(auto) observe(context_type context = {}) const {
        context.time = time_;
        context.dt = 0.0;
        return model_->observe(*state_, std::move(context));
    }

private:
    std::shared_ptr<const ModelType> model_;
    std::shared_ptr<const state_type> state_;
    Identity identity_;
    double time_;
    std::uint64_t value_generation_ = 0;
};

template<class ModelType, class Identity>
auto make_cell(std::shared_ptr<ModelType> model, typename ModelType::state_type state,
               Identity identity, double time = 0.0) {
    using base_model_type = std::remove_const_t<ModelType>;
    return Cell<base_model_type, Identity>(std::move(model), std::move(state), std::move(identity), time);
}

} // namespace cellerator::modeling
