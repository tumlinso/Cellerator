#pragma once

#include <Cellerator/cell.hh>

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

// A retained const State snapshot. Holding this view pins the state object
// after advancement; nested external resources follow the State's contract.
template<class State>
struct DynamicalStateSnapshotView {
    std::shared_ptr<const State> state;
};

namespace detail {
inline void validate_step_clock(double time, double dt) {
    if (!std::isfinite(time) || !std::isfinite(dt))
        throw std::invalid_argument("step time and dt must be finite");
}
inline void validate_result_clock(const std::optional<double>& time) {
    if (time && !std::isfinite(*time)) throw std::invalid_argument("result time must be finite");
}

template<class State, class Identity>
struct dynamical_cell_instance {
    std::shared_ptr<const State> state;
    Identity identity;

    DynamicalStateSnapshotView<State>
    read(ViewTag<DynamicalStateSnapshotView<State>>) const & {
        return {state};
    }
    DynamicalStateSnapshotView<State>
    read(ViewTag<DynamicalStateSnapshotView<State>>) const && = delete;

    CellIdentityView<Identity> read(ViewTag<CellIdentityView<Identity>>) const & {
        return {identity};
    }
    CellIdentityView<Identity> read(ViewTag<CellIdentityView<Identity>>) const && = delete;
};

} // namespace detail

// Explicit state-transition behavior and parameter ownership. Generic learned
// models use Cell directly and do not inherit this time/state contract.
template<class State, class Context, class Dynamics, class Observation = IdentityObservation>
class DynamicalModel final {
public:
    using state_type = State;
    using context_type = Context;
    using parameters_type = typename Context::parameters_type;
    using raw_result_type = std::decay_t<std::invoke_result_t<const Dynamics&, const State&, const Context&>>;
    using result_type = std::conditional_t<std::is_same_v<raw_result_type, State>,
                                            StepResult<State>, raw_result_type>;
    static_assert(std::is_same_v<std::decay_t<decltype(std::declval<result_type>().state)>, State>,
                  "dynamical model result.state must have the declared State type");

    DynamicalModel(Dynamics dynamics, std::shared_ptr<const parameters_type> parameters = {},
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

template<class State, class Context = StepContext<>, class Dynamics,
         class Observation = IdentityObservation>
auto make_dynamical_model(Dynamics dynamics,
                          std::shared_ptr<const typename Context::parameters_type> parameters = {},
                          Observation observation = {}) {
    using model_type = DynamicalModel<State, Context, Dynamics, Observation>;
    return std::make_shared<model_type>(std::move(dynamics), std::move(parameters), std::move(observation));
}

// State, clock, observation, and transition behavior are opt-in through this
// specialized Cell. Results remain immutable snapshots and may hold move-only
// state without copying it.
template<class ModelType, class Identity = std::uint64_t>
class DynamicalCell final {
public:
    using state_type = typename ModelType::state_type;
    using context_type = typename ModelType::context_type;
    using result_type = typename ModelType::result_type;
    using identity_type = Identity;
    using state_snapshot_view_type = DynamicalStateSnapshotView<state_type>;
    using identity_view_type = CellIdentityView<Identity>;
    using model_type = std::remove_const_t<ModelType>;
    using instance_type = detail::dynamical_cell_instance<state_type, Identity>;
    using binding_type = Cell<const model_type, instance_type, const model_type*>;

    DynamicalCell(std::shared_ptr<const ModelType> model, state_type state,
                  Identity identity, double time = 0.0)
        : model_(std::move(model)),
          binding_(model_.get(), instance_type{
              std::make_shared<const state_type>(std::move(state)), std::move(identity)}),
          time_(time) {
        if (!model_) throw std::invalid_argument("dynamical cell model cannot be null");
        detail::validate_step_clock(time_, 0.0);
    }

    DynamicalCell(const DynamicalCell&) = delete;
    DynamicalCell& operator=(const DynamicalCell&) = delete;
    DynamicalCell(DynamicalCell&&) = default;
    DynamicalCell& operator=(DynamicalCell&&) = default;

    const state_type& state() const & noexcept { return *binding_.instance().state; }
    const state_type& state() const && = delete;
    std::shared_ptr<const state_type> state_handle() const noexcept {
        return binding_.instance().state;
    }
    const std::shared_ptr<const ModelType>& model() const & noexcept { return model_; }
    const std::shared_ptr<const ModelType>& model() const && = delete;
    const Identity& identity() const & noexcept { return binding_.instance().identity; }
    const Identity& identity() const && = delete;
    const Identity& id() const & noexcept { return binding_.instance().identity; }
    const Identity& id() const && = delete;
    double time() const noexcept { return time_; }
    std::uint64_t value_generation() const noexcept { return value_generation_; }

    template<class View> static constexpr bool supports() noexcept {
        return binding_type::template supports<View>();
    }
    template<class View> static constexpr bool supports_edit() noexcept {
        return binding_type::template supports_edit<View>();
    }
    template<class View, std::enable_if_t<
        binding_type::template supports<View>(), int> = 0>
    View view() const & { return binding_.template view<View>(); }
    template<class View> void view() const && = delete;

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
        auto result = std::make_shared<result_type>(model_->step(state(), std::move(context)));
        const auto next_time = result->time ? *result->time : time_ + dt;
        detail::validate_result_clock(next_time);
        result->time = next_time;
        // All callback, clock and allocation checks precede adoption. Callbacks
        // that mutate external handles still own their own rollback semantics.
        auto next_state = std::shared_ptr<const state_type>(result, &result->state);
        binding_.instance().state = std::move(next_state);
        time_ = next_time;
        ++value_generation_;
        return result;
    }

    decltype(auto) observe(context_type context = {}) const & {
        context.time = time_;
        context.dt = 0.0;
        return model_->observe(state(), std::move(context));
    }

    void observe(context_type = {}) const && = delete;

private:
    std::shared_ptr<const ModelType> model_;
    binding_type binding_;
    double time_;
    std::uint64_t value_generation_ = 0;
};

template<class ModelType, class Identity>
auto make_dynamical_cell(std::shared_ptr<ModelType> model,
                         typename ModelType::state_type state,
                         Identity identity, double time = 0.0) {
    using base_model_type = std::remove_const_t<ModelType>;
    return DynamicalCell<base_model_type, Identity>(
        std::move(model), std::move(state), std::move(identity), time);
}

// The previous model names remain available as narrow transition-only
// compatibility shims. Generic cells implement typed Cell access contracts.
template<class State, class Context, class Dynamics,
         class Observation = IdentityObservation>
using Model [[deprecated("use DynamicalModel for explicit state transitions")]] =
    DynamicalModel<State, Context, Dynamics, Observation>;

template<class State, class Context = StepContext<>, class Dynamics,
         class Observation = IdentityObservation>
[[deprecated("use make_dynamical_model for explicit state transitions")]]
auto make_model(Dynamics dynamics,
                std::shared_ptr<const typename Context::parameters_type> parameters = {},
                Observation observation = {}) {
    return make_dynamical_model<State, Context>(
        std::move(dynamics), std::move(parameters), std::move(observation));
}

} // namespace cellerator::modeling
