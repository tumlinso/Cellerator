#include "Cellerator/ops/mechanism/gated_product_host.hh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace cellerator::ops::mechanism {

namespace detail {

template<class T>
struct SavedGroup {
    GateEvaluation<T> gate{T{}, T{}, T{}, true};
    T product = T{1};
    T theta = T{};
    T phi = T{};
    T z1 = T{};
};

template<class T>
struct GatedProductHostTapeState {
    std::size_t batch = 0;
    std::size_t coordinates = 0;
    std::size_t conditions = 0;
    std::size_t parameters = 0;
    T beta{};
    std::vector<GatedProductGroup> groups;
    std::vector<std::uint64_t> arguments;
    std::vector<OutputIncidence> outputs;
    std::vector<T> x;
    std::vector<SavedGroup<T>> saved_groups;
    std::size_t forward_participant_slot_reads = 0;
    bool consumed = false;
};

struct GatedProductHostTapeAccess {
    template<class T>
    static GatedProductHostTape<T> make(
        std::unique_ptr<GatedProductHostTapeState<T>> state) noexcept {
        return GatedProductHostTape<T>(std::move(state));
    }

    template<class T>
    static GatedProductHostTapeState<T>* get(
        GatedProductHostTape<T>& tape) noexcept {
        return tape.state_.get();
    }
};

} // namespace detail

template<class T>
GatedProductHostTape<T>::GatedProductHostTape(
    std::unique_ptr<detail::GatedProductHostTapeState<T>> state) noexcept
    : state_(std::move(state)) {}

template<class T>
GatedProductHostTape<T>::GatedProductHostTape(
    GatedProductHostTape&&) noexcept = default;

template<class T>
GatedProductHostTape<T>& GatedProductHostTape<T>::operator=(
    GatedProductHostTape&&) noexcept = default;

template<class T>
GatedProductHostTape<T>::~GatedProductHostTape() = default;

template<class T>
bool GatedProductHostTape<T>::consumed() const noexcept {
    return !state_ || state_->consumed;
}

template<class T>
std::size_t GatedProductHostTape<T>::forward_participant_slot_reads() const noexcept {
    return state_ ? state_->forward_participant_slot_reads : 0;
}

namespace {

std::size_t as_size(std::uint64_t value, const char* what) {
    if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
        if (value > std::numeric_limits<std::size_t>::max())
            throw std::overflow_error(what);
    }
    return static_cast<std::size_t>(value);
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             const char* what) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
        throw std::overflow_error(what);
    return left * right;
}

bool valid_range(std::uint64_t begin, std::uint64_t count,
                 std::uint64_t extent) noexcept {
    return begin <= extent && count <= extent - begin;
}

void validate_topology(const GatedProductTopologyView& topology) {
    if (topology.group_count != 0 && topology.groups == nullptr)
        throw std::invalid_argument("gated-product groups pointer is null");
    if (topology.argument_count != 0 && topology.arguments == nullptr)
        throw std::invalid_argument("gated-product arguments pointer is null");
    if (topology.output_count != 0 && topology.outputs == nullptr)
        throw std::invalid_argument("gated-product outputs pointer is null");

    (void)as_size(topology.group_count, "gated-product group count overflows size_t");
    (void)as_size(topology.argument_count, "gated-product argument count overflows size_t");
    (void)as_size(topology.output_count, "gated-product output count overflows size_t");
    (void)as_size(topology.coordinates, "gated-product coordinate count overflows size_t");
    (void)as_size(topology.conditions, "gated-product condition count overflows size_t");
    (void)as_size(topology.parameters, "gated-product parameter count overflows size_t");

    std::unordered_set<std::uint64_t> ids;
    ids.reserve(as_size(topology.group_count, "gated-product group count overflows size_t"));
    for (std::uint64_t i = 0; i < topology.group_count; ++i) {
        const GatedProductGroup& group = topology.groups[i];
        if (!valid_range(group.argument_begin, group.argument_count,
                         topology.argument_count))
            throw std::invalid_argument("gated-product group argument range is invalid");
        if (!valid_range(group.output_begin, group.output_count,
                         topology.output_count))
            throw std::invalid_argument("gated-product group output range is invalid");
        if (group.parameter_index >= topology.parameters)
            throw std::invalid_argument("gated-product parameter index is out of range");
        if (group.condition_index >= topology.conditions)
            throw std::invalid_argument("gated-product condition index is out of range");
        if (!ids.insert(group.logical_id).second)
            throw std::invalid_argument("gated-product logical group IDs must be unique");
    }

    for (std::uint64_t i = 0; i < topology.argument_count; ++i) {
        if (topology.arguments[i] >= topology.coordinates)
            throw std::invalid_argument("gated-product participant index is out of range");
    }
    for (std::uint64_t i = 0; i < topology.output_count; ++i) {
        const OutputIncidence& output = topology.outputs[i];
        if (output.coordinate >= topology.coordinates)
            throw std::invalid_argument("gated-product output coordinate is out of range");
        if (!std::isfinite(output.scale))
            throw std::invalid_argument("gated-product output scale must be finite");
    }
}

template<class T>
std::vector<T> copy_if_nonempty(std::span<const T> values) {
    return std::vector<T>(values.begin(), values.end());
}

template<class T>
GatedProductHostForward<T> forward_impl(
    const GatedProductTopologyView& topology, std::size_t batch,
    std::span<const T> x, std::span<const T> z,
    std::span<const T> theta, std::span<const T> phi,
    std::span<const T> tau, T beta) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    validate_topology(topology);

    const std::size_t coordinates = as_size(
        topology.coordinates, "gated-product coordinate count overflows size_t");
    const std::size_t conditions = as_size(
        topology.conditions, "gated-product condition count overflows size_t");
    const std::size_t parameters = as_size(
        topology.parameters, "gated-product parameter count overflows size_t");
    const std::size_t group_count = as_size(
        topology.group_count, "gated-product group count overflows size_t");
    const std::size_t x_count = checked_multiply(
        batch, coordinates, "gated-product x extent overflows size_t");
    const std::size_t z_channels = checked_multiply(
        conditions, std::size_t{2}, "gated-product condition extent overflows size_t");
    const std::size_t z_count = checked_multiply(
        batch, z_channels, "gated-product z extent overflows size_t");
    const std::size_t saved_group_count = checked_multiply(
        batch, group_count, "gated-product tape extent overflows size_t");

    if (x.size() != x_count || z.size() != z_count ||
        theta.size() != parameters || phi.size() != parameters ||
        tau.size() != parameters)
        throw std::invalid_argument("gated-product input spans do not match declared extents");

    auto state = std::make_unique<detail::GatedProductHostTapeState<T>>();
    state->batch = batch;
    state->coordinates = coordinates;
    state->conditions = conditions;
    state->parameters = parameters;
    state->beta = beta;
    if (topology.group_count != 0)
        state->groups.assign(topology.groups, topology.groups + group_count);
    if (topology.argument_count != 0) {
        const auto argument_count = as_size(
            topology.argument_count, "gated-product argument count overflows size_t");
        state->arguments.assign(topology.arguments,
                                 topology.arguments + argument_count);
    }
    if (topology.output_count != 0) {
        const auto output_count = as_size(
            topology.output_count, "gated-product output count overflows size_t");
        state->outputs.assign(topology.outputs, topology.outputs + output_count);
    }
    // x is needed by the baseline and is retained for backward's participant
    // products. Gate and parameter primals are saved only for active groups.
    state->x = copy_if_nonempty(x);
    state->saved_groups.resize(saved_group_count);

    std::vector<T> values(x_count, T{});
    for (std::size_t i = 0; i < x_count; ++i)
        values[i] = -beta * state->x[i];

    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t group_index = 0; group_index < group_count; ++group_index) {
            const GatedProductGroup& group = state->groups[group_index];
            detail::SavedGroup<T>& saved =
                state->saved_groups[b * group_count + group_index];
            if (group.output_count == 0) continue;

            const std::size_t parameter = static_cast<std::size_t>(group.parameter_index);
            const std::size_t condition = static_cast<std::size_t>(group.condition_index);
            const std::size_t condition_offset = (b * conditions + condition) * 2;
            const T z0 = z[condition_offset];
            const T z1 = z[condition_offset + 1];
            const T phi_value = phi[parameter];
            const T tau_value = tau[parameter];
            saved.gate = evaluate_gate(z0, z1, phi_value, tau_value);
            if (saved.gate.excluded) continue;

            const T theta_value = theta[parameter];
            T product = T{1};
            const std::size_t argument_begin =
                static_cast<std::size_t>(group.argument_begin);
            const std::size_t argument_count =
                static_cast<std::size_t>(group.argument_count);
            for (std::size_t slot = 0; slot < argument_count; ++slot) {
                const std::size_t coordinate = static_cast<std::size_t>(
                    state->arguments[argument_begin + slot]);
                product *= state->x[b * coordinates + coordinate];
                ++state->forward_participant_slot_reads;
            }

            saved.product = product;
            saved.theta = theta_value;
            saved.phi = phi_value;
            saved.z1 = z1;
            for (std::size_t output_slot = 0;
                 output_slot < static_cast<std::size_t>(group.output_count);
                 ++output_slot) {
                const OutputIncidence& incidence = state->outputs[
                    static_cast<std::size_t>(group.output_begin) + output_slot];
                const std::size_t output_index = b * coordinates +
                    static_cast<std::size_t>(incidence.coordinate);
                values[output_index] += gated_contribution(
                    theta_value, saved.gate, product) * static_cast<T>(incidence.scale);
            }
        }
    }

    return {std::move(values), detail::GatedProductHostTapeAccess::make(std::move(state))};
}

template<class T>
GatedProductHostBackward<T> backward_impl(
    GatedProductHostTape<T>&& tape, std::span<const T> dvalues) {
    detail::GatedProductHostTapeState<T>* state =
        detail::GatedProductHostTapeAccess::get(tape);
    if (state == nullptr || state->consumed)
        throw std::logic_error("gated-product host tape is empty or already consumed");
    const std::size_t output_count = checked_multiply(
        state->batch, state->coordinates,
        "gated-product output cotangent extent overflows size_t");
    if (dvalues.size() != output_count)
        throw std::invalid_argument("gated-product cotangent span has the wrong extent");

    // From this point the call is a one-shot attempt, including allocation
    // failures. Shape errors above leave the tape available for a corrected call.
    state->consumed = true;

    const std::size_t z_count = checked_multiply(
        state->batch,
        checked_multiply(state->conditions, std::size_t{2},
                         "gated-product condition extent overflows size_t"),
        "gated-product z extent overflows size_t");
    GatedProductHostBackward<T> result;
    result.dx.assign(output_count, T{});
    result.dz.assign(z_count, T{});
    result.dtheta.assign(state->parameters, T{});
    result.dphi.assign(state->parameters, T{});
    result.dtau.assign(state->parameters, T{});

    // The baseline is always present, independent of every module gate.
    for (std::size_t i = 0; i < output_count; ++i)
        result.dx[i] = -state->beta * dvalues[i];

    const std::size_t group_count = state->groups.size();
    for (std::size_t b = 0; b < state->batch; ++b) {
        for (std::size_t group_index = 0; group_index < group_count; ++group_index) {
            const GatedProductGroup& group = state->groups[group_index];
            const detail::SavedGroup<T>& saved =
                state->saved_groups[b * group_count + group_index];
            if (group.output_count == 0 || saved.gate.excluded) continue;

            T upstream = T{};
            for (std::size_t output_slot = 0;
                 output_slot < static_cast<std::size_t>(group.output_count);
                 ++output_slot) {
                const OutputIncidence& incidence = state->outputs[
                    static_cast<std::size_t>(group.output_begin) + output_slot];
                const std::size_t output_index = b * state->coordinates +
                    static_cast<std::size_t>(incidence.coordinate);
                upstream += static_cast<T>(incidence.scale) * dvalues[output_index];
            }

            const std::size_t argument_begin =
                static_cast<std::size_t>(group.argument_begin);
            const std::size_t argument_count =
                static_cast<std::size_t>(group.argument_count);
            if (argument_count == std::numeric_limits<std::size_t>::max())
                throw std::overflow_error("gated-product participant prefix extent overflows size_t");
            std::vector<T> slot_values(argument_count);
            std::vector<T> prefix(argument_count + 1, T{1});
            std::vector<T> suffix(argument_count + 1, T{1});
            for (std::size_t slot = 0; slot < argument_count; ++slot) {
                const std::size_t coordinate = static_cast<std::size_t>(
                    state->arguments[argument_begin + slot]);
                slot_values[slot] = state->x[b * state->coordinates + coordinate];
                ++result.participant_slot_reads;
                prefix[slot + 1] = prefix[slot] * slot_values[slot];
            }
            for (std::size_t slot = argument_count; slot > 0; --slot)
                suffix[slot - 1] = suffix[slot] * slot_values[slot - 1];

            const GatedProductPartials<T> partials = local_partials(
                upstream, saved.theta, saved.phi, saved.z1, saved.gate,
                saved.product);
            for (std::size_t slot = 0; slot < argument_count; ++slot) {
                const std::size_t coordinate = static_cast<std::size_t>(
                    state->arguments[argument_begin + slot]);
                result.dx[b * state->coordinates + coordinate] +=
                    partials.participant_factor * prefix[slot] * suffix[slot + 1];
            }

            const std::size_t parameter =
                static_cast<std::size_t>(group.parameter_index);
            const std::size_t condition =
                static_cast<std::size_t>(group.condition_index);
            const std::size_t condition_offset = (b * state->conditions + condition) * 2;
            result.dz[condition_offset] += partials.z0;
            result.dz[condition_offset + 1] += partials.z1;
            result.dtheta[parameter] += partials.theta;
            result.dphi[parameter] += partials.phi;
            result.dtau[parameter] += partials.tau;
        }
    }
    return result;
}

} // namespace

GatedProductHostForward<float> gated_product_forward(
    const GatedProductTopologyView& topology, std::size_t batch,
    std::span<const float> x, std::span<const float> z,
    std::span<const float> theta, std::span<const float> phi,
    std::span<const float> tau, float beta) {
    return forward_impl(topology, batch, x, z, theta, phi, tau, beta);
}

GatedProductHostForward<double> gated_product_forward(
    const GatedProductTopologyView& topology, std::size_t batch,
    std::span<const double> x, std::span<const double> z,
    std::span<const double> theta, std::span<const double> phi,
    std::span<const double> tau, double beta) {
    return forward_impl(topology, batch, x, z, theta, phi, tau, beta);
}

GatedProductHostBackward<float> gated_product_backward(
    GatedProductHostTape<float>&& tape, std::span<const float> dvalues) {
    return backward_impl(std::move(tape), dvalues);
}

GatedProductHostBackward<double> gated_product_backward(
    GatedProductHostTape<double>&& tape, std::span<const double> dvalues) {
    return backward_impl(std::move(tape), dvalues);
}

template class GatedProductHostTape<float>;
template class GatedProductHostTape<double>;

} // namespace cellerator::ops::mechanism
