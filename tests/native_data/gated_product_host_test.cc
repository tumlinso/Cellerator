#include "Cellerator/ops/mechanism/gated_product_host.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mechanism = cellerator::ops::mechanism;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template<class T>
void close(T actual, T expected, T tolerance, const std::string& message) {
    const T scale = std::max(T{1}, std::abs(expected));
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance * scale)
        throw std::runtime_error(message + ": expected " +
                                 std::to_string(expected) + ", got " +
                                 std::to_string(actual));
}

template<class Exception, class Callable>
void throws(Callable&& callable, const std::string& message) {
    try {
        std::forward<Callable>(callable)();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

struct OwnedTopology {
    std::vector<mechanism::GatedProductGroup> groups;
    std::vector<std::uint64_t> arguments;
    std::vector<mechanism::OutputIncidence> outputs;
    std::uint64_t coordinates = 0;
    std::uint64_t conditions = 0;
    std::uint64_t parameters = 0;

    mechanism::GatedProductTopologyView view() const {
        return {groups.data(), groups.size(), arguments.data(), arguments.size(),
                outputs.data(), outputs.size(), coordinates, conditions, parameters};
    }
};

OwnedTopology analytic_topology() {
    return {
        {
            {0, 3, 0, 3, 0, 0, 101},
            {3, 2, 3, 2, 1, 1, 102},
        },
        {0, 1, 1, 1, 2},
        {
            {0, 0.25F}, {0, 0.75F}, {2, 0.5F},
            {0, 2.0F}, {1, -0.5F},
        },
        4, 2, 2,
    };
}

template<class T>
std::vector<T> forward_values(const OwnedTopology& topology,
                              std::size_t batch,
                              const std::vector<T>& x,
                              const std::vector<T>& z,
                              const std::vector<T>& theta,
                              const std::vector<T>& phi,
                              const std::vector<T>& tau,
                              T beta) {
    return mechanism::gated_product_forward(
        topology.view(), batch, x, z, theta, phi, tau, beta).values;
}

template<class T>
T objective(const OwnedTopology& topology,
            const std::vector<T>& x,
            const std::vector<T>& z,
            const std::vector<T>& theta,
            const std::vector<T>& phi,
            const std::vector<T>& tau,
            const std::vector<T>& dvalues,
            T beta) {
    const auto values = forward_values(topology, 1, x, z, theta, phi, tau, beta);
    T result{};
    for (std::size_t i = 0; i < values.size(); ++i)
        result += values[i] * dvalues[i];
    return result;
}

template<class T>
void test_active_analytic_and_finite_difference(T tolerance, T step) {
    const OwnedTopology topology = analytic_topology();
    std::vector<T> x{T{2}, T{3}, T{0.5}, T{0}};
    std::vector<T> z{T{1}, T{0.5}, T{0.3}, T{0.4}};
    std::vector<T> theta{T{0.4}, T{0.6}};
    std::vector<T> phi{T{0.5}, T{-0.25}};
    std::vector<T> tau{T{0.5}, T{-0.2}};
    const std::vector<T> dvalues{T{1}, T{-0.5}, T{0.25}, T{0}};
    const T beta = T{0.25};

    auto forward = mechanism::gated_product_forward(
        topology.view(), 1, x, z, theta, phi, tau, beta);
    close(forward.values[0], T{3.838}, tolerance, "active y[0]");
    close(forward.values[1], T{-0.822}, tolerance, "active y[1]");
    close(forward.values[2], T{1.9}, tolerance, "active y[2]");
    close(forward.values[3], T{0}, tolerance, "active y[3]");
    require(forward.tape.forward_participant_slot_reads() == 5,
            "active forward must read each ordered argument slot");

    // The tape owns its forward contents; mutations to every caller buffer
    // after forward must not change the VJP.
    x.assign(x.size(), T{91});
    z.assign(z.size(), T{-17});
    theta.assign(theta.size(), T{8});
    phi.assign(phi.size(), T{7});
    tau.assign(tau.size(), T{6});
    auto backward = mechanism::gated_product_backward(
        std::move(forward.tape), dvalues);

    close(backward.dx[0], T{2.028125}, tolerance, "active dx[0]");
    close(backward.dx[1], T{3.2705}, tolerance, "active dx[1]");
    close(backward.dx[2], T{0.5855}, tolerance, "active dx[2]");
    close(backward.dx[3], T{0}, tolerance, "active dx[3]");
    close(backward.dtheta[0], T{11.390625}, tolerance, "active dtheta[0]");
    close(backward.dtheta[1], T{0.54}, tolerance, "active dtheta[1]");
    close(backward.dz[0], T{12.15}, tolerance, "active dz0(condition 0)");
    close(backward.dz[1], T{6.075}, tolerance, "active dz1(condition 0)");
    close(backward.dz[2], T{1.62}, tolerance, "active dz0(condition 1)");
    close(backward.dz[3], T{-0.405}, tolerance, "active dz1(condition 1)");
    close(backward.dphi[0], T{6.075}, tolerance, "active dphi[0]");
    close(backward.dphi[1], T{0.648}, tolerance, "active dphi[1]");
    close(backward.dtau[0], T{-12.15}, tolerance, "active dtau[0]");
    close(backward.dtau[1], T{-1.62}, tolerance, "active dtau[1]");
    require(backward.participant_slot_reads == 5,
            "active backward must read each ordered argument slot once");

    // Independent centered differences supplement the analytic values.
    std::vector<T> base_x{T{2}, T{3}, T{0.5}, T{0}};
    std::vector<T> base_z{T{1}, T{0.5}, T{0.3}, T{0.4}};
    std::vector<T> base_theta{T{0.4}, T{0.6}};
    std::vector<T> base_phi{T{0.5}, T{-0.25}};
    std::vector<T> base_tau{T{0.5}, T{-0.2}};
    const auto difference = [&](std::vector<T> plus_x, std::vector<T> plus_z,
                                std::vector<T> plus_theta, std::vector<T> plus_phi,
                                std::vector<T> plus_tau,
                                std::vector<T> minus_x, std::vector<T> minus_z,
                                std::vector<T> minus_theta, std::vector<T> minus_phi,
                                std::vector<T> minus_tau) {
        return (objective(topology, plus_x, plus_z, plus_theta, plus_phi,
                          plus_tau, dvalues, beta) -
                objective(topology, minus_x, minus_z, minus_theta, minus_phi,
                          minus_tau, dvalues, beta)) / (T{2} * step);
    };

    for (std::size_t i = 0; i < base_x.size(); ++i) {
        auto plus = base_x, minus = base_x;
        plus[i] += step;
        minus[i] -= step;
        const T numeric = difference(plus, base_z, base_theta, base_phi, base_tau,
                                     minus, base_z, base_theta, base_phi, base_tau);
        // Re-run an unmutated forward tape for the analytic comparison.
        auto primal = mechanism::gated_product_forward(
            topology.view(), 1, base_x, base_z, base_theta, base_phi, base_tau, beta);
        auto grad = mechanism::gated_product_backward(std::move(primal.tape), dvalues);
        close(numeric, grad.dx[i], tolerance * T{8}, "finite difference dx");
    }
    for (std::size_t i = 0; i < base_z.size(); ++i) {
        auto plus = base_z, minus = base_z;
        plus[i] += step;
        minus[i] -= step;
        const T numeric = difference(base_x, plus, base_theta, base_phi, base_tau,
                                     base_x, minus, base_theta, base_phi, base_tau);
        auto primal = mechanism::gated_product_forward(
            topology.view(), 1, base_x, base_z, base_theta, base_phi, base_tau, beta);
        auto grad = mechanism::gated_product_backward(std::move(primal.tape), dvalues);
        close(numeric, grad.dz[i], tolerance * T{8}, "finite difference dz");
    }
    for (std::size_t i = 0; i < base_theta.size(); ++i) {
        auto plus = base_theta, minus = base_theta;
        plus[i] += step;
        minus[i] -= step;
        const T numeric = difference(base_x, base_z, plus, base_phi, base_tau,
                                     base_x, base_z, minus, base_phi, base_tau);
        auto primal = mechanism::gated_product_forward(
            topology.view(), 1, base_x, base_z, base_theta, base_phi, base_tau, beta);
        auto grad = mechanism::gated_product_backward(std::move(primal.tape), dvalues);
        close(numeric, grad.dtheta[i], tolerance * T{8}, "finite difference dtheta");

        auto plus_phi = base_phi, minus_phi = base_phi;
        plus_phi[i] += step;
        minus_phi[i] -= step;
        const T numeric_phi = difference(base_x, base_z, base_theta, plus_phi, base_tau,
                                         base_x, base_z, base_theta, minus_phi, base_tau);
        auto phi_primal = mechanism::gated_product_forward(
            topology.view(), 1, base_x, base_z, base_theta, base_phi, base_tau, beta);
        auto phi_grad = mechanism::gated_product_backward(
            std::move(phi_primal.tape), dvalues);
        close(numeric_phi, phi_grad.dphi[i], tolerance * T{8}, "finite difference dphi");

        auto plus_tau = base_tau, minus_tau = base_tau;
        plus_tau[i] += step;
        minus_tau[i] -= step;
        const T numeric_tau = difference(base_x, base_z, base_theta, base_phi, plus_tau,
                                         base_x, base_z, base_theta, base_phi, minus_tau);
        auto tau_primal = mechanism::gated_product_forward(
            topology.view(), 1, base_x, base_z, base_theta, base_phi, base_tau, beta);
        auto tau_grad = mechanism::gated_product_backward(
            std::move(tau_primal.tape), dvalues);
        close(numeric_tau, tau_grad.dtau[i], tolerance * T{8}, "finite difference dtau");
    }
}

void test_zero_and_repeated_participants() {
    const OwnedTopology topology{
        {{0, 2, 0, 1, 0, 0, 201}, {2, 2, 1, 1, 0, 0, 202}},
        {0, 1, 0, 0},
        {{0, 1.0F}, {0, 0.5F}},
        2, 1, 1,
    };
    const std::vector<double> x{0.0, 3.0};
    const std::vector<double> z{1.0, 0.0};
    const std::vector<double> theta{2.0};
    const std::vector<double> phi{0.0};
    const std::vector<double> tau{0.0};
    const std::vector<double> dy{1.0, 0.0};
    auto forward = mechanism::gated_product_forward(
        topology.view(), 1, x, z, theta, phi, tau, 0.0);
    close(forward.values[0], 0.0, 1e-12, "zero participant forward");
    auto grad = mechanism::gated_product_backward(std::move(forward.tape), dy);
    close(grad.dx[0], 6.0, 1e-12,
          "single zero factor derivative uses product of the other slot");
    close(grad.dx[1], 0.0, 1e-12,
          "repeated zero factors have zero per-occurrence derivative");
    close(grad.dtheta[0], 0.0, 1e-12,
          "zero participant product has zero parameter derivative");
}

void test_empty_group_and_empty_topology() {
    const OwnedTopology empty_group{
        {{0, 0, 0, 1, 0, 0, 301}}, {}, {{0, 1.0F}}, 1, 1, 1,
    };
    const std::vector<double> x{2.0};
    const std::vector<double> z{1.0, 0.0};
    const std::vector<double> theta{3.0};
    const std::vector<double> phi{0.0};
    const std::vector<double> tau{0.0};
    const std::vector<double> dy{1.0};
    auto forward = mechanism::gated_product_forward(
        empty_group.view(), 1, x, z, theta, phi, tau, 0.5);
    close(forward.values[0], 2.0, 1e-12,
          "empty participant product is one plus the baseline");
    auto grad = mechanism::gated_product_backward(std::move(forward.tape), dy);
    close(grad.dtheta[0], 1.0, 1e-12,
          "empty participant product contributes parameter gradient");
    close(grad.dx[0], -0.5, 1e-12, "empty group keeps baseline gradient");

    const OwnedTopology empty_topology{{}, {}, {}, 2, 0, 0};
    const std::vector<double> baseline_x{2.0, -4.0};
    const std::vector<double> empty;
    auto baseline = mechanism::gated_product_forward(
        empty_topology.view(), 1, baseline_x, empty, empty, empty, empty, 0.25);
    close(baseline.values[0], -0.5, 1e-12, "empty topology baseline y[0]");
    close(baseline.values[1], 1.0, 1e-12, "empty topology baseline y[1]");
    const std::vector<double> seed{2.0, -3.0};
    auto baseline_grad = mechanism::gated_product_backward(
        std::move(baseline.tape), seed);
    close(baseline_grad.dx[0], -0.5, 1e-12, "empty topology baseline dx[0]");
    close(baseline_grad.dx[1], 0.75, 1e-12, "empty topology baseline dx[1]");
    require(baseline_grad.dz.empty() && baseline_grad.dtheta.empty() &&
            baseline_grad.dphi.empty() && baseline_grad.dtau.empty(),
            "empty topology produces empty mechanism gradients");
}

void test_inactive_short_circuit_and_tape_consumption() {
    const OwnedTopology topology{
        {{0, 1, 0, 1, 0, 0, 401}}, {1}, {{0, 1.0F}}, 2, 1, 1,
    };
    const std::vector<double> x{2.0, std::numeric_limits<double>::quiet_NaN()};
    const std::vector<double> z{0.1, 0.5};
    const std::vector<double> theta{std::numeric_limits<double>::quiet_NaN()};
    const std::vector<double> phi{0.5};
    const std::vector<double> tau{0.6}; // u = -0.25, inactive
    const std::vector<double> dy{1.0, 0.0};
    auto forward = mechanism::gated_product_forward(
        topology.view(), 1, x, z, theta, phi, tau, 0.25);
    close(forward.values[0], -0.5, 1e-12,
          "inactive module must not propagate participant/parameter NaNs");
    require(forward.tape.forward_participant_slot_reads() == 0,
            "inactive forward must not read participant slots");
    auto wrong_shape = std::vector<double>{};
    throws<std::invalid_argument>([&] {
        (void)mechanism::gated_product_backward(std::move(forward.tape), wrong_shape);
    }, "backward must reject a malformed cotangent extent");
    require(!forward.tape.consumed(),
            "shape rejection must leave the forward tape available");

    auto grad = mechanism::gated_product_backward(std::move(forward.tape), dy);
    close(grad.dx[0], -0.25, 1e-12, "inactive baseline dx");
    close(grad.dx[1], 0.0, 1e-12, "inactive unused participant dx");
    require(grad.participant_slot_reads == 0,
            "inactive backward must not read participant slots");
    close(grad.dtheta[0], 0.0, 1e-12, "inactive dtheta");
    close(grad.dphi[0], 0.0, 1e-12, "inactive dphi");
    close(grad.dtau[0], 0.0, 1e-12, "inactive dtau");
    close(grad.dz[0], 0.0, 1e-12, "inactive dz0");
    close(grad.dz[1], 0.0, 1e-12, "inactive dz1");
    throws<std::logic_error>([&] {
        (void)mechanism::gated_product_backward(std::move(forward.tape), dy);
    }, "a consumed tape must reject a second backward");
}

void test_topology_and_shape_validation() {
    const std::vector<double> x{1.0, 2.0};
    const std::vector<double> z{1.0, 0.0};
    const std::vector<double> parameter{1.0};

    const OwnedTopology bad_argument{
        {{0, 1, 0, 1, 0, 0, 501}}, {2}, {{0, 1.0F}}, 2, 1, 1,
    };
    throws<std::invalid_argument>([&] {
        (void)mechanism::gated_product_forward(
            bad_argument.view(), 1, x, z, parameter, parameter, parameter, 0.0);
    }, "out-of-range participant must be rejected");

    const OwnedTopology duplicate_ids{
        {{0, 0, 0, 1, 0, 0, 502}, {0, 0, 1, 1, 0, 0, 502}},
        {}, {{0, 1.0F}, {1, 1.0F}}, 2, 1, 1,
    };
    throws<std::invalid_argument>([&] {
        (void)mechanism::gated_product_forward(
            duplicate_ids.view(), 1, x, z, parameter, parameter, parameter, 0.0);
    }, "duplicate logical group IDs must be rejected");

    const OwnedTopology nonfinite_scale{
        {{0, 0, 0, 1, 0, 0, 503}}, {},
        {{0, std::numeric_limits<float>::infinity()}}, 2, 1, 1,
    };
    throws<std::invalid_argument>([&] {
        (void)mechanism::gated_product_forward(
            nonfinite_scale.view(), 1, x, z, parameter, parameter, parameter, 0.0);
    }, "nonfinite output scale must be rejected");

    const OwnedTopology good{{}, {}, {}, 2, 1, 1};
    throws<std::invalid_argument>([&] {
        (void)mechanism::gated_product_forward(
            good.view(), 1, std::span<const double>{x.data(), 1}, z,
            parameter, parameter, parameter, 0.0);
    }, "incorrect input extents must be rejected");

    const OwnedTopology overflow{{}, {}, {}, 2, 0, 0};
    const std::vector<double> none;
    throws<std::overflow_error>([&] {
        (void)mechanism::gated_product_forward(
            overflow.view(), std::numeric_limits<std::size_t>::max(), none,
            none, none, none, none, 0.0);
    }, "overflowing batch extent must be rejected");
}

} // namespace

int main() {
    try {
        test_active_analytic_and_finite_difference<double>(1e-10, 1e-6);
        test_active_analytic_and_finite_difference<float>(2e-4F, 1e-3F);
        test_zero_and_repeated_participants();
        test_empty_group_and_empty_topology();
        test_inactive_short_circuit_and_tape_consumption();
        test_topology_and_shape_validation();
    } catch (const std::exception& error) {
        std::cerr << "gated product host test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "gated product host tests passed\n";
    return 0;
}
