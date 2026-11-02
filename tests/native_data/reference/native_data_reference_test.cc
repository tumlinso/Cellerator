#include "fixture.hh"
#include "Cellerator/geometry/dependency/reads.hh"
#include "Cellerator/ops/mechanism/gated_product_host.hh"

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ref = cellerator::test::native_data_reference;
namespace mech = cellerator::ops::mechanism;
namespace dependency = cellerator::geometry::dependency;
namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

bool close(double actual, double expected, double tolerance = 3e-9) {
    return std::isfinite(actual) && std::isfinite(expected)
        && std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected));
}

void near(double actual, double expected, const char* message) {
    if (!close(actual, expected))
        throw std::runtime_error(std::string(message) + ": actual="
                                 + std::to_string(actual) + " expected="
                                 + std::to_string(expected));
}

double objective(const ref::values& v, int device = -1, bool baseline = true) {
    const auto y = ref::evaluate(v, [&v](std::size_t i) { return v.x[i]; }, device, baseline);
    double value = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i)
        value += ref::loss_seed[i] * y[i];
    return value;
}

void perturb(ref::values& v, std::size_t axis, double delta) {
    if (axis < v.x.size()) {
        v.x[axis] += delta;
        return;
    }
    switch (axis) {
    case 5: v.z0 += delta; break;
    case 6: v.z1 += delta; break;
    case 7: v.theta += delta; break;
    case 8: v.phi += delta; break;
    case 9: v.tau += delta; break;
    default: throw std::out_of_range("reference derivative axis");
    }
}

double centered_difference(const ref::values& v, std::size_t axis,
                           int device = -1, bool baseline = true) {
    constexpr double h = 1e-5;
    auto plus = v;
    auto minus = v;
    perturb(plus, axis, h);
    perturb(minus, axis, -h);
    return (objective(plus, device, baseline) - objective(minus, device, baseline)) / (2.0 * h);
}

std::size_t parameter_result_index(std::size_t derivative_axis) {
    // Variable axes are x0..x4,z0,z1,theta,phi,tau. Expected parameter arrays
    // are ordered theta,z0,z1,phi,tau.
    constexpr std::array<std::size_t, 5> index{{1, 2, 0, 3, 4}};
    return index[derivative_axis - 5];
}

struct never_read_participant {
    std::size_t* reads;
    double operator()(std::size_t) const {
        ++*reads;
        throw std::runtime_error("inactive reference read a participant");
    }
};

struct topology_storage {
    std::array<mech::GatedProductGroup, ref::mechanisms.size()> groups{};
    std::array<std::uint64_t, 9> arguments{};
    std::array<mech::OutputIncidence, 5> outputs{};

    mech::GatedProductTopologyView view() const noexcept {
        return {groups.data(), groups.size(), arguments.data(), arguments.size(),
                outputs.data(), outputs.size(), ref::active_values.x.size(), 1, 1};
    }
};

topology_storage make_topology() {
    topology_storage result;
    std::size_t argument_begin = 0;
    std::size_t output_begin = 0;
    for (std::size_t i = 0; i < ref::mechanisms.size(); ++i) {
        const auto& source = ref::mechanisms[i];
        auto& destination = result.groups[i];
        destination.argument_begin = argument_begin;
        destination.argument_count = source.arity;
        destination.output_begin = output_begin;
        destination.parameter_index = source.parameter_index;
        destination.condition_index = source.condition_index;
        destination.logical_id = source.logical_id;
        for (std::size_t slot = 0; slot < source.arity; ++slot)
            result.arguments[argument_begin + slot] = source.ordered_inputs[slot];
        argument_begin += source.arity;

        for (std::size_t coordinate = 0; coordinate < source.output_scales.size(); ++coordinate) {
            if (source.output_scales[coordinate] == 0.0)
                continue;
            result.outputs[output_begin++] = {
                coordinate, static_cast<float>(source.output_scales[coordinate])};
            ++destination.output_count;
        }
    }
    require(argument_begin == result.arguments.size(), "fixture argument incidence extent");
    require(output_begin == result.outputs.size(), "fixture output incidence extent");
    return result;
}

void compare_host_candidate_to_reference() {
    auto topology = make_topology();
    std::array<double, 5> x = ref::active_values.x;
    std::array<double, 2> z{{ref::active_values.z0, ref::active_values.z1}};
    std::array<double, 1> theta{{ref::active_values.theta}};
    std::array<double, 1> phi{{ref::active_values.phi}};
    std::array<double, 1> tau{{ref::active_values.tau}};
    auto forward = mech::gated_product_forward(
        topology.view(), 1, std::span<const double>(x), std::span<const double>(z),
        std::span<const double>(theta), std::span<const double>(phi),
        std::span<const double>(tau), ref::active_values.beta);
    const auto independent_values = ref::evaluate(ref::active_values,
        [](std::size_t i) { return ref::active_values.x[i]; });
    require(forward.values.size() == independent_values.size(),
            "host candidate output extent");
    for (std::size_t i = 0; i < forward.values.size(); ++i)
        near(forward.values[i], independent_values[i],
             "host candidate forward matches independent reference");
    require(forward.tape.forward_participant_slot_reads() == 9,
            "active local forward reads every ordered participant slot");

    std::array<double, 5> dy = ref::loss_seed;
    bool malformed_rejected = false;
    try {
        (void)mech::gated_product_backward(
            std::move(forward.tape), std::span<const double>(dy.data(), dy.size() - 1));
    } catch (const std::invalid_argument&) {
        malformed_rejected = true;
    }
    require(malformed_rejected && !forward.tape.consumed(),
            "malformed cotangent rejects before consuming the saved tape");

    // The host operation promises an owning tape. Mutate every source buffer and
    // the borrowed topology after forward to catch backward rereads of caller data.
    x.fill(-101.0);
    z.fill(203.0);
    theta.fill(-307.0);
    phi.fill(409.0);
    tau.fill(-503.0);
    topology.arguments.fill(4);
    topology.outputs.fill({4, -7.0f});
    topology.groups[0].logical_id = 0;

    auto backward = mech::gated_product_backward(std::move(forward.tape),
                                                  std::span<const double>(dy));
    require(backward.dx.size() == ref::expected_dx.size(), "host candidate dx extent");
    for (std::size_t i = 0; i < backward.dx.size(); ++i)
        near(backward.dx[i], centered_difference(ref::active_values, i),
             "host candidate dx matches independent finite difference after source mutation");
    require(backward.dz.size() == 2 && backward.dtheta.size() == 1
            && backward.dphi.size() == 1 && backward.dtau.size() == 1,
            "host candidate condition/parameter gradient extents");
    near(backward.dz[0], centered_difference(ref::active_values, 5), "host candidate dz0");
    near(backward.dz[1], centered_difference(ref::active_values, 6), "host candidate dz1");
    near(backward.dtheta[0], centered_difference(ref::active_values, 7), "host candidate dtheta");
    near(backward.dphi[0], centered_difference(ref::active_values, 8), "host candidate dphi");
    near(backward.dtau[0], centered_difference(ref::active_values, 9), "host candidate dtau");
    require(backward.participant_slot_reads == 9,
            "active local backward reads every ordered participant slot");
    require(forward.tape.consumed(), "successful backward consumes the one-shot tape");

    bool replay_rejected = false;
    try {
        (void)mech::gated_product_backward(std::move(forward.tape),
                                            std::span<const double>(dy));
    } catch (const std::exception&) {
        replay_rejected = true;
    }
    require(replay_rejected, "a consumed forward tape cannot be replayed");
}

void compare_inactive_host_candidate_to_reference() {
    auto topology = make_topology();
    std::array<double, 5> x = ref::inactive_values.x;
    std::array<double, 2> z{{ref::inactive_values.z0, ref::inactive_values.z1}};
    std::array<double, 1> theta{{ref::inactive_values.theta}};
    std::array<double, 1> phi{{ref::inactive_values.phi}};
    std::array<double, 1> tau{{ref::inactive_values.tau}};
    auto forward = mech::gated_product_forward(
        topology.view(), 1, std::span<const double>(x), std::span<const double>(z),
        std::span<const double>(theta), std::span<const double>(phi),
        std::span<const double>(tau), ref::inactive_values.beta);
    std::size_t reference_participant_reads = 0;
    const auto expected = ref::evaluate(ref::inactive_values,
        never_read_participant{&reference_participant_reads});
    require(reference_participant_reads == 0,
            "inactive scalar oracle does not read participant slots");
    require(forward.values.size() == expected.size(), "inactive candidate output extent");
    for (std::size_t i = 0; i < expected.size(); ++i)
        near(forward.values[i], expected[i], "inactive candidate output retains baseline");
    require(forward.tape.forward_participant_slot_reads() == 0,
            "inactive host forward skips all participant slots");

    std::array<double, 5> dy = ref::loss_seed;
    auto backward = mech::gated_product_backward(std::move(forward.tape),
                                                  std::span<const double>(dy));
    require(backward.participant_slot_reads == 0,
            "inactive host backward skips all participant slots");
    const std::array<double, 5> expected_dx{{-0.25, 0.0, 0.125, 0.0, 0.0}};
    require(backward.dx.size() == expected_dx.size(), "inactive candidate dx extent");
    for (std::size_t i = 0; i < expected_dx.size(); ++i)
        near(backward.dx[i], expected_dx[i], "inactive candidate baseline dx");
    near(backward.dz[0], 0.0, "inactive candidate dz0 is zero");
    near(backward.dz[1], 0.0, "inactive candidate dz1 is zero");
    near(backward.dtheta[0], 0.0, "inactive candidate dtheta is zero");
    near(backward.dphi[0], 0.0, "inactive candidate dphi is zero");
    near(backward.dtau[0], 0.0, "inactive candidate dtau is zero");

    // At s=0 the squared positive-part gate remains first-order flat.
    z[0] = ref::active_values.z0;
    z[1] = ref::active_values.z1;
    tau[0] = 1.25;
    auto boundary = mech::gated_product_forward(
        topology.view(), 1, std::span<const double>(x), std::span<const double>(z),
        std::span<const double>(theta), std::span<const double>(phi),
        std::span<const double>(tau), ref::active_values.beta);
    require(boundary.tape.forward_participant_slot_reads() == 0,
            "boundary host forward skips participant slots");
    auto boundary_backward = mech::gated_product_backward(
        std::move(boundary.tape), std::span<const double>(dy));
    require(boundary_backward.participant_slot_reads == 0,
            "boundary host backward skips participant slots");
    near(boundary_backward.dz[0], 0.0, "boundary dz0 is zero");
    near(boundary_backward.dz[1], 0.0, "boundary dz1 is zero");
    near(boundary_backward.dtheta[0], 0.0, "boundary dtheta is zero");
    near(boundary_backward.dphi[0], 0.0, "boundary dphi is zero");
    near(boundary_backward.dtau[0], 0.0, "boundary dtau is zero");
}

void active_reference_and_distributed_partials() {
    const auto y = ref::evaluate(ref::active_values,
        [](std::size_t i) { return ref::active_values.x[i]; });
    for (std::size_t i = 0; i < y.size(); ++i)
        near(y[i], ref::expected_outputs[i], "active output reference");

    for (std::size_t axis = 0; axis < 10; ++axis)
        near(centered_difference(ref::active_values, axis),
             axis < 5 ? ref::expected_dx[axis]
                      : ref::expected_dtheta_dz0_dz1_dphi_dtau[parameter_result_index(axis)],
             "active first-order derivative against independent finite difference");

    for (int device : {0, 1}) {
        const auto& expected_dx = device == 0 ? ref::expected_device0_dx : ref::expected_device1_dx;
        const auto& expected_params = device == 0
            ? ref::expected_device0_dtheta_dz0_dz1_dphi_dtau
            : ref::expected_device1_dtheta_dz0_dz1_dphi_dtau;
        for (std::size_t axis = 0; axis < 10; ++axis)
            near(centered_difference(ref::active_values, axis, device, false),
                 axis < 5 ? expected_dx[axis] : expected_params[parameter_result_index(axis)],
                 "per-device gradient partial against independent finite difference");
    }

    // The remote partition contributes nonzero gradients for the shared x1,
    // z0 condition and theta. Their reduced values must match the global result.
    const double remote_dx1 = centered_difference(ref::active_values, 1, 1, false);
    const double remote_dz0 = centered_difference(ref::active_values, 5, 1, false);
    const double remote_dtheta = centered_difference(ref::active_values, 7, 1, false);
    require(remote_dx1 > 0.0 && remote_dz0 > 0.0 && remote_dtheta > 0.0,
            "active remote participant/condition/parameter gradients are nonzero");
    near(centered_difference(ref::active_values, 1, 0, false) + remote_dx1,
         ref::expected_dx[1], "overlapping x1 gradient reduces across devices");
    near(centered_difference(ref::active_values, 5, 0, false) + remote_dz0,
         ref::expected_dtheta_dz0_dz1_dphi_dtau[1], "condition gradient reduces across devices");
    near(centered_difference(ref::active_values, 7, 0, false) + remote_dtheta,
         ref::expected_dtheta_dz0_dz1_dphi_dtau[0], "shared parameter gradient reduces across devices");

    // A zero product still has a nonzero derivative at its single zero factor;
    // the repeated [zero, zero] group has finite zero derivatives per slot.
    near(centered_difference(ref::active_values, 3, 1, false), 0.5625,
         "zero factor retains a nonzero participant derivative");
}

void inactive_gate_and_baseline_ownership() {
    std::size_t participant_reads = 0;
    const auto y = ref::evaluate(ref::inactive_values,
                                 never_read_participant{&participant_reads});
    require(participant_reads == 0, "certified inactive branch skips participant reads");
    near(y[0], -0.5, "inactive output retains nonzero baseline y0 once");
    near(y[1], -0.75, "inactive output retains baseline y1");
    near(y[2], -0.125, "inactive output retains baseline y2 once");
    near(y[3], 0.0, "inactive output retains baseline y3");
    near(y[4], -0.625, "inactive output retains baseline y4");
    for (std::size_t axis = 0; axis < 10; ++axis) {
        const double expected = axis < 3
            ? -ref::inactive_values.beta * ref::loss_seed[axis]
            : 0.0;
        near(centered_difference(ref::inactive_values, axis), expected,
             "inactive first-order derivative keeps only the baseline");
    }

    auto boundary = ref::active_values;
    boundary.tau = 1.25;
    require(ref::gate_argument(boundary) == 0.0, "boundary fixture has s=0");
    require(ref::evaluate(boundary, never_read_participant{&participant_reads})[0]
                == -ref::active_values.beta * ref::active_values.x[0],
            "squared ReLU boundary follows the certified flat branch");
    require(participant_reads == 0, "boundary branch also skips participant reads");
}

void wrong_order_and_one_update_reference() {
    // Equal extents do not make a permuted value binding semantically valid.
    // The real API test will require this same-size wrong-order association to
    // be rejected before launch; this controls the output corruption if ignored.
    auto stale_order = ref::active_values;
    std::swap(stale_order.x[0], stale_order.x[1]);
    require(stale_order.x.size() == ref::active_values.x.size(),
            "wrong-order negative control has the same size");
    const auto correct = ref::evaluate(ref::active_values,
        [](std::size_t i) { return ref::active_values.x[i]; });
    const auto wrong = ref::evaluate(stale_order,
        [&stale_order](std::size_t i) { return stale_order.x[i]; });
    require(!close(wrong[0], correct[0]),
            "ignoring a same-size wrong-order binding changes the result");

    const double reduced_dtheta = centered_difference(ref::active_values, 7);
    auto updated = ref::active_values;
    updated.theta -= 0.01 * reduced_dtheta; // one deterministic canonical SGD update
    near(updated.theta, ref::expected_theta_after_one_sgd_update,
         "one update uses the globally reduced parameter gradient");
    const auto next = ref::evaluate(updated,
        [&updated](std::size_t i) { return updated.x[i]; });
    near(next[0], ref::expected_y0_after_update,
         "post-publication forward uses the refreshed shared parameter");
    near(next[2], ref::expected_y2_after_update,
         "post-publication output contribution matches the reference");

    auto candidate_topology = make_topology();
    std::array<double, 5> next_x = updated.x;
    std::array<double, 2> next_z{{updated.z0, updated.z1}};
    std::array<double, 1> next_theta{{updated.theta}};
    std::array<double, 1> next_phi{{updated.phi}};
    std::array<double, 1> next_tau{{updated.tau}};
    const auto candidate_next = mech::gated_product_forward(
        candidate_topology.view(), 1, std::span<const double>(next_x),
        std::span<const double>(next_z), std::span<const double>(next_theta),
        std::span<const double>(next_phi), std::span<const double>(next_tau), updated.beta);
    near(candidate_next.values[0], next[0],
         "host candidate observes the once-updated canonical parameter");
    near(candidate_next.values[2], next[2],
         "host candidate next forward matches updated reference");

    // A generic zero-valued gate cannot prove that its derivative is zero.
    const double u = 0.0, v = 3.0, participant = 2.0, coefficient = 0.5;
    near(coefficient * u * v * participant, 0.0, "generic zero gate primal");
    near(coefficient * v * participant, 3.0, "generic zero gate has nonzero d/du");
}

void checked_same_size_order_binding() {
    using namespace cellerator::structure;
    const domain_id domain{901, 1};
    const order_id order{902, 1};
    const dependency::SourceRef source_ref{
        dependency::source_id{903, 1}, domain, dependency::representation_id{904, 1}};
    const dependency::SourceDescriptor source{source_ref, LogicalAxis{domain, order, 5}};
    const dependency::SelectionRef selection_ref{
        dependency::selection_id{905, 1}, domain, dependency::SelectionSpace::coordinates};
    dependency::SelectionDescriptor selection{selection_ref, order, 5,
        dependency::SelectionKind::all, 0, 0, {}};
    const dependency::ReadDependency read{source_ref, selection_ref};
    const std::array<dependency::SourceDescriptor, 1> sources{{source}};
    const std::array<dependency::SelectionDescriptor, 1> selections{{selection}};
    const std::array<dependency::ReadDependency, 1> reads{{read}};

    require(static_cast<bool>(dependency::validate_source(source)),
            "valid source association accepted");
    require(static_cast<bool>(dependency::validate_selection(selection)),
            "valid coordinate selection accepted");
    require(static_cast<bool>(dependency::validate_reads(sources, selections, reads)),
            "same-domain, same-order, same-size read association accepted");

    auto wrong_order = selection;
    wrong_order.order = order_id{906, 1};
    const std::array<dependency::SelectionDescriptor, 1> wrong_order_selections{{wrong_order}};
    const auto order_status = dependency::validate_reads(
        sources, wrong_order_selections, reads);
    require(order_status.code == dependency::StatusCode::order_mismatch,
            "equal extent cannot hide a wrong logical order");

    auto wrong_domain = selection;
    wrong_domain.reference.domain = domain_id{907, 1};
    const std::array<dependency::SelectionDescriptor, 1> wrong_domain_selections{{wrong_domain}};
    auto wrong_domain_read = read;
    wrong_domain_read.selection.domain = wrong_domain.reference.domain;
    const std::array<dependency::ReadDependency, 1> wrong_domain_reads{{wrong_domain_read}};
    const auto domain_status = dependency::validate_reads(
        sources, wrong_domain_selections, wrong_domain_reads);
    require(domain_status.code == dependency::StatusCode::domain_mismatch,
            "same-size, wrong-domain selection is rejected separately");
}

} // namespace

int main() try {
    active_reference_and_distributed_partials();
    inactive_gate_and_baseline_ownership();
    wrong_order_and_one_update_reference();
    compare_host_candidate_to_reference();
    compare_inactive_host_candidate_to_reference();
    checked_same_size_order_binding();
    std::cout << "native-data independent host reference passed; production backend=false; gpu=false\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
