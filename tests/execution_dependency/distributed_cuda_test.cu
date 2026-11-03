#include "gated_product_fixture.hh"

#include <Cellerator/execution/dependency/distributed.cuh>

#include <cuda_runtime_api.h>

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace fixture = cellerator::test::execution_dependency;
namespace reference = cellerator::test::native_data_reference;
namespace dependency = cellerator::execution::dependency;

namespace {

constexpr float atol = 2.0e-5f;
constexpr float rtol = 2.0e-5f;

bool parse_devices(int argc, char** argv, std::array<int, 2>& devices) {
    std::array<bool, 2> seen{};
    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        const std::size_t slot = option == "--device0" ? 0
            : option == "--device1" ? 1 : devices.size();
        if (slot == devices.size()) {
            std::cerr << "unknown argument: " << option << '\n';
            return false;
        }
        if (seen[slot]) {
            std::cerr << "duplicate argument: " << option << '\n';
            return false;
        }
        if (++i >= argc) {
            std::cerr << "missing value for " << option << '\n';
            return false;
        }

        const std::string_view value(argv[i]);
        int ordinal = -1;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), ordinal);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || ordinal < 0) {
            std::cerr << "invalid nonnegative CUDA ordinal for " << option << ": " << value << '\n';
            return false;
        }
        devices[slot] = ordinal;
        seen[slot] = true;
    }
    return true;
}

bool validate_devices(const std::array<int, 2>& devices, int device_count) {
    if (devices[0] == devices[1]) {
        std::cerr << "--device0 and --device1 must select distinct CUDA devices\n";
        return false;
    }
    for (std::size_t slot = 0; slot < devices.size(); ++slot) {
        if (devices[slot] >= device_count) {
            std::cerr << "--device" << slot << " selects CUDA ordinal " << devices[slot]
                      << ", but only " << device_count << " device(s) are visible\n";
            return false;
        }
    }
    return true;
}

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

bool close(float actual, float expected) {
    return std::isfinite(actual) && std::isfinite(expected)
        && std::fabs(actual - expected) <= atol + rtol * std::fabs(expected);
}

bool same_bits(float left, float right) {
    return std::memcmp(&left, &right, sizeof(float)) == 0;
}

void near(float actual, float expected, const char* message) {
    if (!close(actual, expected))
        throw std::runtime_error(std::string(message) + ": actual="
                                 + std::to_string(actual) + " expected="
                                 + std::to_string(expected));
}

void exactly(float actual, float expected, const char* message) {
    if (!std::isfinite(actual) || actual != expected)
        throw std::runtime_error(std::string(message) + ": actual="
                                 + std::to_string(actual) + " expected="
                                 + std::to_string(expected));
}

template<class Function>
void require_throws(Function&& function, const char* message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

void validate_semantic_admission(fixture::GatedProductFixture& data,
                                 std::array<int, 2> devices,
                                 dependency::TransferMode mode) {
    const auto catalog = data.source_catalog();
    require(catalog.size() == 6, "fixture exposes all six named sources");
    require(static_cast<bool>(data.semantics->validate_sources(catalog)),
            "fixture source catalog validates against its prepared semantics");
    require(data.semantics->access().revision == 1,
            "fixture binds an explicit access-contract revision");

    auto wrong_domain = catalog;
    ++wrong_domain[0].reference.domain.low;
    require(data.semantics->validate_sources(wrong_domain).code
                == cellerator::geometry::dependency::StatusCode::domain_mismatch,
            "same-size wrong-domain source association is rejected");
    auto wrong_order = catalog;
    ++wrong_order[0].axis.logical_order.low;
    require(data.semantics->validate_sources(wrong_order).code
                == cellerator::geometry::dependency::StatusCode::order_mismatch,
            "same-size wrong-order source association is rejected");

    auto options = data.options(devices, mode);
    options.source_bindings = wrong_domain;
    require_throws([&] {
        dependency::PreparedDistributedGatedProduct invalid(
            options, data.packed_values, data.regulators,
            data.theta, data.phi, data.tau);
    }, "distributed admission rejects a same-size wrong-domain catalog");

    options = data.options(devices, mode);
    options.source_bindings = wrong_order;
    require_throws([&] {
        dependency::PreparedDistributedGatedProduct invalid(
            options, data.packed_values, data.regulators,
            data.theta, data.phi, data.tau);
    }, "distributed admission rejects a same-size wrong-order catalog");

    options = data.options(devices, mode);
    ++options.access_revision;
    require_throws([&] {
        dependency::PreparedDistributedGatedProduct invalid(
            options, data.packed_values, data.regulators,
            data.theta, data.phi, data.tau);
    }, "distributed admission rejects a stale access-contract revision");

    options = data.options(devices, mode);
    auto mismatched_groups = data.groups;
    ++mismatched_groups[0].logical_id;
    options.topology.groups = mismatched_groups.data();
    require_throws([&] {
        dependency::PreparedDistributedGatedProduct invalid(
            options, data.packed_values, data.regulators,
            data.theta, data.phi, data.tau);
    }, "distributed admission rejects same-extent topology drift");
}

std::vector<float> expected_outputs(float theta, bool updated) {
    const auto batch = fixture::fixture_batch;
    const auto coordinates = fixture::fixture_coordinates;
    std::vector<float> expected(batch * coordinates, 0.0f);
    for (std::size_t row = 0; row < batch; ++row) {
        if (row == 0 && !updated) {
            for (std::size_t coordinate = 0;
                 coordinate < reference::expected_outputs.size(); ++coordinate)
                expected[row * coordinates + coordinate] =
                    static_cast<float>(reference::expected_outputs[coordinate]);
        } else if (row == 0) {
            auto values = reference::active_values;
            values.theta = theta;
            const auto logical = reference::evaluate(values,
                [&values](std::size_t coordinate) { return values.x[coordinate]; });
            for (std::size_t coordinate = 0; coordinate < logical.size(); ++coordinate)
                expected[row * coordinates + coordinate] =
                    static_cast<float>(logical[coordinate]);
            near(expected[row * coordinates],
                 static_cast<float>(reference::expected_y0_after_update),
                 "independent oracle post-update y0 fixture");
            near(expected[row * coordinates + 2],
                 static_cast<float>(reference::expected_y2_after_update),
                 "independent oracle post-update y2 fixture");
        } else {
            for (std::size_t coordinate = 0;
                 coordinate < reference::inactive_values.x.size(); ++coordinate)
                expected[row * coordinates + coordinate] = static_cast<float>(
                    -reference::inactive_values.beta * reference::inactive_values.x[coordinate]);
        }
    }
    return expected;
}

std::vector<float> reference_outputs_for_parameters(
    float theta, float phi, float tau) {
    std::vector<float> expected(fixture::fixture_batch * fixture::fixture_coordinates, 0.0f);
    for (std::size_t row = 0; row < fixture::fixture_batch; ++row) {
        auto values = row == 0 ? reference::active_values : reference::inactive_values;
        values.theta = theta;
        values.phi = phi;
        values.tau = tau;
        const auto logical = reference::evaluate(values,
            [&values](std::size_t coordinate) { return values.x[coordinate]; });
        for (std::size_t coordinate = 0; coordinate < logical.size(); ++coordinate)
            expected[row * fixture::fixture_coordinates + coordinate] =
                static_cast<float>(logical[coordinate]);
    }
    return expected;
}

void compare_vector(std::span<const float> actual,
                    std::span<const float> expected,
                    const char* message) {
    require(actual.size() == expected.size(), message);
    for (std::size_t index = 0; index < expected.size(); ++index)
        near(actual[index], expected[index], message);
}

void check_permuted_canonical_rows(std::array<int, 2> devices,
                                   dependency::TransferMode mode) {
    fixture::GatedProductFixture data(false, true);
    require(data.structure->canonical_rows() == std::vector<std::uint64_t>{200, 100},
            "row-identity fixture keeps its nontrivial physical canonical row order");

    dependency::PreparedDistributedGatedProduct runner(
        data.options(devices, mode), data.packed_values, data.regulators,
        data.theta, data.phi, data.tau);
    auto tape = runner.forward();

    // Dense batch axes use ascending canonical row IDs: row 0 is canonical ID
    // 100 (physical row 1, active), row 1 is ID 200 (physical row 0, inactive).
    // These rows deliberately have different x, z and cotangents, so using a
    // canonical ID as a dense batch offset or conflating physical and logical
    // row order cannot accidentally pass.
    std::vector<float> expected(fixture::fixture_batch * fixture::fixture_coordinates, 0.0f);
    for (std::size_t coordinate = 0; coordinate < reference::expected_outputs.size();
         ++coordinate)
        expected[coordinate] = static_cast<float>(reference::expected_outputs[coordinate]);
    constexpr std::array<float, 5> inactive_x{{4.0f, 1.0f, 1.5f, 2.0f, 0.5f}};
    for (std::size_t coordinate = 0; coordinate < inactive_x.size(); ++coordinate)
        expected[fixture::fixture_coordinates + coordinate] = -0.25f * inactive_x[coordinate];
    const auto values = runner.values(tape);
    compare_vector(values, expected,
                   "permuted physical rows return values in ascending canonical row order");

    runner.backward(tape, data.cotangents);
    const auto gradient = runner.gradients(tape);
    for (std::size_t coordinate = 0; coordinate < reference::expected_dx.size(); ++coordinate)
        near(gradient.dx[coordinate], static_cast<float>(reference::expected_dx[coordinate]),
             "canonical ID 100 retains its active input VJP");
    constexpr std::array<float, 5> inactive_seed{{0.5f, 1.0f, 0.25f, 0.0f, 1.0f}};
    for (std::size_t coordinate = 0; coordinate < inactive_seed.size(); ++coordinate)
        exactly(gradient.dx[fixture::fixture_coordinates + coordinate],
                -0.25f * inactive_seed[coordinate],
                "canonical ID 200 retains only its row-specific baseline VJP");
    for (std::size_t coordinate = inactive_seed.size(); coordinate < fixture::fixture_coordinates;
         ++coordinate)
        exactly(gradient.dx[fixture::fixture_coordinates + coordinate], 0.0f,
                "unused canonical ID 200 coordinates have zero VJP");

    near(gradient.dz[0],
         static_cast<float>(reference::expected_dtheta_dz0_dz1_dphi_dtau[1]),
         "active regulator row follows canonical ID 100");
    near(gradient.dz[1],
         static_cast<float>(reference::expected_dtheta_dz0_dz1_dphi_dtau[2]),
         "active regulator channel follows canonical ID 100");
    exactly(gradient.dz[2], 0.0f, "inactive canonical ID 200 z0 VJP is zero");
    exactly(gradient.dz[3], 0.0f, "inactive canonical ID 200 z1 VJP is zero");
    near(gradient.dtheta[0],
         static_cast<float>(reference::expected_dtheta_dz0_dz1_dphi_dtau[0]),
         "only the active canonical row contributes the shared parameter VJP");
    near(gradient.dphi[0],
         static_cast<float>(reference::expected_dtheta_dz0_dz1_dphi_dtau[3]),
         "only the active canonical row contributes the phi VJP");
    near(gradient.dtau[0],
         static_cast<float>(reference::expected_dtheta_dz0_dz1_dphi_dtau[4]),
         "only the active canonical row contributes the tau VJP");
    tape.wait();
}

void check_logical_order_acceptance(std::array<int, 2> devices,
                                   dependency::TransferMode mode,
                                   cellerator::structure::order_id logical_order) {
    fixture::GatedProductFixture data(false, false, logical_order);
    dependency::PreparedDistributedGatedProduct runner(
        data.options(devices, mode), data.packed_values, data.regulators,
        data.theta, data.phi, data.tau);
    auto tape = runner.forward();
    const auto expected = expected_outputs(
        static_cast<float>(reference::active_values.theta), false);
    compare_vector(runner.values(tape), expected,
                   "explicit physical order identities accept valid low logical-order IDs");
    tape.wait();
}

void check_missing_physical_order_rejected(std::array<int, 2> devices) {
    fixture::GatedProductFixture data;
    auto options = data.options(devices, dependency::TransferMode::staged);
    options.physical_orders[1] = {};
    require_throws([&] {
        dependency::PreparedDistributedGatedProduct invalid(
            options, data.packed_values, data.regulators,
            data.theta, data.phi, data.tau);
    }, "distributed preparation rejects a missing physical order identity");
}

void check_unsupported_outputless_group(std::array<int, 2> devices) {
    fixture::GatedProductFixture data;
    data.groups[0].output_count = 0;
    namespace op = cellerator::ops::mechanism;
    namespace gd = cellerator::geometry::dependency;
    const auto& original = *data.semantics;
    const op::GatedProductSourceBindings sources{
        original.source(op::GatedProductSource::state),
        original.source(op::GatedProductSource::regulators),
        original.source(op::GatedProductSource::theta),
        original.source(op::GatedProductSource::phi),
        original.source(op::GatedProductSource::tau),
        original.source(op::GatedProductSource::result)};
    data.semantics = std::make_shared<const op::PreparedGatedProductBinding>(
        data.topology(), sources,
        gd::Applicability{{1201, 0}, {1202, 0}, {1203, 0}, {1204, 0}, {1}},
        gd::foundation_id{1301, 0}, op::operation_id{1302, 0}, op::access_contract_id{1303, 0});
    require_throws([&] {
        dependency::PreparedDistributedGatedProduct invalid(
            data.options(devices, dependency::TransferMode::staged), data.packed_values,
            data.regulators, data.theta, data.phi, data.tau);
    }, "bounded distributed consumer rejects outputless groups before undeclared reads");
}

void check_active_evaluation(fixture::GatedProductFixture& data,
                             std::array<int, 2> devices,
                             dependency::TransferMode mode,
                             dependency::PreparedDistributedGatedProduct& runner) {
    validate_semantic_admission(data, devices, mode);

    dependency::DistributedTransferStats first_stats{};
    std::uint64_t initial_parameter_generation = 0;
    std::array<std::vector<float>, 3> updated_parameters;
    {
        auto tape = runner.forward();
        initial_parameter_generation = tape.parameter_generation();
        require(initial_parameter_generation == runner.parameter_generation(),
                "forward tape retains the parameter generation it read");
        const auto output = runner.values(tape);
        const auto expected = expected_outputs(
            static_cast<float>(reference::active_values.theta), false);
        compare_vector(output, expected,
                       "active/inactive batch outputs match the independent exact oracle");

        // A new value generation cannot overwrite the exact state retained by
        // this active tape. A malformed cotangent likewise fails before it is
        // consumed.
        require_throws([&] { runner.upload_values(data.packed_values, 2); },
                       "a live forward lease rejects value-generation mutation");
        require(!tape.consumed(), "rejected source mutation leaves the tape live");
        require_throws([&] { runner.update(tape, 0.01f); },
                       "parameters cannot update before the forward VJP is complete");
        require(!tape.consumed(), "rejected early update leaves the tape live");
        require_throws([&] {
            runner.backward(tape, std::span<const float>(data.cotangents.data(),
                                                         data.cotangents.size() - 1));
        }, "malformed cotangents fail before tape consumption");
        require(!tape.consumed(), "malformed backward leaves the tape live");
        require_throws([&] { (void)runner.gradients(tape); },
                       "rejected backward cannot publish gradient contents");
        require_throws([&] { runner.update(tape, 0.01f); },
                       "rejected backward cannot update canonical parameters");

        runner.backward(tape, data.cotangents);
        require(tape.consumed(), "successful backward consumes its one-shot tape");
        const auto gradient = runner.gradients(tape);
        require(gradient.dx.size() == fixture::fixture_batch * fixture::fixture_coordinates,
            "state gradient has batch-by-coordinate extent");
        for (std::size_t coordinate = 0; coordinate < reference::expected_dx.size(); ++coordinate) {
            near(gradient.dx[coordinate],
                 static_cast<float>(reference::expected_dx[coordinate]),
                 "active row state VJP matches exact independent oracle");
            near(gradient.dx[fixture::fixture_coordinates + coordinate],
                 static_cast<float>(-reference::active_values.beta
                                    * reference::loss_seed[coordinate]),
                 "inactive row keeps only the baseline state VJP");
        }
        for (std::size_t coordinate = reference::expected_dx.size();
             coordinate < fixture::fixture_coordinates; ++coordinate) {
            near(gradient.dx[coordinate], 0.0f,
                 "unused active-row coordinate has zero VJP");
            near(gradient.dx[fixture::fixture_coordinates + coordinate], 0.0f,
                 "unused inactive-row coordinate has zero VJP");
        }
        require(gradient.dz.size() == fixture::fixture_batch * 2,
                "regulator VJP has batch-condition-channel extent");
        near(gradient.dz[0], static_cast<float>(reference::expected_dtheta_dz0_dz1_dphi_dtau[1]),
             "active condition z0 VJP matches exact oracle");
        near(gradient.dz[1], static_cast<float>(reference::expected_dtheta_dz0_dz1_dphi_dtau[2]),
             "active condition z1 VJP matches exact oracle");
        near(gradient.dz[2], 0.0f, "inactive condition z0 VJP is zero");
        near(gradient.dz[3], 0.0f, "inactive condition z1 VJP is zero");
        require(gradient.dtheta.size() == 1 && gradient.dphi.size() == 1
                    && gradient.dtau.size() == 1,
                "shared parameter gradients have one canonical slot each");
        near(gradient.dtheta[0],
             static_cast<float>(reference::expected_dtheta_dz0_dz1_dphi_dtau[0]),
             "shared parameter VJP includes both device partitions exactly once");
        near(gradient.dphi[0],
             static_cast<float>(reference::expected_dtheta_dz0_dz1_dphi_dtau[3]),
             "gate parameter phi VJP matches exact oracle");
        near(gradient.dtau[0],
             static_cast<float>(reference::expected_dtheta_dz0_dz1_dphi_dtau[4]),
             "gate parameter tau VJP matches exact oracle");

        first_stats = runner.stats();
        require(first_stats.prepared_program_forwards == 2
                && first_stats.prepared_program_backwards == 2,
                "both devices execute forward and backward through prepared programs");
        require(first_stats.condition_bytes > 0,
                "condition-first phase transfers its remote regulator values");
        require(first_stats.participant_bytes > 0,
                "active remote groups transfer their required participant values");
        require(first_stats.parameter_bytes > 0,
                "remote groups consume the shared canonical parameter values");
        require(first_stats.projected_participant_values == 4,
                "only four unique active-row participant values are projected remotely");
    require(reference::expected_device1_dx[1] > 0.0
                && reference::expected_device1_dtheta_dz0_dz1_dphi_dtau[0] > 0.0
                && reference::expected_device1_dtheta_dz0_dz1_dphi_dtau[1] > 0.0,
            "independent oracle has nonzero remote participant, condition, and parameter VJPs");

        require_throws([&] { runner.backward(tape, data.cotangents); },
                       "a one-shot tape rejects a second backward");

        tape.wait();
        constexpr float learning_rate = 0.01f;
        const auto parameters_before_update = runner.parameters();
        runner.update(tape, learning_rate);
        runner.synchronize();
        updated_parameters = runner.parameters();
        require(updated_parameters[0].size() == 1,
                "one canonical theta parameter remains after reduction");
        near(updated_parameters[0][0],
             static_cast<float>(reference::expected_theta_after_one_sgd_update),
             "single-owner SGD uses the reduced parameter gradient once");
        require(!same_bits(updated_parameters[0][0], parameters_before_update[0][0]),
                "scalar learning rate changes theta");
        require(same_bits(updated_parameters[1][0], parameters_before_update[1][0]),
                "scalar learning rate leaves phi bitwise unchanged");
        require(same_bits(updated_parameters[2][0], parameters_before_update[2][0]),
                "scalar learning rate leaves tau bitwise unchanged");
        require(runner.parameter_generation() == initial_parameter_generation + 1,
                "the update publishes one new parameter generation");
        require_throws([&] { runner.update(tape, learning_rate); },
                       "a forward's reduced gradient cannot update parameters twice");
    }

    {
        const auto before_theta_only_forward = runner.stats();
        auto next_tape = runner.forward();
        const auto next_output = runner.values(next_tape);
        const auto next_expected = expected_outputs(updated_parameters[0][0], true);
        compare_vector(next_output, next_expected,
                       "next forward observes the refreshed parameter replica");
        next_tape.wait();
        runner.synchronize();
        const auto after_theta_only_forward = runner.stats();
        require(after_theta_only_forward.projection_refreshes
                    == before_theta_only_forward.projection_refreshes,
                "theta-only update reuses the participant projection on next forward");
        require(after_theta_only_forward.participant_bytes
                    == before_theta_only_forward.participant_bytes,
                "theta-only update transfers no participant values on next forward");
    }

    std::cout << "mode="
              << (mode == dependency::TransferMode::staged ? "staged" : "peer_if_supported")
              << " peer_forward=" << first_stats.peer_forward
              << " peer_reverse=" << first_stats.peer_reverse
              << " condition_bytes=" << first_stats.condition_bytes
              << " participant_bytes=" << first_stats.participant_bytes
              << " parameter_bytes=" << first_stats.parameter_bytes << '\n';
}

void check_gate_parameter_update_and_projection_refresh(
    dependency::TransferMode mode, std::array<int, 2> devices) {
    fixture::GatedProductFixture data;
    dependency::PreparedDistributedGatedProduct runner(
        data.options(devices, mode), data.packed_values, data.regulators,
        data.theta, data.phi, data.tau);
    auto tape = runner.forward();
    runner.backward(tape, data.cotangents);
    const auto gradients = runner.gradients(tape);
    near(gradients.dphi[0],
         static_cast<float>(reference::expected_dtheta_dz0_dz1_dphi_dtau[3]),
         "explicit gate-rate update uses the independent phi gradient");
    near(gradients.dtau[0],
         static_cast<float>(reference::expected_dtheta_dz0_dz1_dphi_dtau[4]),
         "explicit gate-rate update uses the independent tau gradient");
    tape.wait();

    const auto parameters_before_update = runner.parameters();
    constexpr float phi_rate = 0.001f;
    constexpr float tau_rate = 0.001f;
    runner.update(tape, dependency::ParameterLearningRates{0.0f, phi_rate, tau_rate});
    runner.synchronize();
    const auto parameters_after_update = runner.parameters();
    require(same_bits(parameters_after_update[0][0], parameters_before_update[0][0]),
            "zero theta rate leaves theta bitwise unchanged in explicit update");
    near(parameters_after_update[1][0],
         parameters_before_update[1][0] - phi_rate * gradients.dphi[0],
         "explicit phi rate updates the gate parameter");
    near(parameters_after_update[2][0],
         parameters_before_update[2][0] - tau_rate * gradients.dtau[0],
         "explicit tau rate updates the gate threshold");

    const auto before_gate_forward = runner.stats();
    auto refreshed_tape = runner.forward();
    const auto refreshed_output = runner.values(refreshed_tape);
    refreshed_tape.wait();
    runner.synchronize();
    const auto after_gate_forward = runner.stats();
    require(after_gate_forward.projection_refreshes
                > before_gate_forward.projection_refreshes,
            "gate-parameter generation refreshes the participant projection");
    const auto expected = reference_outputs_for_parameters(
        parameters_after_update[0][0], parameters_after_update[1][0],
        parameters_after_update[2][0]);
    compare_vector(refreshed_output, expected,
                   "forward after gate update matches the independent scalar oracle");
    require(!close(refreshed_output[0],
                   static_cast<float>(reference::expected_outputs[0])),
            "gate parameter update changes the active output");
}

void check_all_inactive_transfer_omission(
    dependency::PreparedDistributedGatedProduct& runner,
    const fixture::GatedProductFixture& data,
    const fixture::GatedProductFixture& inactive_values) {
    const auto before = runner.stats();
    runner.upload_regulators(inactive_values.regulators, 2);
    auto tape = runner.forward();
    const auto after_forward = runner.stats();
    require(after_forward.participant_bytes == before.participant_bytes,
            "certified inactive gates transfer no participant values");
    require(after_forward.omitted_participant_transfers
                > before.omitted_participant_transfers,
            "inactive phase records omitted participant-transfer work");

    const auto actual = runner.values(tape);
    std::vector<float> expected(fixture::fixture_batch * fixture::fixture_coordinates, 0.0f);
    for (std::size_t row = 0; row < fixture::fixture_batch; ++row)
        for (std::size_t coordinate = 0; coordinate < reference::active_values.x.size();
             ++coordinate)
            expected[row * fixture::fixture_coordinates + coordinate] =
                static_cast<float>(-reference::active_values.beta
                                   * reference::active_values.x[coordinate]);
    require(actual.size() == expected.size(),
            "all-inactive baseline output has the expected extent");
    for (std::size_t index = 0; index < expected.size(); ++index)
        exactly(actual[index], expected[index],
                "all-inactive forward preserves the exact baseline");

    runner.backward(tape, data.cotangents);
    const auto gradient = runner.gradients(tape);
    for (std::size_t row = 0; row < fixture::fixture_batch; ++row) {
        for (std::size_t coordinate = 0; coordinate < reference::loss_seed.size(); ++coordinate)
            exactly(gradient.dx[row * fixture::fixture_coordinates + coordinate],
                    static_cast<float>(-reference::active_values.beta
                                       * reference::loss_seed[coordinate]),
                    "inactive state VJP retains the exact baseline term");
        for (std::size_t coordinate = reference::loss_seed.size();
             coordinate < fixture::fixture_coordinates; ++coordinate)
            exactly(gradient.dx[row * fixture::fixture_coordinates + coordinate], 0.0f,
                    "inactive unused state coordinate VJP is zero");
    }
    for (float value : gradient.dz)
        exactly(value, 0.0f, "inactive regulator VJPs are exactly zero");
    for (float value : gradient.dtheta)
        exactly(value, 0.0f, "inactive shared-parameter VJP is exactly zero");
    for (float value : gradient.dphi)
        exactly(value, 0.0f, "inactive phi VJP is exactly zero");
    for (float value : gradient.dtau)
        exactly(value, 0.0f, "inactive tau VJP is exactly zero");

    const auto after_backward = runner.stats();
    require(after_backward.participant_bytes == before.participant_bytes,
            "inactive backward also avoids participant transfers");
    require(after_backward.omitted_participant_transfers
                >= after_forward.omitted_participant_transfers,
            "transfer omission counter remains monotonic through backward");
    tape.wait();
}

void check_direct_training_update(std::array<int, 2> devices,
                                  dependency::TransferMode mode) {
    fixture::GatedProductFixture data;
    dependency::PreparedDistributedGatedProduct runner(
        data.options(devices, mode), data.packed_values, data.regulators,
        data.theta, data.phi, data.tau);
    {
        auto tape = runner.forward();
        runner.backward(tape, data.cotangents);
        // Training must enforce completion and status without optional
        // forward-value or gradient downloads.
        runner.update(tape, 0.01f);
        runner.synchronize();
        const auto parameters = runner.parameters();
        near(parameters[0][0], static_cast<float>(reference::expected_theta_after_one_sgd_update),
             "direct backward/update uses the completed distributed VJP");
        require(same_bits(parameters[1][0], data.phi[0])
                && same_bits(parameters[2][0], data.tau[0]),
                "direct scalar update preserves gate parameters");
        near(runner.gradients(tape).dtheta[0],
             static_cast<float>(reference::expected_dtheta_dz0_dz1_dphi_dtau[0]),
             "staged proposal preserves the canonical reduced gradient contents");
    }
    const auto before_failure = runner.parameters();
    const auto revision_before_failure = runner.parameter_generation();
    auto tape = runner.forward();
    std::vector<float> overflowing(data.cotangents.size(), std::numeric_limits<float>::max());
    runner.backward(tape, overflowing);
    require_throws([&] { runner.update(tape, 0.01f); },
                   "direct update rejects asynchronous nonfinite backward status");
    const auto after_failure = runner.parameters();
    require(before_failure == after_failure
            && runner.parameter_generation() == revision_before_failure,
            "rejected asynchronous backward leaves canonical parameters unpublished");
}

void check_reduction_overflow(std::array<int, 2> devices,
                              dependency::TransferMode mode) {
    fixture::GatedProductFixture data;
    const float limit = std::numeric_limits<float>::max();
    const float scale = limit / 8.5f;
    require(reference::expected_device0_dtheta_dz0_dz1_dphi_dtau[0] * scale < limit
            && reference::expected_device1_dtheta_dz0_dz1_dphi_dtau[0] * scale < limit
            && reference::expected_dtheta_dz0_dz1_dphi_dtau[0] * scale > limit,
            "overflow regression has finite local theta VJPs and overflowing reduction");
    {
        dependency::PreparedDistributedGatedProduct runner(
            data.options(devices, mode), data.packed_values, data.regulators,
            data.theta, data.phi, data.tau);
        const auto before = runner.parameters();
        std::vector<float> seed = data.cotangents;
        for (auto& value : seed) value *= scale;
        auto tape = runner.forward();
        runner.backward(tape, seed);
        require_throws([&] { runner.update(tape, 0.01f); },
                       "cross-device VJP overflow prevents parameter publication");
        require(before == runner.parameters() && runner.parameter_generation() == 1,
                "overflowing reduced VJP leaves canonical parameters unchanged");
    }
    data.theta[0] = limit / 11.0f;
    auto high = reference::active_values;
    high.theta = data.theta[0];
    const auto read = [&high](std::size_t coordinate) { return high.x[coordinate]; };
    const auto home = reference::evaluate(high, read, 0, true);
    const auto remote = reference::evaluate(high, read, 1, false);
    require(home[0] < limit && remote[0] < limit && home[0] + remote[0] > limit,
            "forward regression has finite device partials and overflowing sum");
    dependency::PreparedDistributedGatedProduct runner(
        data.options(devices, mode), data.packed_values, data.regulators,
        data.theta, data.phi, data.tau);
    auto tape = runner.forward();
    require_throws([&] { (void)runner.values(tape); },
                   "cross-device output overflow is reported before publication");
}

void check_update_proposal_overflow(std::array<int, 2> devices,
                                    dependency::TransferMode mode) {
    fixture::GatedProductFixture data;
    dependency::PreparedDistributedGatedProduct runner(
        data.options(devices, mode), data.packed_values, data.regulators,
        data.theta, data.phi, data.tau);
    const auto before = runner.parameters();
    const auto revision = runner.parameter_generation();
    auto tape = runner.forward();
    runner.backward(tape, data.cotangents);
    require_throws([&] { runner.update(tape, std::numeric_limits<float>::max()); },
                   "finite learning rate with nonfinite proposal is rejected");
    require(before == runner.parameters() && revision == runner.parameter_generation(),
            "rejected parameter proposal preserves the canonical version");
    require_throws([&] { (void)runner.gradients(tape); },
                   "failed update tape cannot be reused for numerical publication");
}

void run_mode(dependency::TransferMode mode, std::array<int, 2> devices) {
    fixture::GatedProductFixture active;
    fixture::GatedProductFixture inactive(true);
    dependency::PreparedDistributedGatedProduct runner(
        active.options(devices, mode), active.packed_values, active.regulators,
        active.theta, active.phi, active.tau);
    check_active_evaluation(active, devices, mode, runner);
    check_permuted_canonical_rows(devices, mode);
    check_logical_order_acceptance(devices, mode, {0, 1});
    check_logical_order_acceptance(devices, mode, {0, 2});
    check_all_inactive_transfer_omission(runner, active, inactive);
    check_gate_parameter_update_and_projection_refresh(mode, devices);
    check_direct_training_update(devices, mode);
    check_reduction_overflow(devices, mode);
    check_update_proposal_overflow(devices, mode);
}

} // namespace

int main(int argc, char** argv) {
    std::array<int, 2> devices{0, 1};
    if (!parse_devices(argc, argv, devices)) {
        std::cerr << "usage: " << argv[0] << " [--device0 N] [--device1 N]\n";
        return 2;
    }

    int device_count = 0;
    const auto count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_status) << '\n';
        return 1;
    }
    if (device_count < 1) {
        std::cerr << "distributed gated-product test requires visible CUDA devices\n";
        return 1;
    }
    if (!validate_devices(devices, device_count))
        return 2;

    try {
        check_missing_physical_order_rejected(devices);
        check_unsupported_outputless_group(devices);
        run_mode(dependency::TransferMode::staged, devices);
        run_mode(dependency::TransferMode::peer_if_supported, devices);
    } catch (const std::exception& error) {
        std::cerr << "distributed gated-product contract failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
