#include <Cellerator/execution/dependency/distributed.cuh>
#include <Cellerator/geometry/dependency/applicability.hh>
#include <Cellerator/ops/mechanism/gated_product_binding.hh>
#include <Cellerator/tensor/block_masked/structure.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace dependency = cellerator::execution::dependency;
namespace geometry = cellerator::geometry::dependency;
namespace mechanism = cellerator::ops::mechanism;
namespace block_masked = cellerator::tensor::block_masked;
namespace st = cellerator::structure;

namespace {

constexpr std::size_t coordinates = 32;
constexpr std::size_t argument_count = 2 * coordinates;
constexpr float absolute_tolerance = 2.0e-5f;
constexpr float relative_tolerance = 2.0e-5f;
constexpr std::uint64_t state_home_namespace = 0x4350333250484f4dull;
constexpr std::uint64_t state_projection_namespace = 0x4350333250524f4aull;

dependency::PackedAtomIdentities state_home_atom_ids() noexcept {
    return {{state_home_namespace, 1}, {state_home_namespace, 2},
            {state_home_namespace, 3}, {state_home_namespace, 4},
            {state_home_namespace, 5}};
}

dependency::PackedAtomIdentities state_projection_atom_ids() noexcept {
    return {{state_projection_namespace, 1}, {state_projection_namespace, 2},
            {state_projection_namespace, 3}, {state_projection_namespace, 4},
            {state_projection_namespace, 5}};
}

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
        && std::fabs(actual - expected)
            <= absolute_tolerance + relative_tolerance * std::fabs(expected);
}

void near(float actual, float expected, const char* message) {
    if (!close(actual, expected))
        throw std::runtime_error(std::string(message) + ": actual="
                                 + std::to_string(actual) + " expected="
                                 + std::to_string(expected));
}

std::uint16_t one_as_fp16_bits() noexcept {
    const __half one = __float2half_rn(1.0f);
    std::uint16_t bits = 0;
    static_assert(sizeof(one) == sizeof(bits));
    std::memcpy(&bits, &one, sizeof(bits));
    return bits;
}

struct Distributed32Fixture final {
    std::array<mechanism::GatedProductGroup, 2> groups{};
    std::array<std::uint64_t, argument_count> arguments{};
    std::array<mechanism::OutputIncidence, 2> outputs{{{0, 1.0f}, {0, 2.0f}}};
    std::array<std::uint8_t, 2> group_slots{{0, 1}};
    std::shared_ptr<const block_masked::Structure> structure;
    std::shared_ptr<const mechanism::PreparedGatedProductBinding> semantics;
    std::vector<std::uint16_t> packed_values;
    std::array<float, 2> regulators{{1.0f, 0.5f}};
    std::array<float, 1> theta{{0.4f}};
    std::array<float, 1> phi{{0.5f}};
    std::array<float, 1> tau{{0.5f}};

    Distributed32Fixture() {
        make_topology();
        make_structure_and_values();
        make_semantics();
    }

    mechanism::GatedProductTopologyView topology() const noexcept {
        return {groups.data(), groups.size(), arguments.data(), arguments.size(),
                outputs.data(), outputs.size(), coordinates, 1, 1};
    }

    dependency::DistributedGatedProductOptions options(
        std::array<int, 2> devices, dependency::TransferMode transfer_mode) const {
        dependency::DistributedGatedProductOptions result{
            structure, semantics, semantics->sources(), semantics->access().revision,
            topology(), group_slots, devices, transfer_mode, 1, 0.25f};
        result.physical_orders = {st::order_id{1701, 1}, st::order_id{1702, 1}};
        result.atoms = {state_home_atom_ids(), state_projection_atom_ids()};
        result.physical_structures = {
            st::structure_id{2701, 1}, st::structure_id{2702, 1}};
        return result;
    }

private:
    void make_topology() {
        groups[0] = {0, coordinates, 0, 1, 0, 0, 1701};
        groups[1] = {coordinates, coordinates, 1, 1, 0, 0, 1702};
        for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate) {
            arguments[coordinate] = coordinate;
            arguments[coordinates + coordinate] = coordinates - 1 - coordinate;
        }
    }

    void make_structure_and_values() {
        std::vector<std::vector<std::uint32_t>> physical_features_by_row(
            1, std::vector<std::uint32_t>(coordinates));
        std::iota(physical_features_by_row[0].begin(),
                  physical_features_by_row[0].end(), std::uint32_t{0});

        // The packed buffer uses a deliberately non-identity physical-feature map.
        // Values are written by canonical coordinate so the operation still sees
        // a complete vector of exact FP16 ones through the prepared view.
        std::vector<std::uint64_t> canonical_features(coordinates);
        for (std::size_t physical = 0; physical < coordinates; ++physical)
            canonical_features[physical] = (13 * physical) % coordinates;
        auto host_structure = block_masked::Structure::from_rows(
            coordinates, physical_features_by_row, 32, 1,
            std::move(canonical_features));
        structure = std::make_shared<const block_masked::Structure>(
            std::move(host_structure));

        packed_values.resize(structure->stored_values());
        const auto one = one_as_fp16_bits();
        for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate) {
            const auto physical = structure->physical_feature(coordinate);
            packed_values[structure->value_offset(0, physical)] = one;
        }
    }

    void make_semantics() {
        const st::domain_id state_domain{1501, 0};
        const st::order_id state_order{1502, 0};
        const st::domain_id regulator_domain{1511, 0};
        const st::order_id regulator_order{1512, 0};
        const st::domain_id parameter_domain{1521, 0};
        const st::order_id parameter_order{1522, 0};

        const auto source = [](std::uint64_t identity,
                              st::domain_id domain,
                              st::order_id order,
                              std::uint64_t extent) {
            return geometry::SourceDescriptor{
                {{identity, 0}, domain, {identity + 1000, 0}},
                {domain, order, extent}};
        };

        const mechanism::GatedProductSourceBindings bindings{
            source(1601, state_domain, state_order, coordinates),
            source(1602, regulator_domain, regulator_order, 2),
            source(1603, parameter_domain, parameter_order, 1),
            source(1604, parameter_domain, parameter_order, 1),
            source(1605, parameter_domain, parameter_order, 1),
            source(1606, state_domain, state_order, coordinates)};
        const geometry::Applicability applicability{
            {1611, 0}, {1612, 0}, {1613, 0}, {1614, 0}, {1}};
        semantics = std::make_shared<const mechanism::PreparedGatedProductBinding>(
            topology(), bindings, applicability, geometry::foundation_id{1621, 0},
            mechanism::operation_id{1622, 0}, mechanism::access_contract_id{1623, 0}, 1);
    }
};

void run_mode(dependency::TransferMode mode, std::array<int, 2> devices) {
    Distributed32Fixture fixture;
    dependency::PreparedDistributedGatedProduct runner(
        fixture.options(devices, mode), fixture.packed_values, fixture.regulators,
        fixture.theta, fixture.phi, fixture.tau);

    auto tape = runner.forward();
    const auto actual_values = runner.values(tape);
    require(actual_values.size() == coordinates,
            "32-way distributed output has one batch of 32 coordinates");

    // Closed-form values for products of 32 exact ones: u=1 + .5*.5 - .5=.75,
    // so the shared gate is .75^2=.5625. Both output incidences target coord 0.
    near(actual_values[0], 0.425f,
         "overlapping 32-way outputs accumulate over the baseline at coordinate zero");
    for (std::size_t coordinate = 1; coordinate < coordinates; ++coordinate)
        near(actual_values[coordinate], -0.25f,
             "coordinates outside the shared output retain the baseline");

    const auto forward_stats = runner.stats();
    require(forward_stats.projected_participant_values == coordinates,
            "the remote 32-argument group projects each unique participant coordinate once");

    std::array<float, coordinates> cotangents{};
    cotangents[0] = 1.0f;
    runner.backward(tape, cotangents);
    const auto gradient = runner.gradients(tape);
    require(gradient.dx.size() == coordinates,
            "32-way VJP returns all state-coordinate gradients");

    // Each argument derivative is theta*g times its output scale. The two
    // groups therefore add .225 + .45 at every coordinate; coord 0 also gets
    // the -beta baseline derivative.
    near(gradient.dx[0], 0.425f,
         "overlapping 32-way participant VJPs include the baseline at coordinate zero");
    for (std::size_t coordinate = 1; coordinate < coordinates; ++coordinate)
        near(gradient.dx[coordinate], 0.675f,
             "all 32 distinct participant coordinates receive both group contributions");

    require(gradient.dz.size() == 2,
            "one condition returns its two regulator-channel gradients");
    near(gradient.dtheta[0], 1.6875f,
         "shared parameter gradient sums both 32-way group contributions");
    near(gradient.dz[0], 1.8f,
         "regulator z0 gradient sums both overlapping output contributions");
    near(gradient.dz[1], 0.9f,
         "regulator z1 gradient sums both overlapping output contributions");
    near(gradient.dphi[0], 0.9f,
         "shared phi gradient sums both 32-way group contributions");
    near(gradient.dtau[0], -1.8f,
         "shared tau gradient sums both 32-way group contributions");

    tape.wait();
    std::cout << "32-way mode="
              << (mode == dependency::TransferMode::staged
                      ? "staged" : "peer_if_supported")
              << " projected_participant_values="
              << forward_stats.projected_participant_values << '\n';
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
        std::cerr << "distributed 32-way test requires visible CUDA devices\n";
        return 1;
    }
    if (!validate_devices(devices, device_count))
        return 2;

    try {
        run_mode(dependency::TransferMode::staged, devices);
        run_mode(dependency::TransferMode::peer_if_supported, devices);
    } catch (const std::exception& error) {
        std::cerr << "distributed 32-way contract failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
