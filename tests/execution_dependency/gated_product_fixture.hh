#pragma once

#include "../native_data/reference/fixture.hh"

#include <Cellerator/execution/dependency/distributed.cuh>
#include <Cellerator/geometry/dependency/applicability.hh>
#include <Cellerator/ops/mechanism/gated_product_binding.hh>
#include <Cellerator/tensor/block_masked/structure.hh>

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace cellerator::test::execution_dependency {

namespace reference = cellerator::test::native_data_reference;
namespace geometry = cellerator::geometry::dependency;
namespace mechanism = cellerator::ops::mechanism;
namespace block_masked = cellerator::tensor::block_masked;

inline constexpr std::size_t fixture_batch = 2;
inline constexpr std::size_t fixture_coordinates = 32;
inline constexpr std::size_t fixture_conditions = 1;
inline constexpr std::size_t fixture_parameters = 1;

namespace packed_program_identity {
inline constexpr std::uint64_t state_home_namespace = 0x43504750484f4d45ull;
inline constexpr std::uint64_t state_projection_namespace = 0x43504750524f4a31ull;

inline execution::dependency::PackedAtomIdentities state_home() noexcept {
    return {{state_home_namespace, 1}, {state_home_namespace, 2},
            {state_home_namespace, 3}, {state_home_namespace, 4},
            {state_home_namespace, 5}};
}

inline execution::dependency::PackedAtomIdentities state_projection() noexcept {
    return {{state_projection_namespace, 1}, {state_projection_namespace, 2},
            {state_projection_namespace, 3}, {state_projection_namespace, 4},
            {state_projection_namespace, 5}};
}
} // namespace packed_program_identity

inline std::uint16_t fp16_bits(float value) noexcept {
    const __half converted = __float2half_rn(value);
    std::uint16_t bits = 0;
    static_assert(sizeof(converted) == sizeof(bits));
    std::memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

// Two-row, 32-feature caller fixture for the exact four-group reference. The
// first row is active and the second is certified inactive. Logical features
// map through a full, non-identity physical permutation; zeros remain allocated.
// The 27 unused coordinates still exercise baseline ownership over the complete
// state extent without changing the reference module result.
struct GatedProductFixture final {
    std::array<mechanism::GatedProductGroup, reference::mechanisms.size()> groups{};
    std::array<std::uint64_t, 9> arguments{};
    std::array<mechanism::OutputIncidence, 5> outputs{};
    std::array<std::uint8_t, reference::mechanisms.size()> group_slots{{0, 1, 1, 1}};

    std::shared_ptr<const block_masked::Structure> structure;
    std::shared_ptr<const mechanism::PreparedGatedProductBinding> semantics;
    std::vector<float> x;
    std::vector<float> regulators;
    std::vector<float> theta;
    std::vector<float> phi;
    std::vector<float> tau;
    std::vector<std::uint16_t> packed_values;
    std::vector<float> cotangents;

    explicit GatedProductFixture(
        bool all_inactive = false,
        bool row_identity_regression = false,
        cellerator::structure::order_id state_order = {1002, 0}) {
        make_topology();
        make_structure_and_values(all_inactive, row_identity_regression);
        make_semantics(state_order);
    }

    mechanism::GatedProductTopologyView topology() const noexcept {
        return {groups.data(), groups.size(), arguments.data(), arguments.size(),
                outputs.data(), outputs.size(), fixture_coordinates,
                fixture_conditions, fixture_parameters};
    }

    execution::dependency::DistributedGatedProductOptions options(
        std::array<int, 2> devices,
        execution::dependency::TransferMode transfer_mode) const {
        execution::dependency::DistributedGatedProductOptions result{
            structure, semantics, semantics->sources(), semantics->access().revision,
            topology(), group_slots, devices, transfer_mode, 2,
            static_cast<float>(reference::active_values.beta)};
        result.physical_orders = {
            cellerator::structure::order_id{1401, 1},
            cellerator::structure::order_id{1402, 1}};
        result.atoms = {packed_program_identity::state_home(),
                        packed_program_identity::state_projection()};
        result.physical_structures = {
            cellerator::structure::structure_id{1801, 1},
            cellerator::structure::structure_id{1802, 1}};
        return result;
    }

    std::vector<geometry::SourceDescriptor> source_catalog() const {
        const auto descriptors = semantics->sources();
        return {descriptors.begin(), descriptors.end()};
    }

private:
    void make_topology() {
        std::size_t argument_begin = 0;
        std::size_t output_begin = 0;
        for (std::size_t index = 0; index < reference::mechanisms.size(); ++index) {
            const auto& source = reference::mechanisms[index];
            auto& destination = groups[index];
            destination.argument_begin = argument_begin;
            destination.argument_count = source.arity;
            destination.output_begin = output_begin;
            destination.parameter_index = source.parameter_index;
            destination.condition_index = source.condition_index;
            destination.logical_id = source.logical_id;
            for (std::size_t slot = 0; slot < source.arity; ++slot)
                arguments[argument_begin + slot] = source.ordered_inputs[slot];
            argument_begin += source.arity;

            for (std::size_t coordinate = 0; coordinate < source.output_scales.size();
                 ++coordinate) {
                if (source.output_scales[coordinate] == 0.0)
                    continue;
                outputs[output_begin++] = {
                    coordinate, static_cast<float>(source.output_scales[coordinate])};
                ++destination.output_count;
            }
        }
    }

    void make_structure_and_values(bool all_inactive, bool row_identity_regression) {
        std::vector<std::vector<std::uint32_t>> physical_features_by_row(
            fixture_batch, std::vector<std::uint32_t>(fixture_coordinates));
        for (auto& row : physical_features_by_row)
            std::iota(row.begin(), row.end(), std::uint32_t{0});

        // physical position p stores canonical coordinate (13*p) mod 32.
        // 13 is coprime to 32, so this is a complete, deliberately permuted map.
        std::vector<std::uint64_t> canonical_features(fixture_coordinates);
        for (std::size_t physical = 0; physical < fixture_coordinates; ++physical)
            canonical_features[physical] = (13 * physical) % fixture_coordinates;

        std::vector<std::uint64_t> canonical_rows;
        if (row_identity_regression)
            canonical_rows = {200, 100};
        auto host_structure = block_masked::Structure::from_rows(
            fixture_coordinates, physical_features_by_row, 32, 2,
            std::move(canonical_features), std::move(canonical_rows));
        structure = std::make_shared<const block_masked::Structure>(
            std::move(host_structure));

        x.assign(fixture_batch * fixture_coordinates, 0.0f);
        for (std::size_t row = 0; row < fixture_batch; ++row)
            for (std::size_t coordinate = 0; coordinate < reference::active_values.x.size();
                 ++coordinate)
                x[row * fixture_coordinates + coordinate] =
                    static_cast<float>(reference::active_values.x[coordinate]);
        if (row_identity_regression) {
            constexpr std::array<float, 5> second_row{{4.0f, 1.0f, 1.5f, 2.0f, 0.5f}};
            for (std::size_t coordinate = 0; coordinate < second_row.size(); ++coordinate)
                x[fixture_coordinates + coordinate] = second_row[coordinate];
        }

        regulators.resize(fixture_batch * fixture_conditions * 2);
        if (all_inactive) {
            for (std::size_t row = 0; row < fixture_batch; ++row) {
                regulators[row * 2] = static_cast<float>(reference::inactive_values.z0);
                regulators[row * 2 + 1] = static_cast<float>(reference::inactive_values.z1);
            }
        } else {
            regulators[0] = static_cast<float>(reference::active_values.z0);
            regulators[1] = static_cast<float>(reference::active_values.z1);
            regulators[2] = static_cast<float>(reference::inactive_values.z0);
            regulators[3] = static_cast<float>(reference::inactive_values.z1);
        }
        theta = {static_cast<float>(reference::active_values.theta)};
        phi = {static_cast<float>(reference::active_values.phi)};
        tau = {static_cast<float>(reference::active_values.tau)};
        cotangents.assign(fixture_batch * fixture_coordinates, 0.0f);
        for (std::size_t row = 0; row < fixture_batch; ++row)
            for (std::size_t coordinate = 0; coordinate < reference::loss_seed.size();
                 ++coordinate)
                cotangents[row * fixture_coordinates + coordinate] =
                    static_cast<float>(reference::loss_seed[coordinate]);
        if (row_identity_regression) {
            constexpr std::array<float, 5> second_row_seed{{0.5f, 1.0f, 0.25f, 0.0f, 1.0f}};
            for (std::size_t coordinate = 0; coordinate < second_row_seed.size(); ++coordinate)
                cotangents[fixture_coordinates + coordinate] = second_row_seed[coordinate];
        }

        packed_values.resize(structure->stored_values());
        auto sorted_rows = structure->canonical_rows();
        std::sort(sorted_rows.begin(), sorted_rows.end());
        for (std::size_t physical_row = 0; physical_row < fixture_batch; ++physical_row) {
            const auto canonical_row = structure->canonical_rows()[physical_row];
            const auto batch_row = static_cast<std::size_t>(
                std::lower_bound(sorted_rows.begin(), sorted_rows.end(), canonical_row)
                - sorted_rows.begin());
            for (std::size_t coordinate = 0; coordinate < fixture_coordinates; ++coordinate) {
                const auto physical = structure->physical_feature(coordinate);
                const auto offset = structure->value_offset(physical_row, physical);
                packed_values[offset] = fp16_bits(x[batch_row * fixture_coordinates + coordinate]);
            }
        }
    }

    void make_semantics(cellerator::structure::order_id state_order) {
        const structure::domain_id state_domain{1001, 0};
        const structure::domain_id regulator_domain{1011, 0};
        const structure::order_id regulator_order{1012, 0};
        const structure::domain_id parameter_domain{1021, 0};
        const structure::order_id parameter_order{1022, 0};

        const auto source = [](std::uint64_t identity,
                               structure::domain_id domain,
                               structure::order_id order,
                               std::uint64_t extent) {
            return geometry::SourceDescriptor{
                {{identity, 0}, domain, {identity + 1000, 0}},
                {domain, order, extent}};
        };

        mechanism::GatedProductSourceBindings bindings{
            source(1101, state_domain, state_order, fixture_coordinates),
            source(1102, regulator_domain, regulator_order, fixture_conditions * 2),
            source(1103, parameter_domain, parameter_order, fixture_parameters),
            source(1104, parameter_domain, parameter_order, fixture_parameters),
            source(1105, parameter_domain, parameter_order, fixture_parameters),
            source(1106, state_domain, state_order, fixture_coordinates)};
        const geometry::Applicability applicability{
            {1201, 0}, {1202, 0}, {1203, 0}, {1204, 0}, {1}};
        semantics = std::make_shared<const mechanism::PreparedGatedProductBinding>(
            topology(), bindings, applicability, geometry::foundation_id{1301, 0},
            mechanism::operation_id{1302, 0}, mechanism::access_contract_id{1303, 0}, 1);
    }
};

} // namespace cellerator::test::execution_dependency
