#pragma once

#include <Cellerator/compiler/ir/realization/implement_projection_contracts_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::ir::realization::v1 {

enum class packed_operand_role_v1 : std::uint8_t { input = 1u, output, values };
enum class persistence_horizon_v1 : std::uint8_t {
    invocation = 1u, value_generation, structure_epoch, module,
};

struct padding_hole_v1 {
    std::uint64_t begin = 0u;
    std::uint64_t count = 0u;
};

struct packed_operand_v1 {
    stable_identity_v1 identity{};
    stable_identity_v1 source_identity{};
    stable_identity_v1 pack_operation{};
    packed_operand_role_v1 role = packed_operand_role_v1::input;
    persistence_horizon_v1 persistence = persistence_horizon_v1::invocation;
    std::uint64_t source_generation = 0u;
    std::uint64_t element_count = 0u;
    std::uint64_t packed_element_count = 0u;
    std::uint32_t alignment = 1u;
    std::vector<value_position_v1> value_positions;
    std::vector<padding_hole_v1> padding_holes;
};

enum class packed_operand_status_v1 : std::uint8_t {
    ready = 0u, invalid_identity, invalid_shape, invalid_alignment,
    invalid_map, invalid_padding, stale_generation,
};

[[nodiscard]] packed_operand_status_v1 validate_packed_operand_v1(
    const packed_operand_v1& operand, std::string* error = nullptr) noexcept;
[[nodiscard]] packed_operand_status_v1 packed_operand_readiness_v1(
    const packed_operand_v1& operand, std::uint64_t current_generation,
    std::string* error = nullptr) noexcept;

} // namespace cellerator::compiler::ir::realization::v1
