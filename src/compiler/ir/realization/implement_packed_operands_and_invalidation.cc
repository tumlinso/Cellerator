#include <Cellerator/compiler/ir/realization/implement_packed_operands_and_invalidation_v1.hh>

#include <algorithm>
#include <set>

namespace cellerator::compiler::ir::realization::v1 {
namespace {
packed_operand_status_v1 fail(packed_operand_status_v1 status, std::string* error,
    const char* message) noexcept { if (error) *error = message; return status; }
bool power_of_two(std::uint32_t value) noexcept {
    return value && (value & (value - 1u)) == 0u;
}
} // namespace

packed_operand_status_v1 validate_packed_operand_v1(
    const packed_operand_v1& operand, std::string* error) noexcept {
    if (!valid(operand.identity) || !valid(operand.source_identity) ||
        !valid(operand.pack_operation))
        return fail(packed_operand_status_v1::invalid_identity, error,
            "operand, source, and pack-operation identities are required");
    if (!operand.element_count || operand.packed_element_count < operand.element_count)
        return fail(packed_operand_status_v1::invalid_shape, error,
            "packed shape must contain every source element");
    if (!power_of_two(operand.alignment))
        return fail(packed_operand_status_v1::invalid_alignment, error,
            "alignment must be a nonzero power of two");
    std::set<std::uint64_t> logical, physical;
    for (const auto& position : operand.value_positions) {
        if (position.logical_value >= operand.element_count ||
            position.physical_position >= operand.packed_element_count ||
            !logical.insert(position.logical_value).second ||
            !physical.insert(position.physical_position).second)
            return fail(packed_operand_status_v1::invalid_map, error,
                "value positions must be an in-range injection");
    }
    if (logical.size() != operand.element_count)
        return fail(packed_operand_status_v1::invalid_map, error,
            "every source element requires a value position");
    std::vector<bool> holes(operand.packed_element_count, false);
    std::uint64_t hole_count = 0u;
    for (const auto& hole : operand.padding_holes) {
        if (!hole.count || hole.begin >= operand.packed_element_count ||
            hole.count > operand.packed_element_count - hole.begin)
            return fail(packed_operand_status_v1::invalid_padding, error, "invalid padding hole");
        for (std::uint64_t i = hole.begin; i < hole.begin + hole.count; ++i) {
            if (holes[i] || physical.count(i))
                return fail(packed_operand_status_v1::invalid_padding, error,
                    "padding holes overlap values or each other");
            holes[i] = true;
            ++hole_count;
        }
    }
    if (operand.element_count + hole_count != operand.packed_element_count)
        return fail(packed_operand_status_v1::invalid_padding, error,
            "padding holes must explain all packed capacity");
    if (error) error->clear();
    return packed_operand_status_v1::ready;
}

packed_operand_status_v1 packed_operand_readiness_v1(
    const packed_operand_v1& operand, std::uint64_t current_generation,
    std::string* error) noexcept {
    const auto status = validate_packed_operand_v1(operand, error);
    if (status != packed_operand_status_v1::ready) return status;
    if (operand.persistence == persistence_horizon_v1::value_generation &&
        operand.source_generation != current_generation)
        return fail(packed_operand_status_v1::stale_generation, error,
            "packed operand was built for a stale value generation");
    if (error) error->clear();
    return packed_operand_status_v1::ready;
}
} // namespace cellerator::compiler::ir::realization::v1
