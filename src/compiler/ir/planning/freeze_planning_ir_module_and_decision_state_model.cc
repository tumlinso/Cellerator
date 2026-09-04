#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>

#include <cstring>

namespace cellerator::compiler::ir::planning::v1 {
namespace {

struct planning_binary_header_v1 {
    char magic[8];
    std::uint32_t schema_version;
    std::uint32_t decision_count;
    planning_identity_v1 module;
};

constexpr char planning_magic_v1[8] = {'C', 'E', 'P', 'L', 'A', 'N', '1', '\0'};

bool zero(planning_identity_v1 value) noexcept {
    return value.low == 0u && value.high == 0u;
}

bool same(planning_identity_v1 left, planning_identity_v1 right) noexcept {
    return left.low == right.low && left.high == right.high;
}

bool valid_state(decision_state_v1 state) noexcept {
    return static_cast<std::uint8_t>(state) <=
           static_cast<std::uint8_t>(decision_state_v1::fallback);
}

}  // namespace

planning_ir_status_v1 validate_planning_ir_module_v1(
    const planning_ir_module_v1 &module) noexcept {
    if (module.schema_version != planning_ir_schema_version_v1) {
        return planning_ir_status_v1::unsupported_schema;
    }
    if (module.reserved != 0u || module.reserved_count != 0u) {
        return planning_ir_status_v1::nonzero_reserved;
    }
    if (zero(module.module) || (module.decision_count != 0u && module.decisions == nullptr)) {
        return planning_ir_status_v1::invalid_argument;
    }
    for (std::uint32_t index = 0u; index != module.decision_count; ++index) {
        const auto &decision = module.decisions[index];
        if (zero(decision.decision) || zero(decision.candidate) ||
            zero(decision.source_operation)) {
            return planning_ir_status_v1::invalid_identity;
        }
        if (!valid_state(decision.state)) {
            return planning_ir_status_v1::invalid_state;
        }
        if (decision.reserved16 != 0u) {
            return planning_ir_status_v1::nonzero_reserved;
        }
        for (std::uint32_t other = 0u; other != index; ++other) {
            if (same(decision.decision, module.decisions[other].decision)) {
                return planning_ir_status_v1::duplicate_decision;
            }
        }
    }
    return planning_ir_status_v1::ok;
}

planning_ir_status_v1 serialize_planning_decisions_v1(
    const planning_ir_module_v1 &module, void *destination,
    std::size_t capacity, std::size_t *written) noexcept {
    if (written == nullptr) {
        return planning_ir_status_v1::invalid_argument;
    }
    const auto status = validate_planning_ir_module_v1(module);
    if (status != planning_ir_status_v1::ok) {
        return status;
    }
    const std::size_t required = sizeof(planning_binary_header_v1) +
                                 sizeof(decision_record_v1) * module.decision_count;
    *written = required;
    if (destination == nullptr || capacity < required) {
        return planning_ir_status_v1::insufficient_capacity;
    }
    planning_binary_header_v1 header{};
    std::memcpy(header.magic, planning_magic_v1, sizeof(header.magic));
    header.schema_version = module.schema_version;
    header.decision_count = module.decision_count;
    header.module = module.module;
    std::memcpy(destination, &header, sizeof(header));
    std::memcpy(static_cast<std::byte *>(destination) + sizeof(header),
                module.decisions, sizeof(decision_record_v1) * module.decision_count);
    return planning_ir_status_v1::ok;
}

planning_ir_status_v1 deserialize_planning_decisions_v1(
    const void *source, std::size_t source_bytes, decision_record_v1 *decisions,
    std::uint32_t capacity, planning_ir_module_v1 *module) noexcept {
    if (source == nullptr || module == nullptr || source_bytes < sizeof(planning_binary_header_v1)) {
        return planning_ir_status_v1::invalid_argument;
    }
    planning_binary_header_v1 header{};
    std::memcpy(&header, source, sizeof(header));
    if (std::memcmp(header.magic, planning_magic_v1, sizeof(header.magic)) != 0) {
        return planning_ir_status_v1::malformed_binary;
    }
    if (header.schema_version != planning_ir_schema_version_v1) {
        return planning_ir_status_v1::unsupported_schema;
    }
    const std::size_t required = sizeof(header) +
                                 sizeof(decision_record_v1) * header.decision_count;
    if (source_bytes != required) {
        return planning_ir_status_v1::malformed_binary;
    }
    if (header.decision_count > capacity ||
        (header.decision_count != 0u && decisions == nullptr)) {
        return planning_ir_status_v1::insufficient_capacity;
    }
    std::memcpy(decisions, static_cast<const std::byte *>(source) + sizeof(header),
                sizeof(decision_record_v1) * header.decision_count);
    *module = {header.schema_version, 0u, header.module, decisions,
               header.decision_count, 0u};
    return validate_planning_ir_module_v1(*module);
}

}  // namespace cellerator::compiler::ir::planning::v1
