#include <Cellerator/compiler/ir/common/implement_validation_mode_plumbing_v1.hh>

namespace cellerator::compiler::ir {

validation_envelope advance_validation(
    validation_envelope envelope, pipeline_stage next) noexcept {
    envelope.stage = next;
    return envelope;
}

continuation_decision evaluate_validation(const validation_envelope &envelope) {
    if (!envelope.structurally_parseable)
        return {false, "IR is not structurally parseable"};
    if (envelope.mode == trust_mode::verified && !envelope.semantic_checks_passed)
        return {false, "verified IR lacks proof of semantic validity"};
    if (envelope.mode == trust_mode::checked && !envelope.semantic_checks_passed)
        return {false, "checked IR failed semantic validation"};
    if (envelope.mode == trust_mode::unsafe)
        return {true, "unsafe IR bypasses semantic validation"};
    if (envelope.mode == trust_mode::unchecked)
        return {true, "unchecked IR records no semantic trust"};
    return {true, {}};
}

std::uint8_t serialize_trust_mode(trust_mode mode) noexcept {
    return static_cast<std::uint8_t>(mode);
}

bool deserialize_trust_mode(std::uint8_t encoded, trust_mode &mode) noexcept {
    if (encoded > static_cast<std::uint8_t>(trust_mode::unchecked))
        return false;
    mode = static_cast<trust_mode>(encoded);
    return true;
}

std::string_view trust_mode_name(trust_mode mode) noexcept {
    switch (mode) {
    case trust_mode::verified: return "verified";
    case trust_mode::checked: return "checked";
    case trust_mode::trusted: return "trusted";
    case trust_mode::unsafe: return "unsafe";
    case trust_mode::unchecked: return "unchecked";
    }
    return {};
}

} // namespace cellerator::compiler::ir
