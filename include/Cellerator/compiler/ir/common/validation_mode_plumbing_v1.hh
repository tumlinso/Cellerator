#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace cellerator::compiler::ir {

enum class trust_mode : std::uint8_t { verified, checked, trusted, unsafe, unchecked };
enum class pipeline_stage : std::uint8_t { parser, builder, pass, serializer, backend };
struct validation_envelope {
    trust_mode mode{trust_mode::checked};
    pipeline_stage stage{pipeline_stage::parser};
    bool structurally_parseable{};
    bool semantic_checks_passed{};
};
struct continuation_decision { bool continue_pipeline{}; std::string diagnostic; };

validation_envelope advance_validation(
    validation_envelope envelope, pipeline_stage next) noexcept;
continuation_decision evaluate_validation(const validation_envelope &envelope);
std::uint8_t serialize_trust_mode(trust_mode mode) noexcept;
bool deserialize_trust_mode(std::uint8_t encoded, trust_mode &mode) noexcept;
std::string_view trust_mode_name(trust_mode mode) noexcept;

} // namespace cellerator::compiler::ir
